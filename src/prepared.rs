//! Prepared kernels (`malleus::prepared`; deliberately not re-exported at the crate root, so
//! `use malleus::*` consumers see no new names): an [`Executable`] flattened once into register bytecode, executed with
//! no per-call heap allocation, one point at a time or columnar over a batch of points.
//!
//! [`PreparedKernel::new`] resolves every operand to a binding slot, folds each indexing map,
//! dense layout and buffer region into one affine address (`base + sum(stride * coordinate)`),
//! proves every address of the iteration domain lies inside the operand's region, and compiles
//! the ordered statements into two programs over a reusable register file:
//!
//! - a *lazy* program that follows the reference interpreter literally: selects and `and`/`or`
//!   predicates jump, so only the taken branch is evaluated;
//! - an *eager* program for columnar batches: both branches are evaluated and blended per lane.
//!
//! Both programs perform exactly the interpreter's IEEE operations in the interpreter's order
//! (they call the same scalar operation functions, so `f32` kernels round every operation to
//! `f32` and widen on store). Eager evaluation is observationally identical to lazy evaluation
//! because, once addresses are proven in range, expressions are pure and total: a load has no
//! effect and cannot fail, and floating-point operations do not trap. A validated executable
//! therefore never fails at run time; every error is a binding error reported before the first
//! effect. See `EXECUTION.md` for the binding and batch contract.

use std::error::Error;
use std::fmt;

use crate::interpreter::{cast, compare};
use crate::policy::{MathLibrary, binary, fma, unary};
use crate::{
    AccessMode, BinaryOp, BufferBinding, CompareOp, CompositionBinding, CompositionExecutionError,
    CompositionTarget, Executable, ExecutableComposition, ExecutableModule, ExecutionError,
    LocalExecutableRunner, OperandId, Predicate, ReductionOp, ReductionOrder, ScalarExpr,
    ScalarType, StageOperand, Statement, StructuredKernel, UnaryOp,
};

/// Lanes per columnar block of [`PreparedKernel::run_batch`]: each bytecode instruction runs
/// over up to this many batch points before the next instruction is dispatched.
pub const BATCH_BLOCK: usize = 32;

/// A typed refusal to prepare an executable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrepareError {
    /// An operand's affine address is not representable or leaves the operand's region at
    /// some point of the iteration domain. Unreachable for validated kernels; kept typed.
    AddressOutOfRange { kernel: String, operand: usize },
    /// The bytecode cannot index this many instructions or registers.
    ProgramTooLarge { kernel: String },
}

impl fmt::Display for PrepareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kernel preparation failed: {self:?}")
    }
}
impl Error for PrepareError {}

/// A binding error of a prepared call. Every check runs before the first effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallError {
    /// The number of read bindings differs from [`PreparedKernel::read_operands`].
    ReadCount { expected: usize, actual: usize },
    /// The number of writable bindings differs from [`PreparedKernel::write_operands`].
    WriteCount { expected: usize, actual: usize },
    /// Read binding `position` names another operand than the one at that position.
    ReadOperand {
        position: usize,
        expected: OperandId,
        actual: OperandId,
    },
    /// Writable binding `position` names another operand than the one at that position.
    WriteOperand {
        position: usize,
        expected: OperandId,
        actual: OperandId,
    },
    /// A buffer is shorter than `(points - 1) * stride + offset + length` of its operand.
    ShortBuffer {
        operand: OperandId,
        required: usize,
        actual: usize,
    },
    /// `(points - 1) * stride + offset + length` overflows `usize`.
    ExtentOverflow { operand: OperandId },
}

impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "prepared kernel call failed: {self:?}")
    }
}
impl Error for CallError {}

/// A `Read` operand's buffer for a prepared call.
///
/// `stride` is the element offset between consecutive batch points (0 = one buffer shared by
/// every point). It is only consulted by [`PreparedKernel::run_batch`] with more than one point.
#[derive(Clone, Copy, Debug)]
pub struct ReadBinding<'a> {
    pub operand: OperandId,
    pub values: &'a [f64],
    pub stride: usize,
}

impl<'a> ReadBinding<'a> {
    pub fn new(operand: OperandId, values: &'a [f64]) -> Self {
        Self {
            operand,
            values,
            stride: 0,
        }
    }

    pub fn strided(operand: OperandId, values: &'a [f64], stride: usize) -> Self {
        Self {
            operand,
            values,
            stride,
        }
    }
}

/// A `Write`, `ReadWrite` or `Reduce` operand's buffer for a prepared call; see [`ReadBinding`]
/// for `stride`.
#[derive(Debug)]
pub struct WriteBinding<'a> {
    pub operand: OperandId,
    pub values: &'a mut [f64],
    pub stride: usize,
}

impl<'a> WriteBinding<'a> {
    pub fn new(operand: OperandId, values: &'a mut [f64]) -> Self {
        Self {
            operand,
            values,
            stride: 0,
        }
    }

    pub fn strided(operand: OperandId, values: &'a mut [f64], stride: usize) -> Self {
        Self {
            operand,
            values,
            stride,
        }
    }
}

/// Reusable scratch for prepared calls: registers, predicate flags and coordinates.
///
/// A state may serve any number of prepared kernels; it grows (allocates) only when a call
/// needs more scratch than it holds. [`PreparedKernel::state`] returns one already large
/// enough, after which calls on that kernel never allocate.
#[derive(Clone, Debug, Default)]
pub struct PreparedState {
    registers: Vec<f64>,
    flags: Vec<bool>,
    coordinates: Vec<usize>,
}

impl PreparedState {
    pub fn new() -> Self {
        Self::default()
    }

    fn reserve(&mut self, registers: usize, flags: usize, rank: usize) {
        if self.registers.len() < registers {
            self.registers.resize(registers, 0.0);
        }
        if self.flags.len() < flags {
            self.flags.resize(flags, false);
        }
        if self.coordinates.len() < rank {
            self.coordinates.resize(rank, 0);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Read(usize),
    Write(usize),
}

#[derive(Clone, Copy, Debug)]
enum StoreMode {
    Assign,
    Reduce(BinaryOp),
}

/// Affine address `base + sum(stride * coordinate[axis])` over `terms[start..end]`, evaluated
/// in wrapping arithmetic: the true address is proven in range, so the wrapped sum equals it.
#[derive(Clone, Copy, Debug)]
struct Access {
    base: usize,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug)]
enum Instr {
    Const {
        dst: u32,
        value: f64,
    },
    Index {
        dst: u32,
        axis: u32,
    },
    LoadRead {
        dst: u32,
        slot: u32,
        access: u32,
    },
    LoadWrite {
        dst: u32,
        slot: u32,
        access: u32,
    },
    Move {
        dst: u32,
        src: u32,
    },
    Unary {
        op: UnaryOp,
        dst: u32,
        src: u32,
    },
    Binary {
        op: BinaryOp,
        dst: u32,
        lhs: u32,
        rhs: u32,
    },
    Fma {
        dst: u32,
        lhs: u32,
        rhs: u32,
        addend: u32,
    },
    Compare {
        op: CompareOp,
        dst: u32,
        lhs: u32,
        rhs: u32,
    },
    FlagConst {
        dst: u32,
        value: bool,
    },
    FlagMove {
        dst: u32,
        src: u32,
    },
    Not {
        dst: u32,
        src: u32,
    },
    /// Eager programs only.
    And {
        dst: u32,
        lhs: u32,
        rhs: u32,
    },
    /// Eager programs only.
    Or {
        dst: u32,
        lhs: u32,
        rhs: u32,
    },
    /// Eager programs only.
    Blend {
        dst: u32,
        flag: u32,
        if_true: u32,
        if_false: u32,
    },
    /// Lazy programs only.
    JumpIfFalse {
        flag: u32,
        target: u32,
    },
    /// Lazy programs only.
    JumpIfTrue {
        flag: u32,
        target: u32,
    },
    /// Lazy programs only.
    Jump {
        target: u32,
    },
    Store {
        slot: u32,
        access: u32,
        src: u32,
        mode: StoreMode,
    },
}

#[derive(Clone, Debug)]
struct Program {
    code: Vec<Instr>,
    registers: usize,
    flags: usize,
}

/// A kernel compiled once into register bytecode over a reusable register file.
///
/// Preparation proves every affine address in range, so execution cannot fail after the
/// binding checks; the point program follows the interpreter's lazy branches literally and
/// the columnar program blends both branches per lane. `EXECUTION.md` states the full
/// binding, batch and exactness contract.
#[derive(Clone, Debug)]
pub struct PreparedKernel {
    name: String,
    scalar_type: ScalarType,
    math: MathLibrary,
    extents: Vec<usize>,
    order: Vec<usize>,
    empty: bool,
    slots: Vec<Slot>,
    required: Vec<usize>,
    reads: Vec<OperandId>,
    writes: Vec<OperandId>,
    write_region_lengths: Vec<usize>,
    accesses: Vec<Access>,
    terms: Vec<(usize, usize)>,
    lazy: Program,
    eager: Program,
}

trait Precision {
    const TYPE: ScalarType;
}
struct F32;
struct F64;
impl Precision for F32 {
    const TYPE: ScalarType = ScalarType::F32;
}
impl Precision for F64 {
    const TYPE: ScalarType = ScalarType::F64;
}

/// Operand storage seen by the point executor.
trait Memory {
    fn read(&self, slot: usize, index: usize) -> f64;
    fn read_writable(&self, slot: usize, index: usize) -> f64;
    fn write(&mut self, slot: usize, index: usize, value: f64);
}

/// Caller bindings shifted to one batch point.
struct PointMemory<'m, 'r, 'w> {
    reads: &'m [ReadBinding<'r>],
    writes: &'m mut [WriteBinding<'w>],
    point: usize,
}

impl Memory for PointMemory<'_, '_, '_> {
    #[inline(always)]
    fn read(&self, slot: usize, index: usize) -> f64 {
        let binding = &self.reads[slot];
        binding.values[self.point * binding.stride + index]
    }
    #[inline(always)]
    fn read_writable(&self, slot: usize, index: usize) -> f64 {
        let binding = &self.writes[slot];
        binding.values[self.point * binding.stride + index]
    }
    #[inline(always)]
    fn write(&mut self, slot: usize, index: usize, value: f64) {
        let binding = &mut self.writes[slot];
        binding.values[self.point * binding.stride + index] = value;
    }
}

/// One composition stage's private operand copies, indexed by operand id.
struct StageMemory<'m> {
    buffers: &'m mut [Vec<f64>],
    reads: &'m [OperandId],
    writes: &'m [OperandId],
}

impl Memory for StageMemory<'_> {
    #[inline(always)]
    fn read(&self, slot: usize, index: usize) -> f64 {
        self.buffers[self.reads[slot].index()][index]
    }
    #[inline(always)]
    fn read_writable(&self, slot: usize, index: usize) -> f64 {
        self.buffers[self.writes[slot].index()][index]
    }
    #[inline(always)]
    fn write(&mut self, slot: usize, index: usize, value: f64) {
        self.buffers[self.writes[slot].index()][index] = value;
    }
}

impl PreparedKernel {
    /// Validate addressing and compile both programs. Uses the executable's loop order only
    /// when its numeric policy declares a schedule-defined reduction order, exactly as the
    /// interpreter does.
    pub fn new(executable: &Executable) -> Result<Self, PrepareError> {
        let kernel = executable.kernel().as_kernel();
        let extents = kernel.iteration_domain.extents.clone();
        let order = if kernel.numeric_policy.reduction_order == ReductionOrder::Canonical {
            (0..extents.len()).collect()
        } else {
            executable
                .schedule()
                .loop_order
                .iter()
                .map(|axis| axis.index())
                .collect()
        };
        let empty = extents.contains(&0);
        let out_of_range = |operand| PrepareError::AddressOutOfRange {
            kernel: kernel.name.clone(),
            operand,
        };
        let mut slots = Vec::with_capacity(kernel.operands.len());
        let mut required = Vec::with_capacity(kernel.operands.len());
        let (mut reads, mut writes, mut write_region_lengths) = (vec![], vec![], vec![]);
        for (index, operand) in kernel.operands.iter().enumerate() {
            let id = OperandId::new(index);
            required.push(
                operand
                    .region
                    .offset
                    .checked_add(operand.region.length)
                    .ok_or_else(|| out_of_range(index))?,
            );
            if operand.access == AccessMode::Read {
                slots.push(Slot::Read(reads.len()));
                reads.push(id);
            } else {
                slots.push(Slot::Write(writes.len()));
                writes.push(id);
                write_region_lengths.push(operand.region.length);
            }
        }
        let (mut accesses, mut terms) = (Vec::new(), Vec::new());
        for (index, operand) in kernel.operands.iter().enumerate() {
            let map = kernel
                .indexing_maps
                .iter()
                .find(|map| map.operand.index() == index)
                .expect("validated map");
            let mut layout_strides = vec![0i128; operand.shape.len()];
            let mut stride = 1i128;
            for dimension in &operand.layout.minor_to_major {
                layout_strides[*dimension] = stride;
                stride = stride
                    .checked_mul(operand.shape[*dimension] as i128)
                    .ok_or_else(|| out_of_range(index))?;
            }
            let mut base = operand.region.offset as i128;
            let mut axis_strides = vec![0i128; extents.len()];
            for (dimension, expr) in map.results.iter().enumerate() {
                let scale = layout_strides[dimension];
                base = (expr.constant as i128)
                    .checked_mul(scale)
                    .and_then(|term| base.checked_add(term))
                    .ok_or_else(|| out_of_range(index))?;
                for term in &expr.terms {
                    let axis = term.axis.index();
                    axis_strides[axis] = (term.coefficient as i128)
                        .checked_mul(scale)
                        .and_then(|term| axis_strides[axis].checked_add(term))
                        .ok_or_else(|| out_of_range(index))?;
                }
            }
            let start = terms.len();
            if !empty {
                let (mut lowest, mut highest) = (base, base);
                for (axis, stride) in axis_strides.iter().enumerate() {
                    let excursion = stride
                        .checked_mul((extents[axis] - 1) as i128)
                        .ok_or_else(|| out_of_range(index))?;
                    if excursion < 0 {
                        lowest += excursion;
                    } else {
                        highest += excursion;
                    }
                    if *stride != 0 && extents[axis] > 1 {
                        // Two's-complement truncation: wrapping evaluation stays exact.
                        terms.push((axis, *stride as usize));
                    }
                }
                let region = operand.region;
                if lowest < region.offset as i128
                    || highest >= region.offset as i128 + region.length as i128
                {
                    return Err(out_of_range(index));
                }
            }
            accesses.push(Access {
                base: usize::try_from(base).unwrap_or(0),
                start,
                end: terms.len(),
            });
        }
        let lazy = Compiler::compile(kernel, &slots, true);
        let eager = Compiler::compile(kernel, &slots, false);
        if lazy.code.len() >= u32::MAX as usize || eager.code.len() >= u32::MAX as usize {
            return Err(PrepareError::ProgramTooLarge {
                kernel: kernel.name.clone(),
            });
        }
        Ok(Self {
            name: kernel.name.clone(),
            scalar_type: kernel.numeric_policy.scalar_type,
            math: kernel.numeric_policy.level.math(),
            extents,
            order,
            empty,
            slots,
            required,
            reads,
            writes,
            write_region_lengths,
            accesses,
            terms,
            lazy,
            eager,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn scalar_type(&self) -> ScalarType {
        self.scalar_type
    }

    /// `Read` operands in operand-id order: the order of read bindings in every call.
    pub fn read_operands(&self) -> &[OperandId] {
        &self.reads
    }

    /// `Write`, `ReadWrite` and `Reduce` operands in operand-id order: the order of writable
    /// bindings in every call.
    pub fn write_operands(&self) -> &[OperandId] {
        &self.writes
    }

    /// Elements one point's binding of `operand` must provide: `region.offset + region.length`.
    pub fn required_len(&self, operand: OperandId) -> Option<usize> {
        self.required.get(operand.index()).copied()
    }

    /// A state large enough for every call on this kernel, so calls never allocate.
    pub fn state(&self) -> PreparedState {
        let mut state = PreparedState::new();
        self.reserve(&mut state, true);
        state
    }

    /// Checked one-shot call with the interpreter's binding contract and error values:
    /// `Interpreter::run` and this function accept and refuse exactly the same bindings.
    /// Allocates its binding tables and scratch; use [`Self::call`] on hot paths.
    pub fn run(&self, bindings: &mut [BufferBinding<'_>]) -> Result<(), ExecutionError> {
        let mut seen = vec![false; self.required.len()];
        for binding in bindings.iter() {
            let id = binding.operand.index();
            let Some(&required) = self.required.get(id) else {
                return Err(ExecutionError::InvalidBinding(id));
            };
            if seen[id] {
                return Err(ExecutionError::DuplicateBinding(id));
            }
            seen[id] = true;
            if binding.values.len() < required {
                return Err(ExecutionError::InvalidBinding(id));
            }
        }
        if let Some(id) = seen.iter().position(|present| !present) {
            return Err(ExecutionError::MissingBinding(id));
        }
        let mut by_operand: Vec<Option<&mut [f64]>> =
            (0..self.required.len()).map(|_| None).collect();
        for binding in bindings.iter_mut() {
            by_operand[binding.operand.index()] = Some(&mut *binding.values);
        }
        let mut reads = Vec::with_capacity(self.reads.len());
        let mut writes = Vec::with_capacity(self.writes.len());
        for (index, values) in by_operand.into_iter().enumerate() {
            let values = values.expect("completeness checked");
            match self.slots[index] {
                Slot::Read(_) => reads.push(ReadBinding::new(OperandId::new(index), values)),
                Slot::Write(_) => writes.push(WriteBinding::new(OperandId::new(index), values)),
            }
        }
        self.call(&mut self.state(), &reads, &mut writes)
            .expect("bindings satisfy the interpreter contract checked above");
        Ok(())
    }

    /// One reusable call: `reads` binds [`Self::read_operands`] and `writes` binds
    /// [`Self::write_operands`], position by position. Validates counts, identities and
    /// lengths, then executes with no heap allocation once `state` is large enough.
    /// Equivalent to `run_batch(state, 1, reads, writes)`; strides are not consulted.
    pub fn call(
        &self,
        state: &mut PreparedState,
        reads: &[ReadBinding<'_>],
        writes: &mut [WriteBinding<'_>],
    ) -> Result<(), CallError> {
        self.run_batch(state, 1, reads, writes)
    }

    /// Execute `points` calls. The contract is *`points` sequential calls in point order*,
    /// where call `p` sees every binding shifted by `p * stride` elements: a uniform (stride 0)
    /// writable operand is shared, so later points observe (and reductions accumulate onto)
    /// earlier points' effects, and overlapping strided windows behave as the sequential
    /// calls would.
    ///
    /// When every writable binding's stride is at least its operand's region length (written
    /// windows of distinct points are disjoint), points run columnar in blocks of
    /// [`BATCH_BLOCK`] lanes; otherwise they run one after another through the point program.
    /// Both paths are bitwise identical to sequential interpreter calls. All bindings are
    /// checked before the first effect; a call never fails after that.
    pub fn run_batch(
        &self,
        state: &mut PreparedState,
        points: usize,
        reads: &[ReadBinding<'_>],
        writes: &mut [WriteBinding<'_>],
    ) -> Result<(), CallError> {
        self.check(points, reads, writes)?;
        if points == 0 {
            return Ok(());
        }
        let columnar = points > 1
            && writes
                .iter()
                .zip(&self.write_region_lengths)
                .all(|(binding, length)| binding.stride >= *length);
        self.reserve(state, columnar);
        match (self.scalar_type, columnar) {
            (ScalarType::F64, false) => self.sequential::<F64>(state, points, reads, writes),
            (ScalarType::F32, false) => self.sequential::<F32>(state, points, reads, writes),
            (ScalarType::F64, true) => self.blocks::<F64>(state, points, reads, writes),
            (ScalarType::F32, true) => self.blocks::<F32>(state, points, reads, writes),
        }
        Ok(())
    }

    fn reserve(&self, state: &mut PreparedState, columnar: bool) {
        state.reserve(self.lazy.registers, self.lazy.flags, self.extents.len());
        if columnar {
            state.reserve(
                self.eager.registers * BATCH_BLOCK,
                self.eager.flags * BATCH_BLOCK,
                self.extents.len(),
            );
        }
    }

    /// The binding checks of [`run_batch`](Self::run_batch) (`points = 1` for
    /// [`call`](Self::call)) without executing anything. A native backend implementing the
    /// same positional batch ABI reuses them so that it accepts and refuses exactly the
    /// bindings this kernel does.
    pub fn check_bindings(
        &self,
        points: usize,
        reads: &[ReadBinding<'_>],
        writes: &[WriteBinding<'_>],
    ) -> Result<(), CallError> {
        self.check(points, reads, writes)
    }

    fn check(
        &self,
        points: usize,
        reads: &[ReadBinding<'_>],
        writes: &[WriteBinding<'_>],
    ) -> Result<(), CallError> {
        if reads.len() != self.reads.len() {
            return Err(CallError::ReadCount {
                expected: self.reads.len(),
                actual: reads.len(),
            });
        }
        if writes.len() != self.writes.len() {
            return Err(CallError::WriteCount {
                expected: self.writes.len(),
                actual: writes.len(),
            });
        }
        for (position, (binding, expected)) in reads.iter().zip(&self.reads).enumerate() {
            if binding.operand != *expected {
                return Err(CallError::ReadOperand {
                    position,
                    expected: *expected,
                    actual: binding.operand,
                });
            }
            self.check_extent(*expected, binding.values.len(), binding.stride, points)?;
        }
        for (position, (binding, expected)) in writes.iter().zip(&self.writes).enumerate() {
            if binding.operand != *expected {
                return Err(CallError::WriteOperand {
                    position,
                    expected: *expected,
                    actual: binding.operand,
                });
            }
            self.check_extent(*expected, binding.values.len(), binding.stride, points)?;
        }
        Ok(())
    }

    fn check_extent(
        &self,
        operand: OperandId,
        actual: usize,
        stride: usize,
        points: usize,
    ) -> Result<(), CallError> {
        if points == 0 {
            return Ok(());
        }
        let required = (points - 1)
            .checked_mul(stride)
            .and_then(|span| span.checked_add(self.required[operand.index()]))
            .ok_or(CallError::ExtentOverflow { operand })?;
        if actual < required {
            return Err(CallError::ShortBuffer {
                operand,
                required,
                actual,
            });
        }
        Ok(())
    }

    fn sequential<P: Precision>(
        &self,
        state: &mut PreparedState,
        points: usize,
        reads: &[ReadBinding<'_>],
        writes: &mut [WriteBinding<'_>],
    ) {
        for point in 0..points {
            self.execute::<P, _>(
                state,
                &mut PointMemory {
                    reads,
                    writes,
                    point,
                },
            );
        }
    }

    fn blocks<P: Precision>(
        &self,
        state: &mut PreparedState,
        points: usize,
        reads: &[ReadBinding<'_>],
        writes: &mut [WriteBinding<'_>],
    ) {
        let mut first = 0;
        while first < points {
            let lanes = BATCH_BLOCK.min(points - first);
            self.execute_columnar::<P>(state, reads, writes, first, lanes);
            first += lanes;
        }
    }

    /// Run this kernel on one composition stage's private operand copies.
    fn execute_stage(&self, state: &mut PreparedState, buffers: &mut [Vec<f64>]) {
        self.reserve(state, false);
        let mut memory = StageMemory {
            buffers,
            reads: &self.reads,
            writes: &self.writes,
        };
        match self.scalar_type {
            ScalarType::F64 => self.execute::<F64, _>(state, &mut memory),
            ScalarType::F32 => self.execute::<F32, _>(state, &mut memory),
        }
    }

    #[inline(always)]
    fn offset(&self, access: u32, coordinates: &[usize]) -> usize {
        let access = self.accesses[access as usize];
        let mut offset = access.base;
        for &(axis, stride) in &self.terms[access.start..access.end] {
            offset = offset.wrapping_add(coordinates[axis].wrapping_mul(stride));
        }
        offset
    }

    /// Odometer step over the loop order (last axis innermost); false once exhausted.
    #[inline(always)]
    fn advance(&self, coordinates: &mut [usize]) -> bool {
        for &axis in self.order.iter().rev() {
            coordinates[axis] += 1;
            if coordinates[axis] < self.extents[axis] {
                return true;
            }
            coordinates[axis] = 0;
        }
        false
    }

    fn execute<P: Precision, M: Memory>(&self, state: &mut PreparedState, memory: &mut M) {
        if self.empty {
            return;
        }
        let PreparedState {
            registers,
            flags,
            coordinates,
        } = state;
        let coordinates = &mut coordinates[..self.extents.len()];
        coordinates.fill(0);
        loop {
            self.point::<P, M>(registers, flags, coordinates, memory);
            if !self.advance(coordinates) {
                return;
            }
        }
    }

    #[inline(always)]
    fn point<P: Precision, M: Memory>(
        &self,
        r: &mut [f64],
        f: &mut [bool],
        coordinates: &[usize],
        memory: &mut M,
    ) {
        let code = &self.lazy.code;
        let mut pc = 0;
        while let Some(instr) = code.get(pc) {
            pc += 1;
            match *instr {
                Instr::Const { dst, value } => r[dst as usize] = value,
                Instr::Index { dst, axis } => {
                    r[dst as usize] = cast(coordinates[axis as usize] as f64, P::TYPE)
                }
                Instr::LoadRead { dst, slot, access } => {
                    let index = self.offset(access, coordinates);
                    r[dst as usize] = cast(memory.read(slot as usize, index), P::TYPE);
                }
                Instr::LoadWrite { dst, slot, access } => {
                    let index = self.offset(access, coordinates);
                    r[dst as usize] = cast(memory.read_writable(slot as usize, index), P::TYPE);
                }
                Instr::Move { dst, src } => r[dst as usize] = r[src as usize],
                Instr::Unary { op, dst, src } => {
                    r[dst as usize] = unary(op, r[src as usize], P::TYPE, self.math)
                }
                Instr::Binary { op, dst, lhs, rhs } => {
                    r[dst as usize] =
                        binary(op, r[lhs as usize], r[rhs as usize], P::TYPE, self.math)
                }
                Instr::Fma {
                    dst,
                    lhs,
                    rhs,
                    addend,
                } => {
                    r[dst as usize] = fma(
                        r[lhs as usize],
                        r[rhs as usize],
                        r[addend as usize],
                        P::TYPE,
                    )
                }
                Instr::Compare { op, dst, lhs, rhs } => {
                    f[dst as usize] = compare(op, r[lhs as usize], r[rhs as usize])
                }
                Instr::FlagConst { dst, value } => f[dst as usize] = value,
                Instr::FlagMove { dst, src } => f[dst as usize] = f[src as usize],
                Instr::Not { dst, src } => f[dst as usize] = !f[src as usize],
                Instr::And { dst, lhs, rhs } => f[dst as usize] = f[lhs as usize] & f[rhs as usize],
                Instr::Or { dst, lhs, rhs } => f[dst as usize] = f[lhs as usize] | f[rhs as usize],
                Instr::Blend {
                    dst,
                    flag,
                    if_true,
                    if_false,
                } => {
                    r[dst as usize] = if f[flag as usize] {
                        r[if_true as usize]
                    } else {
                        r[if_false as usize]
                    }
                }
                Instr::JumpIfFalse { flag, target } => {
                    if !f[flag as usize] {
                        pc = target as usize;
                    }
                }
                Instr::JumpIfTrue { flag, target } => {
                    if f[flag as usize] {
                        pc = target as usize;
                    }
                }
                Instr::Jump { target } => pc = target as usize,
                Instr::Store {
                    slot,
                    access,
                    src,
                    mode,
                } => {
                    let index = self.offset(access, coordinates);
                    let value = r[src as usize];
                    let slot = slot as usize;
                    match mode {
                        StoreMode::Assign => memory.write(slot, index, value),
                        StoreMode::Reduce(op) => {
                            let old = memory.read_writable(slot, index);
                            memory.write(slot, index, binary(op, old, value, P::TYPE, self.math));
                        }
                    }
                }
            }
        }
    }

    fn execute_columnar<P: Precision>(
        &self,
        state: &mut PreparedState,
        reads: &[ReadBinding<'_>],
        writes: &mut [WriteBinding<'_>],
        first: usize,
        lanes: usize,
    ) {
        if self.empty {
            return;
        }
        let PreparedState {
            registers,
            flags,
            coordinates,
        } = state;
        let coordinates = &mut coordinates[..self.extents.len()];
        coordinates.fill(0);
        let mut block = Block {
            reads,
            writes,
            first,
            lanes,
        };
        loop {
            self.columnar_point::<P>(registers, flags, coordinates, &mut block);
            if !self.advance(coordinates) {
                return;
            }
        }
    }

    /// One iteration point for `lanes` consecutive batch points. Arithmetic runs over all
    /// [`BATCH_BLOCK`] lanes of a register (lanes past `lanes` hold stale, unobserved values);
    /// loads and stores touch only the live lanes.
    #[inline(always)]
    fn columnar_point<P: Precision>(
        &self,
        r: &mut [f64],
        f: &mut [bool],
        coordinates: &[usize],
        block: &mut Block<'_, '_, '_>,
    ) {
        const B: usize = BATCH_BLOCK;
        let Block {
            reads,
            writes,
            first,
            lanes,
        } = block;
        let (first, lanes) = (*first, *lanes);
        let at = |register: u32| register as usize * B;
        for instr in &self.eager.code {
            match *instr {
                Instr::Const { dst, value } => r[at(dst)..at(dst) + B].fill(value),
                Instr::Index { dst, axis } => {
                    r[at(dst)..at(dst) + B].fill(cast(coordinates[axis as usize] as f64, P::TYPE))
                }
                Instr::LoadRead { dst, slot, access } => {
                    let binding = &reads[slot as usize];
                    let index = self.offset(access, coordinates);
                    load_lanes::<P>(
                        &mut r[at(dst)..at(dst) + B],
                        binding.values,
                        binding.stride,
                        first,
                        lanes,
                        index,
                    );
                }
                Instr::LoadWrite { dst, slot, access } => {
                    let binding = &writes[slot as usize];
                    let index = self.offset(access, coordinates);
                    load_lanes::<P>(
                        &mut r[at(dst)..at(dst) + B],
                        binding.values,
                        binding.stride,
                        first,
                        lanes,
                        index,
                    );
                }
                Instr::Move { dst, src } => r.copy_within(at(src)..at(src) + B, at(dst)),
                Instr::Unary { op, dst, src } => {
                    let x: [f64; B] = r[at(src)..at(src) + B].try_into().expect("block");
                    unary_lanes::<P>(op, self.math, &x, &mut r[at(dst)..at(dst) + B]);
                }
                Instr::Binary { op, dst, lhs, rhs } => {
                    let x: [f64; B] = r[at(lhs)..at(lhs) + B].try_into().expect("block");
                    let y: [f64; B] = r[at(rhs)..at(rhs) + B].try_into().expect("block");
                    binary_lanes::<P>(op, self.math, &x, &y, &mut r[at(dst)..at(dst) + B]);
                }
                Instr::Fma {
                    dst,
                    lhs,
                    rhs,
                    addend,
                } => {
                    for lane in 0..B {
                        r[at(dst) + lane] = fma(
                            r[at(lhs) + lane],
                            r[at(rhs) + lane],
                            r[at(addend) + lane],
                            P::TYPE,
                        );
                    }
                }
                Instr::Compare { op, dst, lhs, rhs } => {
                    for lane in 0..B {
                        f[at(dst) + lane] = compare(op, r[at(lhs) + lane], r[at(rhs) + lane]);
                    }
                }
                Instr::FlagConst { dst, value } => f[at(dst)..at(dst) + B].fill(value),
                Instr::FlagMove { dst, src } => f.copy_within(at(src)..at(src) + B, at(dst)),
                Instr::Not { dst, src } => {
                    for lane in 0..B {
                        f[at(dst) + lane] = !f[at(src) + lane];
                    }
                }
                Instr::And { dst, lhs, rhs } => {
                    for lane in 0..B {
                        f[at(dst) + lane] = f[at(lhs) + lane] & f[at(rhs) + lane];
                    }
                }
                Instr::Or { dst, lhs, rhs } => {
                    for lane in 0..B {
                        f[at(dst) + lane] = f[at(lhs) + lane] | f[at(rhs) + lane];
                    }
                }
                Instr::Blend {
                    dst,
                    flag,
                    if_true,
                    if_false,
                } => {
                    for lane in 0..B {
                        r[at(dst) + lane] = if f[at(flag) + lane] {
                            r[at(if_true) + lane]
                        } else {
                            r[at(if_false) + lane]
                        };
                    }
                }
                Instr::JumpIfFalse { .. } | Instr::JumpIfTrue { .. } | Instr::Jump { .. } => {
                    unreachable!("eager programs contain no jumps")
                }
                Instr::Store {
                    slot,
                    access,
                    src,
                    mode,
                } => {
                    let binding = &mut writes[slot as usize];
                    let index = self.offset(access, coordinates);
                    let values = &r[at(src)..at(src) + lanes];
                    for (lane, value) in values.iter().enumerate() {
                        let target = &mut binding.values[(first + lane) * binding.stride + index];
                        *target = match mode {
                            StoreMode::Assign => *value,
                            StoreMode::Reduce(op) => {
                                binary(op, *target, *value, P::TYPE, self.math)
                            }
                        };
                    }
                }
            }
        }
    }
}

/// The live lanes `first..first + lanes` of one columnar block.
struct Block<'m, 'r, 'w> {
    reads: &'m [ReadBinding<'r>],
    writes: &'m mut [WriteBinding<'w>],
    first: usize,
    lanes: usize,
}

#[inline(always)]
fn load_lanes<P: Precision>(
    out: &mut [f64],
    values: &[f64],
    stride: usize,
    first: usize,
    lanes: usize,
    index: usize,
) {
    if stride == 0 {
        out.fill(cast(values[index], P::TYPE));
    } else {
        for (lane, out) in out[..lanes].iter_mut().enumerate() {
            *out = cast(values[(first + lane) * stride + index], P::TYPE);
        }
    }
}

macro_rules! lanes {
    ($out:ident, $lane:ident, $expr:expr) => {
        for ($lane, out) in $out.iter_mut().enumerate() {
            *out = $expr;
        }
    };
}

/// Hoists the operation dispatch out of the lane loop; each arm calls the interpreter's own
/// scalar operation with a constant operation, so the per-lane arithmetic is identical.
#[inline(always)]
fn unary_lanes<P: Precision>(op: UnaryOp, m: MathLibrary, x: &[f64; BATCH_BLOCK], out: &mut [f64]) {
    use UnaryOp::*;
    match op {
        Neg => lanes!(out, l, unary(Neg, x[l], P::TYPE, m)),
        Abs => lanes!(out, l, unary(Abs, x[l], P::TYPE, m)),
        Sqrt => lanes!(out, l, unary(Sqrt, x[l], P::TYPE, m)),
        Exp => lanes!(out, l, unary(Exp, x[l], P::TYPE, m)),
        Ln => lanes!(out, l, unary(Ln, x[l], P::TYPE, m)),
        Sin => lanes!(out, l, unary(Sin, x[l], P::TYPE, m)),
        Cos => lanes!(out, l, unary(Cos, x[l], P::TYPE, m)),
        Tan => lanes!(out, l, unary(Tan, x[l], P::TYPE, m)),
        Floor => lanes!(out, l, unary(Floor, x[l], P::TYPE, m)),
        Ceil => lanes!(out, l, unary(Ceil, x[l], P::TYPE, m)),
    }
}

#[inline(always)]
fn binary_lanes<P: Precision>(
    op: BinaryOp,
    m: MathLibrary,
    x: &[f64; BATCH_BLOCK],
    y: &[f64; BATCH_BLOCK],
    out: &mut [f64],
) {
    use BinaryOp::*;
    match op {
        Add => lanes!(out, l, binary(Add, x[l], y[l], P::TYPE, m)),
        Sub => lanes!(out, l, binary(Sub, x[l], y[l], P::TYPE, m)),
        Mul => lanes!(out, l, binary(Mul, x[l], y[l], P::TYPE, m)),
        Div => lanes!(out, l, binary(Div, x[l], y[l], P::TYPE, m)),
        Pow => lanes!(out, l, binary(Pow, x[l], y[l], P::TYPE, m)),
        Min => lanes!(out, l, binary(Min, x[l], y[l], P::TYPE, m)),
        Max => lanes!(out, l, binary(Max, x[l], y[l], P::TYPE, m)),
        Atan2 => lanes!(out, l, binary(Atan2, x[l], y[l], P::TYPE, m)),
    }
}

#[derive(Clone, Copy)]
struct Value {
    register: u32,
    temporary: bool,
}

/// Tree-to-register compiler. Temporaries are recycled through a free list as soon as their
/// consumer is emitted; `Let` locals keep their register for the rest of the point.
struct Compiler<'a> {
    kernel: &'a StructuredKernel,
    slots: &'a [Slot],
    lazy: bool,
    code: Vec<Instr>,
    free: Vec<u32>,
    registers: u32,
    free_flags: Vec<u32>,
    flags: u32,
    locals: Vec<u32>,
}

impl<'a> Compiler<'a> {
    fn compile(kernel: &'a StructuredKernel, slots: &'a [Slot], lazy: bool) -> Program {
        let mut compiler = Self {
            kernel,
            slots,
            lazy,
            code: Vec::new(),
            free: Vec::new(),
            registers: 0,
            free_flags: Vec::new(),
            flags: 0,
            locals: Vec::new(),
        };
        for statement in &kernel.body.statements {
            match statement {
                Statement::Let { value, .. } => {
                    let value = compiler.expr(value);
                    compiler.locals.push(value.register);
                }
                Statement::Store { operand, value } => {
                    let value = compiler.expr(value);
                    let Slot::Write(slot) = slots[operand.index()] else {
                        unreachable!("validated store")
                    };
                    let mode = match kernel.operands[operand.index()].access {
                        AccessMode::Write | AccessMode::ReadWrite => StoreMode::Assign,
                        AccessMode::Reduce(ReductionOp::Add) => StoreMode::Reduce(BinaryOp::Add),
                        AccessMode::Reduce(ReductionOp::Multiply) => {
                            StoreMode::Reduce(BinaryOp::Mul)
                        }
                        AccessMode::Reduce(ReductionOp::Min) => StoreMode::Reduce(BinaryOp::Min),
                        AccessMode::Reduce(ReductionOp::Max) => StoreMode::Reduce(BinaryOp::Max),
                        AccessMode::Read => unreachable!("validated store"),
                    };
                    compiler.code.push(Instr::Store {
                        slot: slot as u32,
                        access: operand.index() as u32,
                        src: value.register,
                        mode,
                    });
                    compiler.release(value);
                }
            }
        }
        Program {
            code: compiler.code,
            registers: compiler.registers as usize,
            flags: compiler.flags as usize,
        }
    }

    fn register(&mut self) -> u32 {
        self.free.pop().unwrap_or_else(|| {
            self.registers += 1;
            self.registers - 1
        })
    }

    fn release(&mut self, value: Value) {
        if value.temporary {
            self.free.push(value.register);
        }
    }

    fn flag(&mut self) -> u32 {
        self.free_flags.pop().unwrap_or_else(|| {
            self.flags += 1;
            self.flags - 1
        })
    }

    fn here(&self) -> u32 {
        self.code.len() as u32
    }

    fn patch(&mut self, at: usize) {
        let here = self.here();
        match &mut self.code[at] {
            Instr::JumpIfFalse { target, .. }
            | Instr::JumpIfTrue { target, .. }
            | Instr::Jump { target } => *target = here,
            _ => unreachable!("patched instruction is a jump"),
        }
    }

    fn temporary(&mut self, emit: impl FnOnce(u32) -> Instr) -> Value {
        let register = self.register();
        self.code.push(emit(register));
        Value {
            register,
            temporary: true,
        }
    }

    fn expr(&mut self, expr: &ScalarExpr) -> Value {
        let scalar_type = self.kernel.numeric_policy.scalar_type;
        match expr {
            ScalarExpr::Constant(value) => {
                let value = cast(*value, scalar_type);
                self.temporary(|dst| Instr::Const { dst, value })
            }
            ScalarExpr::Index(axis) => {
                let axis = axis.index() as u32;
                self.temporary(|dst| Instr::Index { dst, axis })
            }
            ScalarExpr::Local(local) => Value {
                register: self.locals[local.index()],
                temporary: false,
            },
            ScalarExpr::Load(operand) => {
                let access = operand.index() as u32;
                match self.slots[operand.index()] {
                    Slot::Read(slot) => self.temporary(|dst| Instr::LoadRead {
                        dst,
                        slot: slot as u32,
                        access,
                    }),
                    Slot::Write(slot) => self.temporary(|dst| Instr::LoadWrite {
                        dst,
                        slot: slot as u32,
                        access,
                    }),
                }
            }
            ScalarExpr::Unary { op, arg } => {
                let arg = self.expr(arg);
                self.release(arg);
                self.temporary(|dst| Instr::Unary {
                    op: *op,
                    dst,
                    src: arg.register,
                })
            }
            ScalarExpr::Binary { op, lhs, rhs } => {
                let lhs = self.expr(lhs);
                let rhs = self.expr(rhs);
                self.release(lhs);
                self.release(rhs);
                self.temporary(|dst| Instr::Binary {
                    op: *op,
                    dst,
                    lhs: lhs.register,
                    rhs: rhs.register,
                })
            }
            ScalarExpr::Fma { lhs, rhs, addend } => {
                let lhs = self.expr(lhs);
                let rhs = self.expr(rhs);
                let addend = self.expr(addend);
                self.release(lhs);
                self.release(rhs);
                self.release(addend);
                self.temporary(|dst| Instr::Fma {
                    dst,
                    lhs: lhs.register,
                    rhs: rhs.register,
                    addend: addend.register,
                })
            }
            ScalarExpr::Select {
                condition,
                if_true,
                if_false,
            } if self.lazy => {
                let condition = self.predicate(condition);
                let branch = self.code.len();
                self.code.push(Instr::JumpIfFalse {
                    flag: condition,
                    target: 0,
                });
                self.free_flags.push(condition);
                let dst = self.register();
                let value = self.expr(if_true);
                self.code.push(Instr::Move {
                    dst,
                    src: value.register,
                });
                self.release(value);
                let jump = self.code.len();
                self.code.push(Instr::Jump { target: 0 });
                self.patch(branch);
                let value = self.expr(if_false);
                self.code.push(Instr::Move {
                    dst,
                    src: value.register,
                });
                self.release(value);
                self.patch(jump);
                Value {
                    register: dst,
                    temporary: true,
                }
            }
            ScalarExpr::Select {
                condition,
                if_true,
                if_false,
            } => {
                let flag = self.predicate(condition);
                let if_true = self.expr(if_true);
                let if_false = self.expr(if_false);
                self.free_flags.push(flag);
                self.release(if_true);
                self.release(if_false);
                self.temporary(|dst| Instr::Blend {
                    dst,
                    flag,
                    if_true: if_true.register,
                    if_false: if_false.register,
                })
            }
        }
    }

    fn predicate(&mut self, predicate: &Predicate) -> u32 {
        match predicate {
            Predicate::Constant(value) => {
                let dst = self.flag();
                self.code.push(Instr::FlagConst { dst, value: *value });
                dst
            }
            Predicate::Compare { op, lhs, rhs } => {
                let lhs = self.expr(lhs);
                let rhs = self.expr(rhs);
                self.release(lhs);
                self.release(rhs);
                let dst = self.flag();
                self.code.push(Instr::Compare {
                    op: *op,
                    dst,
                    lhs: lhs.register,
                    rhs: rhs.register,
                });
                dst
            }
            Predicate::Not(value) => {
                let dst = self.predicate(value);
                self.code.push(Instr::Not { dst, src: dst });
                dst
            }
            Predicate::And(lhs, rhs) | Predicate::Or(lhs, rhs) => {
                let is_and = matches!(predicate, Predicate::And(..));
                let dst = self.predicate(lhs);
                if self.lazy {
                    let branch = self.code.len();
                    self.code.push(if is_and {
                        Instr::JumpIfFalse {
                            flag: dst,
                            target: 0,
                        }
                    } else {
                        Instr::JumpIfTrue {
                            flag: dst,
                            target: 0,
                        }
                    });
                    let src = self.predicate(rhs);
                    self.code.push(Instr::FlagMove { dst, src });
                    self.free_flags.push(src);
                    self.patch(branch);
                } else {
                    let src = self.predicate(rhs);
                    self.code.push(if is_and {
                        Instr::And {
                            dst,
                            lhs: dst,
                            rhs: src,
                        }
                    } else {
                        Instr::Or {
                            dst,
                            lhs: dst,
                            rhs: src,
                        }
                    });
                    self.free_flags.push(src);
                }
                dst
            }
        }
    }
}

/// Every kernel of an [`ExecutableModule`], prepared once.
#[derive(Clone, Debug)]
pub struct PreparedModule {
    name: String,
    kernels: Vec<PreparedKernel>,
}

impl PreparedModule {
    pub fn new(module: &ExecutableModule) -> Result<Self, PrepareError> {
        Ok(Self {
            name: module.name().to_owned(),
            kernels: module
                .kernels()
                .iter()
                .map(PreparedKernel::new)
                .collect::<Result<_, _>>()?,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kernels(&self) -> &[PreparedKernel] {
        &self.kernels
    }

    pub fn kernel(&self, name: &str) -> Option<&PreparedKernel> {
        self.kernels.iter().find(|kernel| kernel.name == name)
    }
}

const UNBOUND: usize = usize::MAX;

/// An [`ExecutableComposition`] with every stage prepared.
///
/// [`Self::run`] has `Interpreter::run_composition`'s binding contract, error values and
/// copy-in/copy-out stage semantics (each stage runs on private copies of its operand windows;
/// writable copies are written back in operand order), with the copies, scratch buffers and
/// binding tables owned by a reusable [`CompositionState`] instead of allocated per call.
#[derive(Clone, Debug)]
pub struct PreparedComposition {
    name: String,
    stages: Vec<PreparedKernel>,
    /// First flat operand index of every stage, plus the total.
    offsets: Vec<usize>,
    operands: Vec<StageOperand>,
    shared_of: Vec<Option<usize>>,
    lengths: Vec<usize>,
    writable: Vec<bool>,
    shared_lengths: Vec<usize>,
}

/// Reusable buffers and binding tables of a [`PreparedComposition`].
#[derive(Clone, Debug, Default)]
pub struct CompositionState {
    kernel: PreparedState,
    buffers: Vec<Vec<f64>>,
    scratch: Vec<Vec<f64>>,
    operand_binding: Vec<usize>,
    shared_binding: Vec<usize>,
}

impl PreparedComposition {
    pub fn new(composition: &ExecutableComposition) -> Result<Self, PrepareError> {
        let stages = composition
            .stages()
            .iter()
            .map(PreparedKernel::new)
            .collect::<Result<Vec<_>, _>>()?;
        let mut offsets = vec![0];
        let (mut operands, mut shared_of, mut lengths, mut writable) =
            (vec![], vec![], vec![], vec![]);
        for (stage_index, stage) in composition.stages().iter().enumerate() {
            for (operand_index, definition) in
                stage.kernel().as_kernel().operands.iter().enumerate()
            {
                let operand = StageOperand::new(stage_index, OperandId::new(operand_index));
                operands.push(operand);
                shared_of.push(composition.shared_buffer_of(operand));
                lengths.push(definition.region.offset + definition.region.length);
                writable.push(definition.access.can_write());
            }
            offsets.push(operands.len());
        }
        let shared_lengths = (0..)
            .map_while(|buffer| composition.shared_length(buffer))
            .collect();
        Ok(Self {
            name: composition.name().to_owned(),
            stages,
            offsets,
            operands,
            shared_of,
            lengths,
            writable,
            shared_lengths,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn stages(&self) -> &[PreparedKernel] {
        &self.stages
    }

    /// A state sized for this composition, so calls never allocate.
    pub fn state(&self) -> CompositionState {
        let mut state = CompositionState::default();
        self.reserve(&mut state);
        state
    }

    fn reserve(&self, state: &mut CompositionState) {
        for stage in &self.stages {
            stage.reserve(&mut state.kernel, false);
        }
        let sized = |buffers: &Vec<Vec<f64>>, lengths: &[usize]| {
            buffers.len() == lengths.len()
                && buffers
                    .iter()
                    .zip(lengths)
                    .all(|(buffer, length)| buffer.len() == *length)
        };
        if !sized(&state.buffers, &self.lengths) {
            state.buffers = self
                .lengths
                .iter()
                .map(|length| vec![0.0; *length])
                .collect();
        }
        if !sized(&state.scratch, &self.shared_lengths) {
            state.scratch = self
                .shared_lengths
                .iter()
                .map(|length| vec![0.0; *length])
                .collect();
        }
        state.operand_binding.resize(self.lengths.len(), UNBOUND);
        state
            .shared_binding
            .resize(self.shared_lengths.len(), UNBOUND);
    }

    fn flat(&self, operand: StageOperand) -> Option<usize> {
        let start = *self.offsets.get(operand.stage)?;
        let end = *self.offsets.get(operand.stage + 1)?;
        let flat = start.checked_add(operand.operand.index())?;
        (flat < end).then_some(flat)
    }

    /// Execute every stage in order; see `Interpreter::run_composition` for the contract.
    pub fn run(
        &self,
        state: &mut CompositionState,
        bindings: &mut [CompositionBinding<'_>],
    ) -> Result<(), CompositionExecutionError> {
        self.reserve(state);
        let CompositionState {
            kernel,
            buffers,
            scratch,
            operand_binding,
            shared_binding,
        } = state;
        operand_binding.fill(UNBOUND);
        shared_binding.fill(UNBOUND);
        for (index, binding) in bindings.iter().enumerate() {
            let invalid = || CompositionExecutionError::InvalidBinding(binding.target);
            match binding.target {
                CompositionTarget::Operand(operand) => {
                    let flat = self.flat(operand);
                    if flat.is_some_and(|flat| self.shared_of[flat].is_some()) {
                        return Err(CompositionExecutionError::SharedOperandBound(operand));
                    }
                    let flat = flat.ok_or_else(invalid)?;
                    if binding.values.len() < self.lengths[flat] {
                        return Err(invalid());
                    }
                    if operand_binding[flat] != UNBOUND {
                        return Err(CompositionExecutionError::DuplicateBinding(binding.target));
                    }
                    operand_binding[flat] = index;
                }
                CompositionTarget::Shared(buffer) => {
                    let length = *self.shared_lengths.get(buffer).ok_or_else(invalid)?;
                    if binding.values.len() < length {
                        return Err(invalid());
                    }
                    if shared_binding[buffer] != UNBOUND {
                        return Err(CompositionExecutionError::DuplicateBinding(binding.target));
                    }
                    shared_binding[buffer] = index;
                }
            }
        }
        for ((operand, shared), binding) in self
            .operands
            .iter()
            .zip(&self.shared_of)
            .zip(&*operand_binding)
        {
            if shared.is_none() && *binding == UNBOUND {
                return Err(CompositionExecutionError::MissingBinding(*operand));
            }
        }
        for (buffer, values) in scratch.iter_mut().enumerate() {
            if shared_binding[buffer] == UNBOUND {
                values.fill(0.0);
            }
        }
        for (stage_index, stage) in self.stages.iter().enumerate() {
            let range = self.offsets[stage_index]..self.offsets[stage_index + 1];
            for flat in range.clone() {
                let length = self.lengths[flat];
                let source = match self.shared_of[flat] {
                    Some(buffer) if shared_binding[buffer] == UNBOUND => &scratch[buffer][..length],
                    Some(buffer) => &bindings[shared_binding[buffer]].values[..length],
                    None => &bindings[operand_binding[flat]].values[..length],
                };
                buffers[flat].copy_from_slice(source);
            }
            stage.execute_stage(kernel, &mut buffers[range.clone()]);
            for flat in range.filter(|flat| self.writable[*flat]) {
                let length = self.lengths[flat];
                let target = match self.shared_of[flat] {
                    Some(buffer) if shared_binding[buffer] == UNBOUND => {
                        &mut scratch[buffer][..length]
                    }
                    Some(buffer) => &mut bindings[shared_binding[buffer]].values[..length],
                    None => &mut bindings[operand_binding[flat]].values[..length],
                };
                target.copy_from_slice(&buffers[flat]);
            }
        }
        Ok(())
    }
}

/// [`LocalExecutableRunner`] over [`PreparedKernel::run`], so the local differential
/// campaign (primal, JVP, VJP and parameter-JVP interpreter-vs-executable checks plus the
/// derivative checks) can run through the prepared bytecode.
#[derive(Clone, Copy, Debug, Default)]
pub struct PreparedRunner;

impl LocalExecutableRunner for PreparedRunner {
    fn identity(&self) -> &str {
        "malleus-prepared/bytecode-v1"
    }

    fn run(&self, executable: &Executable, buffers: &mut [Vec<f64>]) -> Result<(), String> {
        let prepared = PreparedKernel::new(executable).map_err(|error| error.to_string())?;
        let mut bindings = buffers
            .iter_mut()
            .enumerate()
            .map(|(index, values)| BufferBinding::new(OperandId::new(index), values))
            .collect::<Vec<_>>();
        prepared
            .run(&mut bindings)
            .map_err(|error| error.to_string())
    }
}
