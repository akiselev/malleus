//! Checked schedule primitives: proven write injectivity, `vectorize(axis)` and
//! `parallel(axis)`.
//!
//! # Legality condition
//!
//! For an axis `a`, let `o_k(p)` be the element offset an iteration point `p` addresses in
//! writable operand `k` (its region offset plus the layout-strided affine map). The schedule
//! may move `a` outermost and run its values as independent lanes iff for every writable
//! operand (`Write`, `ReadWrite`, `Reduce`):
//!
//! `∀ p, p' ∈ domain: p_a ≠ p'_a ⇒ o_k(p) ≠ o_k(p')`
//!
//! Then two iterations with different `a` values never touch the same writable element, so
//! every element sees the same ordered sequence of stores (and reduction updates) as in the
//! canonical sequential order; readable-only operands are separate buffers and unaffected.
//! [`prove_write_injective`] decides the condition by exact integer reasoning on
//! `Σ_b g_b δ_b = 0` with `|δ_b| < n_b`, `δ_a ≠ 0` (`g_b` the strided offset coefficient of
//! axis `b`): a bitset of reachable partial sums over the other axes, refused with
//! [`ScheduleError::NotProvable`] only when that set exceeds its budget. `Write` maps are
//! already injective over the whole domain by validation; the question is real for
//! reductions and read-write operands.
//!
//! [`VectorizedKernel`] executes a proven axis through the prepared batch machinery
//! (`PreparedKernel::run_batch`, columnar when written windows are disjoint) over a derived
//! kernel without the axis; [`VectorizedKernel::run_parallel`] splits the axis over
//! `std::thread::scope` threads, each on private copies of the writable buffers, and copies
//! back exactly the elements its lanes address. Both are bitwise equal to
//! `Interpreter::run` on the reference executable (proved by `tests/schedule.rs`).

use std::error::Error;
use std::fmt;

use crate::prepared::{CallError, PrepareError, PreparedKernel, ReadBinding, WriteBinding};
use crate::{
    AccessMode, AxisId, BufferBinding, BufferRegion, DenseLayout, Executable, ExecutionError,
    IndexExpr, IndexTerm, IndexingMap, IterationDomain, KernelOperand, OperandId, Predicate,
    ReductionOrder, ScalarExpr, Statement, StructuredKernel,
};

/// Largest partial-sum range the exact injectivity decision explores.
pub const INJECTIVITY_BUDGET: u128 = 1 << 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    Invalid(crate::ValidationError),
    InvalidAxis(usize),
    /// Two iterations with different values of the axis address one element of `operand`;
    /// `witness` is a pair of iteration points that collide.
    NotInjective {
        operand: OperandId,
        witness: Witness,
    },
    /// The exact decision exceeded [`INJECTIVITY_BUDGET`].
    NotProvable {
        operand: OperandId,
    },
    /// The body reads the axis as a value (`ScalarExpr::Index`).
    AxisUsedAsValue,
    /// The operand's offset decreases along the axis (batch strides are non-negative).
    NegativeStride {
        operand: OperandId,
    },
    /// A writable operand's axis term is not alone in its most-major dimension.
    UnbatchableLayout {
        operand: OperandId,
    },
    /// `ReductionOrder::ScheduleDefined` kernels execute in schedule order, not canonically.
    ScheduleDefinedOrder,
    Prepare(PrepareError),
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "schedule primitive refused: {self:?}")
    }
}

impl Error for ScheduleError {}

/// Evidence that `axis` satisfies the legality condition: the offset coefficient of the axis
/// in every operand (the batch stride), in operand order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InjectivityProof {
    pub axis: AxisId,
    pub strides: Vec<i128>,
}

/// The affine element offset `constant + Σ coefficient_b · p_b` of one operand.
fn affine_offset(operand: &KernelOperand, map: &IndexingMap, rank: usize) -> (i128, Vec<i128>) {
    let mut strides = vec![0i128; operand.shape.len()];
    let mut stride = 1i128;
    for dimension in &operand.layout.minor_to_major {
        strides[*dimension] = stride;
        stride *= operand.shape[*dimension] as i128;
    }
    let mut constant = operand.region.offset as i128;
    let mut coefficients = vec![0i128; rank];
    for (result, stride) in map.results.iter().zip(&strides) {
        constant += stride * result.constant as i128;
        for term in &result.terms {
            coefficients[term.axis.index()] += stride * term.coefficient as i128;
        }
    }
    (constant, coefficients)
}

fn map_of(kernel: &StructuredKernel, operand: usize) -> &IndexingMap {
    kernel
        .indexing_maps
        .iter()
        .find(|map| map.operand.index() == operand)
        .expect("validated map")
}

/// Decide the legality condition of the module docs for `axis`.
pub fn prove_write_injective(
    kernel: &StructuredKernel,
    axis: AxisId,
) -> Result<InjectivityProof, ScheduleError> {
    crate::validate(kernel.clone()).map_err(ScheduleError::Invalid)?;
    let extents = &kernel.iteration_domain.extents;
    let rank = extents.len();
    if axis.index() >= rank {
        return Err(ScheduleError::InvalidAxis(axis.index()));
    }
    let mut strides = Vec::new();
    for (index, operand) in kernel.operands.iter().enumerate() {
        let (_, coefficients) = affine_offset(operand, map_of(kernel, index), rank);
        strides.push(coefficients[axis.index()]);
        if operand.access == AccessMode::Read || extents[axis.index()] <= 1 {
            continue;
        }
        if extents.contains(&0) {
            continue;
        }
        let id = OperandId::new(index);
        if let Some(witness) = collision(&coefficients, extents, axis.index(), id)? {
            return Err(ScheduleError::NotInjective {
                operand: id,
                witness,
            });
        }
    }
    Ok(InjectivityProof { axis, strides })
}

/// Two iteration points that address one element.
type Witness = (Vec<usize>, Vec<usize>);

/// A pair of points with different `axis` values and equal offsets, if one exists.
fn collision(
    coefficients: &[i128],
    extents: &[usize],
    axis: usize,
    operand: OperandId,
) -> Result<Option<Witness>, ScheduleError> {
    let g_a = coefficients[axis];
    let n_a = extents[axis] as i128;
    // δ_b ranges over (-(n_b - 1)) ..= n_b - 1 for the other axes with nonzero coefficient.
    let others: Vec<usize> = (0..extents.len())
        .filter(|b| *b != axis && coefficients[*b] != 0 && extents[*b] > 1)
        .collect();
    let range: i128 = others
        .iter()
        .map(|b| coefficients[*b].abs() * (extents[*b] as i128 - 1))
        .sum();
    if g_a == 0 {
        // Any δ_a with zero δ elsewhere collides.
        let p = vec![0; extents.len()];
        let mut q = p.clone();
        q[axis] = 1;
        return Ok(Some((p, q)));
    }
    if g_a.abs() > range {
        // |g_a δ_a| ≥ |g_a| > |Σ_{b≠a} g_b δ_b|.
        return Ok(None);
    }
    let width = 2 * range + 1;
    if width as u128 > INJECTIVITY_BUDGET {
        return Err(ScheduleError::NotProvable { operand });
    }
    // reach[s + range] = the δ assignment over `others` reaching partial sum s, if any.
    let mut reach: Vec<Option<Vec<i128>>> = vec![None; width as usize];
    reach[range as usize] = Some(Vec::new());
    for b in &others {
        let n_b = extents[*b] as i128;
        let mut next: Vec<Option<Vec<i128>>> = vec![None; width as usize];
        for (index, assignment) in reach.iter().enumerate() {
            let Some(assignment) = assignment else {
                continue;
            };
            for delta in -(n_b - 1)..=n_b - 1 {
                let sum = index as i128 + coefficients[*b] * delta;
                if (0..width).contains(&sum) && next[sum as usize].is_none() {
                    let mut extended = assignment.clone();
                    extended.push(delta);
                    next[sum as usize] = Some(extended);
                }
            }
        }
        reach = next;
    }
    for delta_a in 1..n_a {
        let target = -g_a * delta_a + range;
        if !(0..width).contains(&target) {
            continue;
        }
        if let Some(assignment) = &reach[target as usize] {
            let mut p = vec![0usize; extents.len()];
            let mut q = vec![0usize; extents.len()];
            q[axis] = delta_a as usize;
            for (b, delta) in others.iter().zip(assignment) {
                if *delta >= 0 {
                    q[*b] = *delta as usize;
                } else {
                    p[*b] = (-*delta) as usize;
                }
            }
            return Ok(Some((p, q)));
        }
    }
    Ok(None)
}

/// Private per-thread copies of every bound buffer.
type Copies = Vec<(OperandId, Vec<f64>)>;

/// A kernel executed with one proven-injective axis as independent batch lanes.
#[derive(Clone, Debug)]
pub struct VectorizedKernel {
    prepared: PreparedKernel,
    axis: AxisId,
    lanes: usize,
    /// Per operand: batch stride (elements per axis step).
    strides: Vec<usize>,
    /// Per operand of the derived kernel: constant offset and coefficients over its axes.
    offsets: Vec<(i128, Vec<i128>)>,
    access: Vec<AccessMode>,
    lane_extents: Vec<usize>,
}

impl VectorizedKernel {
    /// `vectorize(axis)`: prove the legality condition, then derive the lane kernel. Refuses
    /// kernels that read the axis as a value, whose offsets decrease along it, whose writable
    /// operands do not carry the axis alone in their most-major dimension, or that use
    /// schedule-defined reduction order.
    pub fn new(executable: &Executable, axis: AxisId) -> Result<Self, ScheduleError> {
        let kernel = executable.kernel().as_kernel();
        if kernel.numeric_policy.reduction_order == ReductionOrder::ScheduleDefined {
            return Err(ScheduleError::ScheduleDefinedOrder);
        }
        let proof = prove_write_injective(kernel, axis)?;
        if body_reads_axis(kernel, axis) {
            return Err(ScheduleError::AxisUsedAsValue);
        }
        let a = axis.index();
        let lanes = kernel.iteration_domain.extents[a];
        let rank = kernel.iteration_domain.rank();
        let mut strides = Vec::new();
        for (index, stride) in proof.strides.iter().enumerate() {
            if *stride < 0 {
                return Err(ScheduleError::NegativeStride {
                    operand: OperandId::new(index),
                });
            }
            strides.push(*stride as usize);
        }
        let shift =
            |axis_id: AxisId| AxisId::new(axis_id.index() - usize::from(axis_id.index() > a));
        let mut derived = kernel.clone();
        derived.iteration_domain = IterationDomain::new(
            kernel
                .iteration_domain
                .extents
                .iter()
                .enumerate()
                .filter(|(b, _)| *b != a)
                .map(|(_, e)| *e)
                .collect::<Vec<_>>(),
        );
        derived.iterators.remove(a);
        let drop_axis = |expr: &IndexExpr| IndexExpr {
            constant: expr.constant,
            terms: expr
                .terms
                .iter()
                .filter(|term| term.axis != axis)
                .map(|term| IndexTerm {
                    axis: shift(term.axis),
                    coefficient: term.coefficient,
                })
                .collect(),
        };
        for (index, operand) in kernel.operands.iter().enumerate() {
            let map = map_of(kernel, index);
            let id = OperandId::new(index);
            let derived_map = derived
                .indexing_maps
                .iter_mut()
                .find(|m| m.operand == id)
                .expect("validated map");
            let excursion = strides[index] * lanes.saturating_sub(1);
            let derived_operand = &mut derived.operands[index];
            if matches!(operand.access, AccessMode::Write | AccessMode::ReadWrite) {
                // Shrink the most-major dimension that alone carries the axis.
                let major = *operand
                    .layout
                    .minor_to_major
                    .last()
                    .expect("validated write rank");
                let carrying: Vec<usize> = map
                    .results
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.terms.iter().any(|t| t.axis == axis && t.coefficient != 0))
                    .map(|(d, _)| d)
                    .collect();
                let coefficient = map.results[major]
                    .terms
                    .iter()
                    .find(|t| t.axis == axis)
                    .map(|t| t.coefficient);
                let Some(coefficient) = coefficient.filter(|_| carrying == [major]) else {
                    return Err(ScheduleError::UnbatchableLayout { operand: id });
                };
                derived_operand.shape[major] -= coefficient as usize * (lanes - 1);
                derived_operand.region =
                    BufferRegion::new(operand.region.offset, operand.region.length - excursion);
                derived_map.results = map.results.iter().map(drop_axis).collect();
            } else {
                // Flatten to rank 1 over the remaining axes.
                let (constant, coefficients) = affine_offset(operand, map, rank);
                let constant = constant - operand.region.offset as i128;
                let mut maximum = constant;
                let mut terms = Vec::new();
                for (b, coefficient) in coefficients.iter().enumerate() {
                    if b == a || *coefficient == 0 {
                        continue;
                    }
                    maximum += coefficient.max(&0)
                        * (kernel.iteration_domain.extents[b] as i128 - 1).max(0);
                    terms.push(IndexTerm {
                        axis: shift(AxisId::new(b)),
                        coefficient: *coefficient as isize,
                    });
                }
                let length = (maximum + 1).max(1) as usize;
                derived_operand.shape = vec![length];
                derived_operand.layout = DenseLayout::row_major(1);
                derived_operand.region = BufferRegion::new(operand.region.offset, length);
                derived_map.results = vec![IndexExpr {
                    constant: constant as isize,
                    terms,
                }];
            }
        }
        for statement in &mut derived.body.statements {
            let (Statement::Let { value, .. } | Statement::Store { value, .. }) = statement;
            *value = shift_indices(value, a);
        }
        derived.name = format!("{}::vectorize{}", kernel.name, a);
        let validated = crate::validate(derived.clone()).map_err(ScheduleError::Invalid)?;
        let prepared = PreparedKernel::new(&Executable::reference(validated))
            .map_err(ScheduleError::Prepare)?;
        let derived_rank = derived.iteration_domain.rank();
        let offsets = derived
            .operands
            .iter()
            .enumerate()
            .map(|(index, operand)| affine_offset(operand, map_of(&derived, index), derived_rank))
            .collect();
        Ok(Self {
            prepared,
            axis,
            lanes,
            strides,
            offsets,
            access: kernel.operands.iter().map(|o| o.access).collect(),
            lane_extents: derived.iteration_domain.extents.clone(),
        })
    }

    pub fn axis(&self) -> AxisId {
        self.axis
    }

    /// Run all lanes through one `run_batch` call. Bindings follow the interpreter contract.
    pub fn run(&self, bindings: &mut [BufferBinding<'_>]) -> Result<(), ExecutionError> {
        self.run_lanes(bindings, 0, self.lanes)
    }

    fn run_lanes(
        &self,
        bindings: &mut [BufferBinding<'_>],
        first: usize,
        count: usize,
    ) -> Result<(), ExecutionError> {
        let operands = self.access.len();
        let mut by_operand: Vec<Option<&mut [f64]>> = (0..operands).map(|_| None).collect();
        for binding in bindings.iter_mut() {
            let id = binding.operand.index();
            if id >= operands {
                return Err(ExecutionError::InvalidBinding(id));
            }
            if by_operand[id].is_some() {
                return Err(ExecutionError::DuplicateBinding(id));
            }
            by_operand[id] = Some(&mut *binding.values);
        }
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        for (index, values) in by_operand.into_iter().enumerate() {
            let values = values.ok_or(ExecutionError::MissingBinding(index))?;
            let start = (first * self.strides[index]).min(values.len());
            let id = OperandId::new(index);
            if self.access[index] == AccessMode::Read {
                reads.push(ReadBinding::strided(
                    id,
                    &values[start..],
                    self.strides[index],
                ));
            } else {
                writes.push(WriteBinding::strided(
                    id,
                    &mut values[start..],
                    self.strides[index],
                ));
            }
        }
        let mut state = self.prepared.state();
        self.prepared
            .run_batch(&mut state, count, &reads, &mut writes)
            .map_err(|error| {
                let operand = match error {
                    CallError::ReadOperand { actual, .. }
                    | CallError::WriteOperand { actual, .. } => actual.index(),
                    CallError::ShortBuffer { operand, .. }
                    | CallError::ExtentOverflow { operand, .. } => operand.index(),
                    CallError::ReadCount { .. } | CallError::WriteCount { .. } => 0,
                };
                ExecutionError::InvalidBinding(operand)
            })
    }

    /// `parallel(axis)` on the proven axis: `threads` scoped threads each run a contiguous
    /// lane range on private copies of the writable buffers; the elements a lane range
    /// addresses (disjoint across ranges by the legality condition) are copied back in
    /// thread order. Bitwise equal to [`Self::run`] for every thread count. No reduction is
    /// shared between threads: a reduction element is owned by exactly one lane.
    pub fn run_parallel(
        &self,
        bindings: &mut [BufferBinding<'_>],
        threads: usize,
    ) -> Result<(), ExecutionError> {
        let threads = threads.clamp(1, self.lanes.max(1));
        if threads == 1 {
            return self.run(bindings);
        }
        // Validate the bindings once on the shared buffers without effects: a zero-lane run.
        self.run_lanes(bindings, 0, 0)?;
        let chunk = self.lanes.div_ceil(threads);
        let ranges: Vec<(usize, usize)> = (0..threads)
            .map(|t| (t * chunk, ((t + 1) * chunk).min(self.lanes)))
            .filter(|(start, end)| start < end)
            .collect();
        let shared: Vec<(OperandId, &[f64])> = bindings
            .iter()
            .map(|binding| (binding.operand, &*binding.values))
            .collect();
        let results: Vec<Result<Copies, ExecutionError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = ranges
                .iter()
                .map(|(start, end)| {
                    let shared = &shared;
                    scope.spawn(move || {
                        let mut copies: Vec<(OperandId, Vec<f64>)> = shared
                            .iter()
                            .map(|(operand, values)| (*operand, values.to_vec()))
                            .collect();
                        let mut local: Vec<BufferBinding<'_>> = copies
                            .iter_mut()
                            .map(|(operand, values)| BufferBinding::new(*operand, values))
                            .collect();
                        self.run_lanes(&mut local, *start, end - start)?;
                        Ok(copies)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("lane thread panicked"))
                .collect()
        });
        for ((start, end), result) in ranges.iter().zip(results) {
            let copies = result?;
            for binding in bindings.iter_mut() {
                let index = binding.operand.index();
                if self.access[index] == AccessMode::Read {
                    continue;
                }
                let copy = &copies
                    .iter()
                    .find(|(operand, _)| *operand == binding.operand)
                    .expect("copied binding")
                    .1;
                self.for_each_address(index, *start, *end, |address| {
                    binding.values[address] = copy[address];
                });
            }
        }
        Ok(())
    }

    fn for_each_address(&self, operand: usize, start: usize, end: usize, mut f: impl FnMut(usize)) {
        let (constant, coefficients) = &self.offsets[operand];
        let mut point = vec![0usize; self.lane_extents.len()];
        if self.lane_extents.contains(&0) {
            return;
        }
        loop {
            let base: i128 = constant
                + coefficients
                    .iter()
                    .zip(&point)
                    .map(|(c, p)| c * *p as i128)
                    .sum::<i128>();
            for lane in start..end {
                f((base + (lane * self.strides[operand]) as i128) as usize);
            }
            let mut axis = point.len();
            loop {
                if axis == 0 {
                    return;
                }
                axis -= 1;
                point[axis] += 1;
                if point[axis] < self.lane_extents[axis] {
                    break;
                }
                point[axis] = 0;
            }
        }
    }
}

fn body_reads_axis(kernel: &StructuredKernel, axis: AxisId) -> bool {
    fn expr(e: &ScalarExpr, axis: AxisId) -> bool {
        match e {
            ScalarExpr::Index(a) => *a == axis,
            ScalarExpr::Constant(_) | ScalarExpr::Load(_) | ScalarExpr::Local(_) => false,
            ScalarExpr::Unary { arg, .. } => expr(arg, axis),
            ScalarExpr::Binary { lhs, rhs, .. } => expr(lhs, axis) || expr(rhs, axis),
            ScalarExpr::Fma { lhs, rhs, addend } => {
                expr(lhs, axis) || expr(rhs, axis) || expr(addend, axis)
            }
            ScalarExpr::Select {
                condition,
                if_true,
                if_false,
            } => predicate(condition, axis) || expr(if_true, axis) || expr(if_false, axis),
        }
    }
    fn predicate(p: &Predicate, axis: AxisId) -> bool {
        match p {
            Predicate::Constant(_) => false,
            Predicate::Compare { lhs, rhs, .. } => expr(lhs, axis) || expr(rhs, axis),
            Predicate::Not(v) => predicate(v, axis),
            Predicate::And(l, r) | Predicate::Or(l, r) => predicate(l, axis) || predicate(r, axis),
        }
    }
    kernel.body.statements.iter().any(|statement| {
        let (Statement::Let { value, .. } | Statement::Store { value, .. }) = statement;
        expr(value, axis)
    })
}

fn shift_indices(e: &ScalarExpr, removed: usize) -> ScalarExpr {
    let shift = |p: &Predicate| shift_predicate(p, removed);
    match e {
        ScalarExpr::Index(axis) if axis.index() > removed => {
            ScalarExpr::Index(AxisId::new(axis.index() - 1))
        }
        ScalarExpr::Unary { op, arg } => ScalarExpr::unary(*op, shift_indices(arg, removed)),
        ScalarExpr::Binary { op, lhs, rhs } => ScalarExpr::binary(
            *op,
            shift_indices(lhs, removed),
            shift_indices(rhs, removed),
        ),
        ScalarExpr::Fma { lhs, rhs, addend } => ScalarExpr::fma(
            shift_indices(lhs, removed),
            shift_indices(rhs, removed),
            shift_indices(addend, removed),
        ),
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => ScalarExpr::Select {
            condition: Box::new(shift(condition)),
            if_true: Box::new(shift_indices(if_true, removed)),
            if_false: Box::new(shift_indices(if_false, removed)),
        },
        other => other.clone(),
    }
}

fn shift_predicate(p: &Predicate, removed: usize) -> Predicate {
    match p {
        Predicate::Constant(v) => Predicate::Constant(*v),
        Predicate::Compare { op, lhs, rhs } => Predicate::Compare {
            op: *op,
            lhs: Box::new(shift_indices(lhs, removed)),
            rhs: Box::new(shift_indices(rhs, removed)),
        },
        Predicate::Not(v) => Predicate::Not(Box::new(shift_predicate(v, removed))),
        Predicate::And(l, r) => Predicate::And(
            Box::new(shift_predicate(l, removed)),
            Box::new(shift_predicate(r, removed)),
        ),
        Predicate::Or(l, r) => Predicate::Or(
            Box::new(shift_predicate(l, removed)),
            Box::new(shift_predicate(r, removed)),
        ),
    }
}
