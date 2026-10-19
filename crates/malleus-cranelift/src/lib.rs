//! Optional Cranelift JIT backend consuming Malleus's validated executable IR.
//!
//! This crate is a workspace member, never a dependency of the `malleus` core, which stays
//! dependency-light and forbids unsafe code. The only unsafe operations here are the
//! documented JIT ABI boundary. Execution is bitwise identical to the reference interpreter
//! (NaNs equivalent) for every admitted kernel: strict IEEE operations, no implicit
//! contraction or reassociation, explicit `Fma` as one Cranelift `fma`, and every operation
//! whose result depends on the math library (transcendentals, `pow`, `atan2`, `min`, `max`)
//! called out to the reference implementation in `malleus::policy` for the kernel's level.
//!
//! One generated function serves every entry point. Its ABI is
//! `extern "C" fn(table: *const *mut f64, strides: *const usize, points: usize)`: for each
//! point `p` in order it executes the whole kernel with operand `i` based at
//! `table[i] + p * strides[i]` elements. This is exactly the `malleus::prepared` batch
//! contract (`points` sequential calls), so aliasing and overlapping strides need no special
//! case. Single calls pass `points = 1`.

use cranelift_codegen::{
    ir::{
        self, AbiParam, InstBuilder, MemFlags, Value,
        condcodes::{FloatCC, IntCC},
        types,
    },
    isa::TargetIsa,
    settings::{self, Configurable},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use malleus::policy::{self, MathLibrary, NumericLevel};
use malleus::prepared::{CallError, PreparedKernel, ReadBinding, WriteBinding};
use malleus::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub mod fixtures;

/// Identity of this backend's code generator, part of every [`CacheKey`].
pub const BACKEND_VERSION: &str = "malleus-cranelift/0.1.0 cranelift/0.128.3 codegen-v2";
/// The batch ABI every compiled kernel implements (see the crate documentation).
pub const BATCH_ABI: &str = "strided-sequential-points-v1";

/// Explicit refusal or checked-call error; unsupported IR never falls back silently.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported backend capability: {0}")]
    Unsupported(String),
    #[error("Cranelift compilation failed: {0}")]
    Compile(String),
    #[error("invalid buffer binding: {0}")]
    Binding(String),
}
type Result<T> = std::result::Result<T, Error>;
type Native = unsafe extern "C" fn(*const *mut f64, *const usize, usize);

struct CodeMemory(Option<JITModule>);
impl Drop for CodeMemory {
    fn drop(&mut self) {
        if let Some(module) = self.0.take() {
            // SAFETY: no function pointer escapes the code owner; calls borrow the owning
            // `JitKernel`, so none is running or can run once it is dropped.
            unsafe { module.free_memory() }
        }
    }
}
// SAFETY: after `finalize_definitions` the module is never accessed again except by `Drop`
// (which has exclusive ownership); the finalized machine code is immutable and reentrant
// (it keeps no state outside the caller's buffers), so sharing or moving it across threads
// is sound.
unsafe impl Send for CodeMemory {}
// SAFETY: see the `Send` justification; no `&self` method of the module is ever called.
unsafe impl Sync for CodeMemory {}

/// A validated, compiled kernel. Module ownership keeps machine code alive.
pub struct JitKernel {
    executable: Executable,
    prepared: PreparedKernel,
    zero_strides: Vec<usize>,
    function: Native,
    _module: CodeMemory,
}

/// The callout behind every math-library-dependent unary operation: the reference
/// `malleus::policy::unary` on the f64 storage ABI.
extern "C" fn malleus_unary(op: u32, scalar: u32, math: u32, x: f64) -> f64 {
    policy::unary(
        UNARY_OPS[op as usize],
        x,
        SCALARS[scalar as usize],
        MATHS[math as usize],
    )
}

/// The callout behind `pow`, `atan2`, `min` and `max` (and min/max reductions).
extern "C" fn malleus_binary(op: u32, scalar: u32, math: u32, x: f64, y: f64) -> f64 {
    policy::binary(
        BINARY_OPS[op as usize],
        x,
        y,
        SCALARS[scalar as usize],
        MATHS[math as usize],
    )
}

const UNARY_OPS: [UnaryOp; 5] = [
    UnaryOp::Exp,
    UnaryOp::Ln,
    UnaryOp::Sin,
    UnaryOp::Cos,
    UnaryOp::Tan,
];
const BINARY_OPS: [BinaryOp; 4] = [BinaryOp::Pow, BinaryOp::Atan2, BinaryOp::Min, BinaryOp::Max];
const SCALARS: [ScalarType; 2] = [ScalarType::F32, ScalarType::F64];
const MATHS: [MathLibrary; 2] = [MathLibrary::Host, MathLibrary::Portable];

fn native_isa() -> Result<Arc<dyn TargetIsa>> {
    let mut flags = settings::builder();
    flags
        .set("opt_level", "speed")
        .map_err(|e| Error::Compile(e.to_string()))?;
    cranelift_native::builder()
        .map_err(|e| Error::Compile(e.to_string()))?
        .finish(settings::Flags::new(flags))
        .map_err(|e| Error::Compile(e.to_string()))
}

/// The host CPU features Cranelift compiles for, as recorded in every [`CacheKey`].
pub fn cpu_features() -> &'static str {
    static FEATURES: OnceLock<String> = OnceLock::new();
    FEATURES.get_or_init(|| match native_isa() {
        Ok(isa) => {
            let enabled: Vec<String> = isa
                .isa_flags()
                .iter()
                .filter(|v| v.as_bool().unwrap_or(true))
                .map(|v| v.to_string())
                .collect();
            format!("{} {}", isa.triple(), enabled.join(","))
        }
        Err(e) => format!("unavailable: {e}"),
    })
}

impl JitKernel {
    /// Compile strict scalar arithmetic and sequential loops wrapped in the batch point
    /// loop. No SIMD, tiling or parallel schedule is accepted yet. Canonical reduction order
    /// overrides loop permutation.
    pub fn compile(executable: &Executable) -> Result<Self> {
        check_budget(executable)?;
        let prepared = PreparedKernel::new(executable)
            .map_err(|e| Error::Unsupported(format!("preparation refused: {e}")))?;
        let mut builder =
            JITBuilder::with_isa(native_isa()?, cranelift_module::default_libcall_names());
        builder.symbol("malleus_unary", malleus_unary as *const u8);
        builder.symbol("malleus_binary", malleus_binary as *const u8);
        let mut module = CodeMemory(Some(JITModule::new(builder)));
        let function = {
            let module = module.0.as_mut().expect("live code owner");
            let ptr = module.target_config().pointer_type();
            if ptr != types::I64 {
                return Err(Error::Unsupported("64-bit host ABI required".into()));
            }
            let mut unary_sig = module.make_signature();
            unary_sig
                .params
                .extend([types::I32, types::I32, types::I32].map(AbiParam::new));
            unary_sig.params.push(AbiParam::new(types::F64));
            unary_sig.returns.push(AbiParam::new(types::F64));
            let mut binary_sig = unary_sig.clone();
            binary_sig.params.push(AbiParam::new(types::F64));
            let unary_id = declare(module, "malleus_unary", Linkage::Import, &unary_sig)?;
            let binary_id = declare(module, "malleus_binary", Linkage::Import, &binary_sig)?;

            let mut context = module.make_context();
            context.func.signature = module.make_signature();
            context
                .func
                .signature
                .params
                .extend([ptr, ptr, ptr].map(AbiParam::new));
            let id = declare(module, "kernel", Linkage::Local, &context.func.signature)?;
            let mut builder_context = FunctionBuilderContext::new();
            {
                let mut b = FunctionBuilder::new(&mut context.func, &mut builder_context);
                let unary = module.declare_func_in_func(unary_id, b.func);
                let binary = module.declare_func_in_func(binary_id, b.func);
                let entry = b.create_block();
                b.append_block_params_for_function_params(entry);
                b.switch_to_block(entry);
                let [table, strides, points] = b.block_params(entry) else {
                    unreachable!("three parameters")
                };
                let (table, strides, points) = (*table, *strides, *points);
                let k = executable.kernel().as_kernel();
                let operands = k.operands.len();
                let mut bases = Vec::with_capacity(operands);
                let mut steps = Vec::with_capacity(operands);
                for i in 0..operands {
                    bases.push(
                        b.ins()
                            .load(ptr, MemFlags::trusted(), table, (i * 8) as i32),
                    );
                    let stride = b
                        .ins()
                        .load(ptr, MemFlags::trusted(), strides, (i * 8) as i32);
                    steps.push(b.ins().ishl_imm(stride, 3));
                }
                // for p in 0..points: rebase every operand by p * stride, run the kernel.
                let header = b.create_block();
                let body = b.create_block();
                let exit = b.create_block();
                b.append_block_param(header, types::I64);
                let zero = b.ins().iconst(types::I64, 0);
                b.ins().jump(header, &[zero.into()]);
                b.switch_to_block(header);
                let p = b.block_params(header)[0];
                let more = b.ins().icmp(IntCC::UnsignedLessThan, p, points);
                b.ins().brif(more, body, &[], exit, &[]);
                b.switch_to_block(body);
                let buffers = bases
                    .iter()
                    .zip(&steps)
                    .map(|(&base, &step)| {
                        let offset = b.ins().imul(p, step);
                        b.ins().iadd(base, offset)
                    })
                    .collect();
                let mut lower = Lower {
                    b: &mut b,
                    k,
                    unary,
                    binary,
                    buffers,
                    coordinates: vec![],
                    locals: vec![],
                };
                let zero = lower.b.ins().iconst(types::I64, 0);
                lower.coordinates = vec![zero; k.iteration_domain.rank()];
                let order: Vec<usize> =
                    if k.numeric_policy.reduction_order == ReductionOrder::Canonical {
                        (0..k.iteration_domain.rank()).collect()
                    } else {
                        executable
                            .schedule()
                            .loop_order
                            .iter()
                            .map(|a| a.index())
                            .collect()
                    };
                lower.loops(&order, 0)?;
                let next = b.ins().iadd_imm(p, 1);
                b.ins().jump(header, &[next.into()]);
                b.switch_to_block(exit);
                b.ins().return_(&[]);
                b.seal_all_blocks();
                b.finalize();
            }
            module
                .define_function(id, &mut context)
                .map_err(|e| Error::Compile(e.to_string()))?;
            module.clear_context(&mut context);
            module
                .finalize_definitions()
                .map_err(|e| Error::Compile(e.to_string()))?;
            let code = module.get_finalized_function(id);
            // SAFETY: the generated signature is exactly `extern "C" fn(*const *mut f64,
            // *const usize, usize)`. The pointer stays private and this object owns the module
            // for every call.
            unsafe { std::mem::transmute::<*const u8, Native>(code) }
        };
        Ok(Self {
            zero_strides: vec![0; executable.kernel().as_kernel().operands.len()],
            executable: executable.clone(),
            prepared,
            function,
            _module: module,
        })
    }

    /// The executable this kernel was compiled from.
    pub fn executable(&self) -> &Executable {
        &self.executable
    }

    /// Positional read slots, as in `malleus::prepared::PreparedKernel::read_operands`.
    pub fn read_operands(&self) -> &[OperandId] {
        self.prepared.read_operands()
    }

    /// Positional writable slots, as in `PreparedKernel::write_operands`.
    pub fn write_operands(&self) -> &[OperandId] {
        self.prepared.write_operands()
    }

    /// A reusable call state sized for this kernel (one may serve many kernels).
    pub fn state(&self) -> JitState {
        let mut state = JitState::default();
        state.reserve(self.zero_strides.len());
        state
    }

    /// Validate all bindings before entering generated code, including offsets, order,
    /// missing/duplicate operands and lengths. Mutable slice borrows prohibit safe aliasing.
    pub fn prepare<'j, 'b>(
        &'j self,
        bindings: &'b mut [BufferBinding<'_>],
    ) -> Result<PreparedCall<'j, 'b>> {
        let bindings = bindings
            .iter_mut()
            .map(|b| BufferBinding::new(b.operand, &mut *b.values))
            .collect();
        self.prepare_owned(bindings)
    }

    /// Own the binding descriptors while borrowing their buffers. This supports returning
    /// a prepared session without borrowing a temporary descriptor array.
    pub fn prepare_owned<'j, 'v>(
        &'j self,
        bindings: Vec<BufferBinding<'v>>,
    ) -> Result<PreparedCall<'j, 'v>> {
        Ok(PreparedCall {
            kernel: self,
            buffers: BoundBuffers::new(self.executable.kernel().as_kernel(), bindings)?,
        })
    }

    /// Checked convenience call; preparation cost can be amortized with `prepare`.
    pub fn run(&self, bindings: &mut [BufferBinding<'_>]) -> Result<()> {
        self.prepare(bindings)?.run();
        Ok(())
    }

    /// One allocation-free positional call: [`run_batch`](Self::run_batch) with one point.
    pub fn call(
        &self,
        state: &mut JitState,
        reads: &[ReadBinding<'_>],
        writes: &mut [WriteBinding<'_>],
    ) -> std::result::Result<(), CallError> {
        self.run_batch(state, 1, reads, writes)
    }

    /// `points` sequential calls over strided bindings with the `malleus::prepared` batch
    /// contract and refusals (the core `PreparedKernel::check_bindings` decides), executed
    /// by one native loop. No allocation once `state` is sized.
    pub fn run_batch(
        &self,
        state: &mut JitState,
        points: usize,
        reads: &[ReadBinding<'_>],
        writes: &mut [WriteBinding<'_>],
    ) -> std::result::Result<(), CallError> {
        self.prepared.check_bindings(points, reads, writes)?;
        if points == 0 {
            return Ok(());
        }
        state.bind(self.zero_strides.len(), reads, writes);
        // SAFETY: `check_bindings` proved every slot names its operand and holds
        // `(points - 1) * stride + region end` elements, so every generated address of every
        // point is in bounds. `bind` derived the pointers just above from the live borrows
        // `reads` (never stored to: validation rejects stores to Read operands) and `writes`
        // (exclusively borrowed for this call; aliasing between points is the defined
        // sequential order). The state's tables have one entry per operand.
        unsafe { (self.function)(state.pointers.as_ptr(), state.strides.as_ptr(), points) }
        Ok(())
    }
}

fn declare(
    module: &mut JITModule,
    name: &str,
    linkage: Linkage,
    signature: &ir::Signature,
) -> Result<FuncId> {
    module
        .declare_function(name, linkage, signature)
        .map_err(|e| Error::Compile(e.to_string()))
}

/// Reusable pointer and stride tables for [`JitKernel::call`] / [`JitKernel::run_batch`].
/// The pointers are rewritten from the call's bindings on every call and never read
/// between calls.
#[derive(Debug, Default)]
pub struct JitState {
    pointers: Vec<*mut f64>,
    strides: Vec<usize>,
}

// SAFETY: the raw pointers are scratch: every call overwrites them from its own borrowed
// bindings before use and nothing dereferences them afterwards, so moving the state to
// another thread cannot create access to a buffer.
unsafe impl Send for JitState {}

impl JitState {
    fn reserve(&mut self, operands: usize) {
        if self.pointers.len() < operands {
            self.pointers.resize(operands, std::ptr::null_mut());
            self.strides.resize(operands, 0);
        }
    }

    fn bind(
        &mut self,
        operands: usize,
        reads: &[ReadBinding<'_>],
        writes: &mut [WriteBinding<'_>],
    ) {
        self.reserve(operands);
        for binding in reads {
            let id = binding.operand.index();
            self.pointers[id] = binding.values.as_ptr().cast_mut();
            self.strides[id] = binding.stride;
        }
        for binding in writes.iter_mut() {
            let id = binding.operand.index();
            self.pointers[id] = binding.values.as_mut_ptr();
            self.strides[id] = binding.stride;
        }
    }
}

/// Checked, owned `BufferBinding`s plus the raw pointer table handed to generated code.
struct BoundBuffers<'v> {
    bindings: Vec<BufferBinding<'v>>,
    pointers: Vec<*mut f64>,
}

impl<'v> BoundBuffers<'v> {
    fn new(k: &StructuredKernel, bindings: Vec<BufferBinding<'v>>) -> Result<Self> {
        let mut seen = vec![false; k.operands.len()];
        for binding in &bindings {
            let id = binding.operand.index();
            let op = k
                .operands
                .get(id)
                .ok_or_else(|| Error::Binding(format!("unknown operand {id}")))?;
            if std::mem::replace(&mut seen[id], true) {
                return Err(Error::Binding(format!("duplicate operand {id}")));
            }
            let required = op
                .region
                .offset
                .checked_add(op.region.length)
                .ok_or_else(|| Error::Binding("region overflow".into()))?;
            if binding.values.len() < required {
                return Err(Error::Binding(format!("short operand {id}")));
            }
        }
        if seen.iter().any(|present| !present) {
            return Err(Error::Binding("missing operand".into()));
        }
        Ok(Self {
            bindings,
            pointers: vec![std::ptr::null_mut(); k.operands.len()],
        })
    }

    /// Re-derive every pointer from the owned `&mut` slices. `update_input` reborrows those
    /// slices mutably, which invalidates raw pointers derived earlier under Stacked/Tree
    /// Borrows, so pointers are never cached across such a reborrow.
    fn table(&mut self) -> *const *mut f64 {
        for binding in self.bindings.iter_mut() {
            self.pointers[binding.operand.index()] = binding.values.as_mut_ptr();
        }
        self.pointers.as_ptr()
    }

    fn update_input(
        &mut self,
        k: &StructuredKernel,
        id: OperandId,
        offset: usize,
        values: &[f64],
    ) -> Result<()> {
        let op = k
            .operands
            .get(id.index())
            .ok_or_else(|| Error::Binding("unknown input".into()))?;
        if op.access != AccessMode::Read {
            return Err(Error::Binding("update requires Read input".into()));
        }
        let binding = self
            .bindings
            .iter_mut()
            .find(|b| b.operand == id)
            .ok_or_else(|| Error::Binding("missing input".into()))?;
        let end = offset
            .checked_add(values.len())
            .ok_or_else(|| Error::Binding("input offset overflow".into()))?;
        let target = binding
            .values
            .get_mut(offset..end)
            .ok_or_else(|| Error::Binding("input update out of bounds".into()))?;
        target.copy_from_slice(values);
        Ok(())
    }

    fn values(&self, id: OperandId) -> Option<&[f64]> {
        self.bindings
            .iter()
            .find(|b| b.operand == id)
            .map(|b| &*b.values)
    }
}

/// A reusable call holding all buffer borrows and the code owner alive.
pub struct PreparedCall<'j, 'v> {
    kernel: &'j JitKernel,
    buffers: BoundBuffers<'v>,
}
impl PreparedCall<'_, '_> {
    /// Change a readable input without reallocating or invalidating any borrowed buffer.
    /// Output and ReadWrite operands are refused; writes are bounds-checked.
    pub fn update_input(&mut self, id: OperandId, offset: usize, values: &[f64]) -> Result<()> {
        let k = self.kernel.executable.kernel().as_kernel();
        self.buffers.update_input(k, id, offset, values)
    }

    /// Inspect a bound buffer between invocations; no mutable pointer escapes.
    pub fn values(&self, id: OperandId) -> Option<&[f64]> {
        self.buffers.values(id)
    }

    /// Execute without repeated binding allocation/checks. Reductions accumulate into
    /// caller-owned outputs, exactly like the interpreter; they are not reset implicitly.
    pub fn run(&mut self) {
        let table = self.buffers.table();
        // SAFETY: `BoundBuffers::new` checked lengths, uniqueness and completeness; `table`
        // derived the pointers just above from the owned mutable slices, which stay
        // exclusively borrowed by `self` and untouched until the call returns. Validated
        // affine maps plus `check_budget` bound every generated address; `points = 1` with
        // all-zero strides reads exactly one stride entry per operand.
        unsafe { (self.kernel.function)(table, self.kernel.zero_strides.as_ptr(), 1) }
    }
}

fn unsupported(value: impl std::fmt::Debug) -> Error {
    Error::Unsupported(format!("{value:?}"))
}

fn check_budget(e: &Executable) -> Result<()> {
    let s = e.schedule();
    let k = e.kernel().as_kernel();
    if !s.tiles.is_empty() || s.vectorization.is_some() || !s.parallel.is_empty() {
        return Err(unsupported("tiled/vector/parallel schedule"));
    }
    if k.iteration_domain.rank() > 8 || k.body.statements.len() > 256 || k.operands.len() > 256 {
        return Err(unsupported("IR budget"));
    }
    if k.iteration_domain.extents.iter().any(|&x| x > 1_000_000) {
        return Err(unsupported("axis extent budget"));
    }
    if k.iteration_domain
        .extents
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .is_none_or(|n| n > 10_000_000)
    {
        return Err(unsupported("iteration budget"));
    }
    let mut nodes = 0;
    for s in &k.body.statements {
        let (Statement::Let { value, .. } | Statement::Store { value, .. }) = s;
        expression_budget(value, 0, &mut nodes)?;
    }
    for op in &k.operands {
        let n = op
            .shape
            .iter()
            .try_fold(1usize, |n, &x| n.checked_mul(x))
            .ok_or_else(|| unsupported("shape overflow"))?;
        if n > 100_000_000 || op.region.offset > 100_000_000 || op.region.length > 100_000_000 {
            return Err(unsupported("buffer budget"));
        }
    }
    for m in &k.indexing_maps {
        for x in &m.results {
            if x.constant.unsigned_abs() > 1_000_000_000_000
                || x.terms
                    .iter()
                    .any(|t| t.coefficient.unsigned_abs() > 1_000_000)
            {
                return Err(unsupported("affine coefficient budget"));
            }
        }
    }
    Ok(())
}

fn expression_budget(e: &ScalarExpr, d: usize, n: &mut usize) -> Result<()> {
    *n += 1;
    if d > 128 || *n > 4096 {
        return Err(unsupported("expression budget"));
    }
    match e {
        ScalarExpr::Unary { arg, .. } => expression_budget(arg, d + 1, n)?,
        ScalarExpr::Binary { lhs, rhs, .. } => {
            expression_budget(lhs, d + 1, n)?;
            expression_budget(rhs, d + 1, n)?;
        }
        ScalarExpr::Fma { lhs, rhs, addend } => {
            expression_budget(lhs, d + 1, n)?;
            expression_budget(rhs, d + 1, n)?;
            expression_budget(addend, d + 1, n)?;
        }
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => {
            predicate_budget(condition, d + 1, n)?;
            expression_budget(if_true, d + 1, n)?;
            expression_budget(if_false, d + 1, n)?;
        }
        ScalarExpr::Constant(_)
        | ScalarExpr::Index(_)
        | ScalarExpr::Load(_)
        | ScalarExpr::Local(_) => {}
    }
    Ok(())
}
fn predicate_budget(p: &Predicate, d: usize, n: &mut usize) -> Result<()> {
    *n += 1;
    if d > 128 || *n > 4096 {
        return Err(unsupported("predicate budget"));
    }
    match p {
        Predicate::Compare { lhs, rhs, .. } => {
            expression_budget(lhs, d + 1, n)?;
            expression_budget(rhs, d + 1, n)?;
        }
        Predicate::Not(x) => predicate_budget(x, d + 1, n)?,
        Predicate::And(a, b) | Predicate::Or(a, b) => {
            predicate_budget(a, d + 1, n)?;
            predicate_budget(b, d + 1, n)?;
        }
        Predicate::Constant(_) => {}
    }
    Ok(())
}

struct Lower<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    k: &'a StructuredKernel,
    unary: ir::FuncRef,
    binary: ir::FuncRef,
    buffers: Vec<Value>,
    coordinates: Vec<Value>,
    locals: Vec<Value>,
}

impl Lower<'_, '_> {
    fn ty(&self) -> ir::Type {
        match self.k.numeric_policy.scalar_type {
            ScalarType::F32 => types::F32,
            ScalarType::F64 => types::F64,
        }
    }

    /// Calls `malleus_unary`/`malleus_binary` (the reference operation) on f64-ABI values:
    /// f32 values are widened exactly, rounded to f32 inside the callout, and narrowed
    /// exactly afterwards.
    fn callout(&mut self, binary: bool, op: usize, args: &[Value]) -> Value {
        let scalar = match self.k.numeric_policy.scalar_type {
            ScalarType::F32 => 0,
            ScalarType::F64 => 1,
        };
        let math = match self.k.numeric_policy.level.math() {
            MathLibrary::Host => 0,
            MathLibrary::Portable => 1,
        };
        let mut call_args = vec![
            self.b.ins().iconst(types::I32, op as i64),
            self.b.ins().iconst(types::I32, scalar),
            self.b.ins().iconst(types::I32, math),
        ];
        for &a in args {
            call_args.push(if self.ty() == types::F32 {
                self.b.ins().fpromote(types::F64, a)
            } else {
                a
            });
        }
        let callee = if binary { self.binary } else { self.unary };
        let call = self.b.ins().call(callee, &call_args);
        let v = self.b.inst_results(call)[0];
        if self.ty() == types::F32 {
            self.b.ins().fdemote(types::F32, v)
        } else {
            v
        }
    }

    fn binary_callout(&mut self, op: BinaryOp, a: Value, b: Value) -> Value {
        let index = BINARY_OPS
            .iter()
            .position(|o| *o == op)
            .expect("callout op");
        self.callout(true, index, &[a, b])
    }

    fn loops(&mut self, order: &[usize], depth: usize) -> Result<()> {
        if depth == order.len() {
            return self.body();
        }
        let axis = order[depth];
        let header = self.b.create_block();
        let body = self.b.create_block();
        let exit = self.b.create_block();
        self.b.append_block_param(header, types::I64);
        let zero = self.b.ins().iconst(types::I64, 0);
        self.b.ins().jump(header, &[zero.into()]);
        self.b.switch_to_block(header);
        let i = self.b.block_params(header)[0];
        let cond = self.b.ins().icmp_imm(
            IntCC::UnsignedLessThan,
            i,
            self.k.iteration_domain.extents[axis] as i64,
        );
        self.b.ins().brif(cond, body, &[], exit, &[]);
        self.b.switch_to_block(body);
        self.coordinates[axis] = i;
        self.loops(order, depth + 1)?;
        let next = self.b.ins().iadd_imm(i, 1);
        self.b.ins().jump(header, &[next.into()]);
        self.b.switch_to_block(exit);
        Ok(())
    }

    fn address(&mut self, id: OperandId) -> Value {
        let op = &self.k.operands[id.index()];
        let map = self
            .k
            .indexing_maps
            .iter()
            .find(|m| m.operand == id)
            .expect("validated indexing map");
        let mut offset = self.b.ins().iconst(types::I64, op.region.offset as i64);
        let mut stride = 1usize;
        for &dimension in &op.layout.minor_to_major {
            let x = &map.results[dimension];
            let mut index = self.b.ins().iconst(types::I64, x.constant as i64);
            for term in &x.terms {
                let v = self
                    .b
                    .ins()
                    .imul_imm(self.coordinates[term.axis.index()], term.coefficient as i64);
                index = self.b.ins().iadd(index, v);
            }
            let v = self.b.ins().imul_imm(index, stride as i64);
            offset = self.b.ins().iadd(offset, v);
            stride *= op.shape[dimension];
        }
        let bytes = self.b.ins().ishl_imm(offset, 3);
        self.b.ins().iadd(self.buffers[id.index()], bytes)
    }
    fn load(&mut self, id: OperandId) -> Value {
        let addr = self.address(id);
        let v = self.b.ins().load(types::F64, MemFlags::new(), addr, 0);
        if self.ty() == types::F32 {
            self.b.ins().fdemote(types::F32, v)
        } else {
            v
        }
    }
    fn body(&mut self) -> Result<()> {
        self.locals.clear();
        for statement in &self.k.body.statements {
            match statement {
                Statement::Let { value, .. } => {
                    let v = self.expr(value, 0)?;
                    self.locals.push(v);
                }
                Statement::Store { operand, value } => {
                    let mut v = self.expr(value, 0)?;
                    match self.k.operands[operand.index()].access {
                        AccessMode::Reduce(ReductionOp::Add) => {
                            let old = self.load(*operand);
                            v = self.b.ins().fadd(old, v);
                        }
                        AccessMode::Reduce(ReductionOp::Multiply) => {
                            let old = self.load(*operand);
                            v = self.b.ins().fmul(old, v);
                        }
                        AccessMode::Reduce(ReductionOp::Min) => {
                            let old = self.load(*operand);
                            v = self.binary_callout(BinaryOp::Min, old, v);
                        }
                        AccessMode::Reduce(ReductionOp::Max) => {
                            let old = self.load(*operand);
                            v = self.binary_callout(BinaryOp::Max, old, v);
                        }
                        AccessMode::Write | AccessMode::ReadWrite => {}
                        AccessMode::Read => unreachable!("validated store"),
                    }
                    if self.ty() == types::F32 {
                        v = self.b.ins().fpromote(types::F64, v);
                    }
                    let addr = self.address(*operand);
                    self.b.ins().store(MemFlags::new(), v, addr, 0);
                }
            }
        }
        Ok(())
    }
    fn expr(&mut self, e: &ScalarExpr, depth: usize) -> Result<Value> {
        if depth > 128 {
            return Err(unsupported("expression depth"));
        }
        Ok(match e {
            ScalarExpr::Constant(x) => {
                if self.ty() == types::F32 {
                    self.b.ins().f32const(*x as f32)
                } else {
                    self.b.ins().f64const(*x)
                }
            }
            ScalarExpr::Index(a) => {
                let ty = self.ty();
                self.b.ins().fcvt_from_uint(ty, self.coordinates[a.index()])
            }
            ScalarExpr::Local(l) => self.locals[l.index()],
            ScalarExpr::Load(id) => self.load(*id),
            ScalarExpr::Unary { op, arg } => {
                let a = self.expr(arg, depth + 1)?;
                match op {
                    UnaryOp::Neg => self.b.ins().fneg(a),
                    UnaryOp::Abs => self.b.ins().fabs(a),
                    UnaryOp::Sqrt => self.b.ins().sqrt(a),
                    UnaryOp::Floor => self.b.ins().floor(a),
                    UnaryOp::Ceil => self.b.ins().ceil(a),
                    op => {
                        let index = UNARY_OPS.iter().position(|o| o == op).expect("callout op");
                        self.callout(false, index, &[a])
                    }
                }
            }
            ScalarExpr::Binary { op, lhs, rhs } => {
                let a = self.expr(lhs, depth + 1)?;
                let b = self.expr(rhs, depth + 1)?;
                match op {
                    BinaryOp::Add => self.b.ins().fadd(a, b),
                    BinaryOp::Sub => self.b.ins().fsub(a, b),
                    BinaryOp::Mul => self.b.ins().fmul(a, b),
                    BinaryOp::Div => self.b.ins().fdiv(a, b),
                    op => self.binary_callout(*op, a, b),
                }
            }
            ScalarExpr::Fma { lhs, rhs, addend } => {
                let a = self.expr(lhs, depth + 1)?;
                let b = self.expr(rhs, depth + 1)?;
                let c = self.expr(addend, depth + 1)?;
                self.b.ins().fma(a, b, c)
            }
            ScalarExpr::Select {
                condition,
                if_true,
                if_false,
            } => {
                let c = self.predicate(condition, depth + 1)?;
                let t = self.b.create_block();
                let f = self.b.create_block();
                let end = self.b.create_block();
                self.b.append_block_param(end, self.ty());
                self.b.ins().brif(c, t, &[], f, &[]);
                self.b.switch_to_block(t);
                let v = self.expr(if_true, depth + 1)?;
                self.b.ins().jump(end, &[v.into()]);
                self.b.switch_to_block(f);
                let v = self.expr(if_false, depth + 1)?;
                self.b.ins().jump(end, &[v.into()]);
                self.b.switch_to_block(end);
                self.b.block_params(end)[0]
            }
        })
    }
    fn predicate(&mut self, p: &Predicate, depth: usize) -> Result<Value> {
        if depth > 128 {
            return Err(unsupported("predicate depth"));
        }
        Ok(match p {
            Predicate::Constant(b) => self.b.ins().iconst(types::I8, i64::from(*b)),
            Predicate::Compare { op, lhs, rhs } => {
                let a = self.expr(lhs, depth + 1)?;
                let b = self.expr(rhs, depth + 1)?;
                self.b.ins().fcmp(
                    match op {
                        CompareOp::Eq => FloatCC::Equal,
                        CompareOp::NotEq => FloatCC::NotEqual,
                        CompareOp::Less => FloatCC::LessThan,
                        CompareOp::LessEqual => FloatCC::LessThanOrEqual,
                        CompareOp::Greater => FloatCC::GreaterThan,
                        CompareOp::GreaterEqual => FloatCC::GreaterThanOrEqual,
                    },
                    a,
                    b,
                )
            }
            Predicate::Not(x) => {
                let x = self.predicate(x, depth + 1)?;
                self.b.ins().icmp_imm(IntCC::Equal, x, 0)
            }
            Predicate::And(a, b) | Predicate::Or(a, b) => {
                let a = self.predicate(a, depth + 1)?;
                let rhs = self.b.create_block();
                let shortcut = self.b.create_block();
                let end = self.b.create_block();
                self.b.append_block_param(end, types::I8);
                let is_and = matches!(p, Predicate::And(..));
                if is_and {
                    self.b.ins().brif(a, rhs, &[], shortcut, &[]);
                } else {
                    self.b.ins().brif(a, shortcut, &[], rhs, &[]);
                }
                self.b.switch_to_block(shortcut);
                let v = self.b.ins().iconst(types::I8, i64::from(!is_and));
                self.b.ins().jump(end, &[v.into()]);
                self.b.switch_to_block(rhs);
                let v = self.predicate(b, depth + 1)?;
                self.b.ins().jump(end, &[v.into()]);
                self.b.switch_to_block(end);
                self.b.block_params(end)[0]
            }
        })
    }
}

/// Everything compiled code depends on: the same key always maps to interchangeable code.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// `kernel_digest` hex (covers the IR, scalar type and numeric level).
    pub kernel_digest: String,
    /// The complete schedule (canonical JSON of `KernelSchedule`).
    pub schedule: String,
    pub level: NumericLevel,
    pub backend: &'static str,
    pub cpu_features: &'static str,
    pub batch_abi: &'static str,
}

impl CacheKey {
    pub fn new(e: &Executable) -> Self {
        let k = e.kernel().as_kernel();
        Self {
            kernel_digest: kernel_digest(k).hex,
            schedule: format!("{:?}", e.schedule()),
            level: k.numeric_policy.level,
            backend: BACKEND_VERSION,
            cpu_features: cpu_features(),
            batch_abi: BATCH_ABI,
        }
    }
}

/// A thread-safe in-memory compile cache. Compilation happens outside the lock; when two
/// threads race on one key, the first inserted kernel wins and the other is dropped.
#[derive(Default)]
pub struct JitCache {
    kernels: Mutex<HashMap<CacheKey, Arc<JitKernel>>>,
}

impl JitCache {
    /// The cached kernel for `e`, compiling it on a miss.
    pub fn get_or_compile(&self, e: &Executable) -> Result<Arc<JitKernel>> {
        let key = CacheKey::new(e);
        if let Some(kernel) = self.lock().get(&key) {
            return Ok(Arc::clone(kernel));
        }
        let compiled = Arc::new(JitKernel::compile(e)?);
        Ok(Arc::clone(self.lock().entry(key).or_insert(compiled)))
    }

    /// Number of cached kernels.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<CacheKey, Arc<JitKernel>>> {
        self.kernels
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Adapter to Malleus's local differential campaign, caching compiled executables.
#[derive(Default)]
pub struct CampaignRunner {
    cache: JitCache,
}
impl LocalExecutableRunner for CampaignRunner {
    fn identity(&self) -> &str {
        BACKEND_VERSION
    }
    fn run(&self, e: &Executable, buffers: &mut [Vec<f64>]) -> std::result::Result<(), String> {
        let jit = self.cache.get_or_compile(e).map_err(|x| x.to_string())?;
        let mut bindings: Vec<_> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
            .collect();
        jit.run(&mut bindings).map_err(|x| x.to_string())
    }
}

#[cfg(test)]
mod tests {
    //! Pointer handling that runs without generated code, so it also runs under miri.
    use super::*;

    fn kernel() -> StructuredKernel {
        fixtures::polynomial(2)
    }

    /// The cadabra3 miri regression: a pointer cached before `update_input` reborrowed the
    /// slice was UB under Stacked and Tree Borrows; `table` re-derives every pointer.
    #[test]
    fn pointer_table_is_rederived_after_input_updates() {
        let k = kernel();
        let mut buffers = [vec![1., 2.], vec![0.], vec![0.], vec![0.], vec![0.; 2]];
        let bindings = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
            .collect();
        let mut bound = BoundBuffers::new(&k, bindings).unwrap();
        let first = bound.table();
        // SAFETY: test-only stand-in for generated code: entries are fresh pointers into
        // live, length-checked buffers.
        unsafe { assert_eq!(*(*first), 1.) };
        bound.update_input(&k, OperandId::new(0), 0, &[3.]).unwrap();
        let table = bound.table();
        // SAFETY: as above; the table was re-derived after the reborrow in update_input.
        unsafe {
            assert_eq!(*(*table), 3.);
            *(*table.add(4)).add(1) = 7.;
        }
        assert_eq!(bound.values(OperandId::new(4)).unwrap(), [0., 7.]);
        assert!(bound.update_input(&k, OperandId::new(4), 0, &[1.]).is_err());
        assert!(bound.update_input(&k, OperandId::new(0), 2, &[1.]).is_err());
    }

    #[test]
    fn bound_buffers_refuse_bad_bindings() {
        let k = kernel();
        let mut short = [0.];
        assert!(
            BoundBuffers::new(&k, vec![BufferBinding::new(OperandId::new(0), &mut short)]).is_err()
        );
        let (mut a, mut b) = ([0.; 2], [0.; 2]);
        let duplicate = vec![
            BufferBinding::new(OperandId::new(0), &mut a),
            BufferBinding::new(OperandId::new(0), &mut b),
        ];
        assert!(BoundBuffers::new(&k, duplicate).is_err());
        assert!(BoundBuffers::new(&k, vec![]).is_err());
    }

    #[test]
    fn batch_state_points_at_the_current_bindings() {
        let (r0, mut w0) = (vec![1., 2., 3.], vec![0.; 3]);
        let reads = [ReadBinding::strided(OperandId::new(0), &r0, 1)];
        let mut writes = [WriteBinding::strided(OperandId::new(1), &mut w0, 1)];
        let mut state = JitState::default();
        for round in 0..2 {
            state.bind(2, &reads, &mut writes);
            assert_eq!(state.strides[..2], [1, 1]);
            // SAFETY: test-only stand-in for generated code, within both 3-element buffers;
            // the read pointer is only read.
            unsafe {
                for p in 0..3 {
                    let x = *state.pointers[0].add(p);
                    *state.pointers[1].add(p) = x + f64::from(round);
                }
            }
        }
        assert_eq!(*writes[0].values, [2., 3., 4.]);
    }
}
