//! Isolated Cranelift prototype consuming Malleus's existing validated executable IR.
//!
//! The only unsafe operations are the documented JIT ABI boundary. The core Malleus
//! crate remains dependency-neutral and forbids unsafe code. This prototype is ordinary
//! finite-precision execution, not an interval/certified arithmetic backend.

use cranelift_codegen::{
    ir::{
        self, AbiParam, InstBuilder, MemFlags, Value,
        condcodes::{FloatCC, IntCC},
        types,
    },
    settings::{self, Configurable},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use malleus::*;
use std::{cell::RefCell, collections::BTreeMap};

pub mod fixtures;

/// Explicit refusal or checked-call error; unsupported IR never falls back silently.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported prototype capability: {0}")]
    Unsupported(String),
    #[error("Cranelift compilation failed: {0}")]
    Compile(String),
    #[error("invalid buffer binding: {0}")]
    Binding(String),
}
type Result<T> = std::result::Result<T, Error>;
type Native = unsafe extern "C" fn(*const *mut f64);

struct CodeMemory(Option<JITModule>);
impl std::ops::Deref for CodeMemory {
    type Target = JITModule;
    fn deref(&self) -> &JITModule {
        self.0.as_ref().expect("live code owner")
    }
}
impl std::ops::DerefMut for CodeMemory {
    fn deref_mut(&mut self) -> &mut JITModule {
        self.0.as_mut().expect("live code owner")
    }
}
impl Drop for CodeMemory {
    fn drop(&mut self) {
        if let Some(module) = self.0.take() {
            // SAFETY: no function pointer escapes the code owner; prepared calls borrow it.
            // Rust cannot drop this owner while any call is running or can be run afterwards.
            unsafe { module.free_memory() }
        }
    }
}

/// A validated, compiled kernel. Module ownership keeps machine code alive.
pub struct JitKernel {
    executable: Executable,
    function: Native,
    _module: CodeMemory,
}

impl JitKernel {
    /// Compile strict scalar arithmetic and sequential loops. No SIMD, tiling or parallel
    /// schedule is accepted yet. Canonical reduction order overrides loop permutation.
    pub fn compile(executable: &Executable) -> Result<Self> {
        check_budget(executable)?;
        let mut flags = settings::builder();
        flags
            .set("opt_level", "speed")
            .map_err(|e| Error::Compile(e.to_string()))?;
        let isa = cranelift_native::builder()
            .map_err(|e| Error::Compile(e.to_string()))?
            .finish(settings::Flags::new(flags))
            .map_err(|e| Error::Compile(e.to_string()))?;
        let mut module = CodeMemory(Some(JITModule::new(JITBuilder::with_isa(
            isa,
            cranelift_module::default_libcall_names(),
        ))));
        let ptr = module.target_config().pointer_type();
        if ptr != types::I64 {
            return Err(Error::Unsupported("64-bit host ABI required".into()));
        }
        let mut context = module.make_context();
        context.func.signature = module.make_signature();
        context.func.signature.params.push(AbiParam::new(ptr));
        let id = module
            .declare_function("kernel", Linkage::Local, &context.func.signature)
            .map_err(|e| Error::Compile(e.to_string()))?;
        let mut builder_context = FunctionBuilderContext::new();
        {
            let mut b = FunctionBuilder::new(&mut context.func, &mut builder_context);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            let table = b.block_params(entry)[0];
            let buffers = (0..executable.kernel().as_kernel().operands.len())
                .map(|i| b.ins().load(ptr, MemFlags::new(), table, (i * 8) as i32))
                .collect();
            let mut lower = Lower {
                b: &mut b,
                k: executable.kernel().as_kernel(),
                buffers,
                coordinates: vec![],
                locals: vec![],
            };
            let zero = lower.b.ins().iconst(types::I64, 0);
            lower.coordinates = vec![zero; lower.k.iteration_domain.rank()];
            let order: Vec<usize> =
                if lower.k.numeric_policy.reduction_order == ReductionOrder::Canonical {
                    (0..lower.k.iteration_domain.rank()).collect()
                } else {
                    executable
                        .schedule()
                        .loop_order
                        .iter()
                        .map(|a| a.index())
                        .collect()
                };
            lower.loops(&order, 0)?;
            lower.b.ins().return_(&[]);
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
        // SAFETY: the generated signature is exactly extern C (pointer-to-pointer table)->().
        // Pointer stays private, and this object owns the module for every prepared call.
        let function = unsafe { std::mem::transmute::<*const u8, Native>(code) };
        Ok(Self {
            executable: executable.clone(),
            function,
            _module: module,
        })
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
        mut bindings: Vec<BufferBinding<'v>>,
    ) -> Result<PreparedCall<'j, 'v>> {
        let k = self.executable.kernel().as_kernel();
        let mut pointers: Vec<*mut f64> = vec![std::ptr::null_mut(); k.operands.len()];
        for binding in bindings.iter_mut() {
            let id = binding.operand.index();
            let op = k
                .operands
                .get(id)
                .ok_or_else(|| Error::Binding(format!("unknown operand {id}")))?;
            if !pointers[id].is_null() {
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
            pointers[id] = binding.values.as_mut_ptr();
        }
        if pointers.iter().any(|p| p.is_null()) {
            return Err(Error::Binding("missing operand".into()));
        }
        Ok(PreparedCall {
            kernel: self,
            _bindings: bindings,
            pointers,
        })
    }

    /// Checked convenience call; preparation cost can be amortized with `prepare`.
    pub fn run(&self, bindings: &mut [BufferBinding<'_>]) -> Result<()> {
        self.prepare(bindings)?.run();
        Ok(())
    }
}

/// A reusable call holding all buffer borrows and the code owner alive.
pub struct PreparedCall<'j, 'v> {
    kernel: &'j JitKernel,
    _bindings: Vec<BufferBinding<'v>>,
    pointers: Vec<*mut f64>,
}
impl PreparedCall<'_, '_> {
    /// Change a readable input without reallocating or invalidating any borrowed buffer.
    /// Output and ReadWrite operands are refused; writes are bounds-checked.
    pub fn update_input(&mut self, id: OperandId, offset: usize, values: &[f64]) -> Result<()> {
        let k = self.kernel.executable.kernel().as_kernel();
        let op = k
            .operands
            .get(id.index())
            .ok_or_else(|| Error::Binding("unknown input".into()))?;
        if op.access != AccessMode::Read {
            return Err(Error::Binding("update requires Read input".into()));
        }
        let binding = self
            ._bindings
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

    /// Inspect a bound buffer between invocations; no mutable pointer escapes.
    pub fn values(&self, id: OperandId) -> Option<&[f64]> {
        self._bindings
            .iter()
            .find(|b| b.operand == id)
            .map(|b| &*b.values)
    }

    /// Execute without repeated binding allocation/checks. Reductions accumulate into
    /// caller-owned outputs, exactly like the interpreter; they are not reset implicitly.
    pub fn run(&mut self) {
        // Re-derive every pointer from the owned `&mut` slices on each call. `update_input`
        // reborrows those slices mutably, which invalidates raw pointers derived earlier
        // under Stacked/Tree Borrows, so the pointers cached by `prepare_owned` are only
        // used there to detect missing operands.
        for binding in self._bindings.iter_mut() {
            self.pointers[binding.operand.index()] = binding.values.as_mut_ptr();
        }
        // SAFETY: prepare checked lengths/uniqueness; the pointers were derived just above
        // from the original mutable slices, which remain exclusively borrowed by `self` and
        // are not touched again until the call returns. Validated affine maps plus
        // check_budget bound every generated address. This borrow keeps kernel/module and
        // all pointers alive and prevents concurrent reuse of outputs.
        unsafe { (self.kernel.function)(self.pointers.as_ptr()) }
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
        if matches!(
            op.access,
            AccessMode::Reduce(ReductionOp::Min | ReductionOp::Max)
        ) {
            return Err(unsupported(op.access));
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
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => {
            predicate_budget(condition, d + 1, n)?;
            expression_budget(if_true, d + 1, n)?;
            expression_budget(if_false, d + 1, n)?;
        }
        _ => {}
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
        _ => {}
    }
    Ok(())
}

struct Lower<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    k: &'a StructuredKernel,
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
                        AccessMode::Write | AccessMode::ReadWrite => {}
                        a => return Err(unsupported(a)),
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
                    _ => return Err(unsupported(op)),
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
                    _ => return Err(unsupported(op)),
                }
            }
            ScalarExpr::Fma { .. } => return Err(unsupported("fma")),
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

/// Adapter to Malleus's REAL local differential campaign, caching compiled executables.
#[derive(Default)]
pub struct CampaignRunner {
    cache: RefCell<BTreeMap<String, JitKernel>>,
}
impl LocalExecutableRunner for CampaignRunner {
    fn identity(&self) -> &str {
        "malleus-cranelift-experiment/0.128.3-v1"
    }
    fn run(&self, e: &Executable, buffers: &mut [Vec<f64>]) -> std::result::Result<(), String> {
        let key = format!(
            "{}:{:?}",
            kernel_digest(e.kernel().as_kernel()),
            e.schedule()
        );
        let mut cache = self.cache.borrow_mut();
        if !cache.contains_key(&key) {
            cache.insert(
                key.clone(),
                JitKernel::compile(e).map_err(|x| x.to_string())?,
            );
        }
        let jit = cache.get(&key).expect("inserted executable");
        let mut bindings: Vec<_> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
            .collect();
        jit.run(&mut bindings).map_err(|x| x.to_string())
    }
}
