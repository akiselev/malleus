//! Deterministic safe Rust source lowering for local kernels.
use crate::*;
use std::{error::Error, fmt, fmt::Write};

/// Unsupported capability, bounded-work refusal or invalid generated symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringError {
    /// The requested executable exceeds this backend's implemented contract.
    Unsupported(String),
    /// A transformed kernel failed structural validation.
    Compile(String),
}
impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kernel lowering: {self:?}")
    }
}
impl Error for LoweringError {}

/// Emit a standalone safe Rust function using operand-order slice arguments.
/// Read operands take `&[f64]`, all others `&mut [f64]`; f32 arithmetic retains
/// the reference f64 storage ABI. Length checks precede all effects. The function
/// returns `Result<(), &'static str>` and needs no Malleus runtime dependency.
/// Compile with normal Rust FP semantics, never fast-math/reassociation overrides.
///
/// Explicit `Fma` nodes become `mul_add`; nothing else is ever fused. A kernel at level
/// `StrictPortable` calls `::libm::{exp, log, sin, cos, tan, pow, atan2, fmin, fmax}` (and
/// their `f` variants for f32), so the crate compiling its source must depend on exactly the
/// [`policy::PORTABLE_MATH`] pin (`libm = { version = "=0.2.16", default-features = false }`).
/// Kernels at the other levels remain dependency-free.
pub fn lower_rust(e: &Executable, name: &str) -> Result<String, LoweringError> {
    lower_rust_impl(e, name, None)
}

/// Emit a fully unrolled kernel when its entire iteration domain fits `limit`
/// (at most 64 iterations). Traversal order and all FP operations are retained.
/// This is useful for fixed-size local guard batches; expansion is explicitly bounded.
pub fn lower_rust_unrolled(
    e: &Executable,
    name: &str,
    limit: usize,
) -> Result<String, LoweringError> {
    if limit == 0 || limit > 64 {
        return Err(LoweringError::Unsupported(
            "unroll budget must be 1..=64".into(),
        ));
    }
    lower_rust_impl(e, name, Some(limit))
}
/// Options of [`lower_rust_batched`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchedRustOptions {
    /// Points per lane block (`[T; W]` lane arrays), 1..=64.
    pub width: usize,
    /// Also emit an AVX2 copy of the kernel, selected at run time with
    /// `is_x86_feature_detected!("avx2")` on x86_64 (one `unsafe` call of a
    /// `#[target_feature]` function; every other line stays safe Rust). AVX2 does not
    /// change any floating-point result: rustc never contracts or reassociates without
    /// explicit fast-math, and the `fma` feature is not enabled.
    pub avx2_dispatch: bool,
}

impl Default for BatchedRustOptions {
    fn default() -> Self {
        Self {
            width: 8,
            avx2_dispatch: false,
        }
    }
}

/// Emit a standalone batched Rust function with the `malleus::prepared::run_batch` contract:
///
/// `pub fn name(points: usize, b0: &[f64], s0: usize, b1: &mut [f64], s1: usize, ..)`
/// in operand order (`&[f64]` for Read operands, `&mut [f64]` otherwise), where `s{i}` is the
/// element stride between consecutive points. The call is exactly `points` sequential calls,
/// call `p` seeing every buffer shifted by `p * s{i}`. Every binding is checked (with
/// overflow-checked extents) before any effect; `points = 0` executes nothing.
///
/// When `points > 1` and every writable stride is at least its region length (written
/// windows are disjoint, so points are independent), full blocks of `width` points run
/// statement by statement over `[T; W]` lane arrays so LLVM can vectorize vertically; the
/// remaining points, and every call with overlapping or shared writable windows, run one
/// point at a time. Each lane performs exactly the scalar operations of [`lower_rust`]
/// (strict IEEE, `mul_add` only for explicit `Fma`), so both paths are bitwise identical to
/// the interpreter. Portable-level kernels need the [`crate::policy::PORTABLE_MATH`] `libm`
/// dependency, as in [`lower_rust`].
pub fn lower_rust_batched(
    e: &Executable,
    name: &str,
    options: BatchedRustOptions,
) -> Result<String, LoweringError> {
    admit(e, name)?;
    let w = options.width;
    if w == 0 || w > 64 {
        return Err(LoweringError::Unsupported(
            "lane width must be 1..=64".into(),
        ));
    }
    let k = e.kernel().as_kernel();
    let order = loop_order(e);
    let loops = |body: &str| {
        let mut out = String::new();
        for &axis in &order {
            writeln!(
                out,
                "for i{axis} in 0..{}usize {{",
                k.iteration_domain.extents[axis]
            )
            .expect("string write");
        }
        out.push_str(body);
        for _ in &order {
            out.push_str("}\n");
        }
        out
    };
    let mut point = Emitter {
        k,
        nodes: 0,
        mode: Mode::Offset,
    };
    let point_body = loops(&statements(&mut point)?);
    let mut lane = Emitter {
        k,
        nodes: 0,
        mode: Mode::Lane(w),
    };
    let lane_body = loops(&statements(&mut lane)?);

    let mut params = String::from("points: usize, ");
    let mut args = String::from("points, ");
    for (i, o) in k.operands.iter().enumerate() {
        let m = if o.access == AccessMode::Read {
            ""
        } else {
            "mut "
        };
        write!(params, "b{i}: &{m}[f64], s{i}: usize, ").expect("string write");
        write!(args, "b{i}, s{i}, ").expect("string write");
    }
    let mut out = format!(
        "// Generated by Malleus rust-batched/1 (width {w}); strict floating-point operations.\n"
    );
    let attrs = "#[allow(unused_variables, unused_parens, unused_mut, clippy::all)]";
    let inner = if options.avx2_dispatch {
        format!("{name}_impl")
    } else {
        name.to_owned()
    };
    let vis = if options.avx2_dispatch {
        "#[inline(always)]\nfn"
    } else {
        "#[inline]\npub fn"
    };
    writeln!(
        out,
        "{attrs}\n{vis} {inner}({params}) -> Result<(), &'static str> {{"
    )
    .expect("string write");
    out.push_str("if points == 0 {\n");
    for (i, o) in k.operands.iter().enumerate() {
        let required = o.region.offset + o.region.length;
        writeln!(out, "let _ = (b{i}.len(), s{i}, {required}usize);").expect("string write");
    }
    out.push_str("return Ok(());\n}\n");
    for (i, o) in k.operands.iter().enumerate() {
        let required = o.region.offset + o.region.length;
        writeln!(
            out,
            "match (points - 1).checked_mul(s{i}).and_then(|x| x.checked_add({required}usize)) {{ \
             Some(n) if n <= b{i}.len() => {{}} Some(_) => return Err(\"short operand {i}\"), \
             None => return Err(\"extent overflow {i}\") }}"
        )
        .expect("string write");
    }
    let disjoint: Vec<String> = k
        .operands
        .iter()
        .enumerate()
        .filter(|(_, o)| o.access.can_write())
        .map(|(i, o)| format!("s{i} >= {}usize", o.region.length))
        .collect();
    let disjoint = if disjoint.is_empty() {
        "true".to_owned()
    } else {
        disjoint.join(" && ")
    };
    writeln!(
        out,
        "let mut p = 0usize;\nif points > 1 && {disjoint} {{\nwhile p + {w} <= points {{"
    )
    .expect("string write");
    out.push_str(&lane_body);
    writeln!(out, "p += {w};\n}}\n}}\nwhile p < points {{").expect("string write");
    for i in 0..k.operands.len() {
        writeln!(out, "let o{i} = p * s{i};").expect("string write");
    }
    out.push_str(&point_body);
    out.push_str("p += 1;\n}\nOk(())\n}\n");
    if options.avx2_dispatch {
        writeln!(
            out,
            "#[cfg(target_arch = \"x86_64\")]\n#[target_feature(enable = \"avx2\")]\n{attrs}\n\
             fn {name}_avx2({params}) -> Result<(), &'static str> {{ {name}_impl({args}) }}\n\
             {attrs}\npub fn {name}({params}) -> Result<(), &'static str> {{\n\
             #[cfg(target_arch = \"x86_64\")]\n\
             if ::std::is_x86_feature_detected!(\"avx2\") {{\n\
             // SAFETY: the CPU supports AVX2, the only target feature the callee enables.\n\
             return unsafe {{ {name}_avx2({args}) }};\n}}\n{name}_impl({args})\n}}"
        )
        .expect("string write");
    }
    Ok(out)
}

fn lower_rust_impl(
    e: &Executable,
    name: &str,
    unroll: Option<usize>,
) -> Result<String, LoweringError> {
    admit(e, name)?;
    let k = e.kernel().as_kernel();
    let mut out =
        String::from("// Generated by Malleus rust-source/1; strict floating-point operations.\n");
    let mut emit = Emitter {
        k,
        nodes: 0,
        mode: Mode::Plain,
    };
    write!(
        out,
        "#[inline]\n#[allow(unused_variables, unused_parens, clippy::all)]\npub fn {name}("
    )
    .expect("string write");
    for (i, o) in k.operands.iter().enumerate() {
        write!(
            out,
            "b{i}: &{}[f64],",
            if o.access == AccessMode::Read {
                ""
            } else {
                "mut "
            }
        )
        .expect("string write");
    }
    out.push_str(") -> Result<(), &'static str> {\n");
    for (i, o) in k.operands.iter().enumerate() {
        let required = o.region.offset + o.region.length;
        writeln!(
            out,
            "if b{i}.len() < {required} {{ return Err(\"short operand {i}\"); }}"
        )
        .expect("string write");
    }
    let order = loop_order(e);
    if let Some(limit) = unroll {
        let iterations = k.iteration_domain.extents.iter().product::<usize>();
        if iterations > limit {
            return Err(LoweringError::Unsupported(
                "iteration domain exceeds unroll limit".into(),
            ));
        }
    }
    let body = statements(&mut emit)?;
    if unroll.is_some() {
        fn expand(
            out: &mut String,
            body: &str,
            k: &StructuredKernel,
            order: &[usize],
            depth: usize,
            coordinates: &mut Vec<(usize, usize)>,
        ) {
            if depth == order.len() {
                out.push_str("{\n");
                for &(axis, i) in coordinates.iter() {
                    writeln!(out, "let i{axis}={i}usize;").expect("string write");
                }
                out.push_str(body);
                out.push_str("}\n");
                return;
            }
            let axis = order[depth];
            for i in 0..k.iteration_domain.extents[axis] {
                coordinates.push((axis, i));
                expand(out, body, k, order, depth + 1, coordinates);
                coordinates.pop();
            }
        }
        if body
            .len()
            .saturating_mul(k.iteration_domain.extents.iter().product::<usize>())
            > 2_000_000
        {
            return Err(LoweringError::Unsupported(
                "unrolled source size budget".into(),
            ));
        }
        expand(&mut out, &body, k, &order, 0, &mut vec![]);
    } else {
        for &axis in &order {
            writeln!(
                out,
                "for i{axis} in 0..{}usize {{",
                k.iteration_domain.extents[axis]
            )
            .expect("string write");
        }
        out.push_str(&body);
        for _ in order {
            out.push_str("}\n");
        }
    }
    out.push_str("Ok(())\n}\n");
    Ok(out)
}

/// Name, schedule and size admission shared by every Rust entry point.
fn admit(e: &Executable, name: &str) -> Result<(), LoweringError> {
    if name.len() > 128
        || name.is_empty()
        || name == "_"
        || !name
            .bytes()
            .enumerate()
            .all(|(i, c)| c == b'_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
        || [
            "fn", "let", "mod", "pub", "struct", "enum", "impl", "type", "self", "Self", "crate",
            "super", "use", "const", "static", "unsafe", "extern", "async", "await", "move",
            "match", "loop", "for", "while", "in", "if", "else", "return", "break", "continue",
            "true", "false", "ref", "mut", "as", "where", "trait", "dyn", "union", "yield", "try",
            "gen", "abstract", "become", "box", "do", "final", "macro", "override", "priv",
            "typeof", "unsized", "virtual",
        ]
        .contains(&name)
    {
        return Err(LoweringError::Unsupported(
            "invalid Rust function name".into(),
        ));
    }
    let k = e.kernel().as_kernel();
    let s = e.schedule();
    if !s.tiles.is_empty() || s.vectorization.is_some() || !s.parallel.is_empty() {
        return Err(LoweringError::Unsupported(
            "tiled/vector/parallel schedule".into(),
        ));
    }
    if k.iteration_domain.rank() > 8
        || k.operands.len() > 256
        || k.body.statements.len() > 2048
        || k.iteration_domain.extents.iter().any(|&n| n > 1_000_000)
        || k.iteration_domain
            .extents
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .is_none_or(|n| n > 10_000_000)
    {
        return Err(LoweringError::Unsupported("Rust emission budget".into()));
    }
    for o in &k.operands {
        let size = o.shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b));
        if size.is_none_or(|n| n > 100_000_000)
            || o.region.offset > 100_000_000
            || o.region.length > 100_000_000
        {
            return Err(LoweringError::Unsupported("Rust buffer budget".into()));
        }
    }
    Ok(())
}

fn loop_order(e: &Executable) -> Vec<usize> {
    let k = e.kernel().as_kernel();
    if k.numeric_policy.reduction_order == ReductionOrder::Canonical {
        (0..k.iteration_domain.rank()).collect()
    } else {
        e.schedule().loop_order.iter().map(|a| a.index()).collect()
    }
}

/// The kernel body for one iteration point in the emitter's mode.
fn statements(emit: &mut Emitter<'_>) -> Result<String, LoweringError> {
    let k = emit.k;
    let mut body = String::new();
    for statement in &k.body.statements {
        match statement {
            Statement::Let { local, value } => {
                let expr = emit.expr(value, 0)?;
                let l = local.index();
                let ty = emit.ty();
                match emit.mode {
                    Mode::Lane(w) => writeln!(
                        body,
                        "let v{l}: [{ty}; {w}] = ::core::array::from_fn(|j| {expr});"
                    ),
                    _ => writeln!(body, "let v{l}: {ty} = {expr};"),
                }
                .expect("string write");
            }
            Statement::Store { operand, value } => {
                let value = emit.expr(value, 0)?;
                let address = emit.address(*operand)?;
                let i = operand.index();
                let mut value = format!("({value})");
                if let AccessMode::Reduce(op) = k.operands[i].access {
                    let old = format!("(b{i}[{address}] as {})", emit.ty());
                    value = match op {
                        ReductionOp::Add => format!("({old}+{value})"),
                        ReductionOp::Multiply => format!("({old}*{value})"),
                        ReductionOp::Min => emit.call2(BinaryOp::Min, &old, &value),
                        ReductionOp::Max => emit.call2(BinaryOp::Max, &old, &value),
                    };
                }
                let store = format!("b{i}[{address}] = ({value}) as f64;");
                match emit.mode {
                    Mode::Lane(w) => writeln!(body, "for j in 0..{w} {{ {store} }}"),
                    _ => writeln!(body, "{store}"),
                }
                .expect("string write");
            }
        }
    }
    Ok(body)
}
/// How operand addresses and locals are spelled in emitted statements.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// One call: `b{i}[address]`.
    Plain,
    /// One batch point: `b{i}[o{i} + address]` with `o{i} = p * s{i}`.
    Offset,
    /// A block of this many points `p + j`: locals are `[T; W]` lane arrays.
    Lane(usize),
}

struct Emitter<'a> {
    k: &'a StructuredKernel,
    nodes: usize,
    mode: Mode,
}
impl Emitter<'_> {
    fn ty(&self) -> &'static str {
        if self.k.numeric_policy.scalar_type == ScalarType::F32 {
            "f32"
        } else {
            "f64"
        }
    }
    fn portable(&self) -> bool {
        self.k.numeric_policy.level.math() == policy::MathLibrary::Portable
    }
    fn libm_suffix(&self) -> &'static str {
        if self.ty() == "f32" { "f" } else { "" }
    }
    /// `pow`, `min`, `max` or `atan2` of two emitted operands under the kernel's math library.
    fn call2(&self, op: BinaryOp, a: &str, b: &str) -> String {
        if self.portable() {
            let name = match op {
                BinaryOp::Pow => "pow",
                BinaryOp::Min => "fmin",
                BinaryOp::Max => "fmax",
                _ => "atan2",
            };
            format!("::libm::{name}{}({a}, {b})", self.libm_suffix())
        } else {
            let name = match op {
                BinaryOp::Pow => "powf",
                BinaryOp::Min => "min",
                BinaryOp::Max => "max",
                _ => "atan2",
            };
            format!("({a}).{name}({b})")
        }
    }
    fn address(&self, id: OperandId) -> Result<String, LoweringError> {
        let o = &self.k.operands[id.index()];
        let map = self
            .k
            .indexing_maps
            .iter()
            .find(|m| m.operand == id)
            .expect("validated map");
        let mut stride = 1usize;
        let i = id.index();
        let mut address = match self.mode {
            Mode::Plain => o.region.offset.to_string(),
            Mode::Offset => format!("o{i} + {}", o.region.offset),
            Mode::Lane(_) => format!("(p + j) * s{i} + {}", o.region.offset),
        };
        for &d in &o.layout.minor_to_major {
            let x = &map.results[d];
            // i128 arithmetic avoids intermediate signed overflow even for cancelling maps.
            if x.constant.unsigned_abs() > 1_000_000_000_000
                || x.terms
                    .iter()
                    .any(|t| t.coefficient.unsigned_abs() > 1_000_000)
            {
                return Err(LoweringError::Unsupported(
                    "Rust affine coefficient budget".into(),
                ));
            }
            let mut index = format!("{}i128", x.constant);
            for t in &x.terms {
                write!(
                    index,
                    " + ({}i128 * i{} as i128)",
                    t.coefficient,
                    t.axis.index()
                )
                .expect("string write");
            }
            write!(address, " + (({index}) as usize)*{stride}").expect("string write");
            stride = stride
                .checked_mul(o.shape[d])
                .ok_or_else(|| LoweringError::Unsupported("layout overflow".into()))?;
        }
        Ok(address)
    }
    fn budget(&mut self, depth: usize) -> Result<(), LoweringError> {
        self.nodes += 1;
        if depth > 128 || self.nodes > 65_536 {
            Err(LoweringError::Unsupported("Rust expression budget".into()))
        } else {
            Ok(())
        }
    }
    fn expr(&mut self, e: &ScalarExpr, d: usize) -> Result<String, LoweringError> {
        self.budget(d)?;
        Ok(match e {
            ScalarExpr::Constant(x) => {
                if self.ty() == "f32" {
                    format!("f32::from_bits({}u32)", (*x as f32).to_bits())
                } else {
                    format!("f64::from_bits({}u64)", x.to_bits())
                }
            }
            ScalarExpr::Index(a) => format!("(i{} as {})", a.index(), self.ty()),
            ScalarExpr::Local(l) if matches!(self.mode, Mode::Lane(_)) => {
                format!("v{}[j]", l.index())
            }
            ScalarExpr::Local(l) => format!("v{}", l.index()),
            ScalarExpr::Load(id) => {
                format!("(b{}[{}] as {})", id.index(), self.address(*id)?, self.ty())
            }
            ScalarExpr::Unary { op, arg } => {
                let a = self.expr(arg, d + 1)?;
                match op {
                    UnaryOp::Neg => format!("(-({a}))"),
                    op if self.portable() && policy::unary_uses_math(*op) => format!(
                        "::libm::{}{}({a})",
                        match op {
                            UnaryOp::Exp => "exp",
                            UnaryOp::Ln => "log",
                            UnaryOp::Sin => "sin",
                            UnaryOp::Cos => "cos",
                            _ => "tan",
                        },
                        self.libm_suffix()
                    ),
                    other => format!(
                        "({a}).{}()",
                        match other {
                            UnaryOp::Abs => "abs",
                            UnaryOp::Sqrt => "sqrt",
                            UnaryOp::Exp => "exp",
                            UnaryOp::Ln => "ln",
                            UnaryOp::Sin => "sin",
                            UnaryOp::Cos => "cos",
                            UnaryOp::Tan => "tan",
                            UnaryOp::Floor => "floor",
                            UnaryOp::Ceil => "ceil",
                            UnaryOp::Neg => unreachable!(),
                        }
                    ),
                }
            }
            ScalarExpr::Binary { op, lhs, rhs } => {
                let a = self.expr(lhs, d + 1)?;
                let b = self.expr(rhs, d + 1)?;
                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => format!(
                        "(({a}) {} ({b}))",
                        match op {
                            BinaryOp::Add => "+",
                            BinaryOp::Sub => "-",
                            BinaryOp::Mul => "*",
                            _ => "/",
                        }
                    ),
                    other => self.call2(*other, &a, &b),
                }
            }
            ScalarExpr::Fma { lhs, rhs, addend } => format!(
                "({}).mul_add({}, {})",
                self.expr(lhs, d + 1)?,
                self.expr(rhs, d + 1)?,
                self.expr(addend, d + 1)?
            ),
            ScalarExpr::Select {
                condition,
                if_true,
                if_false,
            } => format!(
                "(if {} {{ {} }} else {{ {} }})",
                self.predicate(condition, d + 1)?,
                self.expr(if_true, d + 1)?,
                self.expr(if_false, d + 1)?
            ),
        })
    }
    fn predicate(&mut self, p: &Predicate, d: usize) -> Result<String, LoweringError> {
        self.budget(d)?;
        Ok(match p {
            Predicate::Constant(v) => v.to_string(),
            Predicate::Compare { op, lhs, rhs } => format!(
                "({} {} {})",
                self.expr(lhs, d + 1)?,
                match op {
                    CompareOp::Eq => "==",
                    CompareOp::NotEq => "!=",
                    CompareOp::Less => "<",
                    CompareOp::LessEqual => "<=",
                    CompareOp::Greater => ">",
                    CompareOp::GreaterEqual => ">=",
                },
                self.expr(rhs, d + 1)?
            ),
            Predicate::Not(a) => format!("(!{})", self.predicate(a, d + 1)?),
            Predicate::And(a, b) => format!(
                "({} && {})",
                self.predicate(a, d + 1)?,
                self.predicate(b, d + 1)?
            ),
            Predicate::Or(a, b) => format!(
                "({} || {})",
                self.predicate(a, d + 1)?,
                self.predicate(b, d + 1)?
            ),
        })
    }
}
