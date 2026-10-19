//! Explicit real-interval enclosure lowering into ordinary Malleus f64 IR.
//!
//! Requires round-to-nearest/ties-even and gradual underflow, with no contraction or
//! reassociation. Every arithmetic endpoint is padded outwards by the Rump–Zimmermann–
//! Boldo–Melquiond successor bound, so it reaches at least the neighbouring double. See
//! INTERVALS.md for the proof.
//!
//! Three lowerings share one operand layout ([`IntervalProgram::endpoints`]):
//!
//! - [`lower_intervals`]: naive endpoint interval arithmetic;
//! - [`lower_centered_intervals`]: the mean-value (centered) form `f(c) + f'(X)·(X − c)`
//!   built from the structured JVP, intersected with the naive enclosure;
//! - [`lower_affine_intervals`]: first-order affine arithmetic with one noise symbol per
//!   read operand element and one accumulated error radius, intersected with the naive
//!   enclosure.
//!
//! The intersections make "never wider than the naive enclosure" a theorem for the latter two.
//!
//! # Input contract of lowered kernels
//!
//! A lowered kernel (run by the interpreter, the Rust source backend or any other backend)
//! is sound only when all of the following hold; none is checked at run time:
//!
//! - every input interval is given as ordered endpoints, `lo <= hi`, and neither endpoint is
//!   NaN (infinite endpoints are allowed and widen the result conservatively);
//! - the input interval encloses the exact value of its operand;
//! - the caller has checked [`ieee_environment_supported`] on the thread that runs the
//!   kernel and the floating-point environment does not change during the call.
//!
//! Given these, every output interval encloses the real-number result of the source kernel
//! for every real choice of the inputs within their intervals. Output endpoints are never
//! NaN; overflow and nonfinite intermediate values widen to infinite endpoints.

use crate::LoweringError as Error;
use crate::*;
type Result<T> = std::result::Result<T, Error>;

/// `φ = u(1 + 2u)` with unit roundoff `u = 2^-53` (exactly representable).
const PHI: f64 = f64::EPSILON * 0.5 * (1.0 + f64::EPSILON);
/// `η`, the smallest positive subnormal.
const ETA: f64 = f64::from_bits(1);

#[derive(Clone)]
struct Pair {
    lo: ScalarExpr,
    hi: ScalarExpr,
}

/// Operand `i` becomes endpoints `2*i` and `2*i+1`, preserving shape/layout/region.
/// Callers must supply ordered, non-NaN input endpoints enclosing their exact data and
/// must have checked [`ieee_environment_supported`]; see the module documentation.
pub struct IntervalProgram {
    /// Ordinary validated Malleus IR, runnable by both interpreter and Cranelift.
    pub executable: Executable,
}
impl IntervalProgram {
    /// Endpoint operand identifiers for a source operand.
    pub fn endpoints(id: OperandId) -> [OperandId; 2] {
        [
            OperandId::new(2 * id.index()),
            OperandId::new(2 * id.index() + 1),
        ]
    }
}

/// Lift the real-arithmetic meaning of a strict f64 Read/Write kernel to enclosures.
/// Unsupported operations/effects are refusals, not unbounded optimistic approximations.
///
/// Supported: finite constants, indices, loads, locals, negation, addition, subtraction,
/// multiplication (structural squares keep correlation), explicit `Fma` (enclosing the real
/// `a·b + c`) and selects whose predicate depends only on indices and constants.
pub fn lower_intervals(source: &ValidatedKernel) -> Result<IntervalProgram> {
    let k = source.as_kernel();
    let mut lift = Lift::new(k, NAIVE_BUDGET)?;
    let naive = lift.naive_env(k);
    let stores = lift.region(&naive, &k.body)?;
    for (operand, pair) in stores {
        lift.store(operand, pair);
    }
    lift.finish(k, "outward-interval-v1")
}

/// The centered (mean-value) form `f(c) + Σ_j ∂_j f(X)·(X_j − c_j)`, intersected with the
/// naive enclosure of [`lower_intervals`].
///
/// The centre `c` of every read operand element is a finite double inside its input
/// interval (the rounded midpoint when it lies inside, else a finite endpoint, else 0). For
/// every read operand `j`, the gradient enclosure `∂_j f(X)` is the interval lifting of the
/// structured JVP with `j` alone independent (direction bound to `[1, 1]`, primal loads bound
/// to `X`); it multiplies the outward interval `X_j − c_j`. Soundness: for `x ∈ X` the mean
/// value theorem on the segment `[c, x] ⊆ X` gives `f(x) = f(c) + Σ_j ∂_j f(ξ)·(x_j − c_j)`
/// with `ξ ∈ X`; the supported operations are polynomial in the data (index-only branches
/// are fixed per iteration point), so each `∂_j f(ξ)` lies in its lifted enclosure.
pub fn lower_centered_intervals(source: &ValidatedKernel) -> Result<IntervalProgram> {
    let k = source.as_kernel();
    let mut lift = Lift::new(k, WIDE_BUDGET)?;
    let reads = read_operands(k);
    let writes: Vec<OperandId> = (0..k.operands.len())
        .map(OperandId::new)
        .filter(|id| k.operands[id.index()].access == AccessMode::Write)
        .collect();
    let naive = lift.naive_env(k);
    let naive_stores = lift.region(&naive, &k.body)?;
    let mut point = Env {
        loads: vec![None; k.operands.len()],
        locals: vec![],
    };
    let mut direction = vec![None; k.operands.len()];
    for id in &reads {
        let x = naive.loads[id.index()].clone().expect("read operand");
        let centre = lift.centre(&x);
        let c = Pair {
            lo: centre.clone(),
            hi: centre,
        };
        direction[id.index()] = Some(lift.sub(x, c.clone()));
        point.loads[id.index()] = Some(c);
    }
    let point_stores = lift.region(&point, &k.body)?;
    let mut centered: Vec<(OperandId, Pair)> = writes
        .iter()
        .filter_map(|&o| last(&point_stores, o).map(|p| (o, p)))
        .collect();
    for &j in &reads {
        let product = differentiate(
            k,
            &DerivativeRequest {
                mode: DerivativeMode::Jvp,
                independent_operands: vec![j],
                dependent_operands: writes.clone(),
            },
        )
        .map_err(|e| Error::Unsupported(format!("centered form: {e}")))?;
        let mut tangent = Env {
            loads: vec![None; product.kernel.operands.len()],
            locals: vec![],
        };
        for pair in &product.primal_operands {
            tangent.loads[pair.derivative.index()] = naive.loads[pair.primal.index()].clone();
        }
        for pair in &product.independent_operands {
            tangent.loads[pair.derivative.index()] = Some(Pair {
                lo: c(1.),
                hi: c(1.),
            });
        }
        let gradients = lift.region(&tangent, &product.kernel.body)?;
        let delta = direction[j.index()].clone().expect("read operand");
        for (operand, sum) in &mut centered {
            let Some(g) = product
                .dependent_operands
                .iter()
                .find(|d| d.primal == *operand)
                .and_then(|d| last(&gradients, d.derivative))
            else {
                continue;
            };
            let term = lift.product(g, delta.clone());
            *sum = lift.add(sum.clone(), term);
        }
    }
    for (operand, n) in naive_stores {
        let pair = match last(&centered, operand) {
            Some(centered) => lift.intersect(n, centered),
            None => n,
        };
        lift.store(operand, pair);
    }
    lift.finish(k, "centered-interval-v1")
}

/// First-order affine arithmetic (de Figueiredo–Stolfi), intersected with the naive enclosure
/// of [`lower_intervals`].
///
/// Each value is `x0 + Σ_i x_i·ε_i + t`, `|t| ≤ e`, with one noise symbol `ε_i ∈ [-1, 1]` per
/// read operand (the element it reads at the current iteration point) and one accumulated
/// error radius `e`. All new roundoff and nonlinear terms are added to `e` with upward
/// rounding, so the form encloses the real value; the final range `x0 ± (Σ|x_i| + e)` is
/// rounded outwards. Supported: the [`lower_intervals`] set (neg/add/sub/mul, structural
/// squares, `Fma` as the real `a·b + c`, constants, indices, index-only selects); anything
/// else refuses.
pub fn lower_affine_intervals(source: &ValidatedKernel) -> Result<IntervalProgram> {
    let k = source.as_kernel();
    let mut lift = Lift::new(k, WIDE_BUDGET)?;
    let reads = read_operands(k);
    let naive = lift.naive_env(k);
    let naive_stores = lift.region(&naive, &k.body)?;
    let mut inputs = vec![None; k.operands.len()];
    for (symbol, id) in reads.iter().enumerate() {
        let x = naive.loads[id.index()].clone().expect("read operand");
        inputs[id.index()] = Some(lift.affine_input(&x, symbol, reads.len()));
    }
    let mut env = AffineEnv {
        inputs,
        locals: vec![],
        symbols: reads.len(),
    };
    let mut affine_stores = vec![];
    for statement in &k.body.statements {
        match statement {
            Statement::Let { value, .. } => {
                let form = lift.affine(&mut env, value, 0)?;
                env.locals.push(form);
            }
            Statement::Store { operand, value } => {
                let form = lift.affine(&mut env, value, 0)?;
                affine_stores.push((*operand, form));
            }
        }
    }
    for ((operand, n), (_, form)) in naive_stores.into_iter().zip(affine_stores) {
        let range = lift.affine_range(&form);
        let pair = lift.intersect(n, range);
        lift.store(operand, pair);
    }
    lift.finish(k, "affine-interval-v1")
}

/// Check the required FP environment using runtime operands. It must remain unchanged
/// for the duration of any prepared call. Rust's usual floating-point environment is valid.
pub fn ieee_environment_supported() -> bool {
    use std::hint::black_box as b;
    let x = b(1.0_f64);
    let h = b(f64::EPSILON * 0.5);
    let q = b(f64::EPSILON * 0.75);
    x + h == 1.0
        && -x - h == -1.0
        && x + q == 1.0 + f64::EPSILON
        && -x - q == -1.0 - f64::EPSILON
        && b(f64::MIN_POSITIVE) * b(0.5) == f64::from_bits(1 << 51)
        && b(f64::from_bits(1)) + b(f64::from_bits(1)) == f64::from_bits(2)
}

fn c(x: f64) -> ScalarExpr {
    ScalarExpr::Constant(x)
}
fn bin(op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(op, a, b)
}
fn abs(a: ScalarExpr) -> ScalarExpr {
    ScalarExpr::unary(UnaryOp::Abs, a)
}
fn cmp(op: CompareOp, a: ScalarExpr, b: ScalarExpr) -> Predicate {
    Predicate::Compare {
        op,
        lhs: Box::new(a),
        rhs: Box::new(b),
    }
}
fn finite(v: ScalarExpr) -> Predicate {
    // Ordered comparison: false for NaN as well as for infinities.
    cmp(CompareOp::LessEqual, abs(v), c(f64::MAX))
}
fn choose(p: Predicate, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Select {
        condition: Box::new(p),
        if_true: Box::new(a),
        if_false: Box::new(b),
    }
}
fn extremum(a: ScalarExpr, b: ScalarExpr, minimum: bool) -> ScalarExpr {
    choose(
        cmp(
            if minimum {
                CompareOp::Less
            } else {
                CompareOp::Greater
            },
            a.clone(),
            b.clone(),
        ),
        a,
        b,
    )
}
/// `RN(RN(φ·|v|) + η)`; for finite `v`, `RN(v + it) >= succ(v)` and `RN(v - it) <= pred(v)`.
fn successor_bound(v: ScalarExpr) -> ScalarExpr {
    bin(BinaryOp::Add, bin(BinaryOp::Mul, abs(v), c(PHI)), c(ETA))
}
/// `+inf` without a nonfinite literal (core Malleus rejects those).
fn infinity() -> ScalarExpr {
    bin(BinaryOp::Mul, c(f64::MAX), c(2.))
}
fn last(stores: &[(OperandId, Pair)], operand: OperandId) -> Option<Pair> {
    stores
        .iter()
        .rev()
        .find(|(id, _)| *id == operand)
        .map(|(_, p)| p.clone())
}
fn read_operands(k: &StructuredKernel) -> Vec<OperandId> {
    (0..k.operands.len())
        .map(OperandId::new)
        .filter(|id| k.operands[id.index()].access == AccessMode::Read)
        .collect()
}

#[derive(Clone, Copy)]
struct Budget {
    nodes: usize,
    locals: usize,
}
const NAIVE_BUDGET: Budget = Budget {
    nodes: 512,
    locals: 1024,
};
const WIDE_BUDGET: Budget = Budget {
    nodes: 16384,
    locals: 65536,
};

/// Interval values of a source kernel's loads (by source operand id) and locals.
struct Env {
    loads: Vec<Option<Pair>>,
    locals: Vec<Pair>,
}

/// An affine form `center + Σ coeffs[i]·ε_i + [-err, err]`; `None` is an exact zero.
#[derive(Clone)]
struct Form {
    center: ScalarExpr,
    coeffs: Vec<Option<ScalarExpr>>,
    err: Option<ScalarExpr>,
}

struct AffineEnv {
    inputs: Vec<Option<Form>>,
    locals: Vec<Form>,
    symbols: usize,
}

struct Lift {
    statements: Vec<Statement>,
    stores: Vec<Statement>,
    next_local: usize,
    nodes: usize,
    budget: Budget,
}
impl Lift {
    fn new(k: &StructuredKernel, budget: Budget) -> Result<Self> {
        if k.numeric_policy != NumericPolicy::default() {
            return Err(Error::Unsupported(
                "interval lowering requires strict canonical f64".into(),
            ));
        }
        if k.operands
            .iter()
            .any(|o| !matches!(o.access, AccessMode::Read | AccessMode::Write))
        {
            return Err(Error::Unsupported(
                "interval lowering requires Read/Write operands".into(),
            ));
        }
        if k.iteration_domain.extents.iter().any(|&n| n > 1_000_000) {
            return Err(Error::Unsupported("interval index budget".into()));
        }
        Ok(Lift {
            statements: vec![],
            stores: vec![],
            next_local: 0,
            nodes: 0,
            budget,
        })
    }
    fn naive_env(&self, k: &StructuredKernel) -> Env {
        Env {
            loads: (0..k.operands.len())
                .map(|i| {
                    k.operands[i].access.can_read().then(|| {
                        let [lo, hi] = IntervalProgram::endpoints(OperandId::new(i));
                        Pair {
                            lo: ScalarExpr::Load(lo),
                            hi: ScalarExpr::Load(hi),
                        }
                    })
                })
                .collect(),
            locals: vec![],
        }
    }
    /// Lift a region in a fresh local scope, returning its stores in order.
    fn region(&mut self, env: &Env, region: &KernelRegion) -> Result<Vec<(OperandId, Pair)>> {
        let mut env = Env {
            loads: env.loads.clone(),
            locals: vec![],
        };
        let mut stores = vec![];
        for statement in &region.statements {
            match statement {
                Statement::Let { value, .. } => {
                    let pair = self.expr(&env, value, 0)?;
                    env.locals.push(pair);
                }
                Statement::Store { operand, value } => {
                    stores.push((*operand, self.expr(&env, value, 0)?));
                }
            }
        }
        Ok(stores)
    }
    fn store(&mut self, operand: OperandId, pair: Pair) {
        let [lo, hi] = IntervalProgram::endpoints(operand);
        self.stores.push(Statement::Store {
            operand: lo,
            value: pair.lo,
        });
        self.stores.push(Statement::Store {
            operand: hi,
            value: pair.hi,
        });
    }
    fn finish(mut self, k: &StructuredKernel, suffix: &str) -> Result<IntervalProgram> {
        let mut result = k.clone();
        result.name = format!("{}::{suffix}", k.name);
        result.operands = k
            .operands
            .iter()
            .enumerate()
            .flat_map(|(i, o)| {
                ["lo", "hi"].map(|suffix| {
                    let mut o = o.clone();
                    o.name = format!("endpoint_{i}_{suffix}");
                    o
                })
            })
            .collect();
        result.indexing_maps = k
            .indexing_maps
            .iter()
            .flat_map(|m| {
                IntervalProgram::endpoints(m.operand).map(|operand| {
                    let mut m = m.clone();
                    m.operand = operand;
                    m
                })
            })
            .collect();
        self.statements.append(&mut self.stores);
        result.body.statements = self.statements;
        let validated = validate(result).map_err(|e| Error::Compile(e.to_string()))?;
        Ok(IntervalProgram {
            executable: Executable::reference(validated),
        })
    }
    fn emit(&mut self, value: ScalarExpr) -> ScalarExpr {
        let local = LocalId::new(self.next_local);
        self.next_local += 1;
        self.statements.push(Statement::Let { local, value });
        ScalarExpr::Local(local)
    }
    /// `e(v) = RN(RN(φ·|v|) + η)` for finite `v`, `+inf` otherwise: an upper bound on the
    /// rounding error of any IEEE operation whose rounded result is `v`.
    fn radius(&mut self, v: ScalarExpr) -> ScalarExpr {
        self.emit(choose(finite(v.clone()), successor_bound(v), infinity()))
    }
    /// The outward enclosure `[RN(v − e(v)), RN(v + e(v))]` of one rounded operation `v`.
    fn outward(&mut self, value: ScalarExpr) -> Pair {
        let v = self.emit(value);
        let ok = finite(v.clone());
        let radius = self.emit(successor_bound(v.clone()));
        let lo = self.emit(choose(
            ok.clone(),
            bin(BinaryOp::Sub, v.clone(), radius.clone()),
            bin(BinaryOp::Mul, c(-f64::MAX), c(2.)),
        ));
        let hi = self.emit(choose(ok, bin(BinaryOp::Add, v, radius), infinity()));
        Pair { lo, hi }
    }
    /// An upper bound on the real `a op b` (for the error accumulation of affine forms).
    fn up(&mut self, op: BinaryOp, a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
        self.outward(bin(op, a, b)).hi
    }
    fn up_sum(&mut self, terms: Vec<ScalarExpr>) -> Option<ScalarExpr> {
        let mut terms = terms.into_iter();
        let first = terms.next()?;
        Some(terms.fold(first, |acc, t| self.up(BinaryOp::Add, acc, t)))
    }
    fn add(&mut self, a: Pair, b: Pair) -> Pair {
        let lo = self.outward(bin(BinaryOp::Add, a.lo, b.lo)).lo;
        let hi = self.outward(bin(BinaryOp::Add, a.hi, b.hi)).hi;
        Pair { lo, hi }
    }
    fn sub(&mut self, a: Pair, b: Pair) -> Pair {
        let lo = self.outward(bin(BinaryOp::Sub, a.lo, b.hi)).lo;
        let hi = self.outward(bin(BinaryOp::Sub, a.hi, b.lo)).hi;
        Pair { lo, hi }
    }
    fn intersect(&mut self, a: Pair, b: Pair) -> Pair {
        let lo = self.emit(extremum(a.lo, b.lo, false));
        let hi = self.emit(extremum(a.hi, b.hi, true));
        Pair { lo, hi }
    }
    /// A finite double inside `[lo, hi]` (the input contract guarantees one exists).
    fn centre(&mut self, x: &Pair) -> ScalarExpr {
        let mid = self.emit(bin(
            BinaryOp::Add,
            bin(BinaryOp::Mul, x.lo.clone(), c(0.5)),
            bin(BinaryOp::Mul, x.hi.clone(), c(0.5)),
        ));
        let inside = Predicate::And(
            Box::new(finite(mid.clone())),
            Box::new(Predicate::And(
                Box::new(cmp(CompareOp::GreaterEqual, mid.clone(), x.lo.clone())),
                Box::new(cmp(CompareOp::LessEqual, mid.clone(), x.hi.clone())),
            )),
        );
        self.emit(choose(
            inside,
            mid,
            choose(
                finite(x.lo.clone()),
                x.lo.clone(),
                choose(finite(x.hi.clone()), x.hi.clone(), c(0.)),
            ),
        ))
    }
    fn product(&mut self, a: Pair, b: Pair) -> Pair {
        let mut products = vec![];
        for x in [a.lo, a.hi] {
            for y in [b.lo.clone(), b.hi.clone()] {
                products.push(self.outward(bin(BinaryOp::Mul, x.clone(), y)));
            }
        }
        let mut out = products.remove(0);
        for p in products {
            out.lo = self.emit(extremum(out.lo, p.lo, true));
            out.hi = self.emit(extremum(out.hi, p.hi, false));
        }
        out
    }
    fn square(&mut self, a: Pair) -> Pair {
        let low = self.outward(bin(BinaryOp::Mul, a.lo.clone(), a.lo.clone()));
        let high = self.outward(bin(BinaryOp::Mul, a.hi.clone(), a.hi.clone()));
        let lo = self.emit(choose(
            cmp(CompareOp::GreaterEqual, a.lo, c(0.)),
            low.lo,
            choose(cmp(CompareOp::LessEqual, a.hi, c(0.)), high.lo, c(0.)),
        ));
        let hi = self.emit(extremum(low.hi, high.hi, false));
        Pair { lo, hi }
    }
    fn charge(&mut self, depth: usize) -> Result<()> {
        self.nodes += 1;
        if depth > 64 || self.nodes > self.budget.nodes || self.next_local > self.budget.locals {
            return Err(Error::Unsupported(
                "interval lowering expression budget".into(),
            ));
        }
        Ok(())
    }
    fn expr(&mut self, env: &Env, e: &ScalarExpr, depth: usize) -> Result<Pair> {
        self.charge(depth)?;
        Ok(match e {
            ScalarExpr::Constant(x) if x.is_finite() => Pair { lo: c(*x), hi: c(*x) },
            ScalarExpr::Index(_) => Pair { lo: e.clone(), hi: e.clone() },
            ScalarExpr::Load(id) => env.loads[id.index()].clone().expect("validated readable load"),
            ScalarExpr::Local(id) => env.locals[id.index()].clone(),
            ScalarExpr::Select { condition, if_true, if_false } if index_predicate(condition) => {
                let a = self.expr(env, if_true, depth + 1)?;
                let b = self.expr(env, if_false, depth + 1)?;
                Pair {
                    lo: self.emit(choose((**condition).clone(), a.lo, b.lo)),
                    hi: self.emit(choose((**condition).clone(), a.hi, b.hi)),
                }
            }
            ScalarExpr::Unary { op: UnaryOp::Neg, arg } => {
                let a = self.expr(env, arg, depth + 1)?;
                Pair { lo: ScalarExpr::unary(UnaryOp::Neg, a.hi), hi: ScalarExpr::unary(UnaryOp::Neg, a.lo) }
            }
            ScalarExpr::Fma { lhs, rhs, addend } => {
                let a = self.expr(env, lhs, depth + 1)?;
                let b = self.expr(env, rhs, depth + 1)?;
                let addend = self.expr(env, addend, depth + 1)?;
                let p = self.product(a, b);
                self.add(p, addend)
            }
            ScalarExpr::Binary { op, lhs, rhs } => {
                if *op == BinaryOp::Mul && lhs == rhs {
                    let a = self.expr(env, lhs, depth + 1)?;
                    return Ok(self.square(a));
                }
                let a = self.expr(env, lhs, depth + 1)?;
                let b = self.expr(env, rhs, depth + 1)?;
                match op {
                    BinaryOp::Add => self.add(a, b),
                    BinaryOp::Sub => self.sub(a, b),
                    BinaryOp::Mul => self.product(a, b),
                    _ => return Err(Error::Unsupported(format!("interval binary {op:?}"))),
                }
            }
            _ => return Err(Error::Unsupported("interval expression (only finite constants, indices, loads, locals, neg/add/sub/mul/fma, index selects)".into())),
        })
    }

    fn affine_input(&mut self, x: &Pair, symbol: usize, symbols: usize) -> Form {
        let centre = self.centre(x);
        let above = self.up(BinaryOp::Sub, x.hi.clone(), centre.clone());
        let below = self.up(BinaryOp::Sub, centre.clone(), x.lo.clone());
        let radius = self.emit(extremum(above, below, false));
        let mut coeffs = vec![None; symbols];
        coeffs[symbol] = Some(radius);
        Form {
            center: centre,
            coeffs,
            err: None,
        }
    }
    fn exact_form(&self, center: ScalarExpr, symbols: usize) -> Form {
        Form {
            center,
            coeffs: vec![None; symbols],
            err: None,
        }
    }
    /// `Σ|x_i| + e`, rounded upwards (`None` when the form is exact).
    fn affine_radius(&mut self, x: &Form) -> Option<ScalarExpr> {
        let terms = x
            .coeffs
            .iter()
            .flatten()
            .map(|t| abs(t.clone()))
            .chain(x.err.clone())
            .collect();
        self.up_sum(terms)
    }
    fn affine_range(&mut self, x: &Form) -> Pair {
        match self.affine_radius(x) {
            None => self.outward(x.center.clone()),
            Some(r) => Pair {
                lo: self
                    .outward(bin(BinaryOp::Sub, x.center.clone(), r.clone()))
                    .lo,
                hi: self.outward(bin(BinaryOp::Add, x.center.clone(), r)).hi,
            },
        }
    }
    /// Emit one rounded operation, returning it and its rounding-error bound.
    fn rounded(&mut self, value: ScalarExpr) -> (ScalarExpr, ScalarExpr) {
        let v = self.emit(value);
        let r = self.radius(v.clone());
        (v, r)
    }
    fn affine_linear(&mut self, op: BinaryOp, x: Form, y: Form) -> Form {
        let (center, r0) = self.rounded(bin(op, x.center, y.center));
        let mut errors = vec![r0];
        errors.extend(x.err);
        errors.extend(y.err);
        let coeffs = x
            .coeffs
            .into_iter()
            .zip(y.coeffs)
            .map(|pair| match pair {
                (None, None) => None,
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(if op == BinaryOp::Sub {
                    ScalarExpr::unary(UnaryOp::Neg, b)
                } else {
                    b
                }),
                (Some(a), Some(b)) => {
                    let (v, r) = self.rounded(bin(op, a, b));
                    errors.push(r);
                    Some(v)
                }
            })
            .collect();
        Form {
            center,
            coeffs,
            err: self.up_sum(errors),
        }
    }
    fn affine_product(&mut self, x: Form, y: Form) -> Form {
        let (center, r0) = self.rounded(bin(BinaryOp::Mul, x.center.clone(), y.center.clone()));
        let mut errors = vec![r0];
        let rx = self.affine_radius(&x);
        let ry = self.affine_radius(&y);
        if let (Some(rx), Some(ry)) = (rx, ry) {
            errors.push(self.up(BinaryOp::Mul, rx, ry));
        }
        if let Some(ey) = y.err.clone() {
            errors.push(self.up(BinaryOp::Mul, abs(x.center.clone()), ey));
        }
        if let Some(ex) = x.err.clone() {
            errors.push(self.up(BinaryOp::Mul, abs(y.center.clone()), ex));
        }
        let mut coeffs = vec![];
        for (a, b) in x.coeffs.into_iter().zip(y.coeffs) {
            // x0·y_i + y0·x_i, each product and the sum rounded once.
            let mut part = |coeff: Option<ScalarExpr>, other: &ScalarExpr, lift: &mut Self| {
                coeff.map(|t| {
                    let (v, r) = lift.rounded(bin(BinaryOp::Mul, other.clone(), t));
                    errors.push(r);
                    v
                })
            };
            let p = part(b, &x.center, self);
            let q = part(a, &y.center, self);
            coeffs.push(match (p, q) {
                (Some(p), Some(q)) => {
                    let (v, r) = self.rounded(bin(BinaryOp::Add, p, q));
                    errors.push(r);
                    Some(v)
                }
                (p, q) => p.or(q),
            });
        }
        Form {
            center,
            coeffs,
            err: self.up_sum(errors),
        }
    }
    fn affine_square(&mut self, x: Form) -> Form {
        // x² = x0² + 2·x0·L + 2·x0·t + (L + t)², with (L + t)² ∈ [0, R²] = h ± h.
        let Some(r) = self.affine_radius(&x) else {
            let (center, r0) = self.rounded(bin(BinaryOp::Mul, x.center.clone(), x.center));
            return Form {
                err: Some(r0),
                ..self.exact_form(center, x.coeffs.len())
            };
        };
        let r2 = self.up(BinaryOp::Mul, r.clone(), r);
        let h = self.up(BinaryOp::Mul, r2, c(0.5));
        let (s, rs) = self.rounded(bin(BinaryOp::Mul, x.center.clone(), x.center.clone()));
        let (center, r0) = self.rounded(bin(BinaryOp::Add, s, h.clone()));
        let mut errors = vec![h, rs, r0];
        if let Some(ex) = x.err {
            let twice = bin(BinaryOp::Mul, abs(x.center.clone()), c(2.));
            errors.push(self.up(BinaryOp::Mul, twice, ex));
        }
        let mut coeffs = vec![];
        for coeff in x.coeffs {
            coeffs.push(coeff.map(|t| {
                // RN(2·RN(x0·t)): the doubling is exact unless it overflows, which makes the
                // coefficient infinite and the final range unbounded.
                let (p, rp) = self.rounded(bin(BinaryOp::Mul, x.center.clone(), t));
                errors.push(bin(BinaryOp::Mul, rp, c(2.)));
                self.emit(bin(BinaryOp::Mul, p, c(2.)))
            }));
        }
        Form {
            center,
            coeffs,
            err: self.up_sum(errors),
        }
    }
    fn affine(&mut self, env: &mut AffineEnv, e: &ScalarExpr, depth: usize) -> Result<Form> {
        self.charge(depth)?;
        let n = env.symbols;
        Ok(match e {
            ScalarExpr::Constant(x) if x.is_finite() => self.exact_form(c(*x), n),
            ScalarExpr::Index(_) => self.exact_form(e.clone(), n),
            ScalarExpr::Load(id) => env.inputs[id.index()]
                .clone()
                .expect("validated readable load"),
            ScalarExpr::Local(id) => env.locals[id.index()].clone(),
            ScalarExpr::Select {
                condition,
                if_true,
                if_false,
            } if index_predicate(condition) => {
                let a = self.affine(env, if_true, depth + 1)?;
                let b = self.affine(env, if_false, depth + 1)?;
                let pick = |a: Option<ScalarExpr>, b: Option<ScalarExpr>, lift: &mut Self| {
                    if a.is_none() && b.is_none() {
                        return None;
                    }
                    let (a, b) = (a.unwrap_or(c(0.)), b.unwrap_or(c(0.)));
                    Some(lift.emit(choose((**condition).clone(), a, b)))
                };
                let center = pick(Some(a.center), Some(b.center), self).expect("centre");
                let err = pick(a.err, b.err, self);
                let coeffs = a
                    .coeffs
                    .into_iter()
                    .zip(b.coeffs)
                    .map(|(a, b)| pick(a, b, self))
                    .collect();
                Form {
                    center,
                    coeffs,
                    err,
                }
            }
            ScalarExpr::Unary {
                op: UnaryOp::Neg,
                arg,
            } => {
                let a = self.affine(env, arg, depth + 1)?;
                let neg = |x: ScalarExpr| ScalarExpr::unary(UnaryOp::Neg, x);
                Form {
                    center: neg(a.center),
                    coeffs: a.coeffs.into_iter().map(|t| t.map(neg)).collect(),
                    err: a.err,
                }
            }
            ScalarExpr::Fma { lhs, rhs, addend } => {
                let a = self.affine(env, lhs, depth + 1)?;
                let b = self.affine(env, rhs, depth + 1)?;
                let addend = self.affine(env, addend, depth + 1)?;
                let p = self.affine_product(a, b);
                self.affine_linear(BinaryOp::Add, p, addend)
            }
            ScalarExpr::Binary { op, lhs, rhs } => {
                if *op == BinaryOp::Mul && lhs == rhs {
                    let a = self.affine(env, lhs, depth + 1)?;
                    return Ok(self.affine_square(a));
                }
                let a = self.affine(env, lhs, depth + 1)?;
                let b = self.affine(env, rhs, depth + 1)?;
                match op {
                    BinaryOp::Add | BinaryOp::Sub => self.affine_linear(*op, a, b),
                    BinaryOp::Mul => self.affine_product(a, b),
                    _ => return Err(Error::Unsupported(format!("affine binary {op:?}"))),
                }
            }
            _ => {
                return Err(Error::Unsupported(
                    "affine expression (only finite constants, indices, loads, locals, \
                     neg/add/sub/mul/fma, index selects)"
                        .into(),
                ));
            }
        })
    }
}

// Branches depending only on exact iteration coordinates are determinate for every
// interval input. Data-dependent predicates still require explicit branch enclosures.
fn index_predicate(p: &Predicate) -> bool {
    match p {
        Predicate::Constant(_) => true,
        Predicate::Compare { lhs, rhs, .. } => {
            matches!(**lhs, ScalarExpr::Index(_) | ScalarExpr::Constant(_))
                && matches!(**rhs, ScalarExpr::Index(_) | ScalarExpr::Constant(_))
        }
        Predicate::Not(x) => index_predicate(x),
        Predicate::And(a, b) | Predicate::Or(a, b) => index_predicate(a) && index_predicate(b),
    }
}
