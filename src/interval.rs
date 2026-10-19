//! Explicit real-interval enclosure lowering into ordinary Malleus f64 IR.
//!
//! Requires round-to-nearest/ties-even and gradual underflow, with no contraction or
//! reassociation. Every arithmetic endpoint is padded outwards; this is deliberately
//! wider than nextafter-based interval arithmetic. See INTERVALS.md for the proof.
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
pub fn lower_intervals(source: &ValidatedKernel) -> Result<IntervalProgram> {
    let k = source.as_kernel();
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
    let mut result = k.clone();
    result.name = format!("{}::outward-interval-v1", k.name);
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
    let mut lift = Lift {
        statements: vec![],
        locals: vec![],
        next_local: 0,
        nodes: 0,
    };
    for statement in &k.body.statements {
        match statement {
            Statement::Let { value, .. } => {
                let pair = lift.expr(value, 0)?;
                lift.locals.push(pair);
            }
            Statement::Store { operand, value } => {
                let pair = lift.expr(value, 0)?;
                let [lo, hi] = IntervalProgram::endpoints(*operand);
                lift.statements.push(Statement::Store {
                    operand: lo,
                    value: pair.lo,
                });
                lift.statements.push(Statement::Store {
                    operand: hi,
                    value: pair.hi,
                });
            }
        }
    }
    result.body.statements = lift.statements;
    let validated = validate(result).map_err(|e| Error::Compile(e.to_string()))?;
    Ok(IntervalProgram {
        executable: Executable::reference(validated),
    })
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
fn cmp(op: CompareOp, a: ScalarExpr, b: ScalarExpr) -> Predicate {
    Predicate::Compare {
        op,
        lhs: Box::new(a),
        rhs: Box::new(b),
    }
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

struct Lift {
    statements: Vec<Statement>,
    locals: Vec<Pair>,
    next_local: usize,
    nodes: usize,
}
impl Lift {
    fn emit(&mut self, value: ScalarExpr) -> ScalarExpr {
        let local = LocalId::new(self.next_local);
        self.next_local += 1;
        self.statements.push(Statement::Let { local, value });
        ScalarExpr::Local(local)
    }
    fn outward(&mut self, value: ScalarExpr) -> Pair {
        let v = self.emit(value);
        let abs = ScalarExpr::unary(UnaryOp::Abs, v.clone());
        let finite = cmp(CompareOp::LessEqual, abs.clone(), c(f64::MAX));
        let radius = self.emit(bin(
            BinaryOp::Add,
            bin(BinaryOp::Mul, abs, c(f64::EPSILON)),
            c(f64::MIN_POSITIVE),
        ));
        let lo = self.emit(choose(
            finite.clone(),
            bin(BinaryOp::Sub, v.clone(), radius.clone()),
            bin(BinaryOp::Mul, c(-f64::MAX), c(2.)),
        ));
        let hi = self.emit(choose(
            finite,
            bin(BinaryOp::Add, v, radius),
            bin(BinaryOp::Mul, c(f64::MAX), c(2.)),
        ));
        Pair { lo, hi }
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
    fn expr(&mut self, e: &ScalarExpr, depth: usize) -> Result<Pair> {
        self.nodes += 1;
        if depth > 64 || self.nodes > 512 || self.next_local > 1024 {
            return Err(Error::Unsupported(
                "interval lowering expression budget".into(),
            ));
        }
        Ok(match e {
            ScalarExpr::Constant(x) if x.is_finite() => Pair { lo: c(*x), hi: c(*x) },
            ScalarExpr::Index(_) => Pair { lo: e.clone(), hi: e.clone() },
            ScalarExpr::Load(id) => {
                let [lo, hi] = IntervalProgram::endpoints(*id);
                Pair { lo: ScalarExpr::Load(lo), hi: ScalarExpr::Load(hi) }
            }
            ScalarExpr::Local(id) => self.locals[id.index()].clone(),
            ScalarExpr::Select { condition, if_true, if_false } if index_predicate(condition) => {
                let a=self.expr(if_true,depth+1)?;let b=self.expr(if_false,depth+1)?;
                Pair {lo:self.emit(choose((**condition).clone(),a.lo,b.lo)),hi:self.emit(choose((**condition).clone(),a.hi,b.hi))}
            }
            ScalarExpr::Unary { op: UnaryOp::Neg, arg } => {
                let a = self.expr(arg, depth + 1)?;
                Pair { lo: ScalarExpr::unary(UnaryOp::Neg, a.hi), hi: ScalarExpr::unary(UnaryOp::Neg, a.lo) }
            }
            ScalarExpr::Binary { op, lhs, rhs } => {
                if *op == BinaryOp::Mul && lhs == rhs {
                    let a = self.expr(lhs, depth + 1)?;
                    return Ok(self.square(a));
                }
                let a = self.expr(lhs, depth + 1)?;
                let b = self.expr(rhs, depth + 1)?;
                match op {
                    BinaryOp::Add => {
                        let lo = self.outward(bin(*op, a.lo, b.lo)).lo;
                        let hi = self.outward(bin(*op, a.hi, b.hi)).hi;
                        Pair { lo, hi }
                    }
                    BinaryOp::Sub => {
                        let lo = self.outward(bin(*op, a.lo, b.hi)).lo;
                        let hi = self.outward(bin(*op, a.hi, b.lo)).hi;
                        Pair { lo, hi }
                    }
                    BinaryOp::Mul => self.product(a, b),
                    _ => return Err(Error::Unsupported(format!("interval binary {op:?}"))),
                }
            }
            _ => return Err(Error::Unsupported("interval expression (only finite constants, indices, loads, locals, neg/add/sub/mul)".into())),
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
