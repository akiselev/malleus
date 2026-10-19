//! Derivative and schedule fixtures, random kernels over the full scalar op set, and inputs
//! with IEEE specials.

use super::{Rng, fixtures, same_bits};
use malleus::*;

fn load(i: usize) -> ScalarExpr {
    ScalarExpr::Load(OperandId::new(i))
}

fn b(op: BinaryOp, l: ScalarExpr, r: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(op, l, r)
}

fn u(op: UnaryOp, a: ScalarExpr) -> ScalarExpr {
    ScalarExpr::unary(op, a)
}

fn pointwise(name: &str, n: usize, inputs: usize, statements: Vec<Statement>) -> StructuredKernel {
    let axis = AxisId::new(0);
    let mut operands: Vec<_> = (0..inputs)
        .map(|i| KernelOperand::tensor(format!("x{i}"), vec![n], AccessMode::Read))
        .collect();
    operands.push(KernelOperand::tensor("out", vec![n], AccessMode::Write));
    StructuredKernel {
        name: name.into(),
        iteration_domain: IterationDomain::new(vec![n]),
        iterators: vec![IteratorKind::Parallel],
        operands,
        indexing_maps: (0..=inputs)
            .map(|i| IndexingMap::new(OperandId::new(i), vec![IndexExpr::axis(axis)]))
            .collect(),
        body: KernelRegion { statements },
        numeric_policy: NumericPolicy::default(),
    }
}

/// abs, min, max, floor and a select in one kernel: `out = max(|x0|, min(x1, x0)) +
/// floor(x1) + select(x0 < x1, x0, x1)`.
pub fn nonsmooth(n: usize) -> StructuredKernel {
    pointwise(
        "nonsmooth",
        n,
        2,
        vec![
            Statement::Let {
                local: LocalId::new(0),
                value: b(
                    BinaryOp::Max,
                    u(UnaryOp::Abs, load(0)),
                    b(BinaryOp::Min, load(1), load(0)),
                ),
            },
            Statement::Store {
                operand: OperandId::new(2),
                value: b(
                    BinaryOp::Add,
                    b(
                        BinaryOp::Add,
                        ScalarExpr::Local(LocalId::new(0)),
                        u(UnaryOp::Floor, load(1)),
                    ),
                    select_less(load(0), load(1), load(0), load(1)),
                ),
            },
        ],
    )
}

pub fn select_less(
    l: ScalarExpr,
    r: ScalarExpr,
    if_true: ScalarExpr,
    if_false: ScalarExpr,
) -> ScalarExpr {
    ScalarExpr::Select {
        condition: Box::new(Predicate::Compare {
            op: CompareOp::Less,
            lhs: Box::new(l),
            rhs: Box::new(r),
        }),
        if_true: Box::new(if_true),
        if_false: Box::new(if_false),
    }
}

/// Smooth except for selects: `out = select(x0 < x1, x0*x1, x0/x1) - sqrt(x1*x1 + 1)`.
pub fn select_only(n: usize) -> StructuredKernel {
    pointwise(
        "select-only",
        n,
        2,
        vec![Statement::Store {
            operand: OperandId::new(2),
            value: b(
                BinaryOp::Sub,
                select_less(
                    load(0),
                    load(1),
                    b(BinaryOp::Mul, load(0), load(1)),
                    b(BinaryOp::Div, load(0), load(1)),
                ),
                u(
                    UnaryOp::Sqrt,
                    b(
                        BinaryOp::Add,
                        b(BinaryOp::Mul, load(1), load(1)),
                        ScalarExpr::Constant(1.0),
                    ),
                ),
            ),
        }],
    )
}

/// Explicit fma, exp and a sum: `out = fma(x0, x1, exp(x0)) + x0*x1`.
pub fn fma_exp(n: usize) -> StructuredKernel {
    pointwise(
        "fma-exp",
        n,
        2,
        vec![Statement::Store {
            operand: OperandId::new(2),
            value: b(
                BinaryOp::Add,
                ScalarExpr::fma(load(0), load(1), u(UnaryOp::Exp, load(0))),
                b(BinaryOp::Mul, load(0), load(1)),
            ),
        }],
    )
}

/// `out[i] = min over j of x[i, j] * w[j]` (a min reduction).
pub fn min_reduction(n: usize) -> StructuredKernel {
    let mut k = fixtures::matvec(n);
    k.name = format!("min-reduction-{n}");
    k.operands[2].access = AccessMode::Reduce(ReductionOp::Min);
    k
}

pub fn f32_variant(mut k: StructuredKernel) -> StructuredKernel {
    k.name = format!("{}-f32", k.name);
    k.numeric_policy.scalar_type = ScalarType::F32;
    k
}

/// Every derivative fixture family: the shared consumer-neutral fixtures plus the local ones.
pub fn derivative_fixtures() -> Vec<StructuredKernel> {
    vec![
        fixtures::polynomial(3),
        fixtures::quadratic(4),
        fixtures::matvec(3),
        fixtures::polynomial_batch(3),
        nonsmooth(3),
        select_only(3),
        fma_exp(2),
        min_reduction(2),
        f32_variant(fixtures::polynomial(3)),
        f32_variant(nonsmooth(2)),
    ]
}

/// A request over every readable (independent) and every writable (dependent) operand.
pub fn full_request(k: &StructuredKernel, mode: DerivativeMode) -> DerivativeRequest {
    let pick = |f: fn(AccessMode) -> bool| {
        k.operands
            .iter()
            .enumerate()
            .filter(|(_, o)| f(o.access))
            .map(|(i, _)| OperandId::new(i))
            .collect()
    };
    DerivativeRequest {
        mode,
        independent_operands: pick(AccessMode::can_read),
        dependent_operands: pick(AccessMode::can_write),
    }
}

/// A random scalar value biased towards IEEE specials.
pub fn special(r: &mut Rng) -> f64 {
    match r.below(10) {
        0 => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY][r.below(3) as usize],
        1 => [0.0, -0.0][r.below(2) as usize],
        2 => [1.0, -1.0, 0.5, 2.0][r.below(4) as usize],
        _ => r.finite(),
    }
}

/// Buffers in operand order: readable operands random (with specials), writable operands
/// random too (so `Reduce` accumulators and unwritten elements are observable).
pub fn random_buffers(r: &mut Rng, k: &StructuredKernel) -> Vec<Vec<f64>> {
    k.operands
        .iter()
        .map(|o| {
            (0..o.region.offset + o.region.length)
                .map(|_| special(r))
                .collect()
        })
        .collect()
}

pub fn assert_same(a: &[Vec<f64>], b: &[Vec<f64>], context: &str) {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        for (j, (p, q)) in x.iter().zip(y).enumerate() {
            assert!(
                same_bits(*p, *q),
                "{context}: operand {i}[{j}]: {p:?} vs {q:?}"
            );
        }
    }
}

pub fn run(k: &StructuredKernel, buffers: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    super::interpret(
        &Executable::reference(validate(k.clone()).unwrap()),
        buffers,
    )
}

const CONSTANTS: [f64; 7] = [1.0, -0.0, 0.0, 2.0, 0.5, -1.5, 1.0];

fn random_expr(r: &mut Rng, depth: u32, inputs: usize, locals: usize) -> ScalarExpr {
    if depth == 0 || r.below(5) == 0 {
        return match r.below(6) {
            0 => ScalarExpr::Constant(CONSTANTS[r.below(CONSTANTS.len() as u64) as usize]),
            1 if locals > 0 => ScalarExpr::Local(LocalId::new(r.below(locals as u64) as usize)),
            _ => load(r.below(inputs as u64) as usize),
        };
    }
    let sub = |r: &mut Rng| random_expr(r, depth - 1, inputs, locals);
    match r.below(16) {
        0 => u(UnaryOp::Neg, sub(r)),
        1 => u(UnaryOp::Abs, sub(r)),
        2 => u(UnaryOp::Sqrt, sub(r)),
        3 => u(UnaryOp::Floor, sub(r)),
        4 => b(BinaryOp::Sub, sub(r), sub(r)),
        5 => b(BinaryOp::Div, sub(r), sub(r)),
        6 => b(BinaryOp::Min, sub(r), sub(r)),
        7 => b(BinaryOp::Max, sub(r), sub(r)),
        8 => ScalarExpr::fma(sub(r), sub(r), sub(r)),
        9 => {
            let (l, rr, t, f) = (sub(r), sub(r), sub(r), sub(r));
            select_less(l, rr, t, f)
        }
        10 => {
            // A repeated subexpression for CSE.
            let s = sub(r);
            b(BinaryOp::Mul, s.clone(), s)
        }
        11 | 12 => b(BinaryOp::Add, sub(r), sub(r)),
        _ => b(BinaryOp::Mul, sub(r), sub(r)),
    }
}

/// A random pointwise kernel over three inputs with up to three lets and up to two stores
/// into the output (the first is then dead).
pub fn random_kernel(r: &mut Rng, f32: bool) -> StructuredKernel {
    let n = 1 + r.below(3) as usize;
    let mut statements = Vec::new();
    let lets = r.below(4) as usize;
    for i in 0..lets {
        statements.push(Statement::Let {
            local: LocalId::new(i),
            value: random_expr(r, 3, 3, i),
        });
    }
    for _ in 0..1 + r.below(2) {
        statements.push(Statement::Store {
            operand: OperandId::new(3),
            value: random_expr(r, 3, 3, lets),
        });
    }
    let k = pointwise("random", n, 3, statements);
    if f32 { f32_variant(k) } else { k }
}
