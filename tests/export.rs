//! Symbolic (`malleus::symbolic`) and FPCore (`malleus::fpcore`) export.
//!
//! The symbolic DAG round-trips through JSON, passes its structural check, evaluates in exact
//! dyadic arithmetic to the real semantics of the source (and to the interpreter where every
//! operation is exact), and the export of a JVP kernel equals the forward-mode derivative of
//! the primal export. FPCore output is pinned by goldens and checked structurally; no external
//! FPBench/Racket tool is installed on the build host, so none is run.

mod support;

use std::collections::{HashMap, HashSet};

use malleus::fpcore::{FpcoreError, InputBox, export_fpcore};
use malleus::symbolic::{
    MAX_POINTS, SYMBOLIC_SCHEMA, SymbolicError, SymbolicKernel, SymbolicNode, export_symbolic,
};
use malleus::*;
use support::dyadic::Dyadic;
use support::*;

fn executable(k: StructuredKernel) -> Executable {
    Executable::reference(validate(k).unwrap())
}

fn export(k: &StructuredKernel) -> SymbolicKernel {
    let s = export_symbolic(&executable(k.clone())).unwrap();
    s.check().unwrap();
    s
}

/// Exact value of every node under `input` values (predicates as 0/1 in `bools`).
fn evaluate(s: &SymbolicKernel, mut input: impl FnMut(usize) -> Dyadic) -> Vec<Option<Dyadic>> {
    let mut values: Vec<Option<Dyadic>> = vec![];
    let mut bools: Vec<bool> = vec![];
    for node in &s.nodes {
        let v = |i: usize| values[i].clone().expect("value node");
        let (value, flag) = match *node {
            SymbolicNode::Input { input: i } => (Some(input(i)), false),
            SymbolicNode::Constant(c) => (
                Some(Dyadic::from_parts(c.mantissa, c.exponent as i64)),
                false,
            ),
            SymbolicNode::Unary { op, arg } => (
                Some(match op {
                    UnaryOp::Neg => v(arg).neg(),
                    UnaryOp::Abs if v(arg).cmp(&Dyadic::zero()).is_lt() => v(arg).neg(),
                    UnaryOp::Abs => v(arg),
                    _ => panic!("no exact {op:?}"),
                }),
                false,
            ),
            SymbolicNode::Binary { op, lhs, rhs } => {
                let (a, b) = (v(lhs), v(rhs));
                let less = a.cmp(&b).is_lt();
                (
                    Some(match op {
                        BinaryOp::Add => a.add(&b),
                        BinaryOp::Sub => a.sub(&b),
                        BinaryOp::Mul => a.mul(&b),
                        BinaryOp::Min if less => a,
                        BinaryOp::Min => b,
                        BinaryOp::Max if less => b,
                        BinaryOp::Max => a,
                        _ => panic!("no exact {op:?}"),
                    }),
                    false,
                )
            }
            SymbolicNode::Fma {
                lhs, rhs, addend, ..
            } => (Some(v(lhs).mul(&v(rhs)).add(&v(addend))), false),
            SymbolicNode::Bool { value } => (None, value),
            SymbolicNode::Compare { op, lhs, rhs } => {
                let o = v(lhs).cmp(&v(rhs));
                let flag = match op {
                    CompareOp::Eq => o.is_eq(),
                    CompareOp::NotEq => o.is_ne(),
                    CompareOp::Less => o.is_lt(),
                    CompareOp::LessEqual => o.is_le(),
                    CompareOp::Greater => o.is_gt(),
                    CompareOp::GreaterEqual => o.is_ge(),
                };
                (None, flag)
            }
            SymbolicNode::Not { arg } => (None, !bools[arg]),
            SymbolicNode::And { lhs, rhs } => (None, bools[lhs] && bools[rhs]),
            SymbolicNode::Or { lhs, rhs } => (None, bools[lhs] || bools[rhs]),
            SymbolicNode::Select {
                condition,
                if_true,
                if_false,
            } => (
                Some(v(if bools[condition] { if_true } else { if_false })),
                false,
            ),
        };
        values.push(value);
        bools.push(flag);
    }
    values
}

/// Forward-mode (value, tangent) evaluation over the add/sub/mul/neg/fma/select fragment.
fn evaluate_dual(
    s: &SymbolicKernel,
    mut input: impl FnMut(usize) -> (Dyadic, Dyadic),
) -> Vec<Option<(Dyadic, Dyadic)>> {
    let mut out: Vec<Option<(Dyadic, Dyadic)>> = vec![];
    let values = evaluate(s, |i| input(i).0);
    let mut bools = vec![false; s.nodes.len()];
    for (n, node) in s.nodes.iter().enumerate() {
        let d = |i: usize| out[i].clone().expect("value node");
        let r = match *node {
            SymbolicNode::Input { input: i } => Some(input(i)),
            SymbolicNode::Constant(_) => Some((values[n].clone().unwrap(), Dyadic::zero())),
            SymbolicNode::Unary {
                op: UnaryOp::Neg,
                arg,
            } => Some((d(arg).0.neg(), d(arg).1.neg())),
            SymbolicNode::Binary { op, lhs, rhs } => {
                let ((a, da), (b, db)) = (d(lhs), d(rhs));
                Some(match op {
                    BinaryOp::Add => (a.add(&b), da.add(&db)),
                    BinaryOp::Sub => (a.sub(&b), da.sub(&db)),
                    BinaryOp::Mul => (a.mul(&b), da.mul(&b).add(&a.mul(&db))),
                    _ => panic!("no dual {op:?}"),
                })
            }
            SymbolicNode::Fma {
                lhs, rhs, addend, ..
            } => {
                let ((a, da), (b, db), (c, dc)) = (d(lhs), d(rhs), d(addend));
                Some((a.mul(&b).add(&c), da.mul(&b).add(&a.mul(&db)).add(&dc)))
            }
            SymbolicNode::Select {
                condition,
                if_true,
                if_false,
            } => Some(d(if bools[condition] { if_true } else { if_false })),
            SymbolicNode::Compare { .. } | SymbolicNode::Bool { .. } => {
                // Predicate values come from the plain evaluation.
                bools[n] = predicate_value(s, &values, &bools, n);
                None
            }
            SymbolicNode::Not { arg } => {
                bools[n] = !bools[arg];
                None
            }
            SymbolicNode::And { lhs, rhs } => {
                bools[n] = bools[lhs] && bools[rhs];
                None
            }
            SymbolicNode::Or { lhs, rhs } => {
                bools[n] = bools[lhs] || bools[rhs];
                None
            }
            _ => panic!("no dual {node:?}"),
        };
        out.push(r);
    }
    out
}

fn predicate_value(s: &SymbolicKernel, values: &[Option<Dyadic>], _: &[bool], n: usize) -> bool {
    match s.nodes[n] {
        SymbolicNode::Bool { value } => value,
        SymbolicNode::Compare { op, lhs, rhs } => {
            let o = values[lhs]
                .as_ref()
                .unwrap()
                .cmp(values[rhs].as_ref().unwrap());
            match op {
                CompareOp::Eq => o.is_eq(),
                CompareOp::NotEq => o.is_ne(),
                CompareOp::Less => o.is_lt(),
                CompareOp::LessEqual => o.is_le(),
                CompareOp::Greater => o.is_gt(),
                CompareOp::GreaterEqual => o.is_ge(),
            }
        }
        _ => unreachable!(),
    }
}

/// Data-dependent select, Fma, min/max, abs: `select(x < y, fma(x, y, z), max(x, min(y, z))) - |x|`.
fn piecewise(n: usize) -> StructuredKernel {
    let axis = AxisId::new(0);
    let load = |i| ScalarExpr::Load(OperandId::new(i));
    let mut operands: Vec<_> = ["x", "y", "z"]
        .iter()
        .map(|name| KernelOperand::tensor(*name, vec![n], AccessMode::Read))
        .collect();
    operands.push(KernelOperand::tensor("out", vec![n], AccessMode::Write));
    StructuredKernel {
        name: "piecewise".into(),
        iteration_domain: IterationDomain::new(vec![n]),
        iterators: vec![IteratorKind::Parallel],
        indexing_maps: (0..4)
            .map(|i| IndexingMap::new(OperandId::new(i), vec![IndexExpr::axis(axis)]))
            .collect(),
        operands,
        body: KernelRegion {
            statements: vec![
                Statement::Let {
                    local: LocalId::new(0),
                    value: ScalarExpr::Select {
                        condition: Box::new(Predicate::Compare {
                            op: CompareOp::Less,
                            lhs: Box::new(load(0)),
                            rhs: Box::new(load(1)),
                        }),
                        if_true: Box::new(ScalarExpr::fma(load(0), load(1), load(2))),
                        if_false: Box::new(ScalarExpr::binary(
                            BinaryOp::Max,
                            load(0),
                            ScalarExpr::binary(BinaryOp::Min, load(1), load(2)),
                        )),
                    },
                },
                Statement::Store {
                    operand: OperandId::new(3),
                    value: ScalarExpr::binary(
                        BinaryOp::Sub,
                        ScalarExpr::Local(LocalId::new(0)),
                        ScalarExpr::unary(UnaryOp::Abs, load(0)),
                    ),
                },
            ],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

/// Short dyadics (`k/8`, |k| <= 64) keep every operation of the fixtures exact.
fn small_buffers(k: &StructuredKernel, r: &mut Rng) -> Vec<Vec<f64>> {
    k.operands
        .iter()
        .map(|o| {
            (0..o.region.offset + o.region.length)
                .map(|_| (r.below(129) as f64 - 64.0) / 8.0)
                .collect()
        })
        .collect()
}

fn fixtures() -> Vec<StructuredKernel> {
    vec![
        fixtures::polynomial(3),
        fixtures::quadratic(4),
        fixtures::matvec(3),
        fixtures::polynomial_batch(2),
        piecewise(5),
    ]
}

#[test]
fn symbolic_exports_round_trip_and_check() {
    let mut r = Rng(0x5EB_0001);
    let mut kernels = fixtures();
    kernels.extend((0..50).map(|_| Prog::generate(&mut r, 4).kernel()));
    for k in &kernels {
        let s = export(k);
        assert_eq!(s.schema, SYMBOLIC_SCHEMA);
        assert_eq!(s.kernel_digest, kernel_digest(k));
        let json = serde_json::to_string(&s).unwrap();
        let back: SymbolicKernel = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
        back.check().unwrap();
    }
    // The structural check rejects forward references, kind confusion and unknown schemas.
    let mut s = export(&fixtures::polynomial(1));
    let last = s.nodes.len() - 1;
    s.nodes.swap(0, last);
    assert!(s.check().is_err());
    let mut s = export(&piecewise(1));
    let select = s
        .nodes
        .iter()
        .position(|n| matches!(n, SymbolicNode::Select { .. }))
        .unwrap();
    if let SymbolicNode::Select { condition, .. } = &mut s.nodes[select] {
        *condition = 0;
    }
    assert_eq!(
        s.check(),
        Err(SymbolicError::InvalidArgument { node: select })
    );
    let mut s = export(&fixtures::polynomial(1));
    s.schema = "malleus-symbolic/0".into();
    assert!(matches!(s.check(), Err(SymbolicError::Schema(_))));
}

#[test]
fn symbolic_schema_is_pinned() {
    let s = export(&fixtures::polynomial(1));
    let json = serde_json::to_string(&s).unwrap();
    let expected = r#"{"schema":"malleus-symbolic/1","kernel":"polynomial-1","kernel_digest":{"algorithm":"blake3","hex":"HEX"},"precision":"f64","level":"strict_host","inputs":[{"operand":1,"element":0,"name":"c0"},{"operand":0,"element":0,"name":"t"},{"operand":2,"element":0,"name":"c1"},{"operand":3,"element":0,"name":"c2"}],"nodes":[{"node":"input","input":0},{"node":"input","input":1},{"node":"input","input":2},{"node":"input","input":3},{"node":"binary","op":"mul","lhs":3,"rhs":1},{"node":"binary","op":"add","lhs":2,"rhs":4},{"node":"binary","op":"mul","lhs":1,"rhs":5},{"node":"binary","op":"add","lhs":0,"rhs":6}],"outputs":[{"operand":4,"element":0,"node":7,"name":"out"}]}"#;
    assert_eq!(
        json,
        expected.replace("HEX", &kernel_digest(&fixtures::polynomial(1)).hex)
    );
    // Constants carry their bits and exact dyadic value; the sign of zero is in the bits.
    let mut k = fixtures::polynomial(1);
    k.body.statements = vec![Statement::Store {
        operand: OperandId::new(4),
        value: ScalarExpr::binary(
            BinaryOp::Add,
            ScalarExpr::Constant(0.75),
            ScalarExpr::Constant(-0.0),
        ),
    }];
    let json = serde_json::to_string(&export(&k).nodes).unwrap();
    assert_eq!(
        json,
        r#"[{"node":"constant","bits":4604930618986332160,"mantissa":3,"exponent":-2},{"node":"constant","bits":9223372036854775808,"mantissa":0,"exponent":0},{"node":"binary","op":"add","lhs":0,"rhs":1}]"#
    );
}

#[test]
fn exact_evaluation_equals_the_real_semantics() {
    // Random interval-fragment programs: the DAG's exact value equals the exact value of the
    // source expression at arbitrary finite inputs (no representability needed).
    let mut r = Rng(0x5EB_0002);
    for _ in 0..300 {
        let prog = Prog::generate(&mut r, 4);
        let s = export(&prog.kernel());
        let data: Vec<Vec<f64>> = (0..4)
            .map(|_| (0..prog.n).map(|_| r.finite()).collect())
            .collect();
        let values = evaluate(&s, |i| {
            let input = &s.inputs[i];
            Dyadic::from_f64(data[input.operand.index()][input.element])
        });
        assert_eq!(s.outputs.len(), prog.n);
        for o in &s.outputs {
            let loads: Vec<Dyadic> = (0..4)
                .map(|k| Dyadic::from_f64(data[k][o.element]))
                .collect();
            let exact = prog.exact(&loads, o.element);
            assert!(
                values[o.node].as_ref().unwrap().cmp(&exact).is_eq(),
                "{prog:?}"
            );
        }
    }
}

#[test]
fn exact_evaluation_equals_the_interpreter_where_exact() {
    let mut r = Rng(0x5EB_0003);
    for k in fixtures() {
        for _ in 0..20 {
            let buffers = small_buffers(&k, &mut r);
            let s = export(&k);
            let values = evaluate(&s, |i| {
                let input = &s.inputs[i];
                Dyadic::from_f64(buffers[input.operand.index()][input.element])
            });
            let out = interpret(&executable(k.clone()), buffers);
            for o in &s.outputs {
                let got = out[o.operand.index()][o.element];
                assert_eq!(
                    values[o.node].as_ref().unwrap().cmp_f64(got),
                    Some(std::cmp::Ordering::Equal),
                    "{} {}[{}]",
                    k.name,
                    o.name,
                    o.element
                );
            }
        }
    }
}

#[test]
fn jvp_export_is_the_derivative_of_the_primal_export() {
    let mut r = Rng(0x5EB_0004);
    let mut kernels = vec![
        fixtures::polynomial(3),
        fixtures::quadratic(4),
        fixtures::matvec(2),
    ];
    kernels.extend((0..100).map(|_| Prog::generate(&mut r, 3).kernel()));
    for k in kernels {
        let reads: Vec<OperandId> = (0..k.operands.len())
            .map(OperandId::new)
            .filter(|o| k.operands[o.index()].access == AccessMode::Read)
            .collect();
        let writes: Vec<OperandId> = (0..k.operands.len())
            .map(OperandId::new)
            .filter(|o| k.operands[o.index()].access.can_write())
            .collect();
        let product = differentiate(
            &k,
            &DerivativeRequest {
                mode: DerivativeMode::Jvp,
                independent_operands: reads,
                dependent_operands: writes,
            },
        )
        .unwrap();
        let primal = export(&k);
        let jvp = export(&product.kernel);
        // Values and directions per (primal operand, element); reduction outputs start at 0.25
        // with zero tangent.
        let mut value = HashMap::new();
        let mut direction = HashMap::new();
        let mut at = |m: &mut HashMap<(usize, usize), f64>, key, scale: f64| {
            *m.entry(key).or_insert_with(|| r.finite() * scale)
        };
        let primal_values = evaluate_dual(&primal, |i| {
            let input = &primal.inputs[i];
            let key = (input.operand.index(), input.element);
            if k.operands[key.0].access == AccessMode::Read {
                (
                    Dyadic::from_f64(at(&mut value, key, 1.0)),
                    Dyadic::from_f64(at(&mut direction, key, 1.0)),
                )
            } else {
                (Dyadic::from_f64(0.25), Dyadic::zero())
            }
        });
        let jvp_values = evaluate(&jvp, |i| {
            let input = &jvp.inputs[i];
            let q = input.operand;
            let e = input.element;
            if let Some(p) = product.primal_operands.iter().find(|p| p.derivative == q) {
                Dyadic::from_f64(value[&(p.primal.index(), e)])
            } else if let Some(p) = product
                .independent_operands
                .iter()
                .find(|p| p.derivative == q)
            {
                Dyadic::from_f64(
                    direction
                        .get(&(p.primal.index(), e))
                        .copied()
                        .unwrap_or(0.0),
                )
            } else {
                Dyadic::zero()
            }
        });
        let mut checked = HashSet::new();
        for o in &jvp.outputs {
            let p = product
                .dependent_operands
                .iter()
                .find(|d| d.derivative == o.operand)
                .unwrap()
                .primal;
            let primal_out = primal
                .outputs
                .iter()
                .find(|x| x.operand == p && x.element == o.element)
                .unwrap();
            let tangent = &primal_values[primal_out.node].as_ref().unwrap().1;
            assert!(
                jvp_values[o.node].as_ref().unwrap().cmp(tangent).is_eq(),
                "{}: {}[{}]",
                k.name,
                o.name,
                o.element
            );
            checked.insert((p, o.element));
        }
        assert_eq!(checked.len(), primal.outputs.len(), "{}", k.name);
    }
}

#[test]
fn oversized_domains_refuse() {
    let k = fixtures::polynomial(MAX_POINTS + 1);
    assert_eq!(
        export_symbolic(&executable(k.clone())),
        Err(SymbolicError::PointBudget {
            points: Some(MAX_POINTS + 1)
        })
    );
    assert!(matches!(
        export_fpcore(&executable(k), &[]),
        Err(FpcoreError::Symbolic(SymbolicError::PointBudget { .. }))
    ));
    let bad = InputBox {
        operand: OperandId::new(0),
        element: None,
        lo: 1.0,
        hi: 0.0,
    };
    assert_eq!(
        export_fpcore(&executable(fixtures::polynomial(1)), &[bad]),
        Err(FpcoreError::InvalidBox { index: 0 })
    );
}

const OPERATORS: &[&str] = &[
    "FPCore", "let*", "if", "+", "-", "*", "/", "fabs", "sqrt", "exp", "log", "sin", "cos", "tan",
    "floor", "ceil", "pow", "fmin", "fmax", "atan2", "fma", "==", "!=", "<", "<=", ">", ">=",
    "and", "or", "not",
];

/// Balanced s-expressions; every list head is an FPCore operator (or the argument / `let*`
/// binding list), and every symbol is an operator, argument, binding, property or literal.
fn check_structure(source: &str) {
    let spaced = source
        .replace('(', " ( ")
        .replace(')', " ) ")
        .replace('[', " [ ")
        .replace(']', " ] ");
    let tokens: Vec<&str> = spaced.split_whitespace().collect();
    let mut depth: Vec<&str> = vec![];
    let mut arguments: HashSet<&str> = HashSet::new();
    let mut bound: HashSet<&str> = HashSet::new();
    let mut argument_list = None;
    let mut in_string = false;
    for (i, &t) in tokens.iter().enumerate() {
        if in_string || t.starts_with('"') {
            in_string = !(t.ends_with('"') && (t.len() > 1 || in_string));
            continue;
        }
        let previous = if i > 0 { tokens[i - 1] } else { "" };
        match t {
            "(" | "[" => {
                depth.push(t);
                if previous == "FPCore" {
                    argument_list = Some(depth.len());
                }
            }
            ")" => {
                if argument_list == Some(depth.len()) {
                    argument_list = None;
                }
                assert_eq!(depth.pop(), Some("("), "{source}");
            }
            "]" => assert_eq!(depth.pop(), Some("["), "{source}"),
            _ if argument_list.is_some() => {
                arguments.insert(t);
            }
            _ if previous == "[" => {
                bound.insert(t);
            }
            _ if previous == "(" => {
                assert!(OPERATORS.contains(&t), "unmapped head {t} in {source}");
            }
            _ if t.starts_with(':')
                || t.starts_with(|c: char| c.is_ascii_digit() || c == '-')
                || matches!(t, "TRUE" | "FALSE" | "binary64" | "binary32") => {}
            _ => assert!(
                arguments.contains(t) || bound.contains(t),
                "unknown symbol {t} in {source}"
            ),
        }
    }
    assert!(depth.is_empty(), "{source}");
}

#[test]
fn fpcore_goldens() {
    let boxes = [
        InputBox {
            operand: OperandId::new(0),
            element: None,
            lo: -1.0,
            hi: 1.0,
        },
        InputBox {
            operand: OperandId::new(1),
            element: None,
            lo: 0.5,
            hi: f64::INFINITY,
        },
    ];
    let k = fixtures::polynomial(1);
    let digest = kernel_digest(&k).hex;
    let out = export_fpcore(&executable(k), &boxes).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].source,
        format!(
            "(FPCore (c0_0 t_0 c1_0 c2_0)\n  :name \"polynomial-1 out[0]\"\n  :precision binary64\n  \
             :pre (and (<= 0x1p-1 c0_0) (<= -1 t_0 1))\n  :malleus-kernel-digest \"blake3:{digest}\"\n  \
             :malleus-level \"StrictHost\"\n  (+ c0_0 (* t_0 (+ c1_0 (* c2_0 t_0)))))"
        )
    );
    // Reductions unroll; shared subexpressions become `let*` bindings.
    let k = fixtures::quadratic(2);
    let out = export_fpcore(&executable(k), &[]).unwrap();
    assert_eq!(out.len(), 1);
    let body = out[0].source.lines().last().unwrap();
    assert_eq!(
        body,
        "  (+ (+ sum_0 (* parameter_0 (* x_0 x_0))) (* parameter_0 (* x_1 x_1))))"
    );
    let out = export_fpcore(&executable(piecewise(1)), &[]).unwrap();
    let body = out[0].source.lines().last().unwrap();
    assert_eq!(
        body,
        "  (- (if (< x_0 y_0) (fma x_0 y_0 z_0) (fmax x_0 (fmin y_0 z_0))) (fabs x_0)))"
    );
    let mut k = fixtures::polynomial(1);
    k.numeric_policy.scalar_type = ScalarType::F32;
    k.body.statements = vec![
        Statement::Let {
            local: LocalId::new(0),
            value: ScalarExpr::binary(
                BinaryOp::Mul,
                ScalarExpr::Load(OperandId::new(0)),
                ScalarExpr::Constant(0.1),
            ),
        },
        Statement::Store {
            operand: OperandId::new(4),
            value: ScalarExpr::binary(
                BinaryOp::Add,
                ScalarExpr::Local(LocalId::new(0)),
                ScalarExpr::unary(UnaryOp::Sqrt, ScalarExpr::Local(LocalId::new(0))),
            ),
        },
    ];
    let out = export_fpcore(&executable(k), &[]).unwrap();
    assert!(out[0].source.contains(":precision binary32"));
    let body = out[0].source.lines().last().unwrap();
    assert_eq!(
        body,
        "  (let* ([_t2 (* t_0 0xcccccdp-27)]) (+ _t2 (sqrt _t2))))"
    );
}

#[test]
fn fpcore_is_structurally_valid_for_every_fixture_and_random_kernel() {
    let mut r = Rng(0x5EB_0005);
    let mut kernels = fixtures();
    kernels.extend((0..100).map(|_| Prog::generate(&mut r, 4).kernel()));
    let mut count = 0;
    for k in kernels {
        let boxes: Vec<InputBox> = (0..k.operands.len())
            .map(|i| InputBox {
                operand: OperandId::new(i),
                element: None,
                lo: -2.0,
                hi: 3.0,
            })
            .collect();
        for b in export_fpcore(&executable(k), &boxes).unwrap() {
            check_structure(&b.source);
            count += 1;
        }
    }
    assert!(count > 200);
    // Negative controls: the checker itself rejects unmapped operators and unbalanced lists.
    for bad in [
        "(FPCore (x) (frobnicate x))",
        "(FPCore (x) (+ x y))",
        "(FPCore (x) (+ x 1)",
    ] {
        assert!(
            std::panic::catch_unwind(|| check_structure(bad)).is_err(),
            "{bad}"
        );
    }
}
