//! Numeric policy levels, explicit `Fma`, FMA contraction and portable transcendentals.

mod support;

use malleus::policy::{
    self, ContractionError, MathLibrary, NumericLevel, contract_fma, unary_uses_math,
};
use malleus::prepared::*;
use malleus::*;
use support::{Rng, fixtures, same_bits};

fn load(i: usize) -> ScalarExpr {
    ScalarExpr::Load(OperandId::new(i))
}

/// `out[i] = body(in0[i], .., in{inputs-1}[i])` over `n` points.
fn pointwise(n: usize, inputs: usize, body: ScalarExpr, ty: ScalarType) -> StructuredKernel {
    let axis = AxisId::new(0);
    let mut operands: Vec<_> = (0..inputs)
        .map(|i| KernelOperand::tensor(format!("in{i}"), vec![n], AccessMode::Read))
        .collect();
    operands.push(KernelOperand::tensor("out", vec![n], AccessMode::Write));
    let policy = NumericPolicy {
        scalar_type: ty,
        ..NumericPolicy::default()
    };
    StructuredKernel {
        name: "pointwise".into(),
        iteration_domain: IterationDomain::new(vec![n]),
        iterators: vec![IteratorKind::Parallel],
        indexing_maps: (0..=inputs)
            .map(|i| IndexingMap::new(OperandId::new(i), vec![IndexExpr::axis(axis)]))
            .collect(),
        operands,
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(inputs),
                value: body,
            }],
        },
        numeric_policy: policy,
    }
}

fn interpret(e: &Executable, mut buffers: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut bindings: Vec<_> = buffers
        .iter_mut()
        .enumerate()
        .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
        .collect();
    Interpreter::run(e, &mut bindings).unwrap();
    drop(bindings);
    buffers
}

/// Runs `e` through the interpreter, prepared `run`, and a columnar `run_batch` of the
/// rank-1 kernel split into one-point calls; asserts all three agree bitwise and returns them.
fn all_core_paths(e: &Executable, buffers: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let reference = interpret(e, buffers.clone());
    let prepared = PreparedKernel::new(e).unwrap();
    let mut one_shot = buffers.clone();
    let mut bindings: Vec<_> = one_shot
        .iter_mut()
        .enumerate()
        .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
        .collect();
    prepared.run(&mut bindings).unwrap();
    drop(bindings);
    assert!(same(&reference, &one_shot), "prepared run differs");

    // The same data as `points` scalar calls of the point kernel, batched with stride 1.
    let k = e.kernel().as_kernel();
    let points = k.iteration_domain.extents[0];
    let mut point_kernel = k.clone();
    point_kernel.iteration_domain = IterationDomain::new(vec![]);
    point_kernel.iterators.clear();
    for (operand, map) in point_kernel
        .operands
        .iter_mut()
        .zip(&mut point_kernel.indexing_maps)
    {
        *operand = KernelOperand::scalar(operand.name.clone(), operand.access);
        map.results.clear();
    }
    let point =
        PreparedKernel::new(&Executable::reference(validate(point_kernel).unwrap())).unwrap();
    let mut batched = buffers;
    let mut state = point.state();
    let (mut reads, mut writes) = (vec![], vec![]);
    for (i, values) in batched.iter_mut().enumerate() {
        // Scalar operands are shared by every point.
        let stride = usize::from(!k.operands[i].shape.is_empty());
        if k.operands[i].access == AccessMode::Read {
            reads.push(ReadBinding::strided(OperandId::new(i), values, stride));
        } else {
            writes.push(WriteBinding::strided(OperandId::new(i), values, stride));
        }
    }
    point
        .run_batch(&mut state, points, &reads, &mut writes)
        .unwrap();
    drop((reads, writes));
    assert!(same(&reference, &batched), "columnar batch differs");
    reference
}

fn same(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same_bits(*x, *y)))
}

fn executable(k: StructuredKernel) -> Executable {
    Executable::reference(validate(k).unwrap())
}

#[test]
fn default_level_is_not_serialized_and_old_policies_decode_as_strict_host() {
    let kernel = fixtures::polynomial(8);
    let json = serde_json::to_string(&kernel).unwrap();
    assert!(
        !json.contains("level"),
        "default level must stay out of the bytes"
    );
    let old = serde_json::to_string(&kernel.numeric_policy).unwrap();
    assert_eq!(
        old,
        r#"{"scalar_type":"f64","fma":"forbidden","reassociation":"forbidden","reduction_order":"canonical"}"#
    );
    let decoded: NumericPolicy = serde_json::from_str(&old).unwrap();
    assert_eq!(decoded.level, NumericLevel::StrictHost);

    let mut portable = kernel.clone();
    portable.numeric_policy.level = NumericLevel::StrictPortable;
    let json = serde_json::to_string(&portable.numeric_policy).unwrap();
    assert!(json.ends_with(r#","level":"strict_portable"}"#), "{json}");
    assert_eq!(
        serde_json::from_str::<NumericPolicy>(&json).unwrap(),
        portable.numeric_policy
    );
    // The level is identity: L0, L1 and L2 kernels have three distinct digests.
    let mut contracted = kernel.clone();
    contracted.numeric_policy.level = NumericLevel::Contracted;
    let digests = [&kernel, &portable, &contracted].map(|k| kernel_digest(k).hex);
    assert_ne!(digests[0], digests[1]);
    assert_ne!(digests[0], digests[2]);
    assert_ne!(digests[1], digests[2]);
    // A new `Fma` node encodes as its own variant without touching existing encodings.
    let fma = ScalarExpr::fma(load(0), load(1), ScalarExpr::Constant(2.0));
    assert_eq!(
        serde_json::to_string(&fma).unwrap(),
        r#"{"fma":{"lhs":{"load":0},"rhs":{"load":1},"addend":{"constant":2.0}}}"#
    );
}

#[test]
fn validation_refuses_imprecise_permissions_outside_strict_host() {
    for level in [NumericLevel::StrictPortable, NumericLevel::Contracted] {
        for (fma, reassociation) in [
            (FmaPolicy::Allowed, Reassociation::Forbidden),
            (FmaPolicy::Forbidden, Reassociation::Allowed),
        ] {
            let mut k = fixtures::polynomial(4);
            k.numeric_policy.level = level;
            k.numeric_policy.fma = fma;
            k.numeric_policy.reassociation = reassociation;
            assert_eq!(
                validate(k).unwrap_err(),
                ValidationError::ImpreciseNumericPolicy
            );
        }
    }
    // L1 keeps accepting them (existing identities), and executes strictly.
    let mut k = fixtures::polynomial(4);
    k.numeric_policy.fma = FmaPolicy::Allowed;
    k.numeric_policy.reassociation = Reassociation::Allowed;
    let strict = fixtures::polynomial(4);
    let b = fixtures::buffers(&k);
    assert!(same(
        &interpret(&executable(k), b.clone()),
        &interpret(&executable(strict), b)
    ));
}

#[test]
fn explicit_fma_rounds_once_in_every_core_path() {
    // a * b = 1 - 2^-2p rounds to 1, so the unfused result is 0 and the fused one is -2^-2p.
    for (ty, eps) in [
        (ScalarType::F64, 2f64.powi(-30)),
        (ScalarType::F32, 2f64.powi(-14)),
    ] {
        let fused = pointwise(3, 3, ScalarExpr::fma(load(0), load(1), load(2)), ty);
        let unfused = pointwise(
            3,
            3,
            ScalarExpr::binary(
                BinaryOp::Add,
                ScalarExpr::binary(BinaryOp::Mul, load(0), load(1)),
                load(2),
            ),
            ty,
        );
        let inputs = vec![
            vec![1.0 + eps, -0.0, f64::INFINITY],
            vec![1.0 - eps, 0.0, 0.0],
            vec![-1.0, 0.0, 1.0],
            vec![0.0; 3],
        ];
        let f = all_core_paths(&executable(fused), inputs.clone());
        let u = all_core_paths(&executable(unfused), inputs);
        assert_eq!(f[3][0], -(eps * eps));
        assert_eq!(u[3][0], 0.0);
        // -0 * 0 + 0 = +0 under both; inf * 0 is NaN under both.
        assert!(same_bits(f[3][1], 0.0) && same_bits(u[3][1], 0.0));
        assert!(f[3][2].is_nan() && u[3][2].is_nan());
    }
}

#[test]
fn contraction_rewrites_fixed_sites_and_changes_identity() {
    // c0 + t * (c1 + c2 * t)  ->  fma(t, fma(c2, t, c1), c0)
    let kernel = fixtures::polynomial(8);
    let contraction = contract_fma(&kernel).unwrap();
    assert_eq!(contraction.sites, 2);
    let k = &contraction.kernel;
    assert_eq!(k.numeric_policy.level, NumericLevel::Contracted);
    let Statement::Store { value, .. } = &k.body.statements[0] else {
        panic!("store")
    };
    assert_eq!(
        *value,
        ScalarExpr::fma(load(0), ScalarExpr::fma(load(3), load(0), load(2)), load(1))
    );
    assert_ne!(kernel_digest(k), kernel_digest(&kernel));
    let b = fixtures::buffers(k);
    let out = all_core_paths(&executable(k.clone()), b.clone());
    for (i, &t) in b[0].iter().enumerate() {
        assert_eq!(out[4][i], t.mul_add(b[3][0].mul_add(t, b[2][0]), b[1][0]));
    }

    // Subtraction sites negate exactly: a*b - c -> fma(a, b, -c); c - a*b -> fma(-a, b, c).
    let mul = || ScalarExpr::binary(BinaryOp::Mul, load(0), load(1));
    let neg = |e| ScalarExpr::unary(UnaryOp::Neg, e);
    for (body, want) in [
        (
            ScalarExpr::binary(BinaryOp::Sub, mul(), load(2)),
            ScalarExpr::fma(load(0), load(1), neg(load(2))),
        ),
        (
            ScalarExpr::binary(BinaryOp::Sub, load(2), mul()),
            ScalarExpr::fma(neg(load(0)), load(1), load(2)),
        ),
        (
            ScalarExpr::binary(BinaryOp::Add, load(2), mul()),
            ScalarExpr::fma(load(0), load(1), load(2)),
        ),
    ] {
        let c = contract_fma(&pointwise(2, 3, body, ScalarType::F64)).unwrap();
        assert_eq!(c.sites, 1);
        let Statement::Store { value, .. } = &c.kernel.body.statements[0] else {
            panic!("store")
        };
        assert_eq!(*value, want);
    }
    // Products behind a local and non-product sums stay unfused.
    let mut via_local = pointwise(
        2,
        3,
        ScalarExpr::binary(BinaryOp::Add, ScalarExpr::Local(LocalId::new(0)), load(2)),
        ScalarType::F64,
    );
    via_local.body.statements.insert(
        0,
        Statement::Let {
            local: LocalId::new(0),
            value: mul(),
        },
    );
    assert_eq!(contract_fma(&via_local).unwrap().sites, 0);

    let mut portable = kernel.clone();
    portable.numeric_policy.level = NumericLevel::StrictPortable;
    assert_eq!(
        contract_fma(&portable).unwrap_err(),
        ContractionError::Level(NumericLevel::StrictPortable)
    );
    assert_eq!(
        contract_fma(&contraction.kernel).unwrap_err(),
        ContractionError::Level(NumericLevel::Contracted)
    );
    let mut vague = kernel.clone();
    vague.numeric_policy.fma = FmaPolicy::Allowed;
    assert_eq!(
        contract_fma(&vague).unwrap_err(),
        ContractionError::ImprecisePolicy
    );
    let mut invalid = kernel;
    invalid.name.clear();
    assert!(matches!(
        contract_fma(&invalid),
        Err(ContractionError::Invalid(_))
    ));
}

#[test]
fn derivatives_of_fma_match_the_unfused_derivative() {
    let unfused = pointwise(
        16,
        3,
        ScalarExpr::binary(
            BinaryOp::Add,
            ScalarExpr::binary(BinaryOp::Mul, load(0), load(1)),
            load(2),
        ),
        ScalarType::F64,
    );
    let fused = contract_fma(&unfused).unwrap().kernel;
    let mut r = Rng(0x5eed_f00d);
    for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
        let request = DerivativeRequest {
            mode,
            independent_operands: vec![OperandId::new(0), OperandId::new(1), OperandId::new(2)],
            dependent_operands: vec![OperandId::new(3)],
        };
        let du = differentiate(&unfused, &request).unwrap();
        let df = differentiate(&fused, &request).unwrap();
        assert_eq!(df.kernel.numeric_policy.level, NumericLevel::Contracted);
        let buffers: Vec<Vec<f64>> = du
            .kernel
            .operands
            .iter()
            .map(|o| {
                (0..o.region.offset + o.region.length)
                    .map(|_| r.finite().clamp(-1e100, 1e100))
                    .collect()
            })
            .collect();
        let ou = interpret(&executable(du.kernel.clone()), buffers.clone());
        let of = all_core_paths(&executable(df.kernel.clone()), buffers);
        // Derivative outputs are the strict expressions of the unfused derivative.
        let derivative_outputs: Vec<usize> = match mode {
            DerivativeMode::Jvp => df.dependent_operands.iter(),
            _ => df.independent_operands.iter(),
        }
        .map(|d| d.derivative.index())
        .collect();
        assert!(!derivative_outputs.is_empty());
        for i in derivative_outputs {
            assert!(
                same(&ou[i..=i], &of[i..=i]),
                "{mode:?} derivative operand {i}"
            );
        }
    }
}

fn ulps(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let ordered = |x: f64| {
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN.wrapping_sub(bits)
        } else {
            bits
        }
    };
    ordered(a).abs_diff(ordered(b))
}

fn samples(r: &mut Rng, op: &str, n: usize) -> Vec<(f64, f64)> {
    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1e-310,
    ];
    let mut out: Vec<(f64, f64)> = specials
        .iter()
        .flat_map(|&x| specials.iter().map(move |&y| (x, y)))
        .collect();
    while out.len() < n {
        let u = (r.next() >> 11) as f64 / (1u64 << 53) as f64;
        let v = (r.next() >> 11) as f64 / (1u64 << 53) as f64;
        out.push(match op {
            "exp" => (u * 1420.0 - 710.0, 0.0),
            "ln" => (10f64.powf(u * 600.0 - 300.0), 0.0),
            "sin" | "cos" | "tan" if r.below(4) == 0 => (r.finite(), 0.0),
            "sin" | "cos" | "tan" => (u * 200.0 - 100.0, 0.0),
            "pow" => (u * 100.0, v * 100.0 - 50.0),
            _ => (r.finite(), r.finite()),
        });
    }
    out
}

#[test]
fn portable_math_is_bitwise_libm_in_every_core_path_and_close_to_host_math() {
    let mut r = Rng(0x1234_5678_9abc);
    let unary = [
        ("exp", UnaryOp::Exp),
        ("ln", UnaryOp::Ln),
        ("sin", UnaryOp::Sin),
        ("cos", UnaryOp::Cos),
        ("tan", UnaryOp::Tan),
    ];
    let binary = [
        ("pow", BinaryOp::Pow),
        ("atan2", BinaryOp::Atan2),
        ("min", BinaryOp::Min),
        ("max", BinaryOp::Max),
    ];
    let mut report = vec![];
    for ty in [ScalarType::F64, ScalarType::F32] {
        let cases = unary
            .iter()
            .map(|&(name, op)| (name, ScalarExpr::unary(op, load(0))))
            .chain(
                binary
                    .iter()
                    .map(|&(name, op)| (name, ScalarExpr::binary(op, load(0), load(1)))),
            );
        for (name, body) in cases {
            let s = samples(&mut r, name, 4000);
            let n = s.len();
            let mut k = pointwise(n, 2, body, ty);
            let buffers = vec![
                s.iter().map(|p| p.0).collect::<Vec<_>>(),
                s.iter().map(|p| p.1).collect(),
                vec![0.0; n],
            ];
            k.numeric_policy.level = NumericLevel::StrictPortable;
            let portable = all_core_paths(&executable(k.clone()), buffers.clone());
            k.numeric_policy.level = NumericLevel::StrictHost;
            let host = all_core_paths(&executable(k), buffers);
            let mut worst = 0;
            for (i, &(x, y)) in s.iter().enumerate() {
                let want = match (ty, name) {
                    (ScalarType::F64, "exp") => libm::exp(x),
                    (ScalarType::F64, "ln") => libm::log(x),
                    (ScalarType::F64, "sin") => libm::sin(x),
                    (ScalarType::F64, "cos") => libm::cos(x),
                    (ScalarType::F64, "tan") => libm::tan(x),
                    (ScalarType::F64, "pow") => libm::pow(x, y),
                    (ScalarType::F64, "atan2") => libm::atan2(x, y),
                    (ScalarType::F64, "min") => libm::fmin(x, y),
                    (ScalarType::F64, _) => libm::fmax(x, y),
                    (_, "exp") => libm::expf(x as f32) as f64,
                    (_, "ln") => libm::logf(x as f32) as f64,
                    (_, "sin") => libm::sinf(x as f32) as f64,
                    (_, "cos") => libm::cosf(x as f32) as f64,
                    (_, "tan") => libm::tanf(x as f32) as f64,
                    (_, "pow") => libm::powf(x as f32, y as f32) as f64,
                    (_, "atan2") => libm::atan2f(x as f32, y as f32) as f64,
                    (_, "min") => libm::fminf(x as f32, y as f32) as f64,
                    (_, _) => libm::fmaxf(x as f32, y as f32) as f64,
                };
                assert!(
                    same_bits(portable[2][i], want),
                    "{ty:?} {name}({x:e}, {y:e})"
                );
                // Distance in units of the kernel precision.
                let d = match ty {
                    ScalarType::F64 => ulps(portable[2][i], host[2][i]),
                    ScalarType::F32 => {
                        let (p, h) = (portable[2][i] as f32, host[2][i] as f32);
                        if p.is_nan() || h.is_nan() {
                            if p.is_nan() && h.is_nan() {
                                0
                            } else {
                                u64::MAX
                            }
                        } else {
                            let o = |x: f32| {
                                let b = x.to_bits() as i32;
                                if b < 0 { i32::MIN.wrapping_sub(b) } else { b }
                            };
                            u64::from(o(p).abs_diff(o(h)))
                        }
                    }
                };
                worst = worst.max(d);
            }
            report.push(format!("{ty:?} {name}: max {worst} ulp over {n} samples"));
            // Stated bound: both libraries are faithful (< 1 ulp each), so they differ by at
            // most 2 ulp on finite results and agree on every special value.
            assert!(worst <= 2, "{ty:?} {name}: L0 vs L1 differ by {worst} ulp");
        }
    }
    println!("{}", report.join("\n"));
    assert!(unary.iter().all(|(_, op)| unary_uses_math(*op)));
    assert_eq!(
        policy::unary(UnaryOp::Sqrt, 2.0, ScalarType::F64, MathLibrary::Portable),
        2f64.sqrt()
    );
}
