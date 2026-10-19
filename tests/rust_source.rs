//! Compile emitted safe Rust with rustc, then compare all buffers to the interpreter.
use malleus::*;
use std::fmt::Write;

mod support;
use support::{fixtures, libm_extern_args, rustc_run, rustc_run_with};

#[test]
fn compiled_rust_matches_interpreter_primal_ad_policies_and_ieee_inputs() {
    let mut cases = vec![];
    for ty in [ScalarType::F32, ScalarType::F64] {
        let mut k = fixtures::polynomial_batch(8);
        k.numeric_policy.scalar_type = ty;
        cases.push((
            Executable::reference(validate(k.clone()).unwrap()),
            fixtures::buffers(&k),
        ));
        for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
            let p = differentiate(
                &k,
                &DerivativeRequest {
                    mode,
                    independent_operands: vec![OperandId::new(3)],
                    dependent_operands: vec![OperandId::new(4)],
                },
            )
            .unwrap();
            let b = fixtures::buffers(&p.kernel);
            cases.push((Executable::reference(validate(p.kernel).unwrap()), b));
        }
        for op in [
            ReductionOp::Add,
            ReductionOp::Multiply,
            ReductionOp::Min,
            ReductionOp::Max,
        ] {
            let mut k = fixtures::quadratic(8);
            k.numeric_policy.scalar_type = ty;
            k.operands[2].access = AccessMode::Reduce(op);
            let b = fixtures::buffers(&k);
            cases.push((Executable::reference(validate(k).unwrap()), b));
        }
        let mut k = fixtures::polynomial(8);
        k.numeric_policy.scalar_type = ty;
        let x = ScalarExpr::Load(OperandId::new(0));
        k.body.statements = vec![Statement::Store {
            operand: OperandId::new(4),
            value: ScalarExpr::Select {
                condition: Box::new(Predicate::Compare {
                    op: CompareOp::Less,
                    lhs: Box::new(x.clone()),
                    rhs: Box::new(ScalarExpr::Constant(0.)),
                }),
                if_true: Box::new(ScalarExpr::unary(UnaryOp::Neg, x.clone())),
                if_false: Box::new(ScalarExpr::unary(UnaryOp::Sqrt, x)),
            },
        }];
        let mut b = fixtures::buffers(&k);
        b[0] = vec![
            -0.,
            0.,
            -1.,
            4.,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            0.1,
        ];
        cases.push((Executable::reference(validate(k).unwrap()), b));
        let mut k = fixtures::matvec(3);
        k.numeric_policy.scalar_type = ty;
        let b = fixtures::buffers(&k);
        cases.push((Executable::reference(validate(k).unwrap()), b));
        let mut k = fixtures::polynomial(1);
        k.numeric_policy.scalar_type = ty;
        let eps = 2f64.powi(if ty == ScalarType::F32 { -13 } else { -27 });
        let b = vec![
            vec![1. + eps],
            vec![-1.],
            vec![1. - eps],
            vec![0.],
            vec![0.],
        ];
        cases.push((Executable::reference(validate(k).unwrap()), b));
    }
    let mut k = fixtures::polynomial(8);
    k.indexing_maps[0].results = vec![IndexExpr {
        constant: 7,
        terms: vec![IndexTerm {
            axis: AxisId::new(0),
            coefficient: -1,
        }],
    }];
    k.operands[4].region = BufferRegion::new(3, 8);
    let b = fixtures::buffers(&k);
    cases.push((Executable::reference(validate(k).unwrap()), b));
    let cases: Vec<_> = cases.into_iter().flat_map(|x| [x.clone(), x]).collect();
    let mut code = String::new();
    let mut main = String::from("fn main(){\n");
    let mut expected = vec![];
    for (case, (e, inputs)) in cases.iter().enumerate() {
        let name = format!("kernel_{case}");
        code.push_str(&if case % 2 == 0 {
            lower_rust(e, &name).unwrap()
        } else {
            lower_rust_unrolled(e, &name, 64).unwrap()
        });
        main.push_str("{\n");
        let mut reference = inputs.clone();
        let mut bindings: Vec<_> = reference
            .iter_mut()
            .enumerate()
            .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
            .collect();
        Interpreter::run(e, &mut bindings).unwrap();
        drop(bindings);
        expected.push(reference);
        for (i, b) in inputs.iter().enumerate() {
            writeln!(
                main,
                "let mut b{i}=vec![{}];",
                b.iter()
                    .map(|x| format!("f64::from_bits({}u64)", x.to_bits()))
                    .collect::<Vec<_>>()
                    .join(",")
            )
            .unwrap();
        }
        if case == 0 {
            main.push_str("let before=b4.clone();assert!(kernel_0(&[],&b1,&b2,&b3,&mut b4).is_err());assert_eq!(b4,before);\n");
        }
        write!(main, "{name}(").unwrap();
        for (i, o) in e.kernel().as_kernel().operands.iter().enumerate() {
            write!(
                main,
                "&{}b{i},",
                if o.access == AccessMode::Read {
                    ""
                } else {
                    "mut "
                }
            )
            .unwrap();
        }
        main.push_str(").unwrap();\n");
        writeln!(
            main,
            "println!(\"{{:?}}\",vec![{}]);",
            (0..inputs.len())
                .map(|i| format!("b{i}.iter().map(|x|x.to_bits()).collect::<Vec<_>>()"))
                .collect::<Vec<_>>()
                .join(",")
        )
        .unwrap();
        main.push_str("}\n");
    }
    main.push_str("}\n");
    code.push_str(&main);
    let stdout = rustc_run("rust-source", &code);
    assert_eq!(stdout.lines().count(), expected.len());
    for (line, reference) in stdout.lines().zip(expected) {
        let output: Vec<Vec<u64>> = serde_json::from_str(line).unwrap();
        for (actual, want) in output.iter().flatten().zip(reference.iter().flatten()) {
            assert!(
                *actual == want.to_bits() || (f64::from_bits(*actual).is_nan() && want.is_nan()),
                "{actual:x} != {want:?}"
            );
        }
    }
}

#[test]
fn source_refuses_injection_and_unsupported_schedule() {
    let v = validate(fixtures::polynomial(2)).unwrap();
    let e = Executable::reference(v.clone());
    for name in ["fn", "self", "async", "x);panic!()", "9x", "_", ""] {
        assert!(lower_rust(&e, name).is_err());
    }
    assert!(lower_rust(&e, &"x".repeat(129)).is_err());
    let mut s = KernelSchedule::canonical(&v);
    s.vectorization = Some(VectorizationPlan {
        axis: AxisId::new(0),
        width: 2,
    });
    assert!(lower_rust(&Executable::new(v, s).unwrap(), "kernel").is_err());
}

/// Emits every case as `kernel_<i>`, calls it on its buffers and prints all buffers' bits.
fn emitted_program(cases: &[(Executable, Vec<Vec<f64>>)]) -> String {
    let mut code = String::new();
    let mut main = String::from("fn main() {\n");
    for (case, (e, inputs)) in cases.iter().enumerate() {
        let name = format!("kernel_{case}");
        code.push_str(&lower_rust(e, &name).unwrap());
        main.push_str("{\n");
        for (i, b) in inputs.iter().enumerate() {
            let values: Vec<_> = b
                .iter()
                .map(|x| format!("f64::from_bits({}u64)", x.to_bits()))
                .collect();
            writeln!(main, "let mut b{i}=vec![{}];", values.join(",")).unwrap();
        }
        write!(main, "{name}(").unwrap();
        for (i, o) in e.kernel().as_kernel().operands.iter().enumerate() {
            let m = if o.access == AccessMode::Read {
                ""
            } else {
                "mut "
            };
            write!(main, "&{m}b{i},").unwrap();
        }
        main.push_str(").unwrap();\n");
        let printed: Vec<_> = (0..inputs.len())
            .map(|i| format!("b{i}.iter().map(|x|x.to_bits()).collect::<Vec<_>>()"))
            .collect();
        writeln!(
            main,
            "println!(\"{{:?}}\",vec![{}]);\n}}",
            printed.join(",")
        )
        .unwrap();
    }
    main.push_str("}\n");
    code + &main
}

fn assert_bitwise(stdout: &str, cases: &[(Executable, Vec<Vec<f64>>)]) {
    assert_eq!(stdout.lines().count(), cases.len());
    for (line, (e, inputs)) in stdout.lines().zip(cases) {
        let mut buffers = inputs.clone();
        let mut bindings: Vec<_> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
            .collect();
        Interpreter::run(e, &mut bindings).unwrap();
        drop(bindings);
        let output: Vec<Vec<u64>> = serde_json::from_str(line).unwrap();
        for (actual, want) in output.iter().flatten().zip(buffers.iter().flatten()) {
            assert!(
                *actual == want.to_bits() || (f64::from_bits(*actual).is_nan() && want.is_nan()),
                "{}: {actual:x} != {want:?}",
                e.kernel().as_kernel().name
            );
        }
    }
}

#[test]
fn compiled_portable_math_and_explicit_fma_match_interpreter() {
    let specials = [
        0.0,
        -0.0,
        0.5,
        -2.5,
        1.0,
        3.0,
        1e-310,
        700.0,
        -745.5,
        1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1e22,
        -0.75,
    ];
    let n = specials.len() * specials.len();
    let axis = AxisId::new(0);
    let point = |name: String, body: ScalarExpr, ty: ScalarType, level| {
        let policy = NumericPolicy {
            scalar_type: ty,
            level,
            ..NumericPolicy::default()
        };
        StructuredKernel {
            name,
            iteration_domain: IterationDomain::new(vec![n]),
            iterators: vec![IteratorKind::Parallel],
            operands: vec![
                KernelOperand::tensor("x", vec![n], AccessMode::Read),
                KernelOperand::tensor("y", vec![n], AccessMode::Read),
                KernelOperand::tensor("z", vec![n], AccessMode::Read),
                KernelOperand::tensor("out", vec![n], AccessMode::Write),
            ],
            indexing_maps: (0..4)
                .map(|i| IndexingMap::new(OperandId::new(i), vec![IndexExpr::axis(axis)]))
                .collect(),
            body: KernelRegion {
                statements: vec![Statement::Store {
                    operand: OperandId::new(3),
                    value: body,
                }],
            },
            numeric_policy: policy,
        }
    };
    let x: Vec<f64> = specials.iter().flat_map(|&a| specials.map(|_| a)).collect();
    let y: Vec<f64> = specials.iter().flat_map(|_| specials).collect();
    let z: Vec<f64> = (0..n).map(|i| specials[(i * 7) % specials.len()]).collect();
    let l = |i| ScalarExpr::Load(OperandId::new(i));
    let mut cases = vec![];
    use malleus::policy::NumericLevel::{StrictHost, StrictPortable};
    for ty in [ScalarType::F32, ScalarType::F64] {
        let mut bodies = vec![];
        for op in [
            UnaryOp::Exp,
            UnaryOp::Ln,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Tan,
        ] {
            bodies.push((format!("{op:?}"), ScalarExpr::unary(op, l(0))));
        }
        for op in [BinaryOp::Pow, BinaryOp::Atan2, BinaryOp::Min, BinaryOp::Max] {
            bodies.push((format!("{op:?}"), ScalarExpr::binary(op, l(0), l(1))));
        }
        bodies.push(("fma".into(), ScalarExpr::fma(l(0), l(1), l(2))));
        for (name, body) in bodies {
            for level in [StrictPortable, StrictHost] {
                let k = point(format!("{name}-{ty:?}-{level:?}"), body.clone(), ty, level);
                let buffers = vec![x.clone(), y.clone(), z.clone(), vec![0.0; n]];
                cases.push((Executable::reference(validate(k).unwrap()), buffers));
            }
        }
        // Portable min/max reductions.
        for op in [ReductionOp::Min, ReductionOp::Max] {
            let mut k = fixtures::quadratic(8);
            k.numeric_policy.scalar_type = ty;
            k.numeric_policy.level = StrictPortable;
            k.operands[2].access = AccessMode::Reduce(op);
            let b = fixtures::buffers(&k);
            cases.push((Executable::reference(validate(k).unwrap()), b));
        }
    }
    // The adversarial fma case: 1 - 2^-60 is not representable, so only a fused result is nonzero.
    let e = 2f64.powi(-30);
    let k = point(
        "fma-adversarial".into(),
        ScalarExpr::fma(l(0), l(1), l(2)),
        ScalarType::F64,
        StrictHost,
    );
    let mut adversarial = vec![
        vec![1.0 + e; n],
        vec![1.0 - e; n],
        vec![-1.0; n],
        vec![0.0; n],
    ];
    adversarial[0][1] = 2.0;
    cases.push((Executable::reference(validate(k).unwrap()), adversarial));
    let source = emitted_program(&cases);
    assert!(source.contains("::libm::expf(") && source.contains("::libm::atan2("));
    assert!(source.contains(".mul_add("));
    let stdout = rustc_run_with("rust-portable", &source, &libm_extern_args());
    assert_bitwise(&stdout, &cases);
    let last: Vec<Vec<u64>> = serde_json::from_str(stdout.lines().last().unwrap()).unwrap();
    assert_eq!(f64::from_bits(last[3][0]), -(e * e));
}
