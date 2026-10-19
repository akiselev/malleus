//! Dissimilar direct fixtures plus Malleus's existing derivative campaign.
// Executes JIT-generated machine code, which miri cannot interpret.
#![cfg(not(miri))]
use malleus::*;
use malleus_cranelift::{CampaignRunner, JitKernel, fixtures};

fn execute(e: &Executable, b: &mut [Vec<f64>], jit: Option<&JitKernel>) {
    let mut bindings: Vec<_> = b
        .iter_mut()
        .enumerate()
        .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
        .collect();
    match jit {
        Some(j) => j.run(&mut bindings).unwrap(),
        None => Interpreter::run(e, &mut bindings).unwrap(),
    }
}
fn compare(k: StructuredKernel, inputs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let e = Executable::reference(validate(k).unwrap());
    compare_executable(&e, inputs)
}
fn compare_executable(e: &Executable, mut inputs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let jit = JitKernel::compile(e).unwrap();
    let mut expected = inputs.clone();
    execute(e, &mut expected, None);
    execute(e, &mut inputs, Some(&jit));
    for (a, b) in expected.iter().flatten().zip(inputs.iter().flatten()) {
        assert!(
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            "reference {a:?} native {b:?}"
        );
    }
    inputs
}

#[test]
fn polynomials_match_bitwise_in_both_precisions_and_empty_domains() {
    for n in [0, 1, 8, 257, 4096] {
        for ty in [ScalarType::F64, ScalarType::F32] {
            let mut k = fixtures::polynomial(n.max(1));
            // The core validator rejects a mapped zero-length operand even when there
            // are no iterations. A zero-iteration domain over valid buffers is admitted.
            if n == 0 {
                k.iteration_domain.extents[0] = 0;
            }
            k.numeric_policy.scalar_type = ty;
            let b = fixtures::buffers(&k);
            compare(k, b);
        }
    }
}

#[test]
fn actual_campaign_checks_primal_jvp_vjp_parameter_and_adjoint() {
    let k = fixtures::quadratic(17);
    let name = k.name.clone();
    let module = StructuredModule {
        name: "jit-campaign".into(),
        kernels: vec![k.clone()],
    };
    let case = LocalDifferentialCase {
        kernel: name,
        primal_buffers: fixtures::buffers(&k),
        state_directions: vec![OperandValues {
            operand: OperandId::new(0),
            values: vec![0.25; 17],
        }],
        parameter_directions: vec![OperandValues {
            operand: OperandId::new(1),
            values: vec![0.5],
        }],
        dependent_seeds: vec![OperandValues {
            operand: OperandId::new(2),
            values: vec![0.75],
        }],
        centered_step: 1e-5,
        tolerance: ComparisonTolerance::new(1e-8, 1e-8),
    };
    let runner = CampaignRunner::default();
    let report = run_local_differential_campaign(&module, &[case], &runner).unwrap();
    assert!(report.passed(), "{report:#?}");
    assert_eq!(report.checks.len(), 7);
}

#[test]
fn offset_column_major_and_reversed_affine_indexes_match() {
    let a = AxisId::new(0);
    let b = AxisId::new(1);
    let k = StructuredKernel {
        name: "layout".into(),
        iteration_domain: IterationDomain::new(vec![2, 3]),
        iterators: vec![IteratorKind::Parallel; 2],
        operands: vec![
            KernelOperand::tensor("in", vec![2, 3], AccessMode::Read)
                .with_region(BufferRegion::new(2, 6)),
            KernelOperand::tensor("out", vec![2, 3], AccessMode::Write)
                .with_layout(DenseLayout::column_major(2))
                .with_region(BufferRegion::new(3, 6)),
        ],
        indexing_maps: vec![
            IndexingMap::new(
                OperandId::new(0),
                vec![
                    IndexExpr::axis(a),
                    IndexExpr {
                        constant: 2,
                        terms: vec![IndexTerm {
                            axis: b,
                            coefficient: -1,
                        }],
                    },
                ],
            ),
            IndexingMap::new(
                OperandId::new(1),
                vec![IndexExpr::axis(a), IndexExpr::axis(b)],
            ),
        ],
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(1),
                value: ScalarExpr::Load(OperandId::new(0)),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    };
    let result = compare(
        k,
        vec![vec![90., 91., 1., 2., 3., 4., 5., 6.], vec![-99.; 9]],
    );
    assert_eq!(result[1], [-99., -99., -99., 3., 6., 2., 5., 1., 4.]);
}

fn sum2d() -> StructuredKernel {
    let k = fixtures::quadratic(4);
    StructuredKernel {
        name: "ordered-sum".into(),
        iteration_domain: IterationDomain::new(vec![2, 2]),
        iterators: vec![IteratorKind::Reduction; 2],
        operands: vec![
            KernelOperand::tensor("x", vec![2, 2], AccessMode::Read),
            KernelOperand::scalar("sum", AccessMode::Reduce(ReductionOp::Add)),
        ],
        indexing_maps: vec![
            IndexingMap::new(
                OperandId::new(0),
                vec![
                    IndexExpr::axis(AxisId::new(0)),
                    IndexExpr::axis(AxisId::new(1)),
                ],
            ),
            IndexingMap::scalar(OperandId::new(1)),
        ],
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(1),
                value: ScalarExpr::Load(OperandId::new(0)),
            }],
        },
        ..k
    }
}

#[test]
fn canonical_and_schedule_defined_reductions_retain_their_order() {
    for canonical in [true, false] {
        let mut k = sum2d();
        if !canonical {
            k.numeric_policy.reduction_order = ReductionOrder::ScheduleDefined;
        }
        let validated = validate(k).unwrap();
        let mut schedule = KernelSchedule::canonical(&validated);
        schedule.loop_order.reverse();
        let e = Executable::new(validated, schedule).unwrap();
        let out = compare_executable(&e, vec![vec![1e16, 1., -1e16, 1.], vec![0.]]);
        assert_eq!(out[1][0], if canonical { 1.0 } else { 2.0 });
    }
}

#[test]
fn lazy_select_predicates_and_ieee_inputs_match() {
    for ty in [ScalarType::F64, ScalarType::F32] {
        let mut k = fixtures::polynomial(8);
        k.numeric_policy.scalar_type = ty;
        k.body.statements = vec![
            Statement::Let {
                local: LocalId::new(0),
                value: ScalarExpr::Index(AxisId::new(0)),
            },
            Statement::Store {
                operand: OperandId::new(4),
                value: ScalarExpr::Select {
                    condition: Box::new(Predicate::And(
                        Box::new(Predicate::Not(Box::new(Predicate::Constant(false)))),
                        Box::new(Predicate::Or(
                            Box::new(Predicate::Compare {
                                op: CompareOp::Less,
                                lhs: Box::new(ScalarExpr::Load(OperandId::new(0))),
                                rhs: Box::new(ScalarExpr::Constant(0.0)),
                            }),
                            Box::new(Predicate::Constant(false)),
                        )),
                    )),
                    if_true: Box::new(ScalarExpr::unary(
                        UnaryOp::Abs,
                        ScalarExpr::Load(OperandId::new(0)),
                    )),
                    if_false: Box::new(ScalarExpr::unary(
                        UnaryOp::Sqrt,
                        ScalarExpr::Load(OperandId::new(0)),
                    )),
                },
            },
        ];
        let mut b = fixtures::buffers(&k);
        b[0] = vec![
            -1.,
            -0.,
            0.,
            4.,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            0.1,
        ];
        compare(k, b);
    }
}

#[test]
fn readwrite_and_reusable_calls_preserve_effects() {
    let mut k = fixtures::polynomial(3);
    k.operands[4].access = AccessMode::ReadWrite;
    k.body.statements = vec![Statement::Store {
        operand: OperandId::new(4),
        value: ScalarExpr::binary(
            BinaryOp::Add,
            ScalarExpr::Load(OperandId::new(4)),
            ScalarExpr::Load(OperandId::new(0)),
        ),
    }];
    let e = Executable::reference(validate(k).unwrap());
    let j = JitKernel::compile(&e).unwrap();
    let mut b = [vec![1., 2., 3.], vec![0.], vec![0.], vec![0.], vec![0.; 3]];
    let mut bindings: Vec<_> = b
        .iter_mut()
        .enumerate()
        .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
        .collect();
    {
        let mut call = j.prepare(&mut bindings).unwrap();
        for _ in 0..3 {
            call.run();
        }
    }
    drop(bindings);
    assert_eq!(b[4], [3., 6., 9.]);
}

#[test]
fn binding_errors_are_checked_before_jit_entry() {
    let e = Executable::reference(validate(fixtures::polynomial(3)).unwrap());
    let j = JitKernel::compile(&e).unwrap();
    let mut short = [0.];
    assert!(
        j.run(&mut [BufferBinding::new(OperandId::new(0), &mut short)])
            .is_err()
    );
    let mut a = [0.; 3];
    let mut b = [0.; 3];
    assert!(
        j.run(&mut [
            BufferBinding::new(OperandId::new(0), &mut a),
            BufferBinding::new(OperandId::new(0), &mut b)
        ])
        .is_err()
    );
    assert!(
        j.run(&mut [BufferBinding::new(OperandId::new(999), &mut a)])
            .is_err()
    );
    assert!(j.run(&mut []).is_err());
}

#[test]
fn unsupported_schedules_refuse_explicitly() {
    let mut k = fixtures::polynomial(1);
    k.body.statements = vec![Statement::Store {
        operand: OperandId::new(4),
        value: ScalarExpr::unary(UnaryOp::Exp, ScalarExpr::Load(OperandId::new(0))),
    }];
    // Transcendentals are admitted (called out to the reference implementation).
    let e = Executable::reference(validate(k).unwrap());
    assert!(JitKernel::compile(&e).is_ok());
    let v = validate(fixtures::polynomial(3)).unwrap();
    let mut s = KernelSchedule::canonical(&v);
    s.vectorization = Some(VectorizationPlan {
        axis: AxisId::new(0),
        width: 2,
    });
    assert!(JitKernel::compile(&Executable::new(v, s).unwrap()).is_err());
}

#[test]
fn scalar_point_kernel_executes_once_and_f32_rounds_each_operation() {
    let mut k = fixtures::polynomial(1);
    k.iteration_domain = IterationDomain::default();
    k.iterators.clear();
    for m in &mut k.indexing_maps {
        m.results = if m.operand.index() == 0 || m.operand.index() == 4 {
            vec![IndexExpr::constant(0)]
        } else {
            vec![]
        };
    }
    k.numeric_policy.scalar_type = ScalarType::F32;
    let b = vec![vec![16777217.0], vec![1.0], vec![1.0], vec![0.0], vec![0.0]];
    let out = compare(k, b);
    assert_eq!(out[4], [16777216.0]);
}

#[test]
fn forbidden_fma_is_not_introduced_by_native_optimization() {
    for ty in [ScalarType::F32, ScalarType::F64] {
        let mut k = fixtures::polynomial(1);
        k.numeric_policy.scalar_type = ty;
        let epsilon = if ty == ScalarType::F32 {
            2.0_f64.powi(-13)
        } else {
            2.0_f64.powi(-27)
        };
        let b = vec![
            vec![1.0 + epsilon],
            vec![-1.0],
            vec![1.0 - epsilon],
            vec![0.0],
            vec![0.0],
        ];
        let out = compare(k, b);
        assert_eq!(out[4], [0.0]);
    }
}

#[test]
fn generated_f32_jvp_and_vjp_match_the_interpreter() {
    let mut primal = fixtures::quadratic(17);
    primal.numeric_policy.scalar_type = ScalarType::F32;
    for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
        let p = differentiate(
            &primal,
            &DerivativeRequest {
                mode,
                independent_operands: vec![OperandId::new(0), OperandId::new(1)],
                dependent_operands: vec![OperandId::new(2)],
            },
        )
        .unwrap();
        let b = fixtures::buffers(&p.kernel);
        compare(p.kernel, b);
    }
}

#[test]
fn dense_matvec_matches_in_both_precisions() {
    for n in [1, 3, 17] {
        for ty in [ScalarType::F32, ScalarType::F64] {
            let mut k = fixtures::matvec(n);
            k.numeric_policy.scalar_type = ty;
            let b = fixtures::buffers(&k);
            compare(k, b);
        }
    }
}

#[test]
fn admitted_arithmetic_comparisons_and_multiply_reduction_match() {
    for ty in [ScalarType::F32, ScalarType::F64] {
        let mut template = fixtures::polynomial(12);
        template.numeric_policy.scalar_type = ty;
        let mut b = fixtures::buffers(&template);
        b[0] = vec![
            -0.,
            0.,
            -1.,
            1.,
            -1.5,
            1.5,
            1e-300,
            1e300,
            f64::MIN_POSITIVE,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let mut expressions = vec![];
        for op in [
            UnaryOp::Neg,
            UnaryOp::Abs,
            UnaryOp::Sqrt,
            UnaryOp::Floor,
            UnaryOp::Ceil,
        ] {
            expressions.push(ScalarExpr::unary(op, ScalarExpr::Load(OperandId::new(0))));
        }
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
            expressions.push(ScalarExpr::binary(
                op,
                ScalarExpr::Load(OperandId::new(0)),
                ScalarExpr::Constant(-0.5),
            ));
        }
        for op in [
            CompareOp::Eq,
            CompareOp::NotEq,
            CompareOp::Less,
            CompareOp::LessEqual,
            CompareOp::Greater,
            CompareOp::GreaterEqual,
        ] {
            expressions.push(ScalarExpr::Select {
                condition: Box::new(Predicate::Compare {
                    op,
                    lhs: Box::new(ScalarExpr::Load(OperandId::new(0))),
                    rhs: Box::new(ScalarExpr::Constant(0.)),
                }),
                if_true: Box::new(ScalarExpr::Constant(1.)),
                if_false: Box::new(ScalarExpr::Constant(-1.)),
            });
        }
        for value in expressions {
            let mut k = template.clone();
            k.body.statements = vec![Statement::Store {
                operand: OperandId::new(4),
                value,
            }];
            compare(k, b.clone());
        }
        let mut k = fixtures::quadratic(4);
        k.numeric_policy.scalar_type = ty;
        k.operands[2].access = AccessMode::Reduce(ReductionOp::Multiply);
        let b = vec![vec![0.5, 1.5, 2., 0.25], vec![1.25], vec![2.]];
        compare(k, b);
    }
}
