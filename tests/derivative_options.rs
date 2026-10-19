//! W11 M3 packages 1 and 2: nonsmooth policy, Jacobian mode, IEEE-exact simplification.

mod support;

use malleus::ad::*;
use malleus::prepared::*;
use malleus::simplify::simplify_exact;
use malleus::*;
use support::kernels::*;
use support::{Rng, fixtures};

const MODES: [DerivativeMode; 2] = [DerivativeMode::Jvp, DerivativeMode::Vjp];

#[test]
fn default_options_reproduce_differentiate_and_serialize_empty() {
    for k in derivative_fixtures() {
        for mode in MODES {
            let request = full_request(&k, mode);
            let plain = differentiate(&k, &request).unwrap();
            let with = differentiate_with(&k, &request, &DerivativeOptions::default()).unwrap();
            assert_eq!(plain, with, "{} {mode:?}", k.name);
            assert_eq!(kernel_digest(&plain.kernel), kernel_digest(&with.kernel));
        }
    }
    assert_eq!(
        serde_json::to_string(&DerivativeOptions::default()).unwrap(),
        "{}"
    );
    let refuse = DerivativeOptions {
        nonsmooth: NonsmoothPolicy::Refuse,
        simplification: Simplification::Exact,
    };
    let text = serde_json::to_string(&refuse).unwrap();
    assert_eq!(text, r#"{"nonsmooth":"refuse","simplification":"exact"}"#);
    assert_eq!(
        serde_json::from_str::<DerivativeOptions>(&text).unwrap(),
        refuse
    );
    assert_eq!(
        serde_json::from_str::<DerivativeOptions>("{}").unwrap(),
        DerivativeOptions::default()
    );
}

#[test]
fn refuse_names_every_differentiated_kink_site() {
    let refuse = DerivativeOptions {
        nonsmooth: NonsmoothPolicy::Refuse,
        ..Default::default()
    };
    let k = nonsmooth(3);
    let request = full_request(&k, DerivativeMode::Jvp);
    let site = |statement, kind| KinkSite { statement, kind };
    let expected = vec![
        site(0, KinkKind::Max),
        site(0, KinkKind::Abs),
        site(0, KinkKind::Min),
        site(1, KinkKind::Floor),
    ];
    assert_eq!(kink_sites(&k, &request).unwrap(), expected);
    for mode in [
        DerivativeMode::Jvp,
        DerivativeMode::Vjp,
        DerivativeMode::Jacobian,
    ] {
        assert_eq!(
            differentiate_with(&k, &full_request(&k, mode), &refuse),
            Err(DifferentiationError::NonsmoothRefused(expected.clone())),
            "{mode:?}"
        );
    }

    // Only x1 differentiated: abs(x0) carries no tangent; max, min and floor(x1) do.
    let only_x1 = DerivativeRequest {
        mode: DerivativeMode::Jvp,
        independent_operands: vec![OperandId::new(1)],
        dependent_operands: vec![OperandId::new(2)],
    };
    assert_eq!(
        kink_sites(&k, &only_x1).unwrap(),
        vec![
            site(0, KinkKind::Max),
            site(0, KinkKind::Min),
            site(1, KinkKind::Floor)
        ]
    );

    // Select-only kernels are admitted and unchanged.
    let smooth = select_only(3);
    for mode in MODES {
        let request = full_request(&smooth, mode);
        assert_eq!(kink_sites(&smooth, &request).unwrap(), vec![]);
        assert_eq!(
            differentiate_with(&smooth, &request, &refuse).unwrap(),
            differentiate(&smooth, &request).unwrap()
        );
    }

    // Min reductions are kink sites; a store not reaching a dependent is not.
    let reduction = min_reduction(2);
    assert_eq!(
        kink_sites(&reduction, &full_request(&reduction, DerivativeMode::Vjp)).unwrap(),
        vec![site(0, KinkKind::MinReduction)]
    );
    let mut dead = nonsmooth(2);
    dead.operands
        .push(KernelOperand::tensor("other", vec![2], AccessMode::Write));
    dead.indexing_maps.push(IndexingMap::new(
        OperandId::new(3),
        vec![IndexExpr::axis(AxisId::new(0))],
    ));
    dead.body.statements[1] = Statement::Store {
        operand: OperandId::new(3),
        value: ScalarExpr::Constant(1.0),
    };
    dead.body.statements.push(Statement::Store {
        operand: OperandId::new(2),
        value: ScalarExpr::Load(OperandId::new(0)),
    });
    assert_eq!(
        kink_sites(&dead, &full_request(&dead, DerivativeMode::Jvp)).unwrap(),
        vec![]
    );
}

#[test]
fn jvp_reads_tangents_of_operands_declared_after_outputs() {
    // out = x * y with the output declared first: readable ids shift in the derivative.
    let axis = AxisId::new(0);
    let k = StructuredKernel {
        name: "output-first".into(),
        iteration_domain: IterationDomain::new(vec![1]),
        iterators: vec![IteratorKind::Parallel],
        operands: vec![
            KernelOperand::tensor("out", vec![1], AccessMode::Write),
            KernelOperand::tensor("x", vec![1], AccessMode::Read),
            KernelOperand::tensor("y", vec![1], AccessMode::Read),
        ],
        indexing_maps: (0..3)
            .map(|i| IndexingMap::new(OperandId::new(i), vec![IndexExpr::axis(axis)]))
            .collect(),
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(0),
                value: ScalarExpr::binary(
                    BinaryOp::Mul,
                    ScalarExpr::Load(OperandId::new(1)),
                    ScalarExpr::Load(OperandId::new(2)),
                ),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    };
    let product = differentiate(
        &k,
        &DerivativeRequest {
            mode: DerivativeMode::Jvp,
            independent_operands: vec![OperandId::new(2)],
            dependent_operands: vec![OperandId::new(0)],
        },
    )
    .unwrap();
    // Derivative operands: x, y, d_y, d_out. d out = x * dy = 3 * 1.
    let out = run(
        &product.kernel,
        vec![vec![3.0], vec![5.0], vec![1.0], vec![0.0]],
    );
    assert_eq!(out[3], vec![3.0]);
}

/// Bind identity seed `s` for every independent of a JVP and return its outputs.
fn jvp_at_seed(
    jvp: &DerivativeProduct,
    jacobian: &JacobianProduct,
    primal_buffers: &[Vec<f64>],
    s: usize,
) -> Vec<Vec<f64>> {
    let k = &jvp.kernel;
    let mut buffers: Vec<Vec<f64>> = k
        .operands
        .iter()
        .map(|o| vec![0.0; o.region.offset + o.region.length])
        .collect();
    for pair in &jvp.primal_operands {
        buffers[pair.derivative.index()] = primal_buffers[pair.primal.index()].clone();
    }
    for (pair, columns) in jvp.independent_operands.iter().zip(&jacobian.columns) {
        assert_eq!(pair.primal, columns.primal);
        if (columns.offset..columns.offset + columns.len).contains(&s) {
            // Row-major logical element s - offset, stored through the operand's layout.
            let operand = &k.operands[pair.derivative.index()];
            let mut rest = s - columns.offset;
            let mut index = vec![0; operand.shape.len()];
            for d in (0..operand.shape.len()).rev() {
                index[d] = rest % operand.shape[d];
                rest /= operand.shape[d];
            }
            let mut offset = 0;
            let mut stride = 1;
            for d in &operand.layout.minor_to_major {
                offset += index[*d] * stride;
                stride *= operand.shape[*d];
            }
            buffers[pair.derivative.index()][operand.region.offset + offset] = 1.0;
        }
    }
    run(k, buffers)
}

#[test]
fn jacobian_columns_equal_identity_seed_jvps_bitwise() {
    let mut r = Rng(0x5eed_1ac0);
    let mut kernels: Vec<_> = derivative_fixtures()
        .into_iter()
        .filter(|k| !k.name.starts_with("min-reduction"))
        .collect();
    kernels.extend((0..40).map(|i| random_kernel(&mut r, i % 4 == 0)));
    for k in kernels {
        let jvp = differentiate(&k, &full_request(&k, DerivativeMode::Jvp)).unwrap();
        let request = full_request(&k, DerivativeMode::Jacobian);
        let jacobian = differentiate_jacobian(&k, &request, &DerivativeOptions::default()).unwrap();
        assert_eq!(
            differentiate(&k, &request).unwrap(),
            jacobian.product,
            "differentiate(Jacobian) is the same product"
        );
        let primal = random_buffers(&mut r, &k);
        let jk = &jacobian.product.kernel;
        let mut buffers: Vec<Vec<f64>> = jk
            .operands
            .iter()
            .map(|o| vec![0.0; o.region.offset + o.region.length])
            .collect();
        for pair in &jacobian.product.primal_operands {
            buffers[pair.derivative.index()] = primal[pair.primal.index()].clone();
        }
        let out = run(jk, buffers.clone());
        for s in 0..jacobian.width {
            let seeded = jvp_at_seed(&jvp, &jacobian, &primal, s);
            for (jpair, tpair) in jacobian
                .product
                .dependent_operands
                .iter()
                .zip(&jvp.dependent_operands)
            {
                let tangent = &seeded[tpair.derivative.index()];
                let column: Vec<f64> = (0..tangent.len())
                    .map(|e| out[jpair.derivative.index()][e * jacobian.width + s])
                    .collect();
                // Row-major extension of a row-major output: element e of the tangent is row e.
                let shape = &k.operands[jpair.primal.index()].shape;
                assert_eq!(
                    k.operands[jpair.primal.index()].layout,
                    DenseLayout::row_major(shape.len())
                );
                support::kernels::assert_same(
                    &[column],
                    std::slice::from_ref(tangent),
                    &format!("{} seed {s}", k.name),
                );
            }
        }

        // Through prepared run_batch: 3 independent points with disjoint strides.
        let prepared =
            PreparedKernel::new(&Executable::reference(validate(jk.clone()).unwrap())).unwrap();
        let points = 3;
        let lens: Vec<usize> = jk
            .operands
            .iter()
            .map(|o| o.region.offset + o.region.length)
            .collect();
        let mut batch: Vec<Vec<f64>> = lens.iter().map(|l| vec![0.0; l * points]).collect();
        let mut per_point = Vec::new();
        for p in 0..points {
            let mut b = buffers.clone();
            for pair in &jacobian.product.primal_operands {
                b[pair.derivative.index()] =
                    random_buffers(&mut r, &k)[pair.primal.index()].clone();
            }
            for (i, values) in b.iter().enumerate() {
                batch[i][p * lens[i]..(p + 1) * lens[i]].copy_from_slice(values);
            }
            per_point.push(run(jk, b));
        }
        let (reads, writes): (Vec<_>, Vec<_>) = batch
            .iter_mut()
            .enumerate()
            .partition(|(i, _)| jk.operands[*i].access == AccessMode::Read);
        let reads: Vec<_> = reads
            .into_iter()
            .map(|(i, v)| ReadBinding::strided(OperandId::new(i), &*v, lens[i]))
            .collect();
        let mut writes: Vec<_> = writes
            .into_iter()
            .map(|(i, v)| WriteBinding::strided(OperandId::new(i), v, lens[i]))
            .collect();
        prepared
            .run_batch(&mut prepared.state(), points, &reads, &mut writes)
            .unwrap();
        drop((reads, writes));
        for (p, expected) in per_point.iter().enumerate() {
            let got: Vec<Vec<f64>> = batch
                .iter()
                .enumerate()
                .map(|(i, v)| v[p * lens[i]..(p + 1) * lens[i]].to_vec())
                .collect();
            support::kernels::assert_same(&got, expected, &format!("{} batch point {p}", k.name));
        }
    }
}

#[test]
fn jacobian_layout_and_refusals() {
    let k = fixtures::matvec(3);
    let request = full_request(&k, DerivativeMode::Jacobian);
    let j = differentiate_jacobian(&k, &request, &DerivativeOptions::default()).unwrap();
    assert_eq!(j.width, 12);
    assert_eq!(
        j.columns,
        vec![
            JacobianColumns {
                primal: OperandId::new(0),
                offset: 0,
                len: 9
            },
            JacobianColumns {
                primal: OperandId::new(1),
                offset: 9,
                len: 3
            },
        ]
    );
    assert_eq!(j.seed_axis, AxisId::new(2));
    let out = &j.product.kernel.operands[j.product.dependent_operands[0].derivative.index()];
    assert_eq!(out.shape, vec![3, 12]);
    assert_eq!(out.access, AccessMode::Reduce(ReductionOp::Add));
    // d out_i / d matrix_ij = vector_j, d out_i / d vector_j = matrix_ij.
    let primal = fixtures::buffers(&k);
    let jk = &j.product.kernel;
    let mut buffers: Vec<Vec<f64>> = jk
        .operands
        .iter()
        .map(|o| vec![0.0; o.region.length])
        .collect();
    for pair in &j.product.primal_operands {
        buffers[pair.derivative.index()] = primal[pair.primal.index()].clone();
    }
    let result = run(jk, buffers);
    let jac = &result[j.product.dependent_operands[0].derivative.index()];
    for i in 0..3 {
        for col in 0..3 {
            assert_eq!(jac[i * 12 + i * 3 + col], primal[1][col]);
            assert_eq!(jac[i * 12 + 9 + col], primal[0][i * 3 + col]);
        }
    }
    assert_eq!(
        differentiate_jacobian(
            &k,
            &full_request(&k, DerivativeMode::Jvp),
            &Default::default()
        ),
        Err(DifferentiationError::UnsupportedMode(DerivativeMode::Jvp))
    );
    let wide = fixtures::polynomial(MAX_JACOBIAN_WIDTH + 1);
    assert!(matches!(
        differentiate(&wide, &full_request(&wide, DerivativeMode::Jacobian)),
        Err(DifferentiationError::JacobianTooWide { .. })
    ));
}

fn exact() -> DerivativeOptions {
    DerivativeOptions {
        simplification: Simplification::Exact,
        ..Default::default()
    }
}

#[test]
fn exact_simplification_is_bitwise_on_fixtures_and_random_kernels() {
    let mut r = Rng(0xc0ff_ee11);
    let mut kernels = derivative_fixtures();
    kernels.extend((0..300).map(|i| random_kernel(&mut r, i % 3 == 0)));
    let (mut total, mut changed) = (0, 0);
    for k in kernels {
        // The primal itself.
        let simplified = simplify_exact(&k);
        validate(simplified.kernel.clone()).unwrap();
        for _ in 0..4 {
            let buffers = random_buffers(&mut r, &k);
            assert_same(
                &run(&simplified.kernel, buffers.clone()),
                &run(&k, buffers),
                &format!("{} primal", k.name),
            );
        }
        for mode in [
            DerivativeMode::Jvp,
            DerivativeMode::Vjp,
            DerivativeMode::Jacobian,
        ] {
            let request = full_request(&k, mode);
            let plain = differentiate(&k, &request).unwrap();
            let simple = differentiate_with(&k, &request, &exact()).unwrap();
            assert_eq!(plain.kernel.operands, simple.kernel.operands);
            assert_eq!(plain.kernel.indexing_maps, simple.kernel.indexing_maps);
            total += 1;
            if plain.kernel != simple.kernel {
                changed += 1;
            }
            for _ in 0..4 {
                let buffers = random_buffers(&mut r, &plain.kernel);
                assert_same(
                    &run(&simple.kernel, buffers.clone()),
                    &run(&plain.kernel, buffers),
                    &format!("{} {mode:?}", k.name),
                );
            }
        }
    }
    assert!(
        changed * 2 > total,
        "simplification applied to {changed} of {total}"
    );
}

#[test]
fn simplification_rewrites_only_exact_identities() {
    use support::kernels::select_less;
    let x = || ScalarExpr::Load(OperandId::new(0));
    let c = ScalarExpr::Constant;
    let bin = ScalarExpr::binary;
    let one = |value| {
        let mut k = fixtures::polynomial(1);
        k.body.statements = vec![Statement::Store {
            operand: OperandId::new(4),
            value,
        }];
        simplify_exact(&k)
    };
    for kept in [
        bin(BinaryOp::Add, x(), c(0.0)),
        bin(BinaryOp::Mul, x(), c(0.0)),
        bin(BinaryOp::Sub, x(), c(-0.0)),
        bin(BinaryOp::Div, x(), c(1.0)),
        select_less(x(), c(1.0), x(), x()),
    ] {
        assert_eq!(one(kept).rewrites, 0);
    }
    for rewritten in [
        bin(BinaryOp::Mul, x(), c(1.0)),
        bin(BinaryOp::Mul, c(1.0), x()),
        bin(BinaryOp::Sub, x(), c(0.0)),
        bin(BinaryOp::Add, x(), c(-0.0)),
        bin(BinaryOp::Add, c(-0.0), x()),
        ScalarExpr::unary(UnaryOp::Neg, ScalarExpr::unary(UnaryOp::Neg, x())),
    ] {
        let s = one(rewritten);
        assert_eq!(s.rewrites, 1);
        assert_eq!(
            s.kernel.body.statements,
            vec![Statement::Store {
                operand: OperandId::new(4),
                value: x()
            }]
        );
    }
    // Reduce(Add): a -0 store is dead, a +0 store is not (it turns a -0 accumulator into +0).
    let mut k = fixtures::matvec(2);
    k.body.statements.push(Statement::Store {
        operand: OperandId::new(2),
        value: c(-0.0),
    });
    k.body.statements.push(Statement::Store {
        operand: OperandId::new(2),
        value: c(0.0),
    });
    let s = simplify_exact(&k);
    assert_eq!(s.rewrites, 1);
    assert_eq!(s.kernel.body.statements.len(), 2);
    let buffers = vec![vec![1.0; 4], vec![1.0; 2], vec![-0.0; 2]];
    assert_same(
        &run(&s.kernel, buffers.clone()),
        &run(&k, buffers),
        "reduce stores",
    );
    // CSE shares a repeated product through one new local.
    let sq = bin(BinaryOp::Mul, x(), x());
    let s = one(bin(BinaryOp::Add, sq.clone(), sq));
    assert_eq!(s.rewrites, 1);
    assert!(matches!(s.kernel.body.statements[0], Statement::Let { .. }));
}
