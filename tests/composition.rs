//! Kernel-level `Composed` bind chains: a producer kernel's output buffer feeds a consumer
//! kernel's input operand at execution time, with derivatives composed from per-stage products.

use malleus::{
    AccessMode, AxisId, BinaryOp, BufferBinding, CompositionBinding, CompositionDerivativeRequest,
    CompositionDifferentiationError, CompositionError, CompositionTarget, DerivativeMode,
    DerivativeRequest, Executable, ExecutableComposition, IndexExpr, IndexingMap, Interpreter,
    IterationDomain, IteratorKind, KernelComposition, KernelOperand, KernelRegion, NumericPolicy,
    OperandId, ReductionOp, ScalarExpr, SharedBuffer, StageOperand, Statement, StructuredKernel,
    composition_digest, differentiate, differentiate_composition, kernel_digest, validate,
    validate_composition,
};

const DIM: usize = 2;
const TESTS: usize = 3;

fn op(index: usize) -> OperandId {
    OperandId::new(index)
}

fn load(index: usize) -> ScalarExpr {
    ScalarExpr::Load(op(index))
}

fn mul(lhs: ScalarExpr, rhs: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(BinaryOp::Mul, lhs, rhs)
}

fn add(lhs: ScalarExpr, rhs: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(BinaryOp::Add, lhs, rhs)
}

/// `sigma * dot(g, g)` reduced over the gradient axis: the shape of a `joule_heat` output
/// kernel of a producer model. Operands: 0 `sigma` (scalar), 1 `g` [DIM], 2 `q` (reduce add).
fn producer_kernel() -> StructuredKernel {
    let axis = AxisId::new(0);
    StructuredKernel {
        name: "producer::joule".into(),
        iteration_domain: IterationDomain::new(vec![DIM]),
        iterators: vec![IteratorKind::Reduction],
        operands: vec![
            KernelOperand::scalar("sigma", AccessMode::Read),
            KernelOperand::tensor("g", vec![DIM], AccessMode::Read),
            KernelOperand::scalar("q", AccessMode::Reduce(ReductionOp::Add)),
        ],
        indexing_maps: vec![
            IndexingMap::scalar(op(0)),
            IndexingMap::new(op(1), vec![IndexExpr::axis(axis)]),
            IndexingMap::scalar(op(2)),
        ],
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: op(2),
                value: mul(load(0), mul(load(1), load(1))),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

/// `r[i] = (rho * u + q) * v[i]` over test index `i`: a consumer residual kernel whose bound
/// input `q` is an opaque external scalar. Operands: 0 `rho`, 1 `u`, 2 `q`, 3 `v` [TESTS],
/// 4 `r` [TESTS] (write).
fn consumer_kernel() -> StructuredKernel {
    let axis = AxisId::new(0);
    StructuredKernel {
        name: "consumer::energy".into(),
        iteration_domain: IterationDomain::new(vec![TESTS]),
        iterators: vec![IteratorKind::Parallel],
        operands: vec![
            KernelOperand::scalar("rho", AccessMode::Read),
            KernelOperand::scalar("u", AccessMode::Read),
            KernelOperand::scalar("q", AccessMode::Read),
            KernelOperand::tensor("v", vec![TESTS], AccessMode::Read),
            KernelOperand::tensor("r", vec![TESTS], AccessMode::Write),
        ],
        indexing_maps: vec![
            IndexingMap::scalar(op(0)),
            IndexingMap::scalar(op(1)),
            IndexingMap::scalar(op(2)),
            IndexingMap::new(op(3), vec![IndexExpr::axis(axis)]),
            IndexingMap::new(op(4), vec![IndexExpr::axis(axis)]),
        ],
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: op(4),
                value: mul(add(mul(load(0), load(1)), load(2)), load(3)),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

/// The hand-inlined kernel `r[i] = (rho * u + sigma * (g0*g0 + g1*g1)) * v[i]`.
/// Operands: 0 `sigma`, 1 `g` [DIM], 2 `rho`, 3 `u`, 4 `v` [TESTS], 5 `r` [TESTS].
fn inlined_kernel() -> StructuredKernel {
    let axis = AxisId::new(0);
    StructuredKernel {
        name: "inlined".into(),
        iteration_domain: IterationDomain::new(vec![TESTS]),
        iterators: vec![IteratorKind::Parallel],
        operands: vec![
            KernelOperand::scalar("sigma", AccessMode::Read),
            KernelOperand::scalar("g0", AccessMode::Read),
            KernelOperand::scalar("g1", AccessMode::Read),
            KernelOperand::scalar("rho", AccessMode::Read),
            KernelOperand::scalar("u", AccessMode::Read),
            KernelOperand::tensor("v", vec![TESTS], AccessMode::Read),
            KernelOperand::tensor("r", vec![TESTS], AccessMode::Write),
        ],
        indexing_maps: vec![
            IndexingMap::scalar(op(0)),
            IndexingMap::scalar(op(1)),
            IndexingMap::scalar(op(2)),
            IndexingMap::scalar(op(3)),
            IndexingMap::scalar(op(4)),
            IndexingMap::new(op(5), vec![IndexExpr::axis(axis)]),
            IndexingMap::new(op(6), vec![IndexExpr::axis(axis)]),
        ],
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: op(6),
                value: mul(
                    add(
                        mul(load(3), load(4)),
                        add(
                            mul(load(0), mul(load(1), load(1))),
                            mul(load(0), mul(load(2), load(2))),
                        ),
                    ),
                    load(5),
                ),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

fn composition() -> KernelComposition {
    KernelComposition {
        name: "bound-energy".into(),
        stages: vec![producer_kernel(), consumer_kernel()],
        shared_buffers: vec![SharedBuffer::bind(
            StageOperand::new(0, op(2)),
            StageOperand::new(1, op(2)),
        )],
    }
}

const SIGMA: f64 = 1.7;
const G: [f64; DIM] = [0.3, -1.1];
const RHO: f64 = 2.5;
const U: f64 = 310.0;
const V: [f64; TESTS] = [0.2, 0.5, 0.3];

fn run_composition_primal() -> ([f64; TESTS], f64) {
    let executable = ExecutableComposition::reference(validate_composition(composition()).unwrap());
    let mut sigma = [SIGMA];
    let mut g = G;
    let mut rho = [RHO];
    let mut u = [U];
    let mut v = V;
    let mut r = [0.0; TESTS];
    let mut q = [0.0];
    Interpreter::run_composition(
        &executable,
        &mut [
            CompositionBinding::operand(StageOperand::new(0, op(0)), &mut sigma),
            CompositionBinding::operand(StageOperand::new(0, op(1)), &mut g),
            CompositionBinding::operand(StageOperand::new(1, op(0)), &mut rho),
            CompositionBinding::operand(StageOperand::new(1, op(1)), &mut u),
            CompositionBinding::operand(StageOperand::new(1, op(3)), &mut v),
            CompositionBinding::operand(StageOperand::new(1, op(4)), &mut r),
            CompositionBinding::shared(0, &mut q),
        ],
    )
    .unwrap();
    (r, q[0])
}

fn run_inlined_primal() -> [f64; TESTS] {
    let executable = Executable::reference(validate(inlined_kernel()).unwrap());
    let mut sigma = [SIGMA];
    let mut g0 = [G[0]];
    let mut g1 = [G[1]];
    let mut rho = [RHO];
    let mut u = [U];
    let mut v = V;
    let mut r = [0.0; TESTS];
    Interpreter::run(
        &executable,
        &mut [
            BufferBinding::new(op(0), &mut sigma),
            BufferBinding::new(op(1), &mut g0),
            BufferBinding::new(op(2), &mut g1),
            BufferBinding::new(op(3), &mut rho),
            BufferBinding::new(op(4), &mut u),
            BufferBinding::new(op(5), &mut v),
            BufferBinding::new(op(6), &mut r),
        ],
    )
    .unwrap();
    r
}

#[test]
fn two_kernel_composition_evaluates_equal_to_the_hand_inlined_kernel() {
    let (composed, q) = run_composition_primal();
    let inlined = run_inlined_primal();
    let expected_q = SIGMA * (G[0] * G[0] + G[1] * G[1]);
    assert!((q - expected_q).abs() <= 1.0e-15 * expected_q.abs());
    for (composed, inlined) in composed.iter().zip(&inlined) {
        assert!(
            (composed - inlined).abs() <= 1.0e-13 * inlined.abs(),
            "composed {composed} vs inlined {inlined}"
        );
    }
}

#[test]
fn unbound_shared_buffer_is_scratch_and_shared_members_cannot_be_bound_directly() {
    let executable = ExecutableComposition::reference(validate_composition(composition()).unwrap());
    let mut sigma = [SIGMA];
    let mut g = G;
    let mut rho = [RHO];
    let mut u = [U];
    let mut v = V;
    let mut r = [0.0; TESTS];
    Interpreter::run_composition(
        &executable,
        &mut [
            CompositionBinding::operand(StageOperand::new(0, op(0)), &mut sigma),
            CompositionBinding::operand(StageOperand::new(0, op(1)), &mut g),
            CompositionBinding::operand(StageOperand::new(1, op(0)), &mut rho),
            CompositionBinding::operand(StageOperand::new(1, op(1)), &mut u),
            CompositionBinding::operand(StageOperand::new(1, op(3)), &mut v),
            CompositionBinding::operand(StageOperand::new(1, op(4)), &mut r),
        ],
    )
    .unwrap();
    assert_eq!(r, run_composition_primal().0);

    let mut q = [0.0];
    let error = Interpreter::run_composition(
        &executable,
        &mut [
            CompositionBinding::operand(StageOperand::new(0, op(0)), &mut sigma),
            CompositionBinding::operand(StageOperand::new(0, op(1)), &mut g),
            CompositionBinding::operand(StageOperand::new(0, op(2)), &mut q),
            CompositionBinding::operand(StageOperand::new(1, op(0)), &mut rho),
            CompositionBinding::operand(StageOperand::new(1, op(1)), &mut u),
            CompositionBinding::operand(StageOperand::new(1, op(3)), &mut v),
            CompositionBinding::operand(StageOperand::new(1, op(4)), &mut r),
        ],
    )
    .unwrap_err();
    assert_eq!(
        error,
        malleus::CompositionExecutionError::SharedOperandBound(StageOperand::new(0, op(2)))
    );
    let error = Interpreter::run_composition(
        &executable,
        &mut [CompositionBinding::operand(
            StageOperand::new(0, op(0)),
            &mut sigma,
        )],
    )
    .unwrap_err();
    assert_eq!(
        error,
        malleus::CompositionExecutionError::MissingBinding(StageOperand::new(0, op(1)))
    );
}

#[test]
fn component_kernel_digests_are_unchanged_by_composition_and_differentiation() {
    let producer_digest = kernel_digest(&producer_kernel());
    let consumer_digest = kernel_digest(&consumer_kernel());
    let composition = composition();
    assert_eq!(kernel_digest(&composition.stages[0]), producer_digest);
    assert_eq!(kernel_digest(&composition.stages[1]), consumer_digest);
    let validated = validate_composition(composition.clone()).unwrap();
    assert_eq!(
        kernel_digest(validated.stages()[0].as_kernel()),
        producer_digest
    );
    assert_eq!(
        kernel_digest(validated.stages()[1].as_kernel()),
        consumer_digest
    );
    let product = differentiate_composition(
        &composition,
        &CompositionDerivativeRequest {
            mode: DerivativeMode::Jvp,
            independent_operands: vec![StageOperand::new(0, op(1)), StageOperand::new(1, op(1))],
            dependent_operands: vec![StageOperand::new(1, op(4))],
        },
    )
    .unwrap();
    let reproduced = product.primal_stages[0];
    assert_eq!(reproduced.primal_stage, 0);
    assert_eq!(
        kernel_digest(&product.composition.stages[reproduced.stage]),
        producer_digest
    );
    assert_eq!(kernel_digest(&producer_kernel()), producer_digest);
    assert_eq!(kernel_digest(&consumer_kernel()), consumer_digest);

    let digest = composition_digest(&composition);
    assert_eq!(digest, composition_digest(&composition.clone()));
    let mut renamed = composition.clone();
    renamed.name = "other".into();
    assert_ne!(composition_digest(&renamed), digest);
    let mut rebound = composition.clone();
    rebound.shared_buffers[0].members.reverse();
    assert_ne!(composition_digest(&rebound), digest);
    assert_eq!(digest.algorithm, "blake3");
    assert_eq!(digest.hex.len(), 64);
}

#[test]
fn composition_round_trips_through_serde() {
    let composition = composition();
    let encoded = serde_json::to_string(&composition).unwrap();
    let decoded: KernelComposition = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, composition);
    assert_eq!(
        composition_digest(&decoded),
        composition_digest(&composition)
    );
}

/// Directions for every exposed input of the inlined kernel, shared by the JVP/VJP checks.
const D_SIGMA: f64 = 0.4;
const D_G: [f64; DIM] = [-0.7, 0.25];
const D_RHO: f64 = 1.3;
const D_U: f64 = -2.0;
const SEED: [f64; TESTS] = [0.9, -0.4, 1.6];

/// Tangent of the inlined kernel's output with respect to `(sigma, g, rho, u)` by the
/// single-kernel JVP Malleus already proves against finite differences.
fn inlined_jvp() -> [f64; TESTS] {
    let product = differentiate(
        &inlined_kernel(),
        &DerivativeRequest {
            mode: DerivativeMode::Jvp,
            independent_operands: vec![op(0), op(1), op(2), op(3), op(4)],
            dependent_operands: vec![op(6)],
        },
    )
    .unwrap();
    let executable = Executable::reference(validate(product.kernel).unwrap());
    let mut sigma = [SIGMA];
    let mut g0 = [G[0]];
    let mut g1 = [G[1]];
    let mut rho = [RHO];
    let mut u = [U];
    let mut v = V;
    let mut d_sigma = [D_SIGMA];
    let mut d_g0 = [D_G[0]];
    let mut d_g1 = [D_G[1]];
    let mut d_rho = [D_RHO];
    let mut d_u = [D_U];
    let mut d_r = [0.0; TESTS];
    Interpreter::run(
        &executable,
        &mut [
            BufferBinding::new(op(0), &mut sigma),
            BufferBinding::new(op(1), &mut g0),
            BufferBinding::new(op(2), &mut g1),
            BufferBinding::new(op(3), &mut rho),
            BufferBinding::new(op(4), &mut u),
            BufferBinding::new(op(5), &mut v),
            BufferBinding::new(op(6), &mut d_sigma),
            BufferBinding::new(op(7), &mut d_g0),
            BufferBinding::new(op(8), &mut d_g1),
            BufferBinding::new(op(9), &mut d_rho),
            BufferBinding::new(op(10), &mut d_u),
            BufferBinding::new(op(11), &mut d_r),
        ],
    )
    .unwrap();
    d_r
}

fn composed_jvp(independent: Vec<StageOperand>) -> [f64; TESTS] {
    let product = differentiate_composition(
        &composition(),
        &CompositionDerivativeRequest {
            mode: DerivativeMode::Jvp,
            independent_operands: independent.clone(),
            dependent_operands: vec![StageOperand::new(1, op(4))],
        },
    )
    .unwrap();
    assert_eq!(product.mode, DerivativeMode::Jvp);
    let executable = ExecutableComposition::reference(
        validate_composition(product.composition.clone()).unwrap(),
    );
    // Primal values for every exposed readable operand: the reproduced producer stage and the
    // primal operands of every derivative stage, through the explicit tables.
    let mut primal_values: Vec<(StageOperand, Vec<f64>)> = Vec::new();
    let value_of = |stage: usize, operand: OperandId| -> Vec<f64> {
        match (stage, operand.index()) {
            (0, 0) => vec![SIGMA],
            (0, 1) => G.to_vec(),
            (1, 0) => vec![RHO],
            (1, 1) => vec![U],
            (1, 3) => V.to_vec(),
            other => panic!("unexpected primal operand {other:?}"),
        }
    };
    for stage in &product.primal_stages {
        let kernel = &product.composition.stages[stage.stage];
        for (index, operand) in kernel.operands.iter().enumerate() {
            let target = StageOperand::new(stage.stage, op(index));
            if executable.shared_buffer_of(target).is_some() {
                continue;
            }
            if operand.access.can_read() {
                primal_values.push((target, value_of(stage.primal_stage, op(index))));
            } else {
                primal_values.push((target, vec![0.0; operand.region.length]));
            }
        }
    }
    for stage in &product.derivative_stages {
        for pair in &stage.primal_operands {
            let target = StageOperand::new(stage.stage, pair.derivative);
            if executable.shared_buffer_of(target).is_some() {
                continue;
            }
            primal_values.push((target, value_of(stage.primal_stage, pair.primal)));
        }
    }
    let direction_of = |primal: StageOperand| -> Vec<f64> {
        match (primal.stage, primal.operand.index()) {
            (0, 0) => vec![D_SIGMA],
            (0, 1) => D_G.to_vec(),
            (1, 0) => vec![D_RHO],
            (1, 1) => vec![D_U],
            other => panic!("unexpected independent {other:?}"),
        }
    };
    let mut buffers = primal_values;
    for pair in &product.independent_operands {
        buffers.push((pair.derivative, direction_of(pair.primal)));
    }
    assert_eq!(product.dependent_operands.len(), 1);
    let tangent_target = product.dependent_operands[0].derivative;
    buffers.push((tangent_target, vec![0.0; TESTS]));
    let mut bindings = buffers
        .iter_mut()
        .map(|(target, values)| CompositionBinding::operand(*target, values))
        .collect::<Vec<_>>();
    Interpreter::run_composition(&executable, &mut bindings).unwrap();
    drop(bindings);
    let tangent = buffers
        .iter()
        .find(|(target, _)| *target == tangent_target)
        .map(|(_, values)| values.clone())
        .unwrap();
    let mut result = [0.0; TESTS];
    result.copy_from_slice(&tangent);
    result
}

#[test]
fn jvp_through_the_composition_matches_the_inlined_jvp() {
    let expected = inlined_jvp();
    let composed = composed_jvp(vec![
        StageOperand::new(0, op(0)),
        StageOperand::new(0, op(1)),
        StageOperand::new(1, op(0)),
        StageOperand::new(1, op(1)),
    ]);
    for (composed, expected) in composed.iter().zip(&expected) {
        assert!(
            (composed - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
            "composed {composed} vs inlined {expected}"
        );
    }
}

#[test]
fn cross_block_jvp_is_the_chain_rule_of_producer_and_consumer_tangents() {
    // Direction only in the producer's inputs: the consumer sees its bound input move through
    // the producer's tangent and nothing else.
    let composed = composed_jvp(vec![
        StageOperand::new(0, op(0)),
        StageOperand::new(0, op(1)),
    ]);
    let dq =
        D_SIGMA * (G[0] * G[0] + G[1] * G[1]) + SIGMA * (2.0 * G[0] * D_G[0] + 2.0 * G[1] * D_G[1]);
    for (index, composed) in composed.iter().enumerate() {
        let expected = dq * V[index];
        assert!(
            (composed - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
            "cross block {composed} vs {expected}"
        );
    }
    // Direction only in the consumer's own inputs: the diagonal block.
    let composed = composed_jvp(vec![
        StageOperand::new(1, op(0)),
        StageOperand::new(1, op(1)),
    ]);
    for (index, composed) in composed.iter().enumerate() {
        let expected = (D_RHO * U + RHO * D_U) * V[index];
        assert!(
            (composed - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
            "diagonal block {composed} vs {expected}"
        );
    }
}

#[test]
fn vjp_through_the_composition_satisfies_the_adjoint_identity() {
    let independent = vec![
        StageOperand::new(0, op(0)),
        StageOperand::new(0, op(1)),
        StageOperand::new(1, op(0)),
        StageOperand::new(1, op(1)),
    ];
    let product = differentiate_composition(
        &composition(),
        &CompositionDerivativeRequest {
            mode: DerivativeMode::Vjp,
            independent_operands: independent.clone(),
            dependent_operands: vec![StageOperand::new(1, op(4))],
        },
    )
    .unwrap();
    assert_eq!(product.mode, DerivativeMode::Vjp);
    // Reverse order: the consumer's VJP runs before the producer's.
    let positions = product
        .derivative_stages
        .iter()
        .map(|stage| (stage.primal_stage, stage.stage))
        .collect::<Vec<_>>();
    assert_eq!(positions, vec![(1, 1), (0, 2)]);
    let executable = ExecutableComposition::reference(
        validate_composition(product.composition.clone()).unwrap(),
    );
    let value_of = |stage: usize, operand: OperandId| -> Vec<f64> {
        match (stage, operand.index()) {
            (0, 0) => vec![SIGMA],
            (0, 1) => G.to_vec(),
            (1, 0) => vec![RHO],
            (1, 1) => vec![U],
            (1, 3) => V.to_vec(),
            other => panic!("unexpected primal operand {other:?}"),
        }
    };
    let mut buffers: Vec<(StageOperand, Vec<f64>)> = Vec::new();
    for stage in &product.primal_stages {
        let kernel = &product.composition.stages[stage.stage];
        for (index, operand) in kernel.operands.iter().enumerate() {
            let target = StageOperand::new(stage.stage, op(index));
            if executable.shared_buffer_of(target).is_some() {
                continue;
            }
            if operand.access.can_read() {
                buffers.push((target, value_of(stage.primal_stage, op(index))));
            } else {
                buffers.push((target, vec![0.0; operand.region.length]));
            }
        }
    }
    for stage in &product.derivative_stages {
        for pair in &stage.primal_operands {
            let target = StageOperand::new(stage.stage, pair.derivative);
            if executable.shared_buffer_of(target).is_some() {
                continue;
            }
            buffers.push((target, value_of(stage.primal_stage, pair.primal)));
        }
    }
    buffers.push((product.dependent_operands[0].derivative, SEED.to_vec()));
    let cotangent_targets = product
        .independent_operands
        .iter()
        .map(|pair| {
            let length = match pair.primal.operand.index() {
                1 if pair.primal.stage == 0 => DIM,
                _ => 1,
            };
            buffers.push((pair.derivative, vec![0.0; length]));
            (pair.primal, pair.derivative)
        })
        .collect::<Vec<_>>();
    let mut bindings = buffers
        .iter_mut()
        .map(|(target, values)| CompositionBinding::operand(*target, values))
        .collect::<Vec<_>>();
    Interpreter::run_composition(&executable, &mut bindings).unwrap();
    drop(bindings);
    let cotangent = |primal: StageOperand| -> Vec<f64> {
        let (_, derivative) = cotangent_targets
            .iter()
            .find(|(candidate, _)| *candidate == primal)
            .unwrap();
        buffers
            .iter()
            .find(|(target, _)| target == derivative)
            .map(|(_, values)| values.clone())
            .unwrap()
    };
    let reverse_dot = cotangent(StageOperand::new(0, op(0)))[0] * D_SIGMA
        + cotangent(StageOperand::new(0, op(1)))
            .iter()
            .zip(D_G)
            .map(|(bar, d)| bar * d)
            .sum::<f64>()
        + cotangent(StageOperand::new(1, op(0)))[0] * D_RHO
        + cotangent(StageOperand::new(1, op(1)))[0] * D_U;
    let forward_dot = composed_jvp(independent)
        .iter()
        .zip(SEED)
        .map(|(tangent, seed)| tangent * seed)
        .sum::<f64>();
    assert!(
        (reverse_dot - forward_dot).abs() <= 1.0e-12 * forward_dot.abs().max(1.0),
        "<seed, J dx> = {forward_dot} but <J^T seed, dx> = {reverse_dot}"
    );
}

#[test]
fn fan_out_to_two_consumers_accumulates_the_producer_cotangent() {
    let mut second = consumer_kernel();
    second.name = "consumer::second".into();
    // Second consumer: r2[i] = (rho * u + q) * v[i] with independent buffers.
    let composition = KernelComposition {
        name: "fan-out".into(),
        stages: vec![producer_kernel(), consumer_kernel(), second],
        shared_buffers: vec![SharedBuffer::new(vec![
            StageOperand::new(0, op(2)),
            StageOperand::new(1, op(2)),
            StageOperand::new(2, op(2)),
        ])],
    };
    validate_composition(composition.clone()).unwrap();
    let product = differentiate_composition(
        &composition,
        &CompositionDerivativeRequest {
            mode: DerivativeMode::Vjp,
            independent_operands: vec![StageOperand::new(0, op(0))],
            dependent_operands: vec![StageOperand::new(1, op(4)), StageOperand::new(2, op(4))],
        },
    )
    .unwrap();
    let executable = ExecutableComposition::reference(
        validate_composition(product.composition.clone()).unwrap(),
    );
    let value_of = |stage: usize, operand: OperandId| -> Vec<f64> {
        match (stage, operand.index()) {
            (0, 0) => vec![SIGMA],
            (0, 1) => G.to_vec(),
            (_, 0) => vec![RHO],
            (_, 1) => vec![U],
            (_, 3) => V.to_vec(),
            other => panic!("unexpected primal operand {other:?}"),
        }
    };
    let mut buffers: Vec<(StageOperand, Vec<f64>)> = Vec::new();
    for stage in &product.primal_stages {
        let kernel = &product.composition.stages[stage.stage];
        for (index, operand) in kernel.operands.iter().enumerate() {
            let target = StageOperand::new(stage.stage, op(index));
            if executable.shared_buffer_of(target).is_some() {
                continue;
            }
            buffers.push((
                target,
                if operand.access.can_read() {
                    value_of(stage.primal_stage, op(index))
                } else {
                    vec![0.0; operand.region.length]
                },
            ));
        }
    }
    for stage in &product.derivative_stages {
        for pair in &stage.primal_operands {
            let target = StageOperand::new(stage.stage, pair.derivative);
            if executable.shared_buffer_of(target).is_some() {
                continue;
            }
            buffers.push((target, value_of(stage.primal_stage, pair.primal)));
        }
    }
    for pair in &product.dependent_operands {
        buffers.push((pair.derivative, SEED.to_vec()));
    }
    let bar_sigma = product.independent_operands[0].derivative;
    buffers.push((bar_sigma, vec![0.0]));
    let mut bindings = buffers
        .iter_mut()
        .map(|(target, values)| CompositionBinding::operand(*target, values))
        .collect::<Vec<_>>();
    Interpreter::run_composition(&executable, &mut bindings).unwrap();
    drop(bindings);
    let observed = buffers
        .iter()
        .find(|(target, _)| *target == bar_sigma)
        .unwrap()
        .1[0];
    // d r_k[i] / d sigma = dot(g, g) * v[i] for both consumers, seeded identically.
    let expected = 2.0
        * (G[0] * G[0] + G[1] * G[1])
        * V.iter().zip(SEED).map(|(v, seed)| v * seed).sum::<f64>();
    assert!(
        (observed - expected).abs() <= 1.0e-12 * expected.abs(),
        "accumulated cotangent {observed} vs {expected}"
    );
}

#[test]
fn composition_refusals_are_typed() {
    let mut reversed = composition();
    reversed.stages.reverse();
    reversed.shared_buffers = vec![SharedBuffer::bind(
        StageOperand::new(1, op(2)),
        StageOperand::new(0, op(2)),
    )];
    assert_eq!(
        validate_composition(reversed).unwrap_err(),
        CompositionError::WriterAfterReader {
            buffer: 0,
            writer: StageOperand::new(1, op(2)),
            reader: StageOperand::new(0, op(2)),
        }
    );
    let mut mismatched = composition();
    mismatched.shared_buffers = vec![SharedBuffer::bind(
        StageOperand::new(0, op(2)),
        StageOperand::new(1, op(3)),
    )];
    assert_eq!(
        validate_composition(mismatched).unwrap_err(),
        CompositionError::MemberShape {
            buffer: 0,
            member: StageOperand::new(1, op(3)),
        }
    );
    let mut readers_only = composition();
    readers_only.shared_buffers = vec![SharedBuffer::bind(
        StageOperand::new(0, op(0)),
        StageOperand::new(1, op(0)),
    )];
    assert_eq!(
        validate_composition(readers_only).unwrap_err(),
        CompositionError::NoWriter(0)
    );
    assert_eq!(
        differentiate_composition(
            &composition(),
            &CompositionDerivativeRequest {
                mode: DerivativeMode::Jvp,
                independent_operands: vec![StageOperand::new(1, op(2))],
                dependent_operands: vec![StageOperand::new(1, op(4))],
            },
        )
        .unwrap_err(),
        CompositionDifferentiationError::SharedIndependent(StageOperand::new(1, op(2)))
    );
    assert_eq!(
        differentiate_composition(
            &composition(),
            &CompositionDerivativeRequest {
                mode: DerivativeMode::Jvp,
                independent_operands: vec![StageOperand::new(1, op(0))],
                dependent_operands: vec![StageOperand::new(0, op(2))],
            },
        )
        .unwrap_err(),
        CompositionDifferentiationError::SharedDependent(StageOperand::new(0, op(2)))
    );
    // The consumer's own inputs do not reach a producer-only dependent.
    let mut with_probe = composition();
    with_probe.stages[0].operands.push(KernelOperand::scalar(
        "probe",
        AccessMode::Reduce(ReductionOp::Add),
    ));
    with_probe.stages[0]
        .indexing_maps
        .push(IndexingMap::scalar(op(3)));
    with_probe.stages[0].body.statements.push(Statement::Store {
        operand: op(3),
        value: load(0),
    });
    assert_eq!(
        differentiate_composition(
            &with_probe,
            &CompositionDerivativeRequest {
                mode: DerivativeMode::Jvp,
                independent_operands: vec![StageOperand::new(1, op(0))],
                dependent_operands: vec![StageOperand::new(0, op(3))],
            },
        )
        .unwrap_err(),
        CompositionDifferentiationError::IndependentUnreachable(StageOperand::new(1, op(0)))
    );
    let _ = CompositionTarget::Shared(0);
}
