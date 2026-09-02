//! Facet-pair (two-sided trace) kernels on an interior-penalty jump/average fixture that also
//! carries a side-owned outward flux datum per cell and the facet balance row between them.

use malleus::{
    AccessMode, AxisId, BinaryOp, BufferBinding, CompareOp, DerivativeMode, DerivativeRequest,
    Executable, FacetOperandRole, FacetPairError, FacetPairKernel, FacetSide, FacetSwapError,
    IndexExpr, IndexingMap, Interpreter, IterationDomain, IteratorKind, KernelOperand,
    KernelRegion, NumericPolicy, OperandId, Predicate, ReductionOp, ScalarExpr, Statement,
    StructuredKernel, SwapParity, check_facet_swap_symmetry, differentiate_facet_pair,
    facet_pair_digest, kernel_digest, validate, validate_facet_pair,
};

const DIM: usize = 2;
const ETA: f64 = 4.0;

const U_MINUS: usize = 0;
const U_PLUS: usize = 1;
const GRAD_MINUS: usize = 2;
const GRAD_PLUS: usize = 3;
const V_MINUS: usize = 4;
const V_PLUS: usize = 5;
const NORMAL: usize = 6;
const H: usize = 7;
const K: usize = 8;
const G_MINUS: usize = 9;
const G_PLUS: usize = 10;
const JUMP: usize = 11;
const AVERAGE: usize = 12;
const FLUX_AVERAGE: usize = 13;
const R_MINUS: usize = 14;
const R_PLUS: usize = 15;
const BALANCE: usize = 16;
const FLUX_ROW_MINUS: usize = 17;
const FLUX_ROW_PLUS: usize = 18;

fn op(index: usize) -> OperandId {
    OperandId::new(index)
}

fn load(index: usize) -> ScalarExpr {
    ScalarExpr::Load(op(index))
}

fn c(value: f64) -> ScalarExpr {
    ScalarExpr::Constant(value)
}

fn mul(lhs: ScalarExpr, rhs: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(BinaryOp::Mul, lhs, rhs)
}

fn add(lhs: ScalarExpr, rhs: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(BinaryOp::Add, lhs, rhs)
}

fn sub(lhs: ScalarExpr, rhs: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(BinaryOp::Sub, lhs, rhs)
}

fn div(lhs: ScalarExpr, rhs: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(BinaryOp::Div, lhs, rhs)
}

fn neg(value: ScalarExpr) -> ScalarExpr {
    ScalarExpr::unary(malleus::UnaryOp::Neg, value)
}

/// Contribute a component-independent term once inside the reduction over components.
fn once(value: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Select {
        condition: Box::new(Predicate::Compare {
            op: CompareOp::Eq,
            lhs: Box::new(ScalarExpr::Index(AxisId::new(0))),
            rhs: Box::new(c(0.0)),
        }),
        if_true: Box::new(value),
        if_false: Box::new(c(0.0)),
    }
}

fn cell(side: FacetSide, partner: usize) -> FacetOperandRole {
    FacetOperandRole::Cell {
        side,
        partner: Some(op(partner)),
    }
}

fn facet(parity: SwapParity) -> FacetOperandRole {
    FacetOperandRole::Facet { parity }
}

fn fixture() -> FacetPairKernel {
    let axis = AxisId::new(0);
    let vector = |index: usize| IndexingMap::new(op(index), vec![IndexExpr::axis(axis)]);
    let scalar = |index: usize| IndexingMap::scalar(op(index));
    let jump = sub(load(U_MINUS), load(U_PLUS));
    let average = mul(c(0.5), add(load(U_MINUS), load(U_PLUS)));
    let flux_average_component = mul(
        mul(c(0.5), add(load(GRAD_MINUS), load(GRAD_PLUS))),
        load(NORMAL),
    );
    let penalty = mul(div(c(ETA), load(H)), jump.clone());
    let kernel = StructuredKernel {
        name: "interior-penalty".into(),
        iteration_domain: IterationDomain::new(vec![DIM]),
        iterators: vec![IteratorKind::Reduction],
        operands: vec![
            KernelOperand::scalar("u_minus", AccessMode::Read),
            KernelOperand::scalar("u_plus", AccessMode::Read),
            KernelOperand::tensor("grad_minus", vec![DIM], AccessMode::Read),
            KernelOperand::tensor("grad_plus", vec![DIM], AccessMode::Read),
            KernelOperand::scalar("v_minus", AccessMode::Read),
            KernelOperand::scalar("v_plus", AccessMode::Read),
            KernelOperand::tensor("normal", vec![DIM], AccessMode::Read),
            KernelOperand::scalar("h", AccessMode::Read),
            KernelOperand::scalar("k", AccessMode::Read),
            KernelOperand::scalar("g_minus", AccessMode::Read),
            KernelOperand::scalar("g_plus", AccessMode::Read),
            KernelOperand::scalar("jump", AccessMode::Reduce(ReductionOp::Add)),
            KernelOperand::scalar("average", AccessMode::Reduce(ReductionOp::Add)),
            KernelOperand::scalar("flux_average", AccessMode::Reduce(ReductionOp::Add)),
            KernelOperand::scalar("r_minus", AccessMode::Reduce(ReductionOp::Add)),
            KernelOperand::scalar("r_plus", AccessMode::Reduce(ReductionOp::Add)),
            KernelOperand::scalar("balance", AccessMode::Reduce(ReductionOp::Add)),
            KernelOperand::scalar("flux_row_minus", AccessMode::Reduce(ReductionOp::Add)),
            KernelOperand::scalar("flux_row_plus", AccessMode::Reduce(ReductionOp::Add)),
        ],
        indexing_maps: vec![
            scalar(U_MINUS),
            scalar(U_PLUS),
            vector(GRAD_MINUS),
            vector(GRAD_PLUS),
            scalar(V_MINUS),
            scalar(V_PLUS),
            vector(NORMAL),
            scalar(H),
            scalar(K),
            scalar(G_MINUS),
            scalar(G_PLUS),
            scalar(JUMP),
            scalar(AVERAGE),
            scalar(FLUX_AVERAGE),
            scalar(R_MINUS),
            scalar(R_PLUS),
            scalar(BALANCE),
            scalar(FLUX_ROW_MINUS),
            scalar(FLUX_ROW_PLUS),
        ],
        body: KernelRegion {
            statements: vec![
                Statement::Store {
                    operand: op(JUMP),
                    value: once(jump.clone()),
                },
                Statement::Store {
                    operand: op(AVERAGE),
                    value: once(average),
                },
                Statement::Store {
                    operand: op(FLUX_AVERAGE),
                    value: flux_average_component.clone(),
                },
                // -{k grad u}.n [v] + (eta/h) [u] [v], split into the two side-owned rows.
                Statement::Store {
                    operand: op(R_MINUS),
                    value: add(
                        neg(mul(
                            mul(load(K), flux_average_component.clone()),
                            load(V_MINUS),
                        )),
                        once(mul(penalty.clone(), load(V_MINUS))),
                    ),
                },
                Statement::Store {
                    operand: op(R_PLUS),
                    value: sub(
                        mul(mul(load(K), flux_average_component), load(V_PLUS)),
                        once(mul(penalty, load(V_PLUS))),
                    ),
                },
                // Outward-from-owner flux data: balance sums with unit coefficients.
                Statement::Store {
                    operand: op(BALANCE),
                    value: once(add(load(G_MINUS), load(G_PLUS))),
                },
                // Each owner's flux datum against its own outward normal flux.
                Statement::Store {
                    operand: op(FLUX_ROW_MINUS),
                    value: sub(
                        once(load(G_MINUS)),
                        mul(mul(load(K), load(GRAD_MINUS)), load(NORMAL)),
                    ),
                },
                Statement::Store {
                    operand: op(FLUX_ROW_PLUS),
                    value: add(
                        once(load(G_PLUS)),
                        mul(mul(load(K), load(GRAD_PLUS)), load(NORMAL)),
                    ),
                },
            ],
        },
        numeric_policy: NumericPolicy::default(),
    };
    let roles = vec![
        cell(FacetSide::Minus, U_PLUS),
        cell(FacetSide::Plus, U_MINUS),
        cell(FacetSide::Minus, GRAD_PLUS),
        cell(FacetSide::Plus, GRAD_MINUS),
        cell(FacetSide::Minus, V_PLUS),
        cell(FacetSide::Plus, V_MINUS),
        facet(SwapParity::Odd),
        facet(SwapParity::Even),
        facet(SwapParity::Even),
        cell(FacetSide::Minus, G_PLUS),
        cell(FacetSide::Plus, G_MINUS),
        facet(SwapParity::Odd),
        facet(SwapParity::Even),
        facet(SwapParity::Odd),
        cell(FacetSide::Minus, R_PLUS),
        cell(FacetSide::Plus, R_MINUS),
        facet(SwapParity::Even),
        cell(FacetSide::Minus, FLUX_ROW_PLUS),
        cell(FacetSide::Plus, FLUX_ROW_MINUS),
    ];
    FacetPairKernel { kernel, roles }
}

fn buffers() -> Vec<Vec<f64>> {
    let normal = [0.6, 0.8];
    vec![
        vec![3.0],
        vec![2.5],
        vec![1.5, -0.5],
        vec![-2.0, 0.75],
        vec![0.35],
        vec![0.65],
        normal.to_vec(),
        vec![0.125],
        vec![1.3],
        vec![0.9],
        vec![-1.1],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
    ]
}

fn run(kernel: &StructuredKernel, buffers: &mut [Vec<f64>]) {
    let executable = Executable::reference(validate(kernel.clone()).unwrap());
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(index, values)| BufferBinding::new(op(index), values))
        .collect::<Vec<_>>();
    Interpreter::run(&executable, &mut bindings).unwrap();
}

#[test]
fn fixture_validates_and_evaluates_the_jump_average_and_balance_rows() {
    let facet_kernel = fixture();
    let validated = validate_facet_pair(facet_kernel.clone()).unwrap();
    assert_eq!(validated.roles().len(), facet_kernel.kernel.operands.len());
    let mut values = buffers();
    run(&facet_kernel.kernel, &mut values);
    let n = [0.6, 0.8];
    let gm = [1.5, -0.5];
    let gp = [-2.0, 0.75];
    let flux_average = 0.5 * ((gm[0] + gp[0]) * n[0] + (gm[1] + gp[1]) * n[1]);
    let jump = 3.0 - 2.5;
    assert!((values[JUMP][0] - jump).abs() <= 1.0e-15);
    assert!((values[AVERAGE][0] - 2.75).abs() <= 1.0e-15);
    assert!((values[FLUX_AVERAGE][0] - flux_average).abs() <= 1.0e-15);
    let r_minus = -1.3 * flux_average * 0.35 + (ETA / 0.125) * jump * 0.35;
    let r_plus = 1.3 * flux_average * 0.65 - (ETA / 0.125) * jump * 0.65;
    assert!((values[R_MINUS][0] - r_minus).abs() <= 1.0e-13);
    assert!((values[R_PLUS][0] - r_plus).abs() <= 1.0e-13);
    assert!((values[BALANCE][0] - (0.9 - 1.1)).abs() <= 1.0e-15);
    let flux_row_minus = 0.9 - 1.3 * (gm[0] * n[0] + gm[1] * n[1]);
    let flux_row_plus = -1.1 + 1.3 * (gp[0] * n[0] + gp[1] * n[1]);
    assert!((values[FLUX_ROW_MINUS][0] - flux_row_minus).abs() <= 1.0e-13);
    assert!((values[FLUX_ROW_PLUS][0] - flux_row_plus).abs() <= 1.0e-13);
}

#[test]
fn swap_covariance_is_proven_by_execution_and_a_wrong_parity_is_detected() {
    let validated = validate_facet_pair(fixture()).unwrap();
    let report = check_facet_swap_symmetry(&validated, &buffers(), 1.0e-12).unwrap();
    assert!(report.within_tolerance, "{report:?}");
    assert!(report.max_absolute <= 1.0e-12);
    assert_eq!(report.deviations.len(), 8);

    let mut wrong = fixture();
    wrong.roles[JUMP] = facet(SwapParity::Even);
    let wrong = validate_facet_pair(wrong).unwrap();
    let report = check_facet_swap_symmetry(&wrong, &buffers(), 1.0e-12).unwrap();
    assert!(!report.within_tolerance);
    let offending = report
        .deviations
        .iter()
        .filter(|deviation| deviation.max_absolute > 1.0e-12)
        .map(|deviation| deviation.operand)
        .collect::<Vec<_>>();
    assert_eq!(offending, vec![op(JUMP)]);
    assert!((report.max_absolute - 2.0 * 0.5).abs() <= 1.0e-12);

    let mut unpaired = fixture();
    unpaired.roles[G_MINUS] = FacetOperandRole::Cell {
        side: FacetSide::Minus,
        partner: None,
    };
    unpaired.roles[G_PLUS] = FacetOperandRole::Cell {
        side: FacetSide::Plus,
        partner: None,
    };
    let unpaired = validate_facet_pair(unpaired).unwrap();
    assert_eq!(
        check_facet_swap_symmetry(&unpaired, &buffers(), 1.0e-12).unwrap_err(),
        FacetSwapError::UnpairedOperand(G_MINUS)
    );
}

#[test]
fn role_validation_refuses_inconsistent_pairings() {
    let mut short = fixture();
    short.roles.pop();
    assert_eq!(
        validate_facet_pair(short).unwrap_err(),
        FacetPairError::RoleCount {
            expected: 19,
            actual: 18,
        }
    );
    let mut same_side = fixture();
    same_side.roles[U_PLUS] = cell(FacetSide::Minus, U_MINUS);
    assert_eq!(
        validate_facet_pair(same_side).unwrap_err(),
        FacetPairError::PartnerSide {
            operand: U_MINUS,
            partner: U_PLUS,
        }
    );
    let mut asymmetric = fixture();
    asymmetric.roles[U_PLUS] = cell(FacetSide::Plus, V_MINUS);
    assert_eq!(
        validate_facet_pair(asymmetric).unwrap_err(),
        FacetPairError::PartnerAsymmetric {
            operand: U_MINUS,
            partner: U_PLUS,
        }
    );
    let mut shape = fixture();
    shape.roles[U_MINUS] = cell(FacetSide::Minus, GRAD_PLUS);
    shape.roles[GRAD_PLUS] = cell(FacetSide::Plus, U_MINUS);
    shape.roles[U_PLUS] = FacetOperandRole::Cell {
        side: FacetSide::Plus,
        partner: None,
    };
    shape.roles[GRAD_MINUS] = FacetOperandRole::Cell {
        side: FacetSide::Minus,
        partner: None,
    };
    assert_eq!(
        validate_facet_pair(shape).unwrap_err(),
        FacetPairError::PartnerShape {
            operand: U_MINUS,
            partner: GRAD_PLUS,
        }
    );
    let mut one_sided = fixture();
    for role in &mut one_sided.roles {
        if let FacetOperandRole::Cell { side, partner } = role {
            *side = FacetSide::Minus;
            *partner = None;
        }
    }
    assert_eq!(
        validate_facet_pair(one_sided).unwrap_err(),
        FacetPairError::MissingSide(FacetSide::Plus)
    );
}

#[test]
fn facet_pair_digest_covers_roles_but_leaves_the_kernel_digest_untouched() {
    let facet_kernel = fixture();
    let inner = kernel_digest(&facet_kernel.kernel);
    let digest = facet_pair_digest(&facet_kernel);
    assert_eq!(digest, facet_pair_digest(&fixture()));
    assert_eq!(kernel_digest(&fixture().kernel), inner);
    let mut relabelled = fixture();
    relabelled.roles[H] = facet(SwapParity::Odd);
    assert_ne!(facet_pair_digest(&relabelled), digest);
    assert_eq!(kernel_digest(&relabelled.kernel), inner);
    let encoded = serde_json::to_string(&facet_kernel).unwrap();
    let decoded: FacetPairKernel = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, facet_kernel);
    assert_eq!(facet_pair_digest(&decoded), digest);
}

fn direction(index: usize) -> Vec<f64> {
    match index {
        U_MINUS => vec![0.7],
        U_PLUS => vec![-0.4],
        GRAD_MINUS => vec![0.2, 1.1],
        GRAD_PLUS => vec![-0.9, 0.3],
        G_MINUS => vec![1.5],
        G_PLUS => vec![-0.6],
        _ => unreachable!(),
    }
}

fn seed(index: usize) -> f64 {
    match index {
        R_MINUS => 0.8,
        R_PLUS => -1.2,
        BALANCE => 0.5,
        FLUX_ROW_MINUS => 1.7,
        FLUX_ROW_PLUS => -0.3,
        _ => unreachable!(),
    }
}

const INDEPENDENT: [usize; 6] = [U_MINUS, U_PLUS, GRAD_MINUS, GRAD_PLUS, G_MINUS, G_PLUS];
const DEPENDENT: [usize; 5] = [R_MINUS, R_PLUS, BALANCE, FLUX_ROW_MINUS, FLUX_ROW_PLUS];

fn request(mode: DerivativeMode) -> DerivativeRequest {
    DerivativeRequest {
        mode,
        independent_operands: INDEPENDENT.iter().map(|index| op(*index)).collect(),
        dependent_operands: DEPENDENT.iter().map(|index| op(*index)).collect(),
    }
}

#[test]
fn jvp_and_vjp_of_the_facet_pair_satisfy_the_adjoint_identity_and_keep_roles() {
    let facet_kernel = fixture();
    let jvp = differentiate_facet_pair(&facet_kernel, &request(DerivativeMode::Jvp)).unwrap();
    let vjp = differentiate_facet_pair(&facet_kernel, &request(DerivativeMode::Vjp)).unwrap();
    assert_eq!(jvp.roles.len(), jvp.product.kernel.operands.len());
    assert_eq!(vjp.roles.len(), vjp.product.kernel.operands.len());
    // Roles follow operands into the derivative kernels, partners included.
    let d_u_minus = jvp.product.independent_operands[0].derivative;
    let d_u_plus = jvp.product.independent_operands[1].derivative;
    assert_eq!(
        jvp.roles[d_u_minus.index()],
        FacetOperandRole::Cell {
            side: FacetSide::Minus,
            partner: Some(d_u_plus),
        }
    );
    let bar_r_minus = vjp.product.dependent_operands[0].derivative;
    let bar_r_plus = vjp.product.dependent_operands[1].derivative;
    assert_eq!(
        vjp.roles[bar_r_plus.index()],
        FacetOperandRole::Cell {
            side: FacetSide::Plus,
            partner: Some(bar_r_minus),
        }
    );
    let primal = buffers();
    let prime = |product: &malleus::DerivativeProduct| -> Vec<Vec<f64>> {
        let mut values = vec![Vec::new(); product.kernel.operands.len()];
        for pair in &product.primal_operands {
            values[pair.derivative.index()] = primal[pair.primal.index()].clone();
        }
        values
    };

    let mut forward = prime(&jvp.product);
    for pair in &jvp.product.independent_operands {
        forward[pair.derivative.index()] = direction(pair.primal.index());
    }
    for pair in &jvp.product.dependent_operands {
        forward[pair.derivative.index()] = vec![0.0];
    }
    run(&jvp.product.kernel, &mut forward);
    let forward_dot = jvp
        .product
        .dependent_operands
        .iter()
        .map(|pair| forward[pair.derivative.index()][0] * seed(pair.primal.index()))
        .sum::<f64>();

    let mut reverse = prime(&vjp.product);
    for pair in &vjp.product.dependent_operands {
        reverse[pair.derivative.index()] = vec![seed(pair.primal.index())];
    }
    for pair in &vjp.product.independent_operands {
        reverse[pair.derivative.index()] = vec![0.0; direction(pair.primal.index()).len()];
    }
    run(&vjp.product.kernel, &mut reverse);
    let reverse_dot = vjp
        .product
        .independent_operands
        .iter()
        .map(|pair| {
            reverse[pair.derivative.index()]
                .iter()
                .zip(direction(pair.primal.index()))
                .map(|(bar, d)| bar * d)
                .sum::<f64>()
        })
        .sum::<f64>();
    assert!(
        (forward_dot - reverse_dot).abs() <= 1.0e-12 * forward_dot.abs().max(1.0),
        "<seed, J dx> = {forward_dot} but <J^T seed, dx> = {reverse_dot}"
    );

    // Both derivative kernels are facet-pair kernels in their own right and stay swap
    // covariant: directions and seeds carry the parity of the operands they belong to.
    let mut forward_inputs = prime(&jvp.product);
    for pair in &jvp.product.independent_operands {
        forward_inputs[pair.derivative.index()] = direction(pair.primal.index());
    }
    for pair in &jvp.product.dependent_operands {
        forward_inputs[pair.derivative.index()] = vec![0.0];
    }
    let jvp_kernel = validate_facet_pair(jvp.into_facet_pair()).unwrap();
    let report = check_facet_swap_symmetry(&jvp_kernel, &forward_inputs, 1.0e-12).unwrap();
    assert!(report.within_tolerance, "{report:?}");
    let mut reverse_inputs = prime(&vjp.product);
    for pair in &vjp.product.dependent_operands {
        reverse_inputs[pair.derivative.index()] = vec![seed(pair.primal.index())];
    }
    for pair in &vjp.product.independent_operands {
        reverse_inputs[pair.derivative.index()] = vec![0.0; direction(pair.primal.index()).len()];
    }
    let vjp_kernel = validate_facet_pair(vjp.into_facet_pair()).unwrap();
    let report = check_facet_swap_symmetry(&vjp_kernel, &reverse_inputs, 1.0e-12).unwrap();
    assert!(report.within_tolerance, "{report:?}");
}

#[test]
fn one_sided_derivative_request_drops_the_partner_and_refuses_the_swap_check() {
    let product = differentiate_facet_pair(
        &fixture(),
        &DerivativeRequest {
            mode: DerivativeMode::Jvp,
            independent_operands: vec![op(U_MINUS)],
            dependent_operands: vec![op(R_MINUS), op(R_PLUS)],
        },
    )
    .unwrap();
    let d_u_minus = product.product.independent_operands[0].derivative;
    assert_eq!(
        product.roles[d_u_minus.index()],
        FacetOperandRole::Cell {
            side: FacetSide::Minus,
            partner: None,
        }
    );
    let validated = validate_facet_pair(product.into_facet_pair()).unwrap();
    let mut values = vec![vec![0.0]; validated.kernel().as_kernel().operands.len()];
    for (index, operand) in validated.kernel().as_kernel().operands.iter().enumerate() {
        values[index] = vec![0.0; operand.region.length];
    }
    assert_eq!(
        check_facet_swap_symmetry(&validated, &values, 1.0e-12).unwrap_err(),
        FacetSwapError::UnpairedOperand(d_u_minus.index())
    );
}
