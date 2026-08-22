use malleus::{
    AccessMode, AxisId, BinaryOp, CampaignError, ComparisonTolerance, Executable, IndexExpr,
    IndexingMap, IterationDomain, IteratorKind, KernelOperand, KernelRegion, LocalCheckKind,
    LocalDifferentialCase, LocalExecutableRunner, NumericPolicy, NumericPolicyMutation, OperandId,
    OperandValues, ReductionOp, ScalarExpr, ScalarType, Statement, StructuredKernel,
    StructuredModule, check_numeric_policy_mutation, run_local_differential_campaign,
};

fn quadratic_module() -> StructuredModule {
    let axis = AxisId::new(0);
    let x = OperandId::new(0);
    let parameter = OperandId::new(1);
    let output = OperandId::new(2);
    StructuredModule {
        name: "local-fixture".into(),
        kernels: vec![StructuredKernel {
            name: "quadratic".into(),
            iteration_domain: IterationDomain::new(vec![3]),
            iterators: vec![IteratorKind::Reduction],
            operands: vec![
                KernelOperand::tensor("x", vec![3], AccessMode::Read),
                KernelOperand::scalar("parameter", AccessMode::Read),
                KernelOperand::scalar("output", AccessMode::Reduce(ReductionOp::Add)),
            ],
            indexing_maps: vec![
                IndexingMap::new(x, vec![IndexExpr::axis(axis)]),
                IndexingMap::scalar(parameter),
                IndexingMap::scalar(output),
            ],
            body: KernelRegion {
                statements: vec![Statement::Store {
                    operand: output,
                    value: ScalarExpr::binary(
                        BinaryOp::Mul,
                        ScalarExpr::Load(parameter),
                        ScalarExpr::binary(BinaryOp::Mul, ScalarExpr::Load(x), ScalarExpr::Load(x)),
                    ),
                }],
            },
            numeric_policy: NumericPolicy::default(),
        }],
    }
}

fn case() -> LocalDifferentialCase {
    LocalDifferentialCase {
        kernel: "quadratic".into(),
        primal_buffers: vec![vec![1.5, -2.0, 0.25], vec![3.0], vec![0.0]],
        state_directions: vec![OperandValues {
            operand: OperandId::new(0),
            values: vec![-0.5, 2.0, 4.0],
        }],
        parameter_directions: vec![OperandValues {
            operand: OperandId::new(1),
            values: vec![0.75],
        }],
        dependent_seeds: vec![OperandValues {
            operand: OperandId::new(2),
            values: vec![-1.25],
        }],
        centered_step: 1.0e-5,
        tolerance: ComparisonTolerance::new(2.0e-8, 2.0e-8),
    }
}

fn two_output_module() -> StructuredModule {
    let mut module = quadratic_module();
    let kernel = &mut module.kernels[0];
    let second_output = OperandId::new(kernel.operands.len());
    kernel.operands.push(KernelOperand::scalar(
        "second_output",
        AccessMode::Reduce(ReductionOp::Add),
    ));
    kernel
        .indexing_maps
        .push(IndexingMap::scalar(second_output));
    kernel.body.statements.push(Statement::Store {
        operand: second_output,
        value: ScalarExpr::Load(OperandId::new(0)),
    });
    module
}

fn reduction_order_module() -> StructuredModule {
    let row = AxisId::new(0);
    let column = AxisId::new(1);
    StructuredModule {
        name: "reduction-order-fixture".into(),
        kernels: vec![StructuredKernel {
            name: "sensitive-sum".into(),
            iteration_domain: IterationDomain::new(vec![2, 2]),
            iterators: vec![IteratorKind::Reduction, IteratorKind::Reduction],
            operands: vec![
                KernelOperand::tensor("values", vec![2, 2], AccessMode::Read),
                KernelOperand::scalar("sum", AccessMode::Reduce(ReductionOp::Add)),
            ],
            indexing_maps: vec![
                IndexingMap::new(
                    OperandId::new(0),
                    vec![IndexExpr::axis(row), IndexExpr::axis(column)],
                ),
                IndexingMap::scalar(OperandId::new(1)),
            ],
            body: KernelRegion {
                statements: vec![Statement::Store {
                    operand: OperandId::new(1),
                    value: ScalarExpr::Load(OperandId::new(0)),
                }],
            },
            numeric_policy: NumericPolicy::default(),
        }],
    }
}

fn reduction_order_case() -> LocalDifferentialCase {
    LocalDifferentialCase {
        kernel: "sensitive-sum".into(),
        primal_buffers: vec![vec![1.0e16, 1.0, -1.0e16, 1.0], vec![0.0]],
        state_directions: vec![OperandValues {
            operand: OperandId::new(0),
            values: vec![1.0; 4],
        }],
        parameter_directions: vec![],
        dependent_seeds: vec![OperandValues {
            operand: OperandId::new(1),
            values: vec![1.0],
        }],
        centered_step: 1.0e-5,
        tolerance: ComparisonTolerance::new(0.0, 0.0),
    }
}

struct FaithfulRunner;

impl LocalExecutableRunner for FaithfulRunner {
    fn identity(&self) -> &str {
        "fixture-faithful-executable/1"
    }

    fn run(&self, executable: &Executable, buffers: &mut [Vec<f64>]) -> Result<(), String> {
        independent_fixture_run(executable, buffers)
    }
}

struct BiasedRunner;

impl LocalExecutableRunner for BiasedRunner {
    fn identity(&self) -> &str {
        "fixture-biased-executable/1"
    }

    fn run(&self, executable: &Executable, buffers: &mut [Vec<f64>]) -> Result<(), String> {
        independent_fixture_run(executable, buffers)?;
        for (index, operand) in executable.kernel().as_kernel().operands.iter().enumerate() {
            if operand.access.can_write() {
                buffers[index][operand.region.offset] += 1.0;
            }
        }
        Ok(())
    }
}

struct RefusingRunner;

impl LocalExecutableRunner for RefusingRunner {
    fn identity(&self) -> &str {
        "fixture-refusing-executable/1"
    }

    fn run(&self, _executable: &Executable, _buffers: &mut [Vec<f64>]) -> Result<(), String> {
        Err("backend refused fixture".into())
    }
}

struct NonFiniteRunner;

impl LocalExecutableRunner for NonFiniteRunner {
    fn identity(&self) -> &str {
        "fixture-nonfinite-executable/1"
    }

    fn run(&self, executable: &Executable, buffers: &mut [Vec<f64>]) -> Result<(), String> {
        independent_fixture_run(executable, buffers)?;
        let output = executable
            .kernel()
            .as_kernel()
            .operands
            .iter()
            .position(|operand| operand.access.can_write())
            .unwrap();
        buffers[output][0] = f64::NAN;
        Ok(())
    }
}

struct SelfComparisonRunner;

impl LocalExecutableRunner for SelfComparisonRunner {
    fn identity(&self) -> &str {
        "malleus-interpreter/reference-v1"
    }

    fn run(&self, _executable: &Executable, _buffers: &mut [Vec<f64>]) -> Result<(), String> {
        unreachable!("self-comparison must be refused before execution")
    }
}

fn independent_fixture_run(
    executable: &Executable,
    buffers: &mut [Vec<f64>],
) -> Result<(), String> {
    let kernel = executable.kernel().as_kernel();
    if kernel.iteration_domain.rank() != 1 || kernel.numeric_policy.scalar_type != ScalarType::F64 {
        return Err("fixture runner supports only rank-one f64 kernels".into());
    }
    for coordinate in 0..kernel.iteration_domain.extents[0] {
        let mut locals = Vec::new();
        for statement in &kernel.body.statements {
            match statement {
                Statement::Let { value, .. } => {
                    locals.push(fixture_eval(value, kernel, buffers, coordinate, &locals)?);
                }
                Statement::Store { operand, value } => {
                    let value = fixture_eval(value, kernel, buffers, coordinate, &locals)?;
                    let offset = fixture_offset(kernel, *operand, coordinate)?;
                    match kernel.operands[operand.index()].access {
                        AccessMode::Write | AccessMode::ReadWrite => {
                            buffers[operand.index()][offset] = value;
                        }
                        AccessMode::Reduce(ReductionOp::Add) => {
                            buffers[operand.index()][offset] += value;
                        }
                        AccessMode::Reduce(ReductionOp::Multiply) => {
                            buffers[operand.index()][offset] *= value;
                        }
                        AccessMode::Reduce(ReductionOp::Min) => {
                            buffers[operand.index()][offset] =
                                buffers[operand.index()][offset].min(value);
                        }
                        AccessMode::Reduce(ReductionOp::Max) => {
                            buffers[operand.index()][offset] =
                                buffers[operand.index()][offset].max(value);
                        }
                        AccessMode::Read => {
                            return Err("fixture attempted a read-only store".into());
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn fixture_eval(
    expression: &ScalarExpr,
    kernel: &StructuredKernel,
    buffers: &[Vec<f64>],
    coordinate: usize,
    locals: &[f64],
) -> Result<f64, String> {
    Ok(match expression {
        ScalarExpr::Constant(value) => *value,
        ScalarExpr::Index(_) => coordinate as f64,
        ScalarExpr::Load(operand) => {
            buffers[operand.index()][fixture_offset(kernel, *operand, coordinate)?]
        }
        ScalarExpr::Local(local) => locals[local.index()],
        ScalarExpr::Binary { op, lhs, rhs } => {
            let lhs = fixture_eval(lhs, kernel, buffers, coordinate, locals)?;
            let rhs = fixture_eval(rhs, kernel, buffers, coordinate, locals)?;
            match op {
                BinaryOp::Add => lhs + rhs,
                BinaryOp::Sub => lhs - rhs,
                BinaryOp::Mul => lhs * rhs,
                BinaryOp::Div => lhs / rhs,
                BinaryOp::Pow => lhs.powf(rhs),
                BinaryOp::Min => lhs.min(rhs),
                BinaryOp::Max => lhs.max(rhs),
                BinaryOp::Atan2 => lhs.atan2(rhs),
            }
        }
        ScalarExpr::Unary { .. } | ScalarExpr::Select { .. } => {
            return Err("fixture runner does not support this expression".into());
        }
    })
}

fn fixture_offset(
    kernel: &StructuredKernel,
    operand: OperandId,
    coordinate: usize,
) -> Result<usize, String> {
    let map = kernel
        .indexing_maps
        .iter()
        .find(|map| map.operand == operand)
        .ok_or_else(|| "fixture operand has no map".to_owned())?;
    let relative = match map.results.as_slice() {
        [] => 0,
        [index] => {
            let value = index.terms.iter().fold(index.constant, |value, term| {
                value + term.coefficient * coordinate as isize
            });
            usize::try_from(value).map_err(|_| "fixture index is negative")?
        }
        _ => return Err("fixture runner supports only scalar and rank-one maps".into()),
    };
    Ok(kernel.operands[operand.index()].region.offset + relative)
}

struct ShrinkingRunner;

impl LocalExecutableRunner for ShrinkingRunner {
    fn identity(&self) -> &str {
        "fixture-shrinking-executable/1"
    }

    fn run(&self, _executable: &Executable, buffers: &mut [Vec<f64>]) -> Result<(), String> {
        buffers[0].clear();
        Ok(())
    }
}

#[test]
fn generic_campaign_checks_primal_jvp_vjp_parameter_and_backend_agreement() {
    let mut module = quadratic_module();
    let mut second_kernel = module.kernels[0].clone();
    second_kernel.name = "quadratic-second".into();
    module.kernels.push(second_kernel);
    let mut second_case = case();
    second_case.kernel = "quadratic-second".into();
    let report =
        run_local_differential_campaign(&module, &[case(), second_case], &FaithfulRunner).unwrap();
    assert!(report.passed(), "{report:#?}");
    assert_eq!(report.checks.len(), 14);
    for kind in [
        LocalCheckKind::PrimalInterpreterVsExecutable,
        LocalCheckKind::JvpInterpreterVsExecutable,
        LocalCheckKind::JvpCenteredDifference,
        LocalCheckKind::VjpInterpreterVsExecutable,
        LocalCheckKind::JvpVjpAdjointIdentity,
        LocalCheckKind::ParameterJvpInterpreterVsExecutable,
        LocalCheckKind::ParameterJvpCenteredDifference,
    ] {
        assert!(report.checks.iter().any(|check| check.kind == kind));
    }
}

#[test]
fn executable_mismatch_and_backend_refusal_remain_distinct_outcomes() {
    let report =
        run_local_differential_campaign(&quadratic_module(), &[case()], &BiasedRunner).unwrap();
    assert!(!report.passed());
    assert!(report.checks.iter().any(|check| {
        check.kind == LocalCheckKind::PrimalInterpreterVsExecutable && !check.passed
    }));

    let error = run_local_differential_campaign(&quadratic_module(), &[case()], &RefusingRunner)
        .unwrap_err();
    assert!(matches!(error, CampaignError::Execution { .. }));
    assert!(error.to_string().contains("backend refused fixture"));

    let report =
        run_local_differential_campaign(&quadratic_module(), &[case()], &NonFiniteRunner).unwrap();
    assert!(!report.passed());
    assert!(report.checks[0].max_absolute_error.is_infinite());

    let error = run_local_differential_campaign(&quadratic_module(), &[case()], &ShrinkingRunner)
        .unwrap_err();
    assert!(error.to_string().contains("shortened operand buffer"));
}

#[test]
fn retained_f32_policy_mutation_is_detected() {
    let mut sensitive = case();
    sensitive.primal_buffers[0] = vec![16_777_217.0, 1.0, 2.0];
    sensitive.tolerance = ComparisonTolerance::new(0.0, 0.0);
    let result = check_numeric_policy_mutation(
        &quadratic_module(),
        &sensitive,
        NumericPolicyMutation::DemoteF64ToF32,
    )
    .unwrap();
    assert!(result.passed, "{result:#?}");
    assert!(result.max_absolute_error > 0.0);

    assert_eq!(
        check_numeric_policy_mutation(
            &quadratic_module(),
            &sensitive,
            NumericPolicyMutation::PromoteF32ToF64,
        )
        .unwrap_err(),
        CampaignError::MutationNotApplicable
    );

    let mut promoted_module = quadratic_module();
    promoted_module.kernels[0].numeric_policy.scalar_type = ScalarType::F32;
    let promoted = check_numeric_policy_mutation(
        &promoted_module,
        &sensitive,
        NumericPolicyMutation::PromoteF32ToF64,
    )
    .unwrap();
    assert!(promoted.passed, "{promoted:#?}");
    assert!(promoted.max_absolute_error > 0.0);

    let mut schedule_defined = quadratic_module();
    schedule_defined.kernels[0].numeric_policy.reduction_order =
        malleus::ReductionOrder::ScheduleDefined;
    let demoted = check_numeric_policy_mutation(
        &schedule_defined,
        &sensitive,
        NumericPolicyMutation::DemoteF64ToF32,
    )
    .unwrap();
    assert!(demoted.passed, "{demoted:#?}");

    schedule_defined.kernels[0].numeric_policy.scalar_type = ScalarType::F32;
    let promoted = check_numeric_policy_mutation(
        &schedule_defined,
        &sensitive,
        NumericPolicyMutation::PromoteF32ToF64,
    )
    .unwrap();
    assert!(promoted.passed, "{promoted:#?}");
}

#[test]
fn reduction_order_mutation_executes_an_alternate_schedule() {
    let result = check_numeric_policy_mutation(
        &reduction_order_module(),
        &reduction_order_case(),
        NumericPolicyMutation::ToggleReductionOrder,
    )
    .unwrap();
    assert!(result.passed, "{result:#?}");
    assert_eq!(result.max_absolute_error, 1.0);
}

#[test]
fn campaign_refuses_missing_coverage_and_ambiguous_operand_roles() {
    let error =
        run_local_differential_campaign(&quadratic_module(), &[], &FaithfulRunner).unwrap_err();
    assert!(error.to_string().contains("every module kernel"));

    let empty = StructuredModule {
        name: "empty".into(),
        kernels: vec![],
    };
    let error = run_local_differential_campaign(&empty, &[], &FaithfulRunner).unwrap_err();
    assert!(error.to_string().contains("at least one kernel"));

    let mut overlapping = case();
    overlapping.parameter_directions[0].operand = OperandId::new(0);
    overlapping.parameter_directions[0].values = vec![1.0; 3];
    let error =
        run_local_differential_campaign(&quadratic_module(), &[overlapping], &FaithfulRunner)
            .unwrap_err();
    assert!(error.to_string().contains("must be disjoint"));

    let mut missing_seed = case();
    missing_seed.dependent_seeds.clear();
    let error =
        run_local_differential_campaign(&quadratic_module(), &[missing_seed], &FaithfulRunner)
            .unwrap_err();
    assert!(error.to_string().contains("dependent seed"));

    let mut incomplete_multi_output = case();
    incomplete_multi_output.primal_buffers.push(vec![0.0]);
    let error = run_local_differential_campaign(
        &two_output_module(),
        &[incomplete_multi_output],
        &FaithfulRunner,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("every writable operand exactly once")
    );

    let mut short_buffer = case();
    short_buffer.primal_buffers[0].pop();
    let error =
        run_local_differential_campaign(&quadratic_module(), &[short_buffer], &FaithfulRunner)
            .unwrap_err();
    assert!(error.to_string().contains("invalid buffer"));

    let mut oversized = case();
    oversized.primal_buffers[0].push(99.0);
    oversized.state_directions[0].values.push(0.0);
    assert!(
        run_local_differential_campaign(&quadratic_module(), &[oversized], &FaithfulRunner)
            .unwrap()
            .passed()
    );
}

#[test]
fn campaign_refuses_reference_self_comparison_and_unversioned_identity() {
    let error =
        run_local_differential_campaign(&quadratic_module(), &[case()], &SelfComparisonRunner)
            .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot be its own executable candidate")
    );

    struct UnversionedRunner;
    impl LocalExecutableRunner for UnversionedRunner {
        fn identity(&self) -> &str {
            "anonymous-backend"
        }
        fn run(&self, _: &Executable, _: &mut [Vec<f64>]) -> Result<(), String> {
            unreachable!("invalid identity must be refused before execution")
        }
    }
    let error = run_local_differential_campaign(&quadratic_module(), &[case()], &UnversionedRunner)
        .unwrap_err();
    assert!(error.to_string().contains("implementation/version"));
}
