//! Reusable local differential and backend-comparison campaigns.
//!
//! These checks operate only on caller-supplied buffers for one validated
//! [`StructuredModule`](crate::StructuredModule). They do not interpret scientific meaning,
//! traverse meshes, assemble global operators, solve systems, or retain simulation history.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::{
    BufferBinding, DerivativeMode, DerivativeProduct, DerivativeRequest, Executable, Interpreter,
    KernelSchedule, NumericPolicy, OperandId, ReductionOrder, ScalarType, StructuredKernel,
    StructuredModule, differentiate, validate, validate_module,
};

const REFERENCE_IMPLEMENTATION: &str = "malleus-interpreter/reference-v1";

/// Backend execution boundary used by local interpreter-vs-executable checks.
pub trait LocalExecutableRunner {
    /// Stable diagnostic identity for the executable implementation under test.
    fn identity(&self) -> &str;

    /// Execute one already validated and scheduled local kernel.
    fn run(&self, executable: &Executable, buffers: &mut [Vec<f64>]) -> Result<(), String>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComparisonTolerance {
    pub absolute: f64,
    pub relative: f64,
}

impl ComparisonTolerance {
    pub const fn new(absolute: f64, relative: f64) -> Self {
        Self { absolute, relative }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OperandValues {
    pub operand: OperandId,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalDifferentialCase {
    pub kernel: String,
    /// One buffer per primal operand, in operand-id order.
    pub primal_buffers: Vec<Vec<f64>>,
    /// Directions for state-like independent operands.
    pub state_directions: Vec<OperandValues>,
    /// Directions checked again as a parameter-only JVP campaign.
    pub parameter_directions: Vec<OperandValues>,
    /// Cotangent seeds for every dependent operand.
    pub dependent_seeds: Vec<OperandValues>,
    pub centered_step: f64,
    pub tolerance: ComparisonTolerance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCheckKind {
    PrimalInterpreterVsExecutable,
    JvpInterpreterVsExecutable,
    JvpCenteredDifference,
    VjpInterpreterVsExecutable,
    JvpVjpAdjointIdentity,
    ParameterJvpInterpreterVsExecutable,
    ParameterJvpCenteredDifference,
    NumericPolicyMutation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalCheckResult {
    pub kernel: String,
    pub kind: LocalCheckKind,
    pub reference: String,
    pub candidate: String,
    pub max_absolute_error: f64,
    pub max_relative_error: f64,
    pub tolerance: ComparisonTolerance,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalCampaignReport {
    pub module: String,
    pub runner: String,
    pub checks: Vec<LocalCheckResult>,
}

impl LocalCampaignReport {
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|check| check.passed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericPolicyMutation {
    DemoteF64ToF32,
    PromoteF32ToF64,
    ToggleReductionOrder,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CampaignError {
    InvalidModule(crate::ValidationError),
    InvalidCase(String),
    Differentiation(crate::DifferentiationError),
    Execution { runner: String, message: String },
    MutationNotApplicable,
}

impl fmt::Display for CampaignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModule(error) => write!(f, "invalid campaign module: {error}"),
            Self::InvalidCase(message) => write!(f, "invalid local campaign case: {message}"),
            Self::Differentiation(error) => {
                write!(f, "local differential campaign failed: {error}")
            }
            Self::Execution { runner, message } => {
                write!(f, "local executable runner {runner} failed: {message}")
            }
            Self::MutationNotApplicable => write!(f, "numeric-policy mutation is not applicable"),
        }
    }
}

impl Error for CampaignError {}

/// Run primal, JVP, VJP, parameter-JVP, finite-difference, adjoint, and
/// interpreter-vs-executable checks for every kernel in a structured module.
pub fn run_local_differential_campaign(
    module: &StructuredModule,
    cases: &[LocalDifferentialCase],
    runner: &impl LocalExecutableRunner,
) -> Result<LocalCampaignReport, CampaignError> {
    let validated = validate_module(module.clone()).map_err(CampaignError::InvalidModule)?;
    if module.kernels.is_empty() {
        return Err(invalid_case(
            "campaign module must contain at least one kernel",
        ));
    }
    let runner_identity = runner.identity();
    validate_runner_identity(runner_identity)?;
    let cases_by_kernel = complete_cases(module, cases)?;
    let mut checks = Vec::new();
    for validated_kernel in validated.kernels() {
        let kernel = validated_kernel.as_kernel();
        let case = cases_by_kernel[&kernel.name];
        validate_case(kernel, case)?;
        checks.extend(run_case(kernel, case, runner, runner_identity)?);
    }
    Ok(LocalCampaignReport {
        module: validated.name().to_owned(),
        runner: runner_identity.to_owned(),
        checks,
    })
}

/// Execute the same primal buffers before and after one retained numeric-policy mutation.
/// A passing result means the mutation was detected outside the declared tolerance.
pub fn check_numeric_policy_mutation(
    module: &StructuredModule,
    case: &LocalDifferentialCase,
    mutation: NumericPolicyMutation,
) -> Result<LocalCheckResult, CampaignError> {
    validate_module(module.clone()).map_err(CampaignError::InvalidModule)?;
    let kernel = module
        .kernels
        .iter()
        .find(|kernel| kernel.name == case.kernel)
        .ok_or_else(|| invalid_case("mutation case references an unknown kernel"))?;
    validate_case(kernel, case)?;
    let mut mutated = kernel.clone();
    mutate_policy(&mut mutated.numeric_policy, mutation)?;
    let changes_reduction_order = mutation == NumericPolicyMutation::ToggleReductionOrder;
    let baseline = mutation_executable(kernel.clone(), changes_reduction_order)?;
    let mutated = mutation_executable(mutated, changes_reduction_order)?;
    let mut expected = case.primal_buffers.clone();
    let mut actual = case.primal_buffers.clone();
    run_interpreter(&baseline, &mut expected).map_err(|error| CampaignError::Execution {
        runner: "malleus-interpreter-baseline".into(),
        message: error.to_string(),
    })?;
    run_interpreter(&mutated, &mut actual).map_err(|error| CampaignError::Execution {
        runner: "malleus-interpreter-mutated".into(),
        message: error.to_string(),
    })?;
    let comparison = compare_buffers(kernel, &expected, &actual, case.tolerance);
    Ok(LocalCheckResult {
        kernel: kernel.name.clone(),
        kind: LocalCheckKind::NumericPolicyMutation,
        reference: format!("{:?}", kernel.numeric_policy),
        candidate: format!("{:?}", mutated.kernel().as_kernel().numeric_policy),
        max_absolute_error: comparison.maximum_absolute,
        max_relative_error: comparison.maximum_relative,
        tolerance: case.tolerance,
        passed: !comparison.passed,
    })
}

fn run_case(
    kernel: &StructuredKernel,
    case: &LocalDifferentialCase,
    runner: &impl LocalExecutableRunner,
    runner_identity: &str,
) -> Result<Vec<LocalCheckResult>, CampaignError> {
    let mut checks = Vec::new();
    let primal =
        Executable::reference(validate(kernel.clone()).map_err(CampaignError::InvalidModule)?);
    checks.push(compare_executable(
        kernel,
        &primal,
        &case.primal_buffers,
        runner,
        runner_identity,
        LocalCheckKind::PrimalInterpreterVsExecutable,
        case.tolerance,
    )?);

    let directions = case
        .state_directions
        .iter()
        .chain(&case.parameter_directions)
        .cloned()
        .collect::<Vec<_>>();
    let independents = directions
        .iter()
        .map(|value| value.operand)
        .collect::<Vec<_>>();
    let dependents = case
        .dependent_seeds
        .iter()
        .map(|value| value.operand)
        .collect::<Vec<_>>();
    let jvp = differentiate(
        kernel,
        &DerivativeRequest {
            mode: DerivativeMode::Jvp,
            independent_operands: independents.clone(),
            dependent_operands: dependents.clone(),
        },
    )
    .map_err(CampaignError::Differentiation)?;
    let jvp_buffers = derivative_buffers(kernel, case, &jvp, &directions, &[])?;
    let jvp_executable =
        Executable::reference(validate(jvp.kernel.clone()).map_err(CampaignError::InvalidModule)?);
    checks.push(compare_executable(
        &jvp.kernel,
        &jvp_executable,
        &jvp_buffers,
        runner,
        runner_identity,
        LocalCheckKind::JvpInterpreterVsExecutable,
        case.tolerance,
    )?);
    let mut jvp_reference = jvp_buffers;
    run_interpreter(&jvp_executable, &mut jvp_reference).map_err(|error| {
        CampaignError::Execution {
            runner: REFERENCE_IMPLEMENTATION.into(),
            message: error.to_string(),
        }
    })?;
    checks.push(centered_difference_check(
        kernel,
        case,
        &jvp,
        &jvp_reference,
        &directions,
        LocalCheckKind::JvpCenteredDifference,
    )?);

    let vjp = differentiate(
        kernel,
        &DerivativeRequest {
            mode: DerivativeMode::Vjp,
            independent_operands: independents,
            dependent_operands: dependents,
        },
    )
    .map_err(CampaignError::Differentiation)?;
    let vjp_buffers = derivative_buffers(kernel, case, &vjp, &[], &case.dependent_seeds)?;
    let vjp_executable =
        Executable::reference(validate(vjp.kernel.clone()).map_err(CampaignError::InvalidModule)?);
    checks.push(compare_executable(
        &vjp.kernel,
        &vjp_executable,
        &vjp_buffers,
        runner,
        runner_identity,
        LocalCheckKind::VjpInterpreterVsExecutable,
        case.tolerance,
    )?);
    let mut vjp_reference = vjp_buffers;
    run_interpreter(&vjp_executable, &mut vjp_reference).map_err(|error| {
        CampaignError::Execution {
            runner: REFERENCE_IMPLEMENTATION.into(),
            message: error.to_string(),
        }
    })?;
    checks.push(adjoint_check(
        kernel,
        case,
        &jvp,
        &jvp_reference,
        &vjp,
        &vjp_reference,
        &directions,
    ));

    if !case.parameter_directions.is_empty() {
        let parameter_jvp = differentiate(
            kernel,
            &DerivativeRequest {
                mode: DerivativeMode::Jvp,
                independent_operands: case
                    .parameter_directions
                    .iter()
                    .map(|value| value.operand)
                    .collect(),
                dependent_operands: case
                    .dependent_seeds
                    .iter()
                    .map(|value| value.operand)
                    .collect(),
            },
        )
        .map_err(CampaignError::Differentiation)?;
        let buffers = derivative_buffers(
            kernel,
            case,
            &parameter_jvp,
            &case.parameter_directions,
            &[],
        )?;
        let executable = Executable::reference(
            validate(parameter_jvp.kernel.clone()).map_err(CampaignError::InvalidModule)?,
        );
        checks.push(compare_executable(
            &parameter_jvp.kernel,
            &executable,
            &buffers,
            runner,
            runner_identity,
            LocalCheckKind::ParameterJvpInterpreterVsExecutable,
            case.tolerance,
        )?);
        let mut reference = buffers;
        run_interpreter(&executable, &mut reference).map_err(|error| CampaignError::Execution {
            runner: REFERENCE_IMPLEMENTATION.into(),
            message: error.to_string(),
        })?;
        checks.push(centered_difference_check(
            kernel,
            case,
            &parameter_jvp,
            &reference,
            &case.parameter_directions,
            LocalCheckKind::ParameterJvpCenteredDifference,
        )?);
    }
    Ok(checks)
}

fn complete_cases<'a>(
    module: &StructuredModule,
    cases: &'a [LocalDifferentialCase],
) -> Result<BTreeMap<String, &'a LocalDifferentialCase>, CampaignError> {
    let mut by_kernel = BTreeMap::new();
    for case in cases {
        if case.kernel.trim().is_empty() || by_kernel.insert(case.kernel.clone(), case).is_some() {
            return Err(invalid_case(
                "case kernel names must be nonempty and unique",
            ));
        }
    }
    let expected = module
        .kernels
        .iter()
        .map(|kernel| kernel.name.clone())
        .collect::<BTreeSet<_>>();
    let actual = by_kernel.keys().cloned().collect::<BTreeSet<_>>();
    if expected != actual {
        return Err(invalid_case(
            "campaign cases must cover every module kernel exactly once",
        ));
    }
    Ok(by_kernel)
}

fn validate_case(
    kernel: &StructuredKernel,
    case: &LocalDifferentialCase,
) -> Result<(), CampaignError> {
    if case.kernel != kernel.name
        || !case.centered_step.is_finite()
        || case.centered_step <= 0.0
        || !valid_tolerance(case.tolerance)
        || case.primal_buffers.len() != kernel.operands.len()
    {
        return Err(invalid_case(
            "kernel, positive finite step, tolerance, and one primal buffer per operand are required",
        ));
    }
    for (index, (operand, buffer)) in kernel.operands.iter().zip(&case.primal_buffers).enumerate() {
        let required = operand
            .region
            .offset
            .checked_add(operand.region.length)
            .ok_or_else(|| invalid_case("operand buffer length overflow"))?;
        if buffer.len() < required || buffer.iter().any(|value| !value.is_finite()) {
            return Err(invalid_case(format!(
                "primal operand {index} has an invalid buffer"
            )));
        }
    }
    let states = validate_vectors(kernel, &case.state_directions, true, "state direction")?;
    let parameters = validate_vectors(
        kernel,
        &case.parameter_directions,
        true,
        "parameter direction",
    )?;
    if states.is_empty() && parameters.is_empty() {
        return Err(invalid_case(
            "at least one state or parameter direction is required",
        ));
    }
    if !states.is_disjoint(&parameters) {
        return Err(invalid_case(
            "state and parameter direction operands must be disjoint",
        ));
    }
    let dependents = validate_vectors(kernel, &case.dependent_seeds, false, "dependent seed")?;
    let expected_dependents = kernel
        .operands
        .iter()
        .enumerate()
        .filter(|(_, operand)| operand.access.can_write())
        .map(|(index, _)| OperandId::new(index))
        .collect::<BTreeSet<_>>();
    if dependents != expected_dependents {
        return Err(invalid_case(
            "dependent seeds must cover every writable operand exactly once",
        ));
    }
    Ok(())
}

fn validate_runner_identity(identity: &str) -> Result<(), CampaignError> {
    let versioned = identity
        .rsplit_once('/')
        .is_some_and(|(implementation, version)| {
            !implementation.trim().is_empty()
                && !version.trim().is_empty()
                && identity.trim() == identity
        });
    if !versioned {
        return Err(invalid_case(
            "runner identity must name a versioned implementation as implementation/version",
        ));
    }
    if identity == REFERENCE_IMPLEMENTATION {
        return Err(invalid_case(
            "the reference interpreter cannot be its own executable candidate",
        ));
    }
    Ok(())
}

fn validate_vectors(
    kernel: &StructuredKernel,
    vectors: &[OperandValues],
    readable: bool,
    kind: &str,
) -> Result<BTreeSet<OperandId>, CampaignError> {
    let mut seen = BTreeSet::new();
    for vector in vectors {
        let Some(operand) = kernel.operands.get(vector.operand.index()) else {
            return Err(invalid_case(format!(
                "{kind} references an invalid operand"
            )));
        };
        let permitted = if readable {
            operand.access.can_read()
        } else {
            operand.access.can_write()
        };
        let required = operand
            .region
            .offset
            .checked_add(operand.region.length)
            .ok_or_else(|| invalid_case("vector buffer length overflow"))?;
        if !permitted
            || !seen.insert(vector.operand)
            || vector.values.len() < required
            || vector.values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid_case(format!("{kind} is invalid or duplicated")));
        }
    }
    Ok(seen)
}

fn compare_executable(
    kernel: &StructuredKernel,
    executable: &Executable,
    initial: &[Vec<f64>],
    runner: &impl LocalExecutableRunner,
    runner_identity: &str,
    kind: LocalCheckKind,
    tolerance: ComparisonTolerance,
) -> Result<LocalCheckResult, CampaignError> {
    let mut expected = initial.to_vec();
    let mut actual = initial.to_vec();
    run_interpreter(executable, &mut expected).map_err(|error| CampaignError::Execution {
        runner: REFERENCE_IMPLEMENTATION.into(),
        message: error.to_string(),
    })?;
    runner
        .run(executable, &mut actual)
        .map_err(|message| CampaignError::Execution {
            runner: runner_identity.to_owned(),
            message,
        })?;
    validate_returned_buffers(kernel, &actual).map_err(|message| CampaignError::Execution {
        runner: runner_identity.to_owned(),
        message,
    })?;
    let comparison = compare_buffers(kernel, &expected, &actual, tolerance);
    Ok(check_result(
        kernel,
        kind,
        REFERENCE_IMPLEMENTATION,
        runner_identity,
        comparison,
        tolerance,
    ))
}

fn derivative_buffers(
    primal: &StructuredKernel,
    case: &LocalDifferentialCase,
    product: &DerivativeProduct,
    independent_values: &[OperandValues],
    dependent_values: &[OperandValues],
) -> Result<Vec<Vec<f64>>, CampaignError> {
    let mut buffers = product
        .kernel
        .operands
        .iter()
        .map(|operand| vec![0.0; operand.region.offset + operand.region.length])
        .collect::<Vec<_>>();
    let mut readable_index = 0;
    for (primal_id, operand) in primal.operands.iter().enumerate() {
        if operand.access.can_read() {
            copy_binding(
                &mut buffers[readable_index],
                &case.primal_buffers[primal_id],
            );
            readable_index += 1;
        }
    }
    if product.mode == DerivativeMode::Jvp {
        for pair in &product.independent_operands {
            let values = independent_values
                .iter()
                .find(|values| values.operand == pair.primal)
                .ok_or_else(|| invalid_case("missing derivative independent vector"))?;
            copy_binding(&mut buffers[pair.derivative.index()], &values.values);
        }
    }
    for pair in &product.dependent_operands {
        if product.mode == DerivativeMode::Vjp {
            let values = dependent_values
                .iter()
                .find(|values| values.operand == pair.primal)
                .ok_or_else(|| invalid_case("missing VJP dependent seed"))?;
            copy_binding(&mut buffers[pair.derivative.index()], &values.values);
        }
    }
    Ok(buffers)
}

fn centered_difference_check(
    kernel: &StructuredKernel,
    case: &LocalDifferentialCase,
    product: &DerivativeProduct,
    derivative_buffers: &[Vec<f64>],
    directions: &[OperandValues],
    kind: LocalCheckKind,
) -> Result<LocalCheckResult, CampaignError> {
    let mut plus = case.primal_buffers.clone();
    let mut minus = case.primal_buffers.clone();
    for direction in directions {
        let operand = &kernel.operands[direction.operand.index()];
        for index in operand.region.offset..operand.region.offset + operand.region.length {
            plus[direction.operand.index()][index] += case.centered_step * direction.values[index];
            minus[direction.operand.index()][index] -= case.centered_step * direction.values[index];
        }
    }
    let executable =
        Executable::reference(validate(kernel.clone()).map_err(CampaignError::InvalidModule)?);
    run_interpreter(&executable, &mut plus).map_err(|error| CampaignError::Execution {
        runner: REFERENCE_IMPLEMENTATION.into(),
        message: error.to_string(),
    })?;
    run_interpreter(&executable, &mut minus).map_err(|error| CampaignError::Execution {
        runner: REFERENCE_IMPLEMENTATION.into(),
        message: error.to_string(),
    })?;
    let mut comparison = Comparison::passing();
    for pair in &product.dependent_operands {
        let primal_operand = &kernel.operands[pair.primal.index()];
        let derivative_operand = &product.kernel.operands[pair.derivative.index()];
        for offset in 0..primal_operand.region.length {
            let primal_index = primal_operand.region.offset + offset;
            let derivative_index = derivative_operand.region.offset + offset;
            let finite_difference = (plus[pair.primal.index()][primal_index]
                - minus[pair.primal.index()][primal_index])
                / (2.0 * case.centered_step);
            accumulate_error(
                derivative_buffers[pair.derivative.index()][derivative_index],
                finite_difference,
                case.tolerance,
                &mut comparison,
            );
        }
    }
    Ok(check_result(
        kernel,
        kind,
        "centered-primal-difference",
        "structured-jvp",
        comparison,
        case.tolerance,
    ))
}

fn adjoint_check(
    kernel: &StructuredKernel,
    case: &LocalDifferentialCase,
    jvp: &DerivativeProduct,
    jvp_buffers: &[Vec<f64>],
    vjp: &DerivativeProduct,
    vjp_buffers: &[Vec<f64>],
    directions: &[OperandValues],
) -> LocalCheckResult {
    let mut forward = 0.0;
    for pair in &jvp.dependent_operands {
        let derivative = &jvp.kernel.operands[pair.derivative.index()];
        let seed = case
            .dependent_seeds
            .iter()
            .find(|seed| seed.operand == pair.primal)
            .expect("validated dependent seed");
        for offset in 0..derivative.region.length {
            forward += jvp_buffers[pair.derivative.index()][derivative.region.offset + offset]
                * seed.values[kernel.operands[pair.primal.index()].region.offset + offset];
        }
    }
    let mut reverse = 0.0;
    for pair in &vjp.independent_operands {
        let derivative = &vjp.kernel.operands[pair.derivative.index()];
        let direction = directions
            .iter()
            .find(|direction| direction.operand == pair.primal)
            .expect("validated independent direction");
        for offset in 0..derivative.region.length {
            reverse += vjp_buffers[pair.derivative.index()][derivative.region.offset + offset]
                * direction.values[kernel.operands[pair.primal.index()].region.offset + offset];
        }
    }
    let mut comparison = Comparison::passing();
    accumulate_error(forward, reverse, case.tolerance, &mut comparison);
    check_result(
        kernel,
        LocalCheckKind::JvpVjpAdjointIdentity,
        "dot(seed,jvp)",
        "dot(vjp,direction)",
        comparison,
        case.tolerance,
    )
}

fn compare_buffers(
    kernel: &StructuredKernel,
    expected: &[Vec<f64>],
    actual: &[Vec<f64>],
    tolerance: ComparisonTolerance,
) -> Comparison {
    let mut comparison = Comparison::passing();
    for (index, operand) in kernel.operands.iter().enumerate() {
        for offset in operand.region.offset..operand.region.offset + operand.region.length {
            accumulate_error(
                expected[index][offset],
                actual[index][offset],
                tolerance,
                &mut comparison,
            );
        }
    }
    comparison
}

#[derive(Clone, Copy)]
struct Comparison {
    maximum_absolute: f64,
    maximum_relative: f64,
    passed: bool,
}

impl Comparison {
    const fn passing() -> Self {
        Self {
            maximum_absolute: 0.0,
            maximum_relative: 0.0,
            passed: true,
        }
    }
}

fn accumulate_error(
    expected: f64,
    actual: f64,
    tolerance: ComparisonTolerance,
    comparison: &mut Comparison,
) {
    if !expected.is_finite() || !actual.is_finite() {
        comparison.maximum_absolute = f64::INFINITY;
        comparison.maximum_relative = f64::INFINITY;
        comparison.passed = false;
        return;
    }
    let absolute = (expected - actual).abs();
    let scale = expected.abs().max(actual.abs());
    let relative = if scale == 0.0 { 0.0 } else { absolute / scale };
    comparison.maximum_absolute = comparison.maximum_absolute.max(absolute);
    comparison.maximum_relative = comparison.maximum_relative.max(relative);
    comparison.passed &= absolute <= tolerance.absolute || relative <= tolerance.relative;
}

fn copy_binding(target: &mut [f64], source: &[f64]) {
    target.copy_from_slice(&source[..target.len()]);
}

fn check_result(
    kernel: &StructuredKernel,
    kind: LocalCheckKind,
    reference: &str,
    candidate: &str,
    comparison: Comparison,
    tolerance: ComparisonTolerance,
) -> LocalCheckResult {
    LocalCheckResult {
        kernel: kernel.name.clone(),
        kind,
        reference: reference.into(),
        candidate: candidate.into(),
        max_absolute_error: comparison.maximum_absolute,
        max_relative_error: comparison.maximum_relative,
        tolerance,
        passed: comparison.passed,
    }
}

fn valid_tolerance(tolerance: ComparisonTolerance) -> bool {
    tolerance.absolute.is_finite()
        && tolerance.relative.is_finite()
        && tolerance.absolute >= 0.0
        && tolerance.relative >= 0.0
}

fn validate_returned_buffers(
    kernel: &StructuredKernel,
    buffers: &[Vec<f64>],
) -> Result<(), String> {
    if buffers.len() != kernel.operands.len() {
        return Err("runner changed the operand-buffer count".into());
    }
    for (index, (operand, buffer)) in kernel.operands.iter().zip(buffers).enumerate() {
        let required = operand
            .region
            .offset
            .checked_add(operand.region.length)
            .ok_or_else(|| format!("runner operand {index} length overflow"))?;
        if buffer.len() < required {
            return Err(format!("runner shortened operand buffer {index}"));
        }
    }
    Ok(())
}

fn mutate_policy(
    policy: &mut NumericPolicy,
    mutation: NumericPolicyMutation,
) -> Result<(), CampaignError> {
    match mutation {
        NumericPolicyMutation::DemoteF64ToF32 if policy.scalar_type == ScalarType::F64 => {
            policy.scalar_type = ScalarType::F32;
        }
        NumericPolicyMutation::PromoteF32ToF64 if policy.scalar_type == ScalarType::F32 => {
            policy.scalar_type = ScalarType::F64;
        }
        NumericPolicyMutation::ToggleReductionOrder => {
            policy.reduction_order = match policy.reduction_order {
                ReductionOrder::Canonical => ReductionOrder::ScheduleDefined,
                ReductionOrder::ScheduleDefined => ReductionOrder::Canonical,
            };
        }
        _ => return Err(CampaignError::MutationNotApplicable),
    }
    Ok(())
}

fn mutation_executable(
    kernel: StructuredKernel,
    use_alternate_reduction_schedule: bool,
) -> Result<Executable, CampaignError> {
    let schedule_defined = kernel.numeric_policy.reduction_order == ReductionOrder::ScheduleDefined;
    let validated = validate(kernel).map_err(CampaignError::InvalidModule)?;
    if !use_alternate_reduction_schedule || !schedule_defined {
        return Ok(Executable::reference(validated));
    }
    if validated.as_kernel().iteration_domain.rank() < 2
        || !validated
            .as_kernel()
            .operands
            .iter()
            .any(|operand| matches!(operand.access, crate::AccessMode::Reduce(_)))
    {
        return Err(CampaignError::MutationNotApplicable);
    }
    let mut schedule = KernelSchedule::canonical(&validated);
    schedule.loop_order.reverse();
    Executable::new(validated, schedule)
        .map_err(|error| invalid_case(format!("alternate reduction schedule is invalid: {error}")))
}

fn run_interpreter(
    executable: &Executable,
    buffers: &mut [Vec<f64>],
) -> Result<(), crate::ExecutionError> {
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(index, values)| BufferBinding::new(OperandId::new(index), values.as_mut_slice()))
        .collect::<Vec<_>>();
    Interpreter::run(executable, &mut bindings)
}

fn invalid_case(message: impl Into<String>) -> CampaignError {
    CampaignError::InvalidCase(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_or_relative_tolerance_is_applied_per_component() {
        let tolerance = ComparisonTolerance::new(0.5, 0.001);
        let mut comparison = Comparison::passing();
        // This component passes only its relative tolerance.
        accumulate_error(1000.0, 1001.0, tolerance, &mut comparison);
        // This component passes only its absolute tolerance.
        accumulate_error(0.0, 0.5, tolerance, &mut comparison);
        assert!(comparison.passed);
        assert_eq!(comparison.maximum_absolute, 1.0);
        assert_eq!(comparison.maximum_relative, 1.0);
    }
}
