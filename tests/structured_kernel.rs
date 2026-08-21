use malleus::{
    AccessMode, AxisId, BinaryOp, BufferBinding, Executable, ExecutableModule, IndexExpr,
    IndexingMap, Interpreter, IterationDomain, IteratorKind, KernelOperand, KernelRegion, LocalId,
    NumericPolicy, OperandId, ReductionOp, ScalarExpr, Statement, StructuredKernel,
    StructuredModule, ValidationError, validate, validate_module,
};

fn scale_kernel() -> StructuredKernel {
    let axis = AxisId::new(0);
    let input = OperandId::new(0);
    let output = OperandId::new(1);
    StructuredKernel {
        name: "scale".into(),
        iteration_domain: IterationDomain::new(vec![3]),
        iterators: vec![IteratorKind::Parallel],
        operands: vec![
            KernelOperand::tensor("input", vec![3], AccessMode::Read),
            KernelOperand::tensor("output", vec![3], AccessMode::Write),
        ],
        indexing_maps: vec![
            IndexingMap::new(input, vec![IndexExpr::axis(axis)]),
            IndexingMap::new(output, vec![IndexExpr::axis(axis)]),
        ],
        body: KernelRegion {
            statements: vec![
                Statement::Let {
                    local: LocalId::new(0),
                    value: ScalarExpr::binary(
                        BinaryOp::Mul,
                        ScalarExpr::Load(input),
                        ScalarExpr::Constant(2.0),
                    ),
                },
                Statement::Store {
                    operand: output,
                    value: ScalarExpr::Local(LocalId::new(0)),
                },
            ],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

#[test]
fn validates_compiles_and_interprets_a_structured_module() {
    let module = validate_module(StructuredModule {
        name: "fixture".into(),
        kernels: vec![scale_kernel()],
    })
    .unwrap();
    let executable = ExecutableModule::reference(module);
    let kernel = executable.kernel("scale").unwrap();

    let mut input = [1.0, -2.0, 4.5];
    let mut output = [0.0; 3];
    Interpreter::run(
        kernel,
        &mut [
            BufferBinding::new(OperandId::new(0), &mut input),
            BufferBinding::new(OperandId::new(1), &mut output),
        ],
    )
    .unwrap();

    assert_eq!(output, [2.0, -4.0, 9.0]);
}

#[test]
fn reduction_execution_is_canonical_and_deterministic() {
    let axis = AxisId::new(0);
    let input = OperandId::new(0);
    let sum = OperandId::new(1);
    let kernel = StructuredKernel {
        name: "sum".into(),
        iteration_domain: IterationDomain::new(vec![4]),
        iterators: vec![IteratorKind::Reduction],
        operands: vec![
            KernelOperand::tensor("input", vec![4], AccessMode::Read),
            KernelOperand::scalar("sum", AccessMode::Reduce(ReductionOp::Add)),
        ],
        indexing_maps: vec![
            IndexingMap::new(input, vec![IndexExpr::axis(axis)]),
            IndexingMap::scalar(sum),
        ],
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: sum,
                value: ScalarExpr::Load(input),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    };
    let executable = Executable::reference(validate(kernel).unwrap());
    let mut input = [0.5, 1.5, -2.0, 8.0];
    let mut output = [1.0];
    Interpreter::run(
        &executable,
        &mut [
            BufferBinding::new(input_id(), &mut input),
            BufferBinding::new(sum, &mut output),
        ],
    )
    .unwrap();
    assert_eq!(output, [9.0]);

    fn input_id() -> OperandId {
        OperandId::new(0)
    }
}

#[test]
fn rejects_use_before_definition() {
    let mut kernel = scale_kernel();
    kernel.body.statements[0] = Statement::Let {
        local: LocalId::new(1),
        value: ScalarExpr::Constant(1.0),
    };
    assert_eq!(
        validate(kernel).unwrap_err(),
        ValidationError::InvalidLocal {
            expected: 0,
            actual: 1
        }
    );
}

fn quadratic_reduction_kernel() -> StructuredKernel {
    let axis = AxisId::new(0);
    let x = OperandId::new(0);
    let parameter = OperandId::new(1);
    let output = OperandId::new(2);
    StructuredKernel {
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
    }
}

#[test]
fn structured_jvp_matches_directional_finite_difference() {
    let primal = quadratic_reduction_kernel();
    validate(primal.clone()).unwrap();
    let product = malleus::differentiate(
        &primal,
        &malleus::DerivativeRequest {
            mode: malleus::DerivativeMode::Jvp,
            independent_operands: vec![OperandId::new(0)],
            dependent_operands: vec![OperandId::new(2)],
        },
    )
    .unwrap();
    let executable = Executable::reference(validate(product.kernel).unwrap());
    let mut x = [1.5, -2.0, 0.25];
    let mut parameter = [3.0];
    let mut direction = [-0.5, 2.0, 4.0];
    let mut tangent = [0.0];
    Interpreter::run(
        &executable,
        &mut [
            BufferBinding::new(OperandId::new(0), &mut x),
            BufferBinding::new(OperandId::new(1), &mut parameter),
            BufferBinding::new(OperandId::new(2), &mut direction),
            BufferBinding::new(OperandId::new(3), &mut tangent),
        ],
    )
    .unwrap();

    let epsilon = 1.0e-7;
    let evaluate = |values: [f64; 3]| 3.0 * values.iter().map(|value| value * value).sum::<f64>();
    let plus = std::array::from_fn(|index| x[index] + epsilon * direction[index]);
    let minus = std::array::from_fn(|index| x[index] - epsilon * direction[index]);
    let finite_difference = (evaluate(plus) - evaluate(minus)) / (2.0 * epsilon);
    assert!((tangent[0] - finite_difference).abs() < 2.0e-7);
}

#[test]
fn structured_vjp_satisfies_adjoint_dot_product() {
    let primal = quadratic_reduction_kernel();
    let product = malleus::differentiate(
        &primal,
        &malleus::DerivativeRequest {
            mode: malleus::DerivativeMode::Vjp,
            independent_operands: vec![OperandId::new(0), OperandId::new(1)],
            dependent_operands: vec![OperandId::new(2)],
        },
    )
    .unwrap();
    let executable = Executable::reference(validate(product.kernel).unwrap());
    let mut x = [1.5, -2.0, 0.25];
    let mut parameter = [3.0];
    let mut seed = [-1.25];
    let mut x_cotangent = [0.0; 3];
    let mut parameter_cotangent = [0.0];
    Interpreter::run(
        &executable,
        &mut [
            BufferBinding::new(OperandId::new(0), &mut x),
            BufferBinding::new(OperandId::new(1), &mut parameter),
            BufferBinding::new(OperandId::new(2), &mut seed),
            BufferBinding::new(OperandId::new(3), &mut x_cotangent),
            BufferBinding::new(OperandId::new(4), &mut parameter_cotangent),
        ],
    )
    .unwrap();

    let dx = [-0.5, 2.0, 4.0];
    let dp = 0.75;
    let jvp = 2.0 * parameter[0] * x.iter().zip(dx).map(|(x, dx)| x * dx).sum::<f64>()
        + x.iter().map(|x| x * x).sum::<f64>() * dp;
    let reverse_dot = x_cotangent
        .iter()
        .zip(dx)
        .map(|(cotangent, direction)| cotangent * direction)
        .sum::<f64>()
        + parameter_cotangent[0] * dp;
    assert!((seed[0] * jvp - reverse_dot).abs() < 1.0e-13);
}

#[test]
fn parameter_jvp_is_an_operand_selection_not_a_physics_special_case() {
    let primal = quadratic_reduction_kernel();
    let product = malleus::differentiate(
        &primal,
        &malleus::DerivativeRequest {
            mode: malleus::DerivativeMode::Jvp,
            independent_operands: vec![OperandId::new(1)],
            dependent_operands: vec![OperandId::new(2)],
        },
    )
    .unwrap();
    let executable = Executable::reference(validate(product.kernel).unwrap());
    let mut x = [1.5, -2.0, 0.25];
    let mut parameter = [3.0];
    let mut parameter_direction = [0.75];
    let mut tangent = [0.0];
    Interpreter::run(
        &executable,
        &mut [
            BufferBinding::new(OperandId::new(0), &mut x),
            BufferBinding::new(OperandId::new(1), &mut parameter),
            BufferBinding::new(OperandId::new(2), &mut parameter_direction),
            BufferBinding::new(OperandId::new(3), &mut tangent),
        ],
    )
    .unwrap();
    let expected = x.iter().map(|x| x * x).sum::<f64>() * parameter_direction[0];
    assert!((tangent[0] - expected).abs() < 1.0e-13);
}

#[test]
fn validation_refuses_out_of_bounds_indexes_and_aliasing_writes() {
    let mut bounds = scale_kernel();
    bounds.indexing_maps[0].results[0] = IndexExpr::offset(AxisId::new(0), 1);
    assert_eq!(
        validate(bounds).unwrap_err(),
        ValidationError::IndexOutOfBounds {
            operand: 0,
            dimension: 0,
        }
    );

    let mut aliases = scale_kernel();
    aliases.indexing_maps[1].results[0] = IndexExpr::constant(0);
    assert_eq!(
        validate(aliases).unwrap_err(),
        ValidationError::AliasingWrite(1)
    );

    let mut layout = scale_kernel();
    layout.operands[0].layout.minor_to_major = vec![0, 0];
    assert_eq!(
        validate(layout).unwrap_err(),
        ValidationError::InvalidLayout(0)
    );

    let mut region = scale_kernel();
    region.operands[0].region = malleus::BufferRegion::new(1, 2);
    assert_eq!(
        validate(region).unwrap_err(),
        ValidationError::InvalidRegion(0)
    );
}

#[test]
fn interpreter_honors_declared_f32_operation_precision() {
    let mut kernel = scale_kernel();
    kernel.numeric_policy.scalar_type = malleus::ScalarType::F32;
    let executable = Executable::reference(validate(kernel).unwrap());
    let mut input = [16_777_217.0, 1.0, 2.0];
    let mut output = [0.0; 3];
    Interpreter::run(
        &executable,
        &mut [
            BufferBinding::new(OperandId::new(0), &mut input),
            BufferBinding::new(OperandId::new(1), &mut output),
        ],
    )
    .unwrap();
    assert_eq!(output, [33_554_432.0, 2.0, 4.0]);
}

#[test]
fn interpreter_honors_validated_buffer_region_offsets() {
    let mut kernel = scale_kernel();
    for operand in &mut kernel.operands {
        operand.region = malleus::BufferRegion::new(1, 3);
    }
    let executable = Executable::reference(validate(kernel).unwrap());
    let mut input = [99.0, 1.0, -2.0, 4.5];
    let mut output = [77.0, 0.0, 0.0, 0.0];
    Interpreter::run(
        &executable,
        &mut [
            BufferBinding::new(OperandId::new(0), &mut input),
            BufferBinding::new(OperandId::new(1), &mut output),
        ],
    )
    .unwrap();
    assert_eq!(output, [77.0, 2.0, -4.0, 9.0]);
}
