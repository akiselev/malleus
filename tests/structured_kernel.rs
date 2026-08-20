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
