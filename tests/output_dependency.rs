use malleus::*;

fn kernel(statements: Vec<Statement>) -> StructuredKernel {
    StructuredKernel {
        name: "dependencies".into(),
        iteration_domain: IterationDomain::default(),
        iterators: vec![],
        operands: vec![
            KernelOperand::scalar("input", AccessMode::Read),
            KernelOperand::scalar("output", AccessMode::Write),
            KernelOperand::scalar("scratch", AccessMode::ReadWrite),
        ],
        indexing_maps: (0..3)
            .map(|id| IndexingMap::new(OperandId::new(id), vec![]))
            .collect(),
        body: KernelRegion { statements },
        numeric_policy: NumericPolicy::default(),
    }
}
fn store(id: usize, value: ScalarExpr) -> Statement {
    Statement::Store {
        operand: OperandId::new(id),
        value,
    }
}
fn input() -> ScalarExpr {
    ScalarExpr::Load(OperandId::new(0))
}
fn query(kernel: &StructuredKernel) -> bool {
    primal_output_reads_input(kernel, OperandId::new(0)).unwrap()
}

#[test]
fn unused_and_dead_locals_or_overwritten_stores_do_not_contribute() {
    assert!(!query(&kernel(vec![store(1, ScalarExpr::Constant(2.0))])));
    assert!(!query(&kernel(vec![
        Statement::Let {
            local: LocalId::new(0),
            value: input()
        },
        store(1, ScalarExpr::Constant(2.0))
    ])));
    assert!(!query(&kernel(vec![
        store(1, input()),
        store(1, ScalarExpr::Constant(2.0))
    ])));
    assert!(!query(&kernel(vec![
        store(2, input()),
        store(2, ScalarExpr::Constant(2.0)),
        store(1, ScalarExpr::Load(OperandId::new(2)))
    ])));
}

#[test]
fn transitive_locals_operand_stores_and_all_outputs_contribute() {
    assert!(query(&kernel(vec![
        Statement::Let {
            local: LocalId::new(0),
            value: input()
        },
        Statement::Let {
            local: LocalId::new(1),
            value: ScalarExpr::unary(UnaryOp::Neg, ScalarExpr::Local(LocalId::new(0)))
        },
        store(2, ScalarExpr::Local(LocalId::new(1))),
        store(1, ScalarExpr::Load(OperandId::new(2))),
        store(2, ScalarExpr::Constant(0.0))
    ])));
    assert!(query(&kernel(vec![
        store(1, ScalarExpr::Constant(0.0)),
        store(2, input())
    ])));
    assert!(query(&kernel(vec![store(
        1,
        ScalarExpr::binary(BinaryOp::Sub, input(), input())
    )])));
}

#[test]
fn select_predicates_and_unselected_branches_are_conservative() {
    let predicate = Predicate::Compare {
        op: CompareOp::Greater,
        lhs: Box::new(input()),
        rhs: Box::new(ScalarExpr::Constant(0.0)),
    };
    assert!(query(&kernel(vec![store(
        1,
        ScalarExpr::Select {
            condition: Box::new(Predicate::Not(Box::new(predicate))),
            if_true: Box::new(ScalarExpr::Constant(1.0)),
            if_false: Box::new(ScalarExpr::Constant(1.0))
        }
    )])));
    assert!(query(&kernel(vec![store(
        1,
        ScalarExpr::Select {
            condition: Box::new(Predicate::Constant(false)),
            if_true: Box::new(input()),
            if_false: Box::new(ScalarExpr::Constant(0.0))
        }
    )])));
}

#[test]
fn affine_addresses_and_reductions_preserve_dependencies() {
    let mut k = kernel(vec![store(1, input()), store(1, ScalarExpr::Constant(0.0))]);
    k.iteration_domain = IterationDomain::new(vec![3]);
    k.iterators = vec![IteratorKind::Reduction];
    k.operands[0] = KernelOperand::tensor("input", vec![3], AccessMode::Read);
    k.indexing_maps[0] = IndexingMap::new(OperandId::new(0), vec![IndexExpr::axis(AxisId::new(0))]);
    k.operands[1].access = AccessMode::Reduce(ReductionOp::Add);
    k.operands[2].access = AccessMode::Read;
    assert!(query(&k));
    k.body.statements = vec![store(1, ScalarExpr::Index(AxisId::new(0)))];
    assert!(!query(&k));
}

#[test]
fn invalid_kernel_and_invalid_input_are_typed_errors() {
    let k = kernel(vec![store(1, ScalarExpr::Local(LocalId::new(0)))]);
    assert!(matches!(
        primal_output_reads_input(&k, OperandId::new(0)),
        Err(ValidationError::InvalidLocal { .. })
    ));
    let k = kernel(vec![store(1, ScalarExpr::Constant(0.0))]);
    assert_eq!(
        primal_output_reads_input(&k, OperandId::new(12)),
        Err(ValidationError::InvalidOperand(12))
    );
    assert_eq!(
        primal_output_reads_input(&k, OperandId::new(1)),
        Err(ValidationError::InvalidLoad(1))
    );
}

#[test]
fn partially_written_or_unexecuted_readwrite_inputs_remain_dependent() {
    let mut k = kernel(vec![store(2, ScalarExpr::Constant(0.0))]);
    k.operands[2] = KernelOperand::tensor("scratch", vec![2], AccessMode::ReadWrite);
    k.indexing_maps[2] = IndexingMap::new(OperandId::new(2), vec![IndexExpr::constant(0)]);
    assert!(primal_output_reads_input(&k, OperandId::new(2)).unwrap());
    k.iteration_domain = IterationDomain::new(vec![0]);
    k.iterators = vec![IteratorKind::Serial];
    assert!(primal_output_reads_input(&k, OperandId::new(2)).unwrap());
    assert!(!query(&k));
}
