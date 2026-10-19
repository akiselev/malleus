//! Consumer-neutral arithmetic kernels used identically by interpreter and JIT.

use malleus::*;

/// One quadratic guard-like output per input: c0+c1*t+c2*t².
pub fn polynomial(n: usize) -> StructuredKernel {
    let axis = AxisId::new(0);
    StructuredKernel {
        name: format!("polynomial-{n}"),
        iteration_domain: IterationDomain::new(vec![n]),
        iterators: vec![IteratorKind::Parallel],
        operands: vec![
            KernelOperand::tensor("t", vec![n], AccessMode::Read),
            KernelOperand::scalar("c0", AccessMode::Read),
            KernelOperand::scalar("c1", AccessMode::Read),
            KernelOperand::scalar("c2", AccessMode::Read),
            KernelOperand::tensor("out", vec![n], AccessMode::Write),
        ],
        indexing_maps: vec![
            IndexingMap::new(OperandId::new(0), vec![IndexExpr::axis(axis)]),
            IndexingMap::scalar(OperandId::new(1)),
            IndexingMap::scalar(OperandId::new(2)),
            IndexingMap::scalar(OperandId::new(3)),
            IndexingMap::new(OperandId::new(4), vec![IndexExpr::axis(axis)]),
        ],
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(4),
                value: ScalarExpr::binary(
                    BinaryOp::Add,
                    ScalarExpr::Load(OperandId::new(1)),
                    ScalarExpr::binary(
                        BinaryOp::Mul,
                        ScalarExpr::Load(OperandId::new(0)),
                        ScalarExpr::binary(
                            BinaryOp::Add,
                            ScalarExpr::Load(OperandId::new(2)),
                            ScalarExpr::binary(
                                BinaryOp::Mul,
                                ScalarExpr::Load(OperandId::new(3)),
                                ScalarExpr::Load(OperandId::new(0)),
                            ),
                        ),
                    ),
                ),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

/// Parameter-weighted sum of squares; also exercises Malleus's JVP/VJP generation.
pub fn quadratic(n: usize) -> StructuredKernel {
    let axis = AxisId::new(0);
    StructuredKernel {
        name: format!("quadratic-{n}"),
        iteration_domain: IterationDomain::new(vec![n]),
        iterators: vec![IteratorKind::Reduction],
        operands: vec![
            KernelOperand::tensor("x", vec![n], AccessMode::Read),
            KernelOperand::scalar("parameter", AccessMode::Read),
            KernelOperand::scalar("sum", AccessMode::Reduce(ReductionOp::Add)),
        ],
        indexing_maps: vec![
            IndexingMap::new(OperandId::new(0), vec![IndexExpr::axis(axis)]),
            IndexingMap::scalar(OperandId::new(1)),
            IndexingMap::scalar(OperandId::new(2)),
        ],
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(2),
                value: ScalarExpr::binary(
                    BinaryOp::Mul,
                    ScalarExpr::Load(OperandId::new(1)),
                    ScalarExpr::binary(
                        BinaryOp::Mul,
                        ScalarExpr::Load(OperandId::new(0)),
                        ScalarExpr::Load(OperandId::new(0)),
                    ),
                ),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

/// Simple deterministic inputs in operand order. Derivative directions/seeds default to one.
pub fn buffers(k: &StructuredKernel) -> Vec<Vec<f64>> {
    k.operands
        .iter()
        .map(|o| {
            let len = o.region.offset + o.region.length;
            if o.access.can_read() {
                (0..len).map(|i| 0.25 + (i % 17) as f64 / 32.0).collect()
            } else {
                vec![0.0; len]
            }
        })
        .collect()
}

/// Dense local matrix-vector product with two axes and one reducing output map.
pub fn matvec(n: usize) -> StructuredKernel {
    let (row, col) = (AxisId::new(0), AxisId::new(1));
    StructuredKernel {
        name: format!("matvec-{n}"),
        iteration_domain: IterationDomain::new(vec![n, n]),
        iterators: vec![IteratorKind::Parallel, IteratorKind::Reduction],
        operands: vec![
            KernelOperand::tensor("matrix", vec![n, n], AccessMode::Read),
            KernelOperand::tensor("vector", vec![n], AccessMode::Read),
            KernelOperand::tensor("out", vec![n], AccessMode::Reduce(ReductionOp::Add)),
        ],
        indexing_maps: vec![
            IndexingMap::new(
                OperandId::new(0),
                vec![IndexExpr::axis(row), IndexExpr::axis(col)],
            ),
            IndexingMap::new(OperandId::new(1), vec![IndexExpr::axis(col)]),
            IndexingMap::new(OperandId::new(2), vec![IndexExpr::axis(row)]),
        ],
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(2),
                value: ScalarExpr::binary(
                    BinaryOp::Mul,
                    ScalarExpr::Load(OperandId::new(0)),
                    ScalarExpr::Load(OperandId::new(1)),
                ),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

/// Batched polynomials with independent coefficient buffers and one shared parameter.
/// The explicit square lets interval lowering retain correlation across zero.
pub fn polynomial_batch(n: usize) -> StructuredKernel {
    let a = AxisId::new(0);
    StructuredKernel {
        name: format!("polynomial-batch-{n}"),
        iteration_domain: IterationDomain::new(vec![n]),
        iterators: vec![IteratorKind::Parallel],
        operands: vec![
            KernelOperand::tensor("c0", vec![n], AccessMode::Read),
            KernelOperand::tensor("c1", vec![n], AccessMode::Read),
            KernelOperand::tensor("c2", vec![n], AccessMode::Read),
            KernelOperand::scalar("t", AccessMode::Read),
            KernelOperand::tensor("value", vec![n], AccessMode::Write),
        ],
        indexing_maps: (0..5)
            .map(|i| {
                if i == 3 {
                    IndexingMap::scalar(OperandId::new(i))
                } else {
                    IndexingMap::new(OperandId::new(i), vec![IndexExpr::axis(a)])
                }
            })
            .collect(),
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(4),
                value: ScalarExpr::binary(
                    BinaryOp::Add,
                    ScalarExpr::Load(OperandId::new(0)),
                    ScalarExpr::binary(
                        BinaryOp::Add,
                        ScalarExpr::binary(
                            BinaryOp::Mul,
                            ScalarExpr::Load(OperandId::new(1)),
                            ScalarExpr::Load(OperandId::new(3)),
                        ),
                        ScalarExpr::binary(
                            BinaryOp::Mul,
                            ScalarExpr::Load(OperandId::new(2)),
                            ScalarExpr::binary(
                                BinaryOp::Mul,
                                ScalarExpr::Load(OperandId::new(3)),
                                ScalarExpr::Load(OperandId::new(3)),
                            ),
                        ),
                    ),
                ),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    }
}
