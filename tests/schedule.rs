//! W11 M3 package 3: proven write injectivity, `vectorize(axis)` and `parallel(axis)`.

mod support;

use malleus::schedule::*;
use malleus::*;
use support::kernels::*;
use support::{Rng, fixtures};

/// `out[c·i + k] += x[i, k]` over `i < n`, `k < m`: injective in `i` iff `c ≥ m`.
fn strided_sum(n: usize, m: usize, c: isize) -> StructuredKernel {
    let (i, k) = (AxisId::new(0), AxisId::new(1));
    let len = c as usize * (n - 1) + m;
    StructuredKernel {
        name: format!("strided-sum-{n}-{m}-{c}"),
        iteration_domain: IterationDomain::new(vec![n, m]),
        iterators: vec![IteratorKind::Parallel, IteratorKind::Reduction],
        operands: vec![
            KernelOperand::tensor("x", vec![n, m], AccessMode::Read),
            KernelOperand::tensor("out", vec![len], AccessMode::Reduce(ReductionOp::Add)),
        ],
        indexing_maps: vec![
            IndexingMap::new(
                OperandId::new(0),
                vec![IndexExpr::axis(i), IndexExpr::axis(k)],
            ),
            IndexingMap::new(
                OperandId::new(1),
                vec![IndexExpr {
                    constant: 0,
                    terms: vec![
                        IndexTerm {
                            axis: i,
                            coefficient: c,
                        },
                        IndexTerm {
                            axis: k,
                            coefficient: 1,
                        },
                    ],
                }],
            ),
        ],
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(1),
                value: ScalarExpr::binary(
                    BinaryOp::Mul,
                    ScalarExpr::Load(OperandId::new(0)),
                    ScalarExpr::Load(OperandId::new(0)),
                ),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

/// `out[i, j] = a[j] * b[i] - a[j]` in row-major or column-major output storage.
fn outer(n: usize, m: usize, column_major: bool) -> StructuredKernel {
    let (i, j) = (AxisId::new(0), AxisId::new(1));
    let mut out = KernelOperand::tensor("out", vec![n, m], AccessMode::Write);
    if column_major {
        out = out.with_layout(DenseLayout::column_major(2));
    }
    StructuredKernel {
        name: format!("outer-{column_major}"),
        iteration_domain: IterationDomain::new(vec![n, m]),
        iterators: vec![IteratorKind::Parallel, IteratorKind::Parallel],
        operands: vec![
            KernelOperand::tensor("a", vec![m], AccessMode::Read),
            KernelOperand::tensor("b", vec![n], AccessMode::Read),
            out,
        ],
        indexing_maps: vec![
            IndexingMap::new(OperandId::new(0), vec![IndexExpr::axis(j)]),
            IndexingMap::new(OperandId::new(1), vec![IndexExpr::axis(i)]),
            IndexingMap::new(
                OperandId::new(2),
                vec![IndexExpr::axis(i), IndexExpr::axis(j)],
            ),
        ],
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(2),
                value: ScalarExpr::binary(
                    BinaryOp::Sub,
                    ScalarExpr::binary(
                        BinaryOp::Mul,
                        ScalarExpr::Load(OperandId::new(0)),
                        ScalarExpr::Load(OperandId::new(1)),
                    ),
                    ScalarExpr::Load(OperandId::new(0)),
                ),
            }],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

fn offset(k: &StructuredKernel, operand: usize, point: &[usize]) -> usize {
    let o = &k.operands[operand];
    let map = &k
        .indexing_maps
        .iter()
        .find(|m| m.operand.index() == operand)
        .unwrap()
        .results;
    let index: Vec<usize> = map
        .iter()
        .map(|e| {
            (e.constant
                + e.terms
                    .iter()
                    .map(|t| t.coefficient * point[t.axis.index()] as isize)
                    .sum::<isize>()) as usize
        })
        .collect();
    let (mut offset, mut stride) = (0, 1);
    for d in &o.layout.minor_to_major {
        offset += index[*d] * stride;
        stride *= o.shape[*d];
    }
    o.region.offset + offset
}

#[test]
fn injectivity_is_decided_exactly_with_witnesses() {
    let matvec = fixtures::matvec(4);
    let proof = prove_write_injective(&matvec, AxisId::new(0)).unwrap();
    assert_eq!(proof.strides, vec![4, 0, 1]);
    let Err(ScheduleError::NotInjective { operand, witness }) =
        prove_write_injective(&matvec, AxisId::new(1))
    else {
        panic!("the reduction axis of matvec is not injective");
    };
    assert_eq!(operand, OperandId::new(2));
    assert_ne!(witness.0[1], witness.1[1]);
    assert_eq!(
        offset(&matvec, 2, &witness.0),
        offset(&matvec, 2, &witness.1)
    );

    // c ≥ m is injective (the bitset decision), c < m collides with a verified witness.
    for (n, m, c, injective) in [
        (4, 2, 2, true),
        (4, 3, 3, true),
        (4, 3, 2, false),
        (5, 4, 1, false),
        (3, 5, 7, true),
    ] {
        let k = strided_sum(n, m, c);
        match prove_write_injective(&k, AxisId::new(0)) {
            Ok(_) => assert!(injective, "{}", k.name),
            Err(ScheduleError::NotInjective { witness, .. }) => {
                assert!(!injective, "{}", k.name);
                assert_ne!(witness.0[0], witness.1[0]);
                assert_eq!(
                    offset(&k, 1, &witness.0),
                    offset(&k, 1, &witness.1),
                    "{}",
                    k.name
                );
            }
            other => panic!("{}: {other:?}", k.name),
        }
    }
    // Write maps are injective along every axis by validation.
    for axis in 0..2 {
        prove_write_injective(&outer(3, 4, false), AxisId::new(axis)).unwrap();
    }
    assert_eq!(
        prove_write_injective(&matvec, AxisId::new(2)),
        Err(ScheduleError::InvalidAxis(2))
    );
}

fn check_vectorized(k: &StructuredKernel, axis: usize, r: &mut Rng) {
    let executable = Executable::reference(validate(k.clone()).unwrap());
    let vectorized = VectorizedKernel::new(&executable, AxisId::new(axis))
        .unwrap_or_else(|e| panic!("{} axis {axis}: {e:?}", k.name));
    for _ in 0..3 {
        let buffers = random_buffers(r, k);
        let expected = support::interpret(&executable, buffers.clone());
        for threads in [0, 1, 2, 3, 8] {
            let mut got = buffers.clone();
            let mut bindings: Vec<_> = got
                .iter_mut()
                .enumerate()
                .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
                .collect();
            if threads == 0 {
                vectorized.run(&mut bindings).unwrap();
            } else {
                vectorized.run_parallel(&mut bindings, threads).unwrap();
            }
            drop(bindings);
            assert_same(
                &got,
                &expected,
                &format!("{} axis {axis} threads {threads}", k.name),
            );
        }
    }
}

#[test]
fn vectorize_and_parallel_are_bitwise_sequential() {
    let mut r = Rng(0x7ec7_0123);
    check_vectorized(&fixtures::matvec(5), 0, &mut r);
    check_vectorized(&fixtures::polynomial(7), 0, &mut r);
    check_vectorized(&fixtures::polynomial_batch(9), 0, &mut r);
    check_vectorized(&outer(4, 3, false), 0, &mut r);
    check_vectorized(&outer(4, 3, true), 1, &mut r);
    check_vectorized(&strided_sum(5, 3, 3), 0, &mut r);
    check_vectorized(&nonsmooth(6), 0, &mut r);
    check_vectorized(&f32_variant(select_only(5)), 0, &mut r);
    check_vectorized(&min_reduction(4), 0, &mut r);
    // Derivative kernels: JVP (Write) and VJP (Reduce(Add) cotangents of pointwise inputs).
    for k in [fixtures::polynomial_batch(6), fma_exp(5), nonsmooth(4)] {
        for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
            let request = DerivativeRequest {
                mode,
                independent_operands: vec![OperandId::new(0), OperandId::new(1)],
                dependent_operands: vec![OperandId::new(k.operands.len() - 1)],
            };
            let product = differentiate(&k, &request).unwrap();
            check_vectorized(&product.kernel, 0, &mut r);
        }
    }
    for i in 0..40 {
        let k = random_kernel(&mut r, i % 4 == 0);
        check_vectorized(&k, 0, &mut r);
    }
}

#[test]
fn vectorize_refusals_are_typed() {
    let reference = |k: &StructuredKernel| Executable::reference(validate(k.clone()).unwrap());
    let matvec = fixtures::matvec(3);
    assert!(matches!(
        VectorizedKernel::new(&reference(&matvec), AxisId::new(1)),
        Err(ScheduleError::NotInjective { .. })
    ));
    // The inner axis of a row-major output is not its most-major dimension.
    assert_eq!(
        VectorizedKernel::new(&reference(&outer(3, 4, false)), AxisId::new(1)).err(),
        Some(ScheduleError::UnbatchableLayout {
            operand: OperandId::new(2)
        })
    );
    // A body reading the axis as a value.
    let mut indexed = fixtures::polynomial(3);
    indexed.body.statements[0] = Statement::Store {
        operand: OperandId::new(4),
        value: ScalarExpr::Index(AxisId::new(0)),
    };
    assert_eq!(
        VectorizedKernel::new(&reference(&indexed), AxisId::new(0)).err(),
        Some(ScheduleError::AxisUsedAsValue)
    );
    // A reversed read map.
    let mut reversed = fixtures::polynomial(3);
    reversed.indexing_maps[0].results[0] = IndexExpr {
        constant: 2,
        terms: vec![IndexTerm {
            axis: AxisId::new(0),
            coefficient: -1,
        }],
    };
    assert_eq!(
        VectorizedKernel::new(&reference(&reversed), AxisId::new(0)).err(),
        Some(ScheduleError::NegativeStride {
            operand: OperandId::new(0)
        })
    );
    let mut scheduled = fixtures::matvec(3);
    scheduled.numeric_policy.reduction_order = ReductionOrder::ScheduleDefined;
    assert_eq!(
        VectorizedKernel::new(&reference(&scheduled), AxisId::new(0)).err(),
        Some(ScheduleError::ScheduleDefinedOrder)
    );
    // Short buffers are refused before any effect.
    let v = VectorizedKernel::new(&reference(&matvec), AxisId::new(0)).unwrap();
    let (mut a, mut b, mut c) = (vec![0.0; 9], vec![0.0; 3], vec![7.0; 2]);
    let mut bindings = [
        BufferBinding::new(OperandId::new(0), &mut a),
        BufferBinding::new(OperandId::new(1), &mut b),
        BufferBinding::new(OperandId::new(2), &mut c),
    ];
    assert_eq!(
        v.run_parallel(&mut bindings, 2),
        Err(ExecutionError::InvalidBinding(2))
    );
    assert_eq!(c, vec![7.0; 2]);
}
