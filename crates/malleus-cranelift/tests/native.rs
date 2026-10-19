//! Policy levels, explicit Fma, callouts, the native batch loop and the compile cache.
use malleus::policy::{NumericLevel, contract_fma};
use malleus::prepared::*;
use malleus::*;
use malleus_cranelift::{CacheKey, JitCache, JitKernel, fixtures};
use std::sync::Arc;

fn same(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
        && a.iter().map(Vec::len).eq(b.iter().map(Vec::len))
}

fn interpret(e: &Executable, mut buffers: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut bindings: Vec<_> = buffers
        .iter_mut()
        .enumerate()
        .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
        .collect();
    Interpreter::run(e, &mut bindings).unwrap();
    drop(bindings);
    buffers
}

fn split<'a>(
    k: &StructuredKernel,
    buffers: &'a mut [Vec<f64>],
    strides: &[usize],
) -> (Vec<ReadBinding<'a>>, Vec<WriteBinding<'a>>) {
    let (mut reads, mut writes) = (vec![], vec![]);
    for (i, values) in buffers.iter_mut().enumerate() {
        let id = OperandId::new(i);
        if k.operands[i].access == AccessMode::Read {
            reads.push(ReadBinding::strided(id, values, strides[i]));
        } else {
            writes.push(WriteBinding::strided(id, values, strides[i]));
        }
    }
    (reads, writes)
}

/// Checked `run`, positional `call` and the interpreter agree bitwise.
fn compare(e: &Executable, buffers: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let jit = JitKernel::compile(e).unwrap();
    let reference = interpret(e, buffers.clone());
    let mut checked = buffers.clone();
    let mut bindings: Vec<_> = checked
        .iter_mut()
        .enumerate()
        .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
        .collect();
    jit.run(&mut bindings).unwrap();
    drop(bindings);
    assert!(
        same(&reference, &checked),
        "{}: checked run",
        e.kernel().as_kernel().name
    );
    let k = e.kernel().as_kernel();
    let mut positional = buffers;
    let zeros = vec![0; k.operands.len()];
    let (reads, mut writes) = split(k, &mut positional, &zeros);
    let mut state = jit.state();
    jit.call(&mut state, &reads, &mut writes).unwrap();
    drop((reads, writes));
    assert!(same(&reference, &positional), "{}: positional call", k.name);
    reference
}

fn pointwise(
    name: String,
    n: usize,
    body: ScalarExpr,
    ty: ScalarType,
    level: NumericLevel,
) -> Executable {
    let axis = AxisId::new(0);
    let k = StructuredKernel {
        name,
        iteration_domain: IterationDomain::new(vec![n]),
        iterators: vec![IteratorKind::Parallel],
        operands: vec![
            KernelOperand::tensor("x", vec![n], AccessMode::Read),
            KernelOperand::tensor("y", vec![n], AccessMode::Read),
            KernelOperand::tensor("z", vec![n], AccessMode::Read),
            KernelOperand::tensor("out", vec![n], AccessMode::Write),
        ],
        indexing_maps: (0..4)
            .map(|i| IndexingMap::new(OperandId::new(i), vec![IndexExpr::axis(axis)]))
            .collect(),
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(3),
                value: body,
            }],
        },
        numeric_policy: NumericPolicy {
            scalar_type: ty,
            level,
            ..NumericPolicy::default()
        },
    };
    Executable::reference(validate(k).unwrap())
}

fn load(i: usize) -> ScalarExpr {
    ScalarExpr::Load(OperandId::new(i))
}

#[test]
fn callouts_fma_and_min_max_reductions_match_the_interpreter_at_every_level() {
    let specials = [
        0.0,
        -0.0,
        0.5,
        -2.5,
        1.0,
        3.0,
        1e-310,
        700.0,
        -745.5,
        1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1e22,
        -0.75,
        2.0f64.powi(-30),
    ];
    let n = specials.len() * specials.len();
    let x: Vec<f64> = specials.iter().flat_map(|&a| specials.map(|_| a)).collect();
    let y: Vec<f64> = specials.iter().flat_map(|_| specials).collect();
    let z: Vec<f64> = (0..n).map(|i| specials[(i * 7) % specials.len()]).collect();
    let mut bodies = vec![];
    for op in [
        UnaryOp::Exp,
        UnaryOp::Ln,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Tan,
    ] {
        bodies.push((format!("{op:?}"), ScalarExpr::unary(op, load(0))));
    }
    for op in [BinaryOp::Pow, BinaryOp::Atan2, BinaryOp::Min, BinaryOp::Max] {
        bodies.push((format!("{op:?}"), ScalarExpr::binary(op, load(0), load(1))));
    }
    bodies.push(("fma".into(), ScalarExpr::fma(load(0), load(1), load(2))));
    for ty in [ScalarType::F32, ScalarType::F64] {
        for level in [NumericLevel::StrictPortable, NumericLevel::StrictHost] {
            for (name, body) in &bodies {
                let e = pointwise(
                    format!("{name}-{ty:?}-{level:?}"),
                    n,
                    body.clone(),
                    ty,
                    level,
                );
                compare(&e, vec![x.clone(), y.clone(), z.clone(), vec![0.0; n]]);
            }
            for op in [ReductionOp::Min, ReductionOp::Max] {
                let mut k = fixtures::quadratic(16);
                k.numeric_policy.scalar_type = ty;
                k.numeric_policy.level = level;
                k.operands[2].access = AccessMode::Reduce(op);
                let mut b = fixtures::buffers(&k);
                b[0] = specials.to_vec();
                b[2] = vec![-0.0];
                compare(&Executable::reference(validate(k).unwrap()), b);
            }
        }
    }
    // Adversarial fma: only the single-rounding result is nonzero.
    let e = 2f64.powi(-30);
    let k = pointwise(
        "fma".into(),
        1,
        ScalarExpr::fma(load(0), load(1), load(2)),
        ScalarType::F64,
        NumericLevel::StrictHost,
    );
    let out = compare(
        &k,
        vec![vec![1.0 + e], vec![1.0 - e], vec![-1.0], vec![0.0]],
    );
    assert_eq!(out[3][0], -(e * e));
}

#[test]
fn contracted_and_portable_derivative_kernels_match() {
    let contracted = contract_fma(&fixtures::polynomial(8)).unwrap().kernel;
    let mut portable = fixtures::quadratic(8);
    portable.numeric_policy.level = NumericLevel::StrictPortable;
    let Statement::Store { value, .. } = &mut portable.body.statements[0] else {
        panic!("store")
    };
    *value = ScalarExpr::unary(
        UnaryOp::Sin,
        ScalarExpr::binary(BinaryOp::Pow, value.clone(), ScalarExpr::Constant(0.5)),
    );
    for (k, independent, dependent) in [(contracted, 0, 4), (portable, 0, 2)] {
        compare(
            &Executable::reference(validate(k.clone()).unwrap()),
            fixtures::buffers(&k),
        );
        for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
            let p = differentiate(
                &k,
                &DerivativeRequest {
                    mode,
                    independent_operands: vec![OperandId::new(independent)],
                    dependent_operands: vec![OperandId::new(dependent)],
                },
            )
            .unwrap();
            let b = fixtures::buffers(&p.kernel);
            compare(&Executable::reference(validate(p.kernel).unwrap()), b);
        }
    }
}

/// `points` sequential interpreter calls, call `p` seeing every buffer shifted by `p * stride`.
fn sequential(
    e: &Executable,
    mut buffers: Vec<Vec<f64>>,
    strides: &[usize],
    points: usize,
) -> Vec<Vec<f64>> {
    for p in 0..points {
        let mut bindings: Vec<_> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, v)| BufferBinding::new(OperandId::new(i), &mut v[p * strides[i]..]))
            .collect();
        Interpreter::run(e, &mut bindings).unwrap();
    }
    buffers
}

#[test]
fn native_batches_match_sequential_calls_for_every_stride_pattern() {
    let mut reduce = fixtures::quadratic(3);
    reduce.operands[2].access = AccessMode::Reduce(ReductionOp::Max);
    let kernels = [
        fixtures::polynomial(4),
        fixtures::polynomial_batch(3),
        reduce,
    ];
    for k in kernels {
        let e = Executable::reference(validate(k.clone()).unwrap());
        let jit = JitKernel::compile(&e).unwrap();
        let prepared = PreparedKernel::new(&e).unwrap();
        let lengths: Vec<usize> = k
            .operands
            .iter()
            .map(|o| o.region.offset + o.region.length)
            .collect();
        // Disjoint, shared (0) and overlapping (1) writable strides; reads disjoint or shared.
        for pattern in 0..3 {
            let strides: Vec<usize> = k
                .operands
                .iter()
                .zip(&lengths)
                .enumerate()
                .map(|(i, (o, &len))| match (o.access.can_write(), pattern) {
                    (true, 0) => len,
                    (true, 1) => 0,
                    (true, _) => 1,
                    (false, _) if i % 2 == 0 => len,
                    (false, _) => 0,
                })
                .collect();
            for points in [0, 1, 2, 33, 70] {
                let buffers: Vec<Vec<f64>> = lengths
                    .iter()
                    .zip(&strides)
                    .enumerate()
                    .map(|(i, (&len, &s))| {
                        (0..len + s * points.max(1))
                            .map(|j| 0.25 + ((i * 31 + j * 7) % 23) as f64 / 8.0)
                            .collect()
                    })
                    .collect();
                let reference = sequential(&e, buffers.clone(), &strides, points);
                let mut native = buffers.clone();
                let (reads, mut writes) = split(&k, &mut native, &strides);
                let mut state = jit.state();
                jit.run_batch(&mut state, points, &reads, &mut writes)
                    .unwrap();
                drop((reads, writes));
                assert!(
                    same(&reference, &native),
                    "{} pattern {pattern} points {points}",
                    k.name
                );
                let mut core = buffers;
                let (reads, mut writes) = split(&k, &mut core, &strides);
                let mut state = prepared.state();
                prepared
                    .run_batch(&mut state, points, &reads, &mut writes)
                    .unwrap();
                drop((reads, writes));
                assert!(same(&reference, &core));
            }
        }
    }
}

#[test]
fn positional_refusals_match_the_core_prepared_kernel_without_effects() {
    let k = fixtures::polynomial(4);
    let e = Executable::reference(validate(k.clone()).unwrap());
    let jit = JitKernel::compile(&e).unwrap();
    let prepared = PreparedKernel::new(&e).unwrap();
    assert_eq!(jit.read_operands(), prepared.read_operands());
    assert_eq!(jit.write_operands(), prepared.write_operands());
    let base = fixtures::buffers(&k);
    type Mutation = Box<dyn Fn(&mut Vec<ReadBinding>, &mut Vec<WriteBinding>)>;
    let cases: Vec<Mutation> = vec![
        Box::new(|r, _| {
            r.pop();
        }),
        Box::new(|_, w| {
            w.clear();
        }),
        Box::new(|r, _| r.swap(0, 1)),
        Box::new(|_, w| w[0].operand = OperandId::new(0)),
        Box::new(|r, _| {
            let v = r[0].values;
            r[0].values = &v[..2];
        }),
        Box::new(|_, w| w[0].stride = usize::MAX),
    ];
    for (case, mutate) in cases.iter().enumerate() {
        let zeros = vec![0; k.operands.len()];
        let (mut a, mut b) = (base.clone(), base.clone());
        let (mut ra, mut wa) = split(&k, &mut a, &zeros);
        let (mut rb, mut wb) = split(&k, &mut b, &zeros);
        mutate(&mut ra, &mut wa);
        mutate(&mut rb, &mut wb);
        let native = jit
            .run_batch(&mut jit.state(), 2, &ra, &mut wa)
            .unwrap_err();
        let core = prepared
            .run_batch(&mut prepared.state(), 2, &rb, &mut wb)
            .unwrap_err();
        assert_eq!(native, core, "case {case}");
        drop((ra, wa, rb, wb));
        assert!(same(&a, &base), "case {case} had an effect");
    }
}

#[test]
fn compile_cache_keys_cover_identity_and_are_shared_across_threads() {
    let e = Executable::reference(validate(fixtures::polynomial(4)).unwrap());
    let mut portable = fixtures::polynomial(4);
    portable.numeric_policy.level = NumericLevel::StrictPortable;
    let ep = Executable::reference(validate(portable).unwrap());
    let key = CacheKey::new(&e);
    assert_eq!(key, CacheKey::new(&e));
    assert_ne!(key, CacheKey::new(&ep));
    assert!(!key.cpu_features.is_empty() && !key.cpu_features.starts_with("unavailable"));
    let cache = Arc::new(JitCache::default());
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let (cache, e) = (Arc::clone(&cache), e.clone());
            std::thread::spawn(move || cache.get_or_compile(&e).unwrap())
        })
        .collect();
    let kernels: Vec<Arc<JitKernel>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(cache.len(), 1);
    let again = cache.get_or_compile(&e).unwrap();
    assert!(kernels.iter().all(|k| Arc::ptr_eq(k, &again)));
    cache.get_or_compile(&ep).unwrap();
    assert_eq!(cache.len(), 2);
}
