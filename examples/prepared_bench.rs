//! Interpreter vs prepared one-shot vs prepared reusable vs batched execution.
//!
//! `cargo run --release --example prepared_bench`. Every timing is the median of seven
//! batches; a batch repeats one call enough times to run for roughly 20 ms and reports the
//! mean time per call. "Batched" runs the same kernel as one `run_batch` over 64 strided
//! independent points (100,000 for the property kernel) and reports time per point. Outputs of
//! all four paths are compared bitwise before timing. Reductions keep accumulating across
//! repeats in every path, as their execution contract requires.

// The test fixture file is shared; this benchmark uses a subset of it.
#[allow(dead_code)]
#[path = "../tests/support/fixtures.rs"]
mod fixtures;

use std::hint::black_box;
use std::time::Instant;

use malleus::*;

fn median_ns(mut call: impl FnMut()) -> f64 {
    let start = Instant::now();
    let mut calibration = 0u64;
    while start.elapsed().as_millis() < 5 {
        call();
        calibration += 1;
    }
    let repeats = (calibration * 4).max(1);
    let mut samples: Vec<f64> = (0..7)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..repeats {
                call();
            }
            start.elapsed().as_nanos() as f64 / repeats as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[3]
}

fn same(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

fn split<'a>(
    k: &StructuredKernel,
    buffers: &'a mut [Vec<f64>],
    strides: &[usize],
) -> (Vec<ReadBinding<'a>>, Vec<WriteBinding<'a>>) {
    let (mut reads, mut writes) = (vec![], vec![]);
    for (index, values) in buffers.iter_mut().enumerate() {
        let operand = OperandId::new(index);
        if k.operands[index].access == AccessMode::Read {
            reads.push(ReadBinding::strided(operand, values, strides[index]));
        } else {
            writes.push(WriteBinding::strided(operand, values, strides[index]));
        }
    }
    (reads, writes)
}

/// A Finitum-like point property: Arrhenius-scaled conductivity with a clamped temperature
/// and its temperature derivative, two outputs from five scalar inputs.
fn property() -> StructuredKernel {
    use ScalarExpr as E;
    let load = |i| E::Load(OperandId::new(i));
    let local = |i| E::Local(LocalId::new(i));
    let bin = E::binary;
    let names = ["temperature", "k0", "activation", "t_min", "alpha"];
    let mut operands: Vec<_> = names
        .iter()
        .map(|n| KernelOperand::scalar(*n, AccessMode::Read))
        .collect();
    operands.push(KernelOperand::scalar("k", AccessMode::Write));
    operands.push(KernelOperand::scalar("dk_dt", AccessMode::Write));
    StructuredKernel {
        name: "property".into(),
        iteration_domain: IterationDomain::new(vec![]),
        iterators: vec![],
        indexing_maps: (0..7)
            .map(|i| IndexingMap::scalar(OperandId::new(i)))
            .collect(),
        operands,
        body: KernelRegion {
            statements: vec![
                // t = max(T, t_min)
                Statement::Let {
                    local: LocalId::new(0),
                    value: bin(BinaryOp::Max, load(0), load(3)),
                },
                // a = exp(-activation / t)
                Statement::Let {
                    local: LocalId::new(1),
                    value: E::unary(
                        UnaryOp::Exp,
                        E::unary(UnaryOp::Neg, bin(BinaryOp::Div, load(2), local(0))),
                    ),
                },
                // g = 1 + alpha * t
                Statement::Let {
                    local: LocalId::new(2),
                    value: bin(
                        BinaryOp::Add,
                        E::Constant(1.0),
                        bin(BinaryOp::Mul, load(4), local(0)),
                    ),
                },
                Statement::Store {
                    operand: OperandId::new(5),
                    value: bin(
                        BinaryOp::Mul,
                        load(1),
                        bin(BinaryOp::Mul, local(1), local(2)),
                    ),
                },
                // dk/dT = k0 * a * (activation / t^2 * g + alpha) when T > t_min, else 0
                Statement::Store {
                    operand: OperandId::new(6),
                    value: E::Select {
                        condition: Box::new(Predicate::Compare {
                            op: CompareOp::Greater,
                            lhs: Box::new(load(0)),
                            rhs: Box::new(load(3)),
                        }),
                        if_true: Box::new(bin(
                            BinaryOp::Mul,
                            bin(BinaryOp::Mul, load(1), local(1)),
                            bin(
                                BinaryOp::Add,
                                bin(
                                    BinaryOp::Mul,
                                    bin(
                                        BinaryOp::Div,
                                        load(2),
                                        bin(BinaryOp::Mul, local(0), local(0)),
                                    ),
                                    local(2),
                                ),
                                load(4),
                            ),
                        )),
                        if_false: Box::new(E::Constant(0.0)),
                    },
                },
            ],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

fn property_buffers(points: usize) -> Vec<Vec<f64>> {
    vec![
        (0..points)
            .map(|p| 250.0 + (p % 997) as f64 * 0.5)
            .collect(),
        vec![2.5],
        vec![1200.0],
        vec![300.0],
        vec![1e-3],
        vec![0.0; points],
        vec![0.0; points],
    ]
}

struct Row {
    name: &'static str,
    interpreter: f64,
    one_shot: f64,
    reusable: f64,
    batched: f64,
}

fn measure(name: &'static str, kernel: StructuredKernel, buffers: Vec<Vec<f64>>) -> Row {
    let e = Executable::reference(validate(kernel).unwrap());
    let k = e.kernel().as_kernel();
    let prepared = PreparedKernel::new(&e).unwrap();
    let mut state = prepared.state();
    let zeros = vec![0; buffers.len()];

    // Bitwise agreement of the three single-call paths on one fresh call.
    let mut reference = buffers.clone();
    let mut bindings: Vec<_> = reference
        .iter_mut()
        .enumerate()
        .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
        .collect();
    Interpreter::run(&e, &mut bindings).unwrap();
    drop(bindings);
    let mut one_shot = buffers.clone();
    let mut bindings: Vec<_> = one_shot
        .iter_mut()
        .enumerate()
        .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
        .collect();
    prepared.run(&mut bindings).unwrap();
    drop(bindings);
    let mut reusable = buffers.clone();
    let (reads, mut writes) = split(k, &mut reusable, &zeros);
    prepared.call(&mut state, &reads, &mut writes).unwrap();
    drop((reads, writes));
    assert!(
        same(&reference, &one_shot) && same(&reference, &reusable),
        "{name}"
    );

    let mut work = buffers.clone();
    let mut bindings: Vec<_> = work
        .iter_mut()
        .enumerate()
        .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
        .collect();
    let interpreter =
        median_ns(|| Interpreter::run(black_box(&e), black_box(&mut bindings)).unwrap());
    let one_shot = median_ns(|| prepared.run(black_box(&mut bindings)).unwrap());
    drop(bindings);
    let (reads, mut writes) = split(k, &mut work, &zeros);
    let reusable = median_ns(|| {
        prepared
            .call(
                black_box(&mut state),
                black_box(&reads),
                black_box(&mut writes),
            )
            .unwrap()
    });
    drop((reads, writes));

    const POINTS: usize = 64;
    let strides: Vec<usize> = buffers.iter().map(Vec::len).collect();
    let fresh: Vec<Vec<f64>> = buffers.iter().map(|b| b.repeat(POINTS)).collect();
    let mut batch = fresh.clone();
    let (reads, mut writes) = split(k, &mut batch, &strides);
    prepared
        .run_batch(&mut state, POINTS, &reads, &mut writes)
        .unwrap();
    drop((reads, writes));
    let replicated: Vec<Vec<f64>> = reference.iter().map(|b| b.repeat(POINTS)).collect();
    assert!(same(&replicated, &batch), "{name} batch");
    let (reads, mut writes) = split(k, &mut batch, &strides);
    let batched = median_ns(|| {
        prepared
            .run_batch(
                black_box(&mut state),
                POINTS,
                black_box(&reads),
                black_box(&mut writes),
            )
            .unwrap()
    }) / POINTS as f64;
    Row {
        name,
        interpreter,
        one_shot,
        reusable,
        batched,
    }
}

fn property_row() -> Row {
    const CALLS: usize = 100_000;
    let e = Executable::reference(validate(property()).unwrap());
    let k = e.kernel().as_kernel();
    let prepared = PreparedKernel::new(&e).unwrap();
    let mut state = prepared.state();
    let all = property_buffers(CALLS);
    let point = |p: usize| -> Vec<Vec<f64>> {
        vec![
            vec![all[0][p]],
            all[1].clone(),
            all[2].clone(),
            all[3].clone(),
            all[4].clone(),
            vec![0.0],
            vec![0.0],
        ]
    };
    let mut scalar = point(0);
    let mut sweep_interpreter = || {
        for &temperature in &all[0] {
            scalar[0][0] = temperature;
            let mut bindings: Vec<_> = scalar
                .iter_mut()
                .enumerate()
                .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
                .collect();
            Interpreter::run(&e, &mut bindings).unwrap();
            black_box(&bindings);
        }
    };
    let interpreter = median_ns(&mut sweep_interpreter) / CALLS as f64;
    let mut scalar = point(0);
    let one_shot = median_ns(|| {
        for &temperature in &all[0] {
            scalar[0][0] = temperature;
            let mut bindings: Vec<_> = scalar
                .iter_mut()
                .enumerate()
                .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
                .collect();
            prepared.run(&mut bindings).unwrap();
            black_box(&bindings);
        }
    }) / CALLS as f64;
    let (mut k_out, mut dk_out) = (vec![0.0; CALLS], vec![0.0; CALLS]);
    let reusable = median_ns(|| {
        let (mut k_value, mut dk_value) = ([0.0], [0.0]);
        for p in 0..CALLS {
            let reads = [
                ReadBinding::new(OperandId::new(0), &all[0][p..p + 1]),
                ReadBinding::new(OperandId::new(1), &all[1]),
                ReadBinding::new(OperandId::new(2), &all[2]),
                ReadBinding::new(OperandId::new(3), &all[3]),
                ReadBinding::new(OperandId::new(4), &all[4]),
            ];
            let mut writes = [
                WriteBinding::new(OperandId::new(5), &mut k_value),
                WriteBinding::new(OperandId::new(6), &mut dk_value),
            ];
            prepared.call(&mut state, &reads, &mut writes).unwrap();
            k_out[p] = k_value[0];
            dk_out[p] = dk_value[0];
        }
    }) / CALLS as f64;
    let mut batch = all.clone();
    let strides = [1, 0, 0, 0, 0, 1, 1];
    let (reads, mut writes) = split(k, &mut batch, &strides);
    let batched = median_ns(|| {
        prepared
            .run_batch(
                black_box(&mut state),
                CALLS,
                black_box(&reads),
                black_box(&mut writes),
            )
            .unwrap()
    }) / CALLS as f64;
    drop((reads, writes));
    for p in (0..CALLS).step_by(997) {
        let expected = {
            let mut b = point(p);
            let mut bindings: Vec<_> = b
                .iter_mut()
                .enumerate()
                .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
                .collect();
            Interpreter::run(&e, &mut bindings).unwrap();
            drop(bindings);
            b
        };
        for (got, want) in [(batch[5][p], expected[5][0]), (batch[6][p], expected[6][0])]
            .into_iter()
            .chain([(k_out[p], expected[5][0]), (dk_out[p], expected[6][0])])
        {
            assert_eq!(got.to_bits(), want.to_bits(), "property point {p}");
        }
    }
    Row {
        name: "Property kernel, 10^5 point calls",
        interpreter,
        one_shot,
        reusable,
        batched,
    }
}

fn derivative(mode: DerivativeMode) -> StructuredKernel {
    differentiate(
        &fixtures::quadratic(1024),
        &DerivativeRequest {
            mode,
            independent_operands: vec![OperandId::new(0)],
            dependent_operands: vec![OperandId::new(2)],
        },
    )
    .unwrap()
    .kernel
}

fn main() {
    let jvp = derivative(DerivativeMode::Jvp);
    let vjp = derivative(DerivativeMode::Vjp);
    let rows = [
        measure(
            "One guard polynomial",
            fixtures::polynomial(1),
            fixtures::buffers(&fixtures::polynomial(1)),
        ),
        measure(
            "64 guard evaluations",
            fixtures::polynomial(64),
            fixtures::buffers(&fixtures::polynomial(64)),
        ),
        measure(
            "4,096 guard evaluations",
            fixtures::polynomial(4096),
            fixtures::buffers(&fixtures::polynomial(4096)),
        ),
        measure(
            "Quadratic JVP, 1,024 inputs",
            jvp.clone(),
            fixtures::buffers(&jvp),
        ),
        measure(
            "Quadratic VJP, 1,024 inputs",
            vjp.clone(),
            fixtures::buffers(&vjp),
        ),
        measure(
            "Dense matvec 64x64",
            fixtures::matvec(64),
            fixtures::buffers(&fixtures::matvec(64)),
        ),
        property_row(),
    ];
    println!(
        "| Kernel | Interpreter | Prepared one-shot | Prepared reusable | Batched (per point) |"
    );
    println!("|---|---:|---:|---:|---:|");
    let show = |ns: f64| {
        if ns >= 1e6 {
            format!("{:.2} ms", ns / 1e6)
        } else if ns >= 1e3 {
            format!("{:.2} us", ns / 1e3)
        } else {
            format!("{ns:.1} ns")
        }
    };
    for row in rows {
        println!(
            "| {} | {} | {} | {} | {} |",
            row.name,
            show(row.interpreter),
            show(row.one_shot),
            show(row.reusable),
            show(row.batched)
        );
    }
}
