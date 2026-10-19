//! Interpreter vs core prepared/batched vs Cranelift checked/prepared/positional/batched vs
//! batched Rust source, on identical Malleus executables. Every path is compared bitwise with
//! the interpreter before it is timed. Prints JSON on stdout and a Markdown table on stderr.
//!
//! `cargo run -p malleus-cranelift --release --locked --quiet > results-m2.json`
//! (needs `rustc` on `PATH`, or `RUSTC`, for the batched Rust column).

use malleus::prepared::*;
use malleus::rust::{BatchedRustOptions, lower_rust_batched};
use malleus::*;
use malleus_cranelift::{JitKernel, fixtures};
use serde_json::json;
use std::fmt::Write as _;
use std::{hint::black_box, process::Command, time::Instant};

/// Points per batched call.
const POINTS: usize = 64;

fn median(mut f: impl FnMut(), n: usize) -> f64 {
    let mut t = [0.; 7];
    for x in &mut t {
        let start = Instant::now();
        for _ in 0..n {
            f();
        }
        *x = start.elapsed().as_nanos() as f64 / n as f64;
    }
    t.sort_by(f64::total_cmp);
    t[3]
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

/// Compiles the batched Rust source of `e` with a timing `main` (same median method) and
/// returns ns per batched call plus the output bits after one call.
fn batched_rust(
    e: &Executable,
    batch: &[Vec<f64>],
    strides: &[usize],
    count: usize,
) -> Result<(f64, Vec<u64>), String> {
    let k = e.kernel().as_kernel();
    let mut src = lower_rust_batched(e, "kernel", BatchedRustOptions::default())
        .map_err(|e| e.to_string())?;
    src.push_str("fn main() {\n");
    // Rebuild the batch with the same formula as `main` (literals would swamp rustc).
    for (i, o) in k.operands.iter().enumerate() {
        let len = o.region.offset + o.region.length;
        let base = if o.access.can_read() {
            "0.25 + (j % 17) as f64 / 32.0"
        } else {
            "0.0"
        };
        writeln!(
            src,
            "let mut b{i}: Vec<f64> = (0..{POINTS}).flat_map(|p| (0..{len}usize).map(move |j| {base} + p as f64 / 64.0)).collect();"
        )
        .unwrap();
    }
    let args: String = k
        .operands
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let m = if o.access == AccessMode::Read {
                ""
            } else {
                "mut "
            };
            format!("std::hint::black_box(&{m}b{i}), {},", strides[i])
        })
        .collect();
    let outputs: Vec<_> = (0..batch.len())
        .map(|i| format!("b{i}.iter().map(|x| x.to_bits()).collect::<Vec<_>>()"))
        .collect();
    writeln!(
        src,
        "kernel({POINTS}, {args}).unwrap();\nlet bits: Vec<Vec<u64>> = vec![{}];\n\
         let mut t = [0f64; 7];\nfor x in &mut t {{ let s = std::time::Instant::now();\n\
         for _ in 0..{count} {{ kernel({POINTS}, {args}).unwrap(); }}\n\
         *x = s.elapsed().as_nanos() as f64 / {count} as f64; }}\n\
         t.sort_by(f64::total_cmp);\nprintln!(\"{{}}\", t[3]);\nprintln!(\"{{:?}}\", bits.concat());\n}}",
        outputs.join(",")
    )
    .unwrap();
    let dir = std::env::temp_dir().join(format!("malleus-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("bench.rs"), &src).map_err(|e| e.to_string())?;
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let built = Command::new(rustc)
        .args(["--edition=2024", "-O"])
        .arg(dir.join("bench.rs"))
        .arg("-o")
        .arg(dir.join("bench"))
        .output()
        .map_err(|e| e.to_string())?;
    if !built.status.success() {
        return Err(String::from_utf8_lossy(&built.stderr).into_owned());
    }
    let run = Command::new(dir.join("bench"))
        .output()
        .map_err(|e| e.to_string())?;
    std::fs::remove_dir_all(&dir).ok();
    let out = String::from_utf8(run.stdout).map_err(|e| e.to_string())?;
    let mut lines = out.lines();
    let ns: f64 = lines
        .next()
        .ok_or("no timing")?
        .parse()
        .map_err(|_| "bad timing")?;
    let bits: Vec<u64> =
        serde_json::from_str(lines.next().ok_or("no bits")?).map_err(|e| e.to_string())?;
    Ok((ns, bits))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut kernels = vec![];
    for n in [1, 64, 4096] {
        kernels.push((format!("guard-polynomial-{n}"), fixtures::polynomial(n)));
    }
    for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
        let p = differentiate(
            &fixtures::quadratic(1024),
            &DerivativeRequest {
                mode,
                independent_operands: vec![OperandId::new(0), OperandId::new(1)],
                dependent_operands: vec![OperandId::new(2)],
            },
        )?;
        kernels.push((format!("quadratic-{mode:?}-1024"), p.kernel));
    }
    kernels.push(("matvec-64x64".into(), fixtures::matvec(64)));
    let mut results = vec![];
    let mut table = String::from(
        "| Kernel | Interpreter | M1 prepared | M1 batch/pt | JIT checked | JIT prepared | JIT positional | JIT batch/pt | Batched Rust/pt | JIT compile |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for (label, k) in kernels {
        let e = Executable::reference(validate(k.clone())?);
        let start = Instant::now();
        let jit = JitKernel::compile(&e)?;
        let compile_ns = start.elapsed().as_nanos() as f64;
        let prepared = PreparedKernel::new(&e)?;
        let size: usize = k
            .operands
            .iter()
            .map(|o| o.region.offset + o.region.length)
            .sum();
        let count = (200_000 / size.max(1)).clamp(20, 20_000);
        let base = fixtures::buffers(&k);

        // Bitwise checks of every single-call path against the interpreter.
        let run = |f: &mut dyn FnMut(&mut Vec<Vec<f64>>)| {
            let mut b = base.clone();
            f(&mut b);
            b
        };
        let reference = run(&mut |b| {
            let mut bind: Vec<_> = b
                .iter_mut()
                .enumerate()
                .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
                .collect();
            Interpreter::run(&e, &mut bind).unwrap();
        });
        let zeros = vec![0; k.operands.len()];
        let checks = [
            run(&mut |b| {
                let (r, mut w) = split(&k, b, &zeros);
                prepared.call(&mut prepared.state(), &r, &mut w).unwrap();
            }),
            run(&mut |b| {
                let mut bind: Vec<_> = b
                    .iter_mut()
                    .enumerate()
                    .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
                    .collect();
                jit.run(&mut bind).unwrap();
            }),
            run(&mut |b| {
                let (r, mut w) = split(&k, b, &zeros);
                jit.call(&mut jit.state(), &r, &mut w).unwrap();
            }),
        ];
        if !checks.iter().all(|c| same(&reference, c)) {
            return Err(format!("single-call mismatch in {label}").into());
        }

        let mut buffers = base.clone();
        let mut bindings: Vec<_> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
            .collect();
        let interpreter_ns = median(
            || Interpreter::run(black_box(&e), black_box(&mut bindings)).unwrap(),
            count,
        );
        let checked_ns = median(|| jit.run(black_box(&mut bindings)).unwrap(), count);
        let mut call = jit.prepare(&mut bindings)?;
        let jit_prepared_ns = median(|| black_box(&mut call).run(), count);
        drop(call);
        drop(bindings);
        let (reads, mut writes) = split(&k, &mut buffers, &zeros);
        let mut state = prepared.state();
        let m1_ns = median(
            || {
                prepared
                    .call(&mut state, black_box(&reads), black_box(&mut writes))
                    .unwrap()
            },
            count,
        );
        let mut jstate = jit.state();
        let positional_ns = median(
            || {
                jit.call(&mut jstate, black_box(&reads), black_box(&mut writes))
                    .unwrap()
            },
            count,
        );
        drop((reads, writes));

        // POINTS independent points: every operand strided by its own extent.
        let strides: Vec<usize> = k
            .operands
            .iter()
            .map(|o| o.region.offset + o.region.length)
            .collect();
        let batch: Vec<Vec<f64>> = base
            .iter()
            .map(|b| {
                (0..POINTS)
                    .flat_map(|p| b.iter().map(move |x| x + p as f64 / 64.0))
                    .collect()
            })
            .collect();
        let mut sequential = batch.clone();
        for p in 0..POINTS {
            let mut bind: Vec<_> = sequential
                .iter_mut()
                .enumerate()
                .map(|(i, v)| BufferBinding::new(OperandId::new(i), &mut v[p * strides[i]..]))
                .collect();
            Interpreter::run(&e, &mut bind)?;
        }
        let batch_count = (count / POINTS).max(5);
        let mut core_batch = batch.clone();
        let (reads, mut writes) = split(&k, &mut core_batch, &strides);
        prepared.run_batch(&mut state, POINTS, &reads, &mut writes)?;
        drop((reads, writes));
        let mut jit_batch = batch.clone();
        let (reads, mut writes) = split(&k, &mut jit_batch, &strides);
        jit.run_batch(&mut jstate, POINTS, &reads, &mut writes)?;
        drop((reads, writes));
        if !same(&sequential, &core_batch) || !same(&sequential, &jit_batch) {
            return Err(format!("batch mismatch in {label}").into());
        }
        let mut timed = batch.clone();
        let (reads, mut writes) = split(&k, &mut timed, &strides);
        let m1_batch_ns = median(
            || {
                prepared
                    .run_batch(
                        &mut state,
                        POINTS,
                        black_box(&reads),
                        black_box(&mut writes),
                    )
                    .unwrap()
            },
            batch_count,
        ) / POINTS as f64;
        let jit_batch_ns = median(
            || {
                jit.run_batch(
                    &mut jstate,
                    POINTS,
                    black_box(&reads),
                    black_box(&mut writes),
                )
                .unwrap()
            },
            batch_count,
        ) / POINTS as f64;
        drop((reads, writes));
        let (rust_ns, rust_bits) = batched_rust(&e, &batch, &strides, batch_count)?;
        let want: Vec<u64> = sequential.iter().flatten().map(|x| x.to_bits()).collect();
        if rust_bits.len() != want.len()
            || rust_bits.iter().zip(&want).any(|(a, b)| {
                a != b && !(f64::from_bits(*a).is_nan() && f64::from_bits(*b).is_nan())
            })
        {
            return Err(format!("batched Rust mismatch in {label}").into());
        }
        let rust_ns = rust_ns / POINTS as f64;
        writeln!(
            table,
            "| {label} | {interpreter_ns:.1} | {m1_ns:.1} | {m1_batch_ns:.1} | {checked_ns:.1} | {jit_prepared_ns:.1} | {positional_ns:.1} | {jit_batch_ns:.1} | {rust_ns:.1} | {:.0} us |",
            compile_ns / 1000.0
        )?;
        results.push(
            json!({"kernel": label, "interpreter_ns": interpreter_ns, "m1_prepared_ns": m1_ns,
            "m1_batch_per_point_ns": m1_batch_ns, "jit_checked_ns": checked_ns,
            "jit_prepared_ns": jit_prepared_ns, "jit_positional_ns": positional_ns,
            "jit_batch_per_point_ns": jit_batch_ns, "batched_rust_per_point_ns": rust_ns,
            "jit_compile_ns": compile_ns, "bitwise_match": true}),
        );
    }
    eprintln!("{table}");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "cranelift": "0.128.3",
            "build": "cargo --release (opt-level 3); batched Rust via rustc -O, width 8, no AVX2 dispatch",
            "timing": "median of 7 batches, ns per call; batch columns: one call over 64 independent strided points, per point",
            "results": results}))?
    );
    Ok(())
}
