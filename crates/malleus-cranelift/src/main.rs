//! Compile-cost and warm interpreter/native comparison on identical Malleus executables.

use malleus::*;
use malleus_cranelift::{JitKernel, fixtures};
use serde_json::json;
use std::{hint::black_box, time::Instant};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut kernels = vec![];
    for n in [1, 8, 64, 1024, 4096] {
        kernels.push(("guard-polynomial".to_owned(), fixtures::polynomial(n), n));
    }
    for n in [8, 1024] {
        let primal = fixtures::quadratic(n);
        kernels.push(("quadratic-primal".into(), primal.clone(), n));
        for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
            let p = differentiate(
                &primal,
                &DerivativeRequest {
                    mode,
                    independent_operands: vec![OperandId::new(0), OperandId::new(1)],
                    dependent_operands: vec![OperandId::new(2)],
                },
            )?;
            kernels.push((format!("quadratic-{mode:?}"), p.kernel, n));
        }
    }
    for n in [16, 64] {
        kernels.push((format!("matvec-{n}x{n}"), fixtures::matvec(n), n * n));
    }
    let mut results = vec![];
    for (label, k, n) in kernels {
        let e = Executable::reference(validate(k)?);
        let jit = JitKernel::compile(&e)?;
        let mut compiled = Vec::with_capacity(21);
        let compile_ns = median(
            || {
                compiled.push(black_box(
                    JitKernel::compile(black_box(&e)).expect("tested compile"),
                ));
            },
            3,
        );
        drop(compiled);
        let base = fixtures::buffers(e.kernel().as_kernel());
        let mut reference = base.clone();
        let mut native = base.clone();
        {
            let mut b: Vec<_> = reference
                .iter_mut()
                .enumerate()
                .map(|(i, x)| BufferBinding::new(OperandId::new(i), x))
                .collect();
            Interpreter::run(&e, &mut b)?;
        }
        {
            let mut b: Vec<_> = native
                .iter_mut()
                .enumerate()
                .map(|(i, x)| BufferBinding::new(OperandId::new(i), x))
                .collect();
            jit.run(&mut b)?;
        }
        let matched = reference
            .iter()
            .flatten()
            .zip(native.iter().flatten())
            .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
        if !matched {
            return Err(format!("mismatch in {label}-{n}").into());
        }
        let count = if n <= 8 {
            5000
        } else if n <= 64 {
            1000
        } else {
            50
        };
        let mut buffers = base.clone();
        let mut bindings: Vec<_> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
            .collect();
        let interpreter_ns = median(
            || {
                Interpreter::run(black_box(&e), black_box(&mut bindings))
                    .expect("tested interpreter");
            },
            count,
        );
        let checked_jit_ns = median(
            || {
                jit.run(black_box(&mut bindings)).expect("tested binding");
            },
            count,
        );
        let mut prepared = jit.prepare(&mut bindings)?;
        let prepared_jit_ns = median(
            || {
                black_box(&mut prepared).run();
            },
            count,
        );
        results.push(json!({"kernel":label,"points":n,"compile_ns":compile_ns,
            "interpreter_ns":interpreter_ns,"checked_jit_ns":checked_jit_ns,
            "prepared_jit_ns":prepared_jit_ns,"checked_speedup":interpreter_ns/checked_jit_ns,
            "prepared_speedup":interpreter_ns/prepared_jit_ns,
            "checked_break_even_calls":if interpreter_ns>checked_jit_ns{Some((compile_ns/(interpreter_ns-checked_jit_ns)).ceil())}else{None},
            "bitwise_match":matched}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
        "cranelift":"0.128.3 (direct dependency pins; full transitive versions in Cargo.lock)",
        "timing":"median of 7 batches; compile batches 3 kernels; interpreter/JIT use equal repeat counts",
        "host_build":"prototype opt-level=1; core interpreter opt-level=3; Cranelift generated code opt_level=speed",
        "scope":"plain f32/f64 execution, not a certified interval backend or full CAD Boolean",
        "results":results}))?
    );
    Ok(())
}
