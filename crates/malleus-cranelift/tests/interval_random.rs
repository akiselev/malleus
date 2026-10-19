//! Random interval programs: the Cranelift JIT must match the reference interpreter bit for
//! bit, including infinite and extreme endpoints. The generator is shared with the core
//! soundness and source-differential tests.
// Executes JIT-generated machine code, which miri cannot interpret.
#![cfg(not(miri))]

#[path = "../../../tests/support/mod.rs"]
mod support;

use malleus::*;
use malleus_cranelift::{Error, JitKernel};
use support::*;

#[test]
fn jit_matches_interpreter_on_random_interval_programs() {
    assert!(malleus::interval::ieee_environment_supported());
    let mut r = Rng(0x717_C0DE_0000_0001);
    let mut refused = 0;
    for case in 0..300 {
        let prog = Prog::generate(&mut r, 4);
        let program = prog.lower();
        // The JIT refuses oversized lowered programs with a typed budget error; any other
        // failure is a bug.
        let jit = match JitKernel::compile(&program.executable) {
            Ok(jit) => jit,
            Err(Error::Unsupported(_)) => {
                refused += 1;
                continue;
            }
            Err(e) => panic!("case {case}: {e}"),
        };
        for _ in 0..3 {
            let inputs = random_inputs(&mut r, prog.n, true);
            let buffers = endpoint_buffers(prog.n, &inputs);
            let want = interpret(&program.executable, buffers.clone());
            let mut got = buffers;
            let mut bindings: Vec<_> = got
                .iter_mut()
                .enumerate()
                .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
                .collect();
            jit.run(&mut bindings).unwrap();
            drop(bindings);
            for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
                assert!(
                    same_bits(*g, *w),
                    "case {case}: jit {g:e} ({:#x}) vs interpreter {w:e} ({:#x}); {prog:?} {inputs:?}",
                    g.to_bits(),
                    w.to_bits()
                );
            }
        }
    }
    assert!(
        refused < 30,
        "{refused} of 300 programs (expected a handful) refused by the JIT budget"
    );
}

#[test]
fn jit_matches_interpreter_on_centered_and_affine_enclosures() {
    use malleus::interval::{lower_affine_intervals, lower_centered_intervals};
    let mut r = Rng(0x717_C0DE_0000_0002);
    let mut compared = 0;
    for case in 0..60 {
        let prog = Prog::generate(&mut r, 3);
        let kernel = validate(prog.kernel()).unwrap();
        for lower in [lower_centered_intervals, lower_affine_intervals] {
            let (Ok(program), inputs) = (lower(&kernel), random_inputs(&mut r, prog.n, true))
            else {
                continue;
            };
            let Ok(jit) = JitKernel::compile(&program.executable) else {
                continue;
            };
            let buffers = endpoint_buffers(prog.n, &inputs);
            let want = interpret(&program.executable, buffers.clone());
            let mut got = buffers;
            let mut bindings: Vec<_> = got
                .iter_mut()
                .enumerate()
                .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
                .collect();
            jit.run(&mut bindings).unwrap();
            drop(bindings);
            for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
                assert!(
                    same_bits(*g, *w),
                    "case {case}: jit {g:e} vs interpreter {w:e}"
                );
            }
            compared += 1;
        }
    }
    eprintln!("compared {compared} of 120 lowered programs (the rest exceed a budget)");
    assert!(compared > 30, "only {compared} programs compared");
}
