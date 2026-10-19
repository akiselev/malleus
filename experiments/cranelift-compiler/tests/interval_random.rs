//! Random interval programs: the Cranelift JIT must match the reference interpreter bit for
//! bit, including infinite and extreme endpoints. The generator is shared with the core
//! soundness and source-differential tests.

#[path = "../../../tests/support/mod.rs"]
mod support;

use malleus::*;
use malleus_cranelift_experiment::{Error, JitKernel};
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
