//! Differential check of lowered interval kernels: the reference interpreter and the
//! compiled strict Rust source (looped and unrolled) must agree bit for bit.
//!
//! Many random programs are batched into one rustc invocation to keep compile time small.
//! Inputs include infinite endpoints, which the lowering widens conservatively; NaNs are
//! compared as a class.

mod support;

use malleus::{lower_rust, lower_rust_unrolled};
use std::fmt::Write;
use support::*;

const KERNELS: usize = 32;
const INPUT_SETS: usize = 3;

#[test]
fn interpreter_and_rust_source_agree_bitwise_on_random_interval_programs() {
    assert!(malleus::interval::ieee_environment_supported());
    let mut r = Rng(0xD1FF_E4E5_71A7_0001);
    let mut code = String::new();
    let mut main = String::from("fn main() {\n");
    let mut expected: Vec<Vec<Vec<f64>>> = vec![];
    for case in 0..KERNELS {
        let prog = Prog::generate(&mut r, 2);
        let program = prog.lower();
        let name = format!("kernel_{case}");
        code.push_str(&if case % 2 == 0 {
            lower_rust(&program.executable, &name).unwrap()
        } else {
            lower_rust_unrolled(&program.executable, &name, 64).unwrap()
        });
        for _ in 0..INPUT_SETS {
            let inputs = random_inputs(&mut r, prog.n, true);
            let buffers = endpoint_buffers(prog.n, &inputs);
            main.push_str("{\n");
            for (i, b) in buffers.iter().enumerate() {
                let values = b
                    .iter()
                    .map(|x| format!("f64::from_bits({}u64)", x.to_bits()))
                    .collect::<Vec<_>>()
                    .join(",");
                let kw = if i < 8 { "" } else { "mut " };
                writeln!(main, "let {kw}b{i}=vec![{values}];").unwrap();
            }
            let args = (0..10)
                .map(|i| format!("&{}b{i}", if i < 8 { "" } else { "mut " }))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(main, "{name}({args}).unwrap();").unwrap();
            main.push_str("println!(\"{:?}\",vec![b8.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),b9.iter().map(|x|x.to_bits()).collect::<Vec<_>>()]);\n}\n");
            let out = interpret(&program.executable, buffers);
            assert!(
                out[8].iter().chain(&out[9]).all(|x| !x.is_nan()),
                "NaN output endpoint: {prog:?} {inputs:?}"
            );
            expected.push(vec![out[8].clone(), out[9].clone()]);
        }
    }
    main.push_str("}\n");
    code.push_str(&main);
    let stdout = rustc_run("interval-differential", &code);
    assert_eq!(stdout.lines().count(), expected.len());
    for (n, (line, want)) in stdout.lines().zip(&expected).enumerate() {
        let got: Vec<Vec<u64>> = serde_json::from_str(line).unwrap();
        for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
            assert!(
                same_bits(f64::from_bits(*g), *w),
                "run {n}: source {:e} ({g:#x}) vs interpreter {w:e} ({:#x})",
                f64::from_bits(*g),
                w.to_bits()
            );
        }
    }
}
