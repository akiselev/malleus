//! Centered-form and affine enclosures (`malleus::interval::{lower_centered_intervals,
//! lower_affine_intervals}`) contain the exact real value on random programs and boxes, are
//! never wider than the naive enclosure (a theorem: both intersect with it), and are reported
//! against it on narrow boxes.
//!
//! Input contract: endpoints are ordered and non-NaN, and the caller has checked
//! `ieee_environment_supported()`.

mod support;

use malleus::interval::{
    IntervalProgram, ieee_environment_supported, lower_affine_intervals, lower_centered_intervals,
    lower_intervals,
};
use malleus::*;
use support::dyadic::Dyadic;
use support::*;

type Lowering = fn(&ValidatedKernel) -> Result<IntervalProgram, LoweringError>;
const UPGRADES: [(&str, Lowering); 2] = [
    ("centered", lower_centered_intervals),
    ("affine", lower_affine_intervals),
];

fn width(lo: f64, hi: f64) -> f64 {
    hi - lo
}

/// Moderate centres with radii from 2^-4 down to 2^-30.
fn narrow_inputs(r: &mut Rng, n: usize) -> Inputs {
    (0..n)
        .map(|_| {
            std::array::from_fn(|_| {
                let m = (r.below(2001) as f64 - 1000.0) / 256.0;
                let radius = 2f64.powi(-(4 + r.below(27) as i32));
                (m - radius, m + radius)
            })
        })
        .collect()
}

/// Check soundness and the naive-containment theorem; return per-lowering width ratios
/// (sum, count) over elements where the naive width is finite and positive.
fn check(r: &mut Rng, cases: usize, narrow: bool) -> [(f64, usize); 2] {
    let mut ratios = [(0.0, 0); 2];
    let mut refused = [0; 2];
    for _ in 0..cases {
        let prog = Prog::generate(r, 4);
        let kernel = validate(prog.kernel()).unwrap();
        let inputs = if narrow {
            narrow_inputs(r, prog.n)
        } else {
            let unbounded = r.below(4) == 0;
            random_inputs(r, prog.n, unbounded)
        };
        let buffers = endpoint_buffers(prog.n, &inputs);
        let naive = interpret(
            &lower_intervals(&kernel).unwrap().executable,
            buffers.clone(),
        );
        for (k, (name, lower)) in UPGRADES.iter().enumerate() {
            // Expanded derivatives can exceed the lowering budget: a typed refusal.
            let program = match lower(&kernel) {
                Ok(program) => program,
                Err(LoweringError::Unsupported(_)) => {
                    refused[k] += 1;
                    continue;
                }
                Err(e) => panic!("{name}: {e}"),
            };
            let out = interpret(&program.executable, buffers.clone());
            let (lo, hi) = (&out[8], &out[9]);
            if !inputs
                .iter()
                .flatten()
                .any(|(a, b)| a.is_infinite() || b.is_infinite())
                && let Some(v) = first_violation(&prog, &inputs, lo, hi, r)
            {
                panic!("{name} soundness violation: {v}");
            }
            for i in 0..prog.n {
                assert!(!lo[i].is_nan() && !hi[i].is_nan(), "{name}: NaN endpoint");
                assert!(
                    naive[8][i] <= lo[i] && hi[i] <= naive[9][i],
                    "{name}: wider than naive on {prog:?} {inputs:?}"
                );
                let w = width(naive[8][i], naive[9][i]);
                if w.is_finite() && w > 0.0 {
                    ratios[k].0 += width(lo[i], hi[i]) / w;
                    ratios[k].1 += 1;
                }
            }
        }
    }
    assert!(
        refused.iter().all(|&n| n * 50 < cases),
        "budget refusals {refused:?} of {cases}"
    );
    ratios
}

#[test]
fn upgraded_enclosures_contain_the_exact_real_value() {
    assert!(ieee_environment_supported());
    let mut r = Rng(0xCE47_E8ED_AFF1_4E00);
    let wide = check(&mut r, 600, false);
    let narrow = check(&mut r, 600, true);
    for (k, (name, _)) in UPGRADES.iter().enumerate() {
        eprintln!(
            "{name}: mean width / naive width = {:.4} on wide boxes ({} elements), {:.4} on narrow boxes ({} elements)",
            wide[k].0 / wide[k].1 as f64,
            wide[k].1,
            narrow[k].0 / narrow[k].1 as f64,
            narrow[k].1
        );
        assert!(narrow[k].1 > 300, "{name}: too few finite naive enclosures");
    }
}

/// Infinite input endpoints: soundness is checked at finite sample points of each box.
#[test]
fn unbounded_inputs_never_produce_nan_or_optimistic_bounds() {
    let mut r = Rng(0x1A2B_3C4D);
    for _ in 0..300 {
        let prog = Prog::generate(&mut r, 3);
        let kernel = validate(prog.kernel()).unwrap();
        let inputs = random_inputs(&mut r, prog.n, true);
        let buffers = endpoint_buffers(prog.n, &inputs);
        let naive = interpret(
            &lower_intervals(&kernel).unwrap().executable,
            buffers.clone(),
        );
        // Finite sample boxes inside each (possibly unbounded) input.
        let clipped: Inputs = inputs
            .iter()
            .map(|e| e.map(|(a, b)| (a.max(-f64::MAX), b.min(f64::MAX))))
            .collect();
        for (name, lower) in UPGRADES {
            let Ok(program) = lower(&kernel) else {
                continue;
            };
            let out = interpret(&program.executable, buffers.clone());
            if let Some(v) = first_violation(&prog, &clipped, &out[8], &out[9], &mut r) {
                panic!("{name}: {v}");
            }
            for i in 0..prog.n {
                assert!(!out[8][i].is_nan() && !out[9][i].is_nan());
                assert!(naive[8][i] <= out[8][i] && out[9][i] <= naive[9][i]);
            }
        }
    }
}

fn scalar_kernel(inputs: usize, value: ScalarExpr) -> StructuredKernel {
    let mut operands: Vec<_> = (0..inputs)
        .map(|i| KernelOperand::scalar(format!("x{i}"), AccessMode::Read))
        .collect();
    operands.push(KernelOperand::scalar("out", AccessMode::Write));
    StructuredKernel {
        name: "scalar".into(),
        iteration_domain: IterationDomain::default(),
        iterators: vec![],
        indexing_maps: (0..=inputs)
            .map(|i| IndexingMap::scalar(OperandId::new(i)))
            .collect(),
        operands,
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: OperandId::new(inputs),
                value,
            }],
        },
        numeric_policy: NumericPolicy::default(),
    }
}

fn enclose(lower: Lowering, k: &StructuredKernel, boxes: &[(f64, f64)]) -> (f64, f64) {
    let mut buffers: Vec<Vec<f64>> = boxes
        .iter()
        .flat_map(|&(a, b)| [vec![a], vec![b]])
        .collect();
    buffers.extend([vec![f64::NAN], vec![f64::NAN]]);
    let out = interpret(
        &lower(&validate(k.clone()).unwrap()).unwrap().executable,
        buffers,
    );
    let n = out.len();
    (out[n - 2][0], out[n - 1][0])
}

#[test]
fn dependency_is_reduced_on_a_known_example() {
    // f(x) = x·(1 − x) on [0.4, 0.6]: exact range [0.24, 0.25]; naive [0.16, 0.36];
    // centered 0.25 + [−0.2, 0.2]·[−0.1, 0.1] = [0.23, 0.27]; affine 0.25 ± 0.01.
    let x = ScalarExpr::Load(OperandId::new(0));
    let k = scalar_kernel(
        1,
        ScalarExpr::binary(
            BinaryOp::Mul,
            x.clone(),
            ScalarExpr::binary(BinaryOp::Sub, ScalarExpr::Constant(1.0), x),
        ),
    );
    let b = [(0.4, 0.6)];
    let naive = enclose(lower_intervals, &k, &b);
    let centered = enclose(lower_centered_intervals, &k, &b);
    let affine = enclose(lower_affine_intervals, &k, &b);
    eprintln!("x(1-x) on [0.4, 0.6]: naive {naive:?}, centered {centered:?}, affine {affine:?}");
    assert!(naive.0 <= 0.16 && naive.1 >= 0.36);
    for (lo, hi) in [centered, affine] {
        assert!(lo <= 0.24 && hi >= 0.25);
    }
    assert!(width(centered.0, centered.1) < 0.0401);
    assert!(width(affine.0, affine.1) < 0.0201);
}

#[test]
fn fma_encloses_the_real_multiply_add() {
    let load = |i| ScalarExpr::Load(OperandId::new(i));
    let k = scalar_kernel(3, ScalarExpr::fma(load(0), load(1), load(2)));
    let mut r = Rng(0xF3A_0001);
    for _ in 0..2000 {
        let boxes: Vec<(f64, f64)> = (0..3).map(|_| r.interval(false)).collect();
        for lower in [
            lower_intervals,
            lower_centered_intervals,
            lower_affine_intervals,
        ] {
            let (lo, hi) = enclose(lower, &k, &boxes);
            assert!(!lo.is_nan() && !hi.is_nan());
            for corner in 0..8 {
                let p: Vec<Dyadic> = (0..3)
                    .map(|j| {
                        Dyadic::from_f64(if corner >> j & 1 == 0 {
                            boxes[j].0
                        } else {
                            boxes[j].1
                        })
                    })
                    .collect();
                let exact = p[0].mul(&p[1]).add(&p[2]);
                assert!(exact.within(lo, hi), "{boxes:?}: [{lo:e}, {hi:e}]");
            }
        }
    }
}

#[test]
fn unsupported_operations_and_policies_refuse() {
    let load = |i| ScalarExpr::Load(OperandId::new(i));
    let refused = [
        scalar_kernel(2, ScalarExpr::binary(BinaryOp::Div, load(0), load(1))),
        scalar_kernel(1, ScalarExpr::unary(UnaryOp::Sin, load(0))),
        scalar_kernel(1, ScalarExpr::unary(UnaryOp::Abs, load(0))),
    ];
    for (name, lower) in UPGRADES {
        for k in &refused {
            assert!(
                matches!(
                    lower(&validate(k.clone()).unwrap()),
                    Err(LoweringError::Unsupported(_))
                ),
                "{name} admitted {k:?}"
            );
        }
        let mut k = scalar_kernel(1, load(0));
        k.numeric_policy.scalar_type = ScalarType::F32;
        assert!(lower(&validate(k).unwrap()).is_err());
        assert!(lower(&validate(fixtures::quadratic(3)).unwrap()).is_err());
    }
}
