//! Randomized enclosure soundness of `malleus::interval::lower_intervals`.
//!
//! Random kernels over neg/add/sub/mul/correlated squares/constants/loads/locals/index
//! branches are lowered, run on the reference interpreter, and every output interval is
//! checked against the exact (dyadic rational) value of the original real expression at the
//! interval corners, midpoint and random interior points. Inputs include subnormals,
//! near-MAX values, +-0, point intervals and wide intervals.
//!
//! Input contract: endpoints are ordered and non-NaN, and the caller has checked
//! `ieee_environment_supported()`.

mod support;

use malleus::interval::ieee_environment_supported;
use support::dyadic::Dyadic;
use support::*;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const CASES: usize = 2000;

/// A point interval `[x, x]` with a moderate exponent so results stay finite and normal.
fn point(r: &mut Rng) -> (f64, f64) {
    let mag = r.below(40) as i32 - 20;
    let x = (r.below(2001) as f64 - 1000.0) / 7.0 * 2f64.powi(mag);
    (x, x)
}

fn step_inward(mut x: f64, ulps: u32, up: bool) -> f64 {
    for _ in 0..ulps {
        x = if up { x.next_up() } else { x.next_down() };
    }
    x
}

#[test]
fn dyadic_arithmetic_matches_i128_and_f64_decomposition() {
    let mut r = Rng(SEED);
    for _ in 0..2000 {
        let (a, b) = (r.next() as i64 >> 40, r.next() as i64 >> 40);
        let (da, db) = (Dyadic::from_i64(a), Dyadic::from_i64(b));
        assert_eq!(
            da.mul(&db).cmp_f64((a as i128 * b as i128) as f64),
            Some(std::cmp::Ordering::Equal)
        );
        assert_eq!(
            da.add(&db).cmp_f64((a + b) as f64),
            Some(std::cmp::Ordering::Equal)
        );
        assert_eq!(
            da.sub(&db).cmp_f64((a - b) as f64),
            Some(std::cmp::Ordering::Equal)
        );
    }
    // Exact where f64 rounds: 0.1 * 3 is strictly between the neighbouring doubles of 0.3.
    let tenth = Dyadic::from_f64(0.1).mul(&Dyadic::from_i64(3));
    let three_tenths = Dyadic::from_f64(0.1 * 3.0);
    assert!(!tenth.sub(&three_tenths).is_zero());
    // Extreme exponents cancel exactly.
    let big = Dyadic::from_f64(f64::MAX).mul(&Dyadic::from_f64(f64::MAX));
    assert!(big.sub(&big).is_zero());
    let tiny = Dyadic::from_f64(f64::from_bits(1)).mul(&Dyadic::from_f64(f64::from_bits(1)));
    assert!(tiny.cmp_f64(0.0) == Some(std::cmp::Ordering::Greater));
    assert!(tiny.add(&big).cmp_f64(f64::MAX) == Some(std::cmp::Ordering::Greater));
    assert!(Dyadic::from_f64(-0.0).neg().is_zero());
}

#[test]
fn lowered_intervals_contain_the_exact_real_value() {
    assert!(ieee_environment_supported());
    let mut r = Rng(SEED);
    let (mut cases, mut finite, mut elements) = (0, 0, 0);
    for _ in 0..CASES {
        let prog = Prog::generate(&mut r, 4);
        let inputs = random_inputs(&mut r, prog.n, false);
        let lowered = prog.lower();
        let out = interpret(&lowered.executable, endpoint_buffers(prog.n, &inputs));
        let (lo, hi) = (&out[8], &out[9]);
        if let Some(v) = first_violation(&prog, &inputs, lo, hi, &mut r) {
            panic!("soundness violation: {v}");
        }
        cases += 1;
        for i in 0..prog.n {
            assert!(
                !lo[i].is_nan() && !hi[i].is_nan(),
                "NaN endpoint: {prog:?} {inputs:?}"
            );
            elements += 1;
            finite += (lo[i].is_finite() && hi[i].is_finite()) as usize;
        }
    }
    // Guard against vacuous passes where every enclosure is (-inf, inf).
    assert!(
        finite * 2 > elements,
        "only {finite}/{elements} finite enclosures"
    );
    assert_eq!(cases, CASES);
}

#[test]
fn checker_rejects_inward_shrunk_bounds() {
    // Negative control: the identical checker must flag bounds pulled in by 3 ulps. Point
    // inputs make every bound tight against the exact value, so the shrink must be seen.
    let mut r = Rng(SEED ^ 0xC0FFEE);
    let single_ops: [fn() -> Ex; 4] = [
        || Ex::Add(Box::new(Ex::Load(0)), Box::new(Ex::Load(1))),
        || Ex::Sub(Box::new(Ex::Load(0)), Box::new(Ex::Load(1))),
        || Ex::Mul(Box::new(Ex::Load(0)), Box::new(Ex::Load(1))),
        || Ex::Square(Box::new(Ex::Load(0))),
    ];
    let (mut checked, mut caught) = (0, 0);
    for make in single_ops {
        for _ in 0..100 {
            let prog = Prog {
                n: 1,
                lets: vec![],
                store: make(),
            };
            let inputs: Inputs = vec![std::array::from_fn(|_| point(&mut r))];
            let out = interpret(&prog.lower().executable, endpoint_buffers(1, &inputs));
            // Unshrunk bounds pass.
            assert!(first_violation(&prog, &inputs, &out[8], &out[9], &mut r).is_none());
            if !(out[8][0].is_finite() && out[9][0].is_finite()) {
                continue;
            }
            checked += 1;
            let lo = [step_inward(out[8][0], 3, true)];
            let hi = [step_inward(out[9][0], 3, false)];
            caught += first_violation(&prog, &inputs, &lo, &hi, &mut r).is_some() as usize;
        }
    }
    assert!(checked > 300);
    assert_eq!(caught, checked, "checker missed shrunk bounds");

    // Same control on the general corpus: shrinking must be caught in most cases.
    let (mut total, mut flagged) = (0, 0);
    for _ in 0..400 {
        let prog = Prog::generate(&mut r, 3);
        let inputs: Inputs = (0..prog.n)
            .map(|_| std::array::from_fn(|_| point(&mut r)))
            .collect();
        let out = interpret(&prog.lower().executable, endpoint_buffers(prog.n, &inputs));
        if out[8].iter().chain(&out[9]).any(|x| !x.is_finite()) {
            continue;
        }
        total += 1;
        let lo: Vec<_> = out[8].iter().map(|&x| step_inward(x, 3, true)).collect();
        let hi: Vec<_> = out[9].iter().map(|&x| step_inward(x, 3, false)).collect();
        flagged += first_violation(&prog, &inputs, &lo, &hi, &mut r).is_some() as usize;
    }
    assert!(
        total > 100 && flagged * 2 > total,
        "flagged {flagged}/{total}"
    );
}
