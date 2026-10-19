//! The outward padding of `malleus::interval` is the Rump–Zimmermann–Boldo–Melquiond
//! successor bound: every padded endpoint reaches at least the neighbouring double of the
//! rounded value, on adversarial near-underflow, near-overflow and exact-tie inputs.
//!
//! Input contract: endpoints are ordered and non-NaN, and the caller has checked
//! `ieee_environment_supported()`.

mod support;

use malleus::interval::{ieee_environment_supported, lower_intervals};
use malleus::*;
use support::dyadic::Dyadic;
use support::*;

/// Point-interval kernel over `x`, `y` with outputs `x + y`, `x - y`, `x * y` and `x + 0`.
fn ops_kernel(n: usize) -> StructuredKernel {
    let axis = AxisId::new(0);
    let load = |i| ScalarExpr::Load(OperandId::new(i));
    let mut operands = vec![
        KernelOperand::tensor("x", vec![n], AccessMode::Read),
        KernelOperand::tensor("y", vec![n], AccessMode::Read),
    ];
    for name in ["sum", "difference", "product", "identity"] {
        operands.push(KernelOperand::tensor(name, vec![n], AccessMode::Write));
    }
    let values = [
        ScalarExpr::binary(BinaryOp::Add, load(0), load(1)),
        ScalarExpr::binary(BinaryOp::Sub, load(0), load(1)),
        ScalarExpr::binary(BinaryOp::Mul, load(0), load(1)),
        ScalarExpr::binary(BinaryOp::Add, load(0), ScalarExpr::Constant(0.0)),
    ];
    StructuredKernel {
        name: "padding".into(),
        iteration_domain: IterationDomain::new(vec![n]),
        iterators: vec![IteratorKind::Parallel],
        indexing_maps: (0..6)
            .map(|i| IndexingMap::new(OperandId::new(i), vec![IndexExpr::axis(axis)]))
            .collect(),
        operands,
        body: KernelRegion {
            statements: values
                .into_iter()
                .enumerate()
                .map(|(i, value)| Statement::Store {
                    operand: OperandId::new(2 + i),
                    value,
                })
                .collect(),
        },
        numeric_policy: NumericPolicy::default(),
    }
}

/// Run the lowered kernel on point intervals; returns `[lo, hi]` buffers per output.
fn run(xs: &[f64], ys: &[f64]) -> Vec<Vec<f64>> {
    let n = xs.len();
    let program = lower_intervals(&validate(ops_kernel(n)).unwrap()).unwrap();
    let mut buffers = vec![xs.to_vec(), xs.to_vec(), ys.to_vec(), ys.to_vec()];
    buffers.extend((0..8).map(|_| vec![f64::NAN; n]));
    interpret(&program.executable, buffers).split_off(4)
}

/// Every power of two from the smallest subnormal to 2^1023, both signs, with neighbours,
/// plus the extreme magnitudes.
fn binade_edges() -> Vec<f64> {
    let mut v = vec![0.0, f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)];
    for e in -1074..=1023 {
        let p = 2f64.powi(e);
        v.extend([p, p.next_up(), p.next_down(), 3.0 * p]);
    }
    let negatives: Vec<f64> = v.iter().map(|x| -x).collect();
    v.extend(negatives);
    v.retain(|x| x.is_finite());
    v
}

/// The bound the module used before the successor bound: `RN(|v|·2^-52 + MIN_POSITIVE)`.
fn previous(v: f64) -> (f64, f64) {
    let r = v.abs() * f64::EPSILON + f64::MIN_POSITIVE;
    (v - r, v + r)
}

#[test]
fn padded_endpoints_reach_the_neighbouring_doubles() {
    assert!(ieee_environment_supported());
    let mut values = binade_edges();
    let mut r = Rng(0x5EED_F5CC);
    values.extend((0..20_000).map(|_| r.finite()));
    let zeros = vec![0.0; values.len()];
    let out = run(&values, &zeros);
    let (lo, hi) = (&out[6], &out[7]);
    let (mut tight, mut tighter) = (0, 0);
    for (i, &v) in values.iter().enumerate() {
        // `x + 0` rounds to `v` itself (or +0 for -0).
        let v = v + 0.0;
        assert!(
            lo[i] <= v.next_down() && hi[i] >= v.next_up(),
            "{v:e}: [{:e}, {:e}]",
            lo[i],
            hi[i]
        );
        let (old_lo, old_hi) = previous(v);
        assert!(
            old_lo <= lo[i] && hi[i] <= old_hi,
            "{v:e}: wider than before"
        );
        tight += (lo[i] == v.next_down() && hi[i] == v.next_up()) as usize;
        tighter += (old_lo < lo[i] || hi[i] < old_hi) as usize;
    }
    eprintln!(
        "padding: {tight}/{} exactly [pred, succ]; {tighter} strictly tighter than the previous bound",
        values.len()
    );
    assert!(tight * 100 >= values.len() * 99, "only {tight} tight");
}

#[test]
fn adversarial_operations_stay_enclosed() {
    assert!(ieee_environment_supported());
    let tiny = f64::from_bits(1);
    let mut set = vec![
        0.0,
        tiny,
        2.0 * tiny,
        3.0 * tiny,
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE.next_up(),
        f64::MIN_POSITIVE.next_down(),
        f64::MIN_POSITIVE * 0.5,
        2f64.powi(-537),
        2f64.powi(-538) * 1.5,
        2f64.powi(-53),
        2f64.powi(-52),
        1.0,
        1.0f64.next_up(),
        1.0f64.next_down(),
        1.5,
        3.0,
        0.1,
        1.0 / 3.0,
        2f64.powi(970),
        2f64.powi(971),
        2f64.powi(1023),
        f64::MAX,
        f64::MAX.next_down(),
        1.3407807929942596e154,
        2f64.powi(512),
    ];
    let negatives: Vec<f64> = set.iter().map(|x| -x).collect();
    set.extend(negatives);
    let (mut xs, mut ys) = (vec![], vec![]);
    for &x in &set {
        for &y in &set {
            xs.push(x);
            ys.push(y);
        }
    }
    let out = run(&xs, &ys);
    let mut finite = 0;
    for i in 0..xs.len() {
        let (x, y) = (Dyadic::from_f64(xs[i]), Dyadic::from_f64(ys[i]));
        for (k, exact) in [x.add(&y), x.sub(&y), x.mul(&y)].iter().enumerate() {
            let (lo, hi) = (out[2 * k][i], out[2 * k + 1][i]);
            assert!(
                exact.within(lo, hi),
                "op {k} on {:e}, {:e}: [{lo:e}, {hi:e}]",
                xs[i],
                ys[i]
            );
            finite += (lo.is_finite() && hi.is_finite()) as usize;
        }
    }
    assert!(finite * 2 > 3 * xs.len(), "only {finite} finite enclosures");
}
