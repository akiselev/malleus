//! W11 M3 package 4: SMT translation validation of IR-to-IR passes with z3.
#![cfg(not(miri))]

mod support;

use std::path::Path;
use std::time::Duration;

use malleus::ad::{DerivativeOptions, Simplification};
use malleus::simplify::simplify_exact;
use malleus::smt::*;
use malleus::*;
use support::fixtures;
use support::kernels::*;

const TIMEOUT: Duration = Duration::from_secs(20);

fn z3() -> &'static Path {
    let path = Path::new("/usr/bin/z3");
    assert!(
        path.exists(),
        "z3 is required for SMT translation validation tests and was not found at /usr/bin/z3"
    );
    path
}

fn exact() -> DerivativeOptions {
    DerivativeOptions {
        simplification: Simplification::Exact,
        ..Default::default()
    }
}

#[test]
fn post_ad_simplification_is_validated() {
    let mut validated = 0;
    for k in [
        fixtures::polynomial(2),
        fixtures::quadratic(2),
        fixtures::polynomial_batch(2),
        nonsmooth(2),
        select_only(2),
        fma_exp(2),
        min_reduction(2),
        f32_variant(fixtures::polynomial(2)),
    ] {
        for mode in [
            DerivativeMode::Jvp,
            DerivativeMode::Vjp,
            DerivativeMode::Jacobian,
        ] {
            let request = full_request(&k, mode);
            let plain = differentiate(&k, &request).unwrap();
            let simple = malleus::ad::differentiate_with(&k, &request, &exact()).unwrap();
            if plain.kernel == simple.kernel {
                continue;
            }
            let outcome = validate_rewrite(z3(), &plain.kernel, &simple.kernel, TIMEOUT).unwrap();
            assert_eq!(outcome, TvOutcome::Validated, "{} {mode:?}", k.name);
            validated += 1;
        }
    }
    assert!(
        validated >= 6,
        "only {validated} simplified derivative kernels validated"
    );
}

#[test]
fn simplification_of_random_kernels_with_specials_is_validated() {
    let mut r = support::Rng(0x5a7_0001);
    let (mut validated, mut attempted, mut undecided) = (0, 0, 0);
    for i in 0..24 {
        let k = random_kernel(&mut r, i % 3 == 0);
        let s = simplify_exact(&k);
        if s.rewrites == 0 {
            continue;
        }
        attempted += 1;
        match validate_rewrite(z3(), &k, &s.kernel, TIMEOUT).unwrap() {
            TvOutcome::Validated => validated += 1,
            TvOutcome::Counterexample { model } => panic!("{:?}\n{model}", k.body),
            // A timeout claims nothing; it is not counted as validated.
            TvOutcome::NotValidated { .. } => undecided += 1,
        }
    }
    eprintln!(
        "random simplifications: {validated} validated, {undecided} undecided of {attempted}"
    );
    assert!(
        validated >= 8,
        "only {validated} random simplifications validated"
    );
}

#[test]
fn unsound_rewrites_get_counterexamples() {
    let x = || ScalarExpr::Load(OperandId::new(0));
    // Counterexample search bit-blasts the operations; Float32 keeps it fast.
    let store = |value| {
        let mut k = f32_variant(fixtures::polynomial(1));
        k.body.statements = vec![Statement::Store {
            operand: OperandId::new(4),
            value,
        }];
        k
    };
    for (before, after) in [
        // x + 0 → x is wrong at x = -0.
        (
            ScalarExpr::binary(BinaryOp::Add, x(), ScalarExpr::Constant(0.0)),
            x(),
        ),
        // x * 0 → 0 is wrong at NaN, infinities and negative x.
        (
            ScalarExpr::binary(BinaryOp::Mul, x(), ScalarExpr::Constant(0.0)),
            ScalarExpr::Constant(0.0),
        ),
        // Reassociation.
        (
            ScalarExpr::binary(
                BinaryOp::Add,
                ScalarExpr::binary(BinaryOp::Add, x(), ScalarExpr::Load(OperandId::new(1))),
                ScalarExpr::Load(OperandId::new(2)),
            ),
            ScalarExpr::binary(
                BinaryOp::Add,
                x(),
                ScalarExpr::binary(
                    BinaryOp::Add,
                    ScalarExpr::Load(OperandId::new(1)),
                    ScalarExpr::Load(OperandId::new(2)),
                ),
            ),
        ),
    ] {
        let outcome = validate_rewrite(z3(), &store(before), &store(after), TIMEOUT).unwrap();
        assert!(
            matches!(outcome, TvOutcome::Counterexample { .. }),
            "{outcome:?}"
        );
    }
    // A +0 store into an additive reduction is not dead.
    let k = f32_variant(fixtures::matvec(2));
    let mut with_zero = k.clone();
    with_zero.body.statements.push(Statement::Store {
        operand: OperandId::new(2),
        value: ScalarExpr::Constant(0.0),
    });
    assert!(matches!(
        validate_rewrite(z3(), &with_zero, &k, TIMEOUT).unwrap(),
        TvOutcome::Counterexample { .. }
    ));
}

#[test]
fn fma_contraction_is_validated_against_declared_fused_semantics() {
    for k in [
        fixtures::polynomial(2),
        fixtures::polynomial_batch(2),
        f32_variant(fixtures::polynomial(2)),
        fma_exp(2),
    ] {
        let contraction = malleus::policy::contract_fma(&k).unwrap();
        assert!(contraction.sites > 0, "{}", k.name);
        assert_eq!(
            validate_contraction(z3(), &k, &contraction.kernel, TIMEOUT).unwrap(),
            TvOutcome::Validated,
            "{}",
            k.name
        );
    }
    // Against the unfused reference semantics the contracted kernel differs.
    let k = f32_variant(fixtures::polynomial(1));
    let contraction = malleus::policy::contract_fma(&k).unwrap();
    let query = equivalence_query(&k, Semantics::Reference, &contraction.kernel).unwrap();
    assert!(matches!(
        check(z3(), &query, TIMEOUT).unwrap(),
        TvOutcome::Counterexample { .. }
    ));
}

#[test]
fn queries_state_their_encoding_and_refuse_mismatches() {
    let k = fixtures::polynomial(1);
    let query = equivalence_query(&k, Semantics::Reference, &k).unwrap();
    assert!(query.contains("(declare-const in0 Float64)"));
    assert!(query.contains("(not (= a_s0 b_s0))"));
    assert_eq!(check(z3(), &query, TIMEOUT).unwrap(), TvOutcome::Validated);
    let other = fixtures::polynomial(2);
    assert_eq!(
        equivalence_query(&k, Semantics::Reference, &other),
        Err(SmtError::Mismatch("operands"))
    );
    let mut read_write = k.clone();
    read_write.operands[4].access = AccessMode::ReadWrite;
    assert_eq!(
        equivalence_query(&read_write, Semantics::Reference, &read_write),
        Err(SmtError::ReadWriteOperand(4))
    );
    assert!(matches!(
        check(Path::new("/nonexistent/z3"), &query, TIMEOUT),
        Err(SmtError::Solver(_))
    ));
}
