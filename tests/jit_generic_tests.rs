//! Integration tests for the ported JIT: opcode emission (Layer 1) and
//! Cranelift codegen of the scalar opcodes (Layer 2).
//!
//! These are the self-contained layers of solverang's original
//! `tests/jit_generic_tests.rs` — the ones that exercise ONLY the JIT's public
//! surface (`OpcodeEmitter`, `ConstraintOp`, `CompiledConstraints`,
//! `JITCompiler`, `Reg`). The original file's later layers (Expr-AST parsing,
//! round-trip vs interpreted `Problem`, RuntimeConst baking, end-to-end solves,
//! `JITSolver`) all depend on solverang's `#[auto_jacobian]` macro, the
//! `Problem` trait, and `JITSolver` — none of which are in the M1 re-homing
//! scope, so they are deferred to the eventual solverang→malleus rewire.
//!
//! Ported verbatim from solverang; imports rewritten to malleus's flat surface.

#![cfg(feature = "jit")]

use malleus::{CompiledConstraints, ConstraintOp, JITCompiler, OpcodeEmitter, Reg};

// ============================================================================
// LAYER 1: New opcodes exist in ConstraintOp and OpcodeEmitter
// ============================================================================

mod layer1_opcodes {
    use super::*;

    #[test]
    fn emitter_has_exp() {
        let mut e = OpcodeEmitter::new();
        let x = e.load_var(0);
        let r = e.exp(x);
        let ops = e.ops();
        assert!(
            matches!(ops.last(), Some(ConstraintOp::Exp { dst, src }) if *dst == r && *src == x),
            "OpcodeEmitter::exp should emit ConstraintOp::Exp"
        );
    }

    #[test]
    fn emitter_has_ln() {
        let mut e = OpcodeEmitter::new();
        let x = e.load_var(0);
        let r = e.ln(x);
        let ops = e.ops();
        assert!(
            matches!(ops.last(), Some(ConstraintOp::Ln { dst, src }) if *dst == r && *src == x),
            "OpcodeEmitter::ln should emit ConstraintOp::Ln"
        );
    }

    #[test]
    fn emitter_has_pow() {
        let mut e = OpcodeEmitter::new();
        let base = e.load_var(0);
        let exp = e.load_var(1);
        let r = e.pow(base, exp);
        let ops = e.ops();
        assert!(
            matches!(
                ops.last(),
                Some(ConstraintOp::Pow { dst, base: b, exp: ex })
                    if *dst == r && *b == base && *ex == exp
            ),
            "OpcodeEmitter::pow should emit ConstraintOp::Pow"
        );
    }

    #[test]
    fn emitter_has_tan() {
        let mut e = OpcodeEmitter::new();
        let x = e.load_var(0);
        let r = e.tan(x);
        let ops = e.ops();
        assert!(
            matches!(ops.last(), Some(ConstraintOp::Tan { dst, src }) if *dst == r && *src == x),
            "OpcodeEmitter::tan should emit ConstraintOp::Tan"
        );
    }

    #[test]
    fn new_opcodes_register_tracking() {
        // Verify uses_register and defines_register work for new opcodes
        let r0 = Reg::new(0);
        let r1 = Reg::new(1);
        let r2 = Reg::new(2);

        let exp_op = ConstraintOp::Exp { dst: r1, src: r0 };
        assert!(exp_op.uses_register(r0));
        assert!(!exp_op.uses_register(r1));
        assert!(exp_op.defines_register(r1));

        let ln_op = ConstraintOp::Ln { dst: r1, src: r0 };
        assert!(ln_op.uses_register(r0));
        assert!(ln_op.defines_register(r1));

        let pow_op = ConstraintOp::Pow {
            dst: r2,
            base: r0,
            exp: r1,
        };
        assert!(pow_op.uses_register(r0));
        assert!(pow_op.uses_register(r1));
        assert!(pow_op.defines_register(r2));

        let tan_op = ConstraintOp::Tan { dst: r1, src: r0 };
        assert!(tan_op.uses_register(r0));
        assert!(tan_op.defines_register(r1));
    }
}

// ============================================================================
// LAYER 2: Cranelift codegen compiles new opcodes to native code
// ============================================================================

mod layer2_cranelift {
    use super::*;

    /// Helper: build opcodes manually, compile, evaluate, check result.
    fn jit_eval_residual(ops: Vec<ConstraintOp>, n_vars: usize, vars: &[f64]) -> f64 {
        let max_reg = ops
            .iter()
            .filter_map(|op| match op {
                ConstraintOp::LoadVar { dst, .. }
                | ConstraintOp::LoadConst { dst, .. }
                | ConstraintOp::Add { dst, .. }
                | ConstraintOp::Sub { dst, .. }
                | ConstraintOp::Mul { dst, .. }
                | ConstraintOp::Div { dst, .. }
                | ConstraintOp::Neg { dst, .. }
                | ConstraintOp::Sqrt { dst, .. }
                | ConstraintOp::Sin { dst, .. }
                | ConstraintOp::Cos { dst, .. }
                | ConstraintOp::Atan2 { dst, .. }
                | ConstraintOp::Abs { dst, .. }
                | ConstraintOp::Max { dst, .. }
                | ConstraintOp::Min { dst, .. }
                | ConstraintOp::Exp { dst, .. }
                | ConstraintOp::Ln { dst, .. }
                | ConstraintOp::Pow { dst, .. }
                | ConstraintOp::Tan { dst, .. } => Some(dst.index()),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        let mut cc = CompiledConstraints::new(n_vars, 1);
        cc.residual_ops = ops;
        cc.max_register = max_reg;

        let mut compiler = JITCompiler::new().expect("compiler creation failed");
        let jit_fn = compiler.compile(&cc).expect("compilation failed");

        let mut residuals = [0.0];
        jit_fn.evaluate_residuals(vars, &mut residuals);
        residuals[0]
    }

    #[test]
    fn jit_exp() {
        // e^2.0 ≈ 7.389056
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Exp {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[2.0]);
        assert!(
            (result - 2.0_f64.exp()).abs() < 1e-10,
            "exp(2.0) = {}, JIT got {}",
            2.0_f64.exp(),
            result
        );
    }

    #[test]
    fn jit_exp_zero() {
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Exp {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[0.0]);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "exp(0) should be 1.0, got {}",
            result
        );
    }

    #[test]
    fn jit_ln() {
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Ln {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[std::f64::consts::E]);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "ln(e) should be 1.0, got {}",
            result
        );
    }

    #[test]
    fn jit_ln_identity() {
        // ln(exp(3.5)) == 3.5
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Exp {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::Ln {
                dst: Reg::new(2),
                src: Reg::new(1),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(2),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[3.5]);
        assert!(
            (result - 3.5).abs() < 1e-10,
            "ln(exp(3.5)) should be 3.5, got {}",
            result
        );
    }

    #[test]
    fn jit_pow() {
        // 2.0^3.0 = 8.0
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::LoadConst {
                dst: Reg::new(1),
                value: 3.0,
            },
            ConstraintOp::Pow {
                dst: Reg::new(2),
                base: Reg::new(0),
                exp: Reg::new(1),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(2),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[2.0]);
        assert!(
            (result - 8.0).abs() < 1e-10,
            "2^3 should be 8.0, got {}",
            result
        );
    }

    #[test]
    fn jit_pow_fractional() {
        // 9.0^0.5 = 3.0
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::LoadConst {
                dst: Reg::new(1),
                value: 0.5,
            },
            ConstraintOp::Pow {
                dst: Reg::new(2),
                base: Reg::new(0),
                exp: Reg::new(1),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(2),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[9.0]);
        assert!(
            (result - 3.0).abs() < 1e-10,
            "9^0.5 should be 3.0, got {}",
            result
        );
    }

    #[test]
    fn jit_tan() {
        // tan(pi/4) = 1.0
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Tan {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        let pi_over_4 = std::f64::consts::FRAC_PI_4;
        let result = jit_eval_residual(ops, 1, &[pi_over_4]);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "tan(pi/4) should be 1.0, got {}",
            result
        );
    }

    #[test]
    fn jit_sin_full_range() {
        // This tests that sin uses libm, not Taylor (Taylor fails at large angles)
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Sin {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        // sin(5.0) ≈ -0.9589... — well outside Taylor approximation range
        let result = jit_eval_residual(ops, 1, &[5.0]);
        assert!(
            (result - 5.0_f64.sin()).abs() < 1e-10,
            "sin(5.0) should be {}, got {} (Taylor approx gives wrong answer here)",
            5.0_f64.sin(),
            result
        );
    }

    #[test]
    fn jit_cos_full_range() {
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::Cos {
                dst: Reg::new(1),
                src: Reg::new(0),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(1),
            },
        ];
        let result = jit_eval_residual(ops, 1, &[5.0]);
        assert!(
            (result - 5.0_f64.cos()).abs() < 1e-10,
            "cos(5.0) should be {}, got {}",
            5.0_f64.cos(),
            result
        );
    }

    #[test]
    fn jit_atan2_full_range() {
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::LoadVar {
                dst: Reg::new(1),
                var_idx: 1,
            },
            ConstraintOp::Atan2 {
                dst: Reg::new(2),
                y: Reg::new(0),
                x: Reg::new(1),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(2),
            },
        ];
        // atan2(-1, -1) = -3*pi/4 ≈ -2.356 — in third quadrant, Taylor approx fails
        let result = jit_eval_residual(ops, 2, &[-1.0, -1.0]);
        let expected = (-1.0_f64).atan2(-1.0);
        assert!(
            (result - expected).abs() < 1e-10,
            "atan2(-1, -1) should be {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn jit_compound_exp_ln_expression() {
        // Compute: ln(x[0]) * exp(x[1]) - should exercise both new opcodes together
        // ln(e) * exp(0) = 1.0 * 1.0 = 1.0
        let ops = vec![
            ConstraintOp::LoadVar {
                dst: Reg::new(0),
                var_idx: 0,
            },
            ConstraintOp::LoadVar {
                dst: Reg::new(1),
                var_idx: 1,
            },
            ConstraintOp::Ln {
                dst: Reg::new(2),
                src: Reg::new(0),
            },
            ConstraintOp::Exp {
                dst: Reg::new(3),
                src: Reg::new(1),
            },
            ConstraintOp::Mul {
                dst: Reg::new(4),
                a: Reg::new(2),
                b: Reg::new(3),
            },
            ConstraintOp::StoreResidual {
                residual_idx: 0,
                src: Reg::new(4),
            },
        ];
        let result = jit_eval_residual(ops, 2, &[std::f64::consts::E, 0.0]);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "ln(e) * exp(0) should be 1.0, got {}",
            result
        );
    }
}
