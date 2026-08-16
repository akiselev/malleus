//! `malleus` — the differentiable, retargetable graph compiler.
//!
//! The one differentiation core for physics residuals and ML surrogates: a
//! matrix-free operator/kernel IR with IR-level AD, e-graph rewriting, and
//! retargetable codegen (CPU now; GPU/WASM/FPGA later).
//!
//! # M1 scope — pure re-homing
//!
//! This first slice is a **faithful port** of solverang's `jit/` module into
//! its own crate, with **NO new capability**. It moves the scalar opcode IR
//! ([`ConstraintOp`] / [`CompiledConstraints`]) plus the Cranelift JIT codegen
//! backend ([`JITCompiler`] / [`JITFunction`] / [`CompiledNewtonStep`]) and the
//! fluent [`OpcodeEmitter`] into malleus, keeping the original tests so the port
//! is provably faithful.
//!
//! ## Deferred to later milestones (M2+)
//!
//! Explicitly **not** attempted here — these are separate milestones:
//! - Reverse-mode automatic differentiation.
//! - E-graph rewriting / equality saturation.
//! - Revolve checkpointing.
//! - Non-CPU backends (GPU / WASM / FPGA).
//! - The numeric-contracts `LinearOperator` generation.
//! - Rewiring solverang to depend on malleus (malleus is a standalone workspace
//!   root during build-out; folded into the federation at a later barrier).
//!
//! # Architecture
//!
//! 1. **Opcode emission**: build an opcode stream with [`OpcodeEmitter`],
//!    producing a [`CompiledConstraints`] ready for compilation.
//! 2. **Compilation + execution**: [`JITCompiler`] compiles the opcode stream
//!    to native x86_64/aarch64 code via Cranelift, returning a [`JITFunction`]
//!    with `evaluate_residuals()` / `evaluate_jacobian()` (and fused / dense
//!    variants), or a whole-Newton-step [`CompiledNewtonStep`].
//!
//! # Feature flag
//!
//! The JIT lives behind the `jit` feature (default-on), mirroring solverang.
//!
//! # Platform support
//!
//! JIT compilation is supported on x86_64 (Linux, macOS, Windows) and aarch64
//! (Linux, macOS); use [`jit_available`] to query at runtime.
//!
//! # Unsafe
//!
//! The crate uses `#![deny(unsafe_op_in_unsafe_fn)]` rather than forbidding
//! unsafe outright: a Cranelift JIT must call JIT'd function pointers, which is
//! inherently `unsafe`. Those calls (and the code-pointer transmutes that mint
//! the fn pointers) are the crate's only `unsafe` sites, each kept in a
//! tightly-scoped block with a `// SAFETY:` comment. See `STATUS.md`.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "jit")]
mod compiled_newton;
#[cfg(feature = "jit")]
mod cranelift;
#[cfg(feature = "jit")]
mod lower;
#[cfg(feature = "jit")]
mod opcodes;

#[cfg(feature = "jit")]
pub use compiled_newton::CompiledNewtonStep;
#[cfg(feature = "jit")]
pub use cranelift::{JITCompiler, JITError, JITFunction};
#[cfg(feature = "jit")]
pub use lower::OpcodeEmitter;
#[cfg(feature = "jit")]
pub use opcodes::{
    CompiledConstraints, ConstraintOp, HessianEntry, JacobianEntry, Reg, ValidationError,
};

/// When to JIT-compile constraint evaluation.
#[cfg(feature = "jit")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JitMode {
    /// Compile when the estimated work exceeds `jit_threshold` and the
    /// platform supports JIT (the default).
    #[default]
    Auto,
    /// Always compile, regardless of problem size. Useful for benchmarking.
    ForceJit,
    /// Never compile; always use interpreted evaluation. Useful for debugging.
    ForceInterpreted,
}

/// Configuration for JIT-enabled solving.
#[cfg(feature = "jit")]
#[derive(Clone, Debug)]
pub struct JITConfig {
    /// Threshold for JIT compilation (constraints * estimated_iterations).
    ///
    /// Problems with estimated work below this threshold use interpreted evaluation.
    /// Default: 1000
    pub jit_threshold: usize,

    /// Estimated number of iterations for threshold calculation.
    ///
    /// Default: 50
    pub estimated_iterations: usize,

    /// Maximum number of solver iterations.
    ///
    /// Default: 200
    pub max_iterations: usize,

    /// Convergence tolerance for residual norm.
    ///
    /// Default: 1e-8
    pub tolerance: f64,

    /// When to JIT-compile (auto by threshold, forced on, or forced off).
    ///
    /// Default: [`JitMode::Auto`]
    pub mode: JitMode,
}

#[cfg(feature = "jit")]
impl Default for JITConfig {
    fn default() -> Self {
        Self {
            jit_threshold: 1000,
            estimated_iterations: 50,
            max_iterations: 200,
            tolerance: 1e-8,
            mode: JitMode::Auto,
        }
    }
}

#[cfg(feature = "jit")]
impl JITConfig {
    /// Create a configuration that always uses JIT compilation.
    pub fn always_jit() -> Self {
        Self {
            mode: JitMode::ForceJit,
            ..Default::default()
        }
    }

    /// Create a configuration that always uses interpreted evaluation.
    pub fn always_interpreted() -> Self {
        Self {
            mode: JitMode::ForceInterpreted,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for large problems.
    pub fn for_large_problems() -> Self {
        Self {
            jit_threshold: 500,
            max_iterations: 500,
            tolerance: 1e-10,
            ..Default::default()
        }
    }
}

/// Check if JIT compilation is available on this platform.
#[cfg(feature = "jit")]
pub fn jit_available() -> bool {
    // Cranelift supports x86_64 and aarch64
    cfg!(any(target_arch = "x86_64", target_arch = "aarch64"))
}
