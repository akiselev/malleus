//! `malleus` — differentiable, retargetable graph compiler.
//!
//! Malleus is the finite-precision pointwise/local compilation boundary in the scientific
//! stack. Resolvent owns scientific/discrete meaning; Sinbad owns field/topology/runtime
//! state; Solverang owns numerical algorithms.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "jit")]
mod compiled_newton;
#[cfg(feature = "jit")]
mod cranelift;
#[cfg(feature = "jit")]
mod lower;
#[cfg(feature = "jit")]
mod opcodes;
#[cfg(feature = "resolvent")]
mod resolvent_bridge;
mod scientific;

#[cfg(feature = "jit")]
pub use compiled_newton::CompiledNewtonStep;
#[cfg(feature = "jit")]
pub use cranelift::{JITCompiler, JITError, JITFunction};
#[cfg(feature = "jit")]
pub use lower::OpcodeEmitter;
#[cfg(feature = "jit")]
pub use opcodes::{CompiledConstraints, ConstraintOp, HessianEntry, JacobianEntry, Reg, ValidationError};
#[cfg(feature = "resolvent")]
pub use resolvent_bridge::{PlanCoverage, ResolventLoweringError, coverage, lower_pointwise_plan};
pub use scientific::{BindingLayout, ConstitutiveKernelContract, ElementKernelContract, GuardPolicy, KernelBlockId, LinearTable, LocalBinding, LocalInputKind, PropertyKernel, PropertyKernelBundle, PropertyKernelError, ValidityGuard, shared_evaluations};
#[cfg(feature = "resolvent")]
pub use scientific::{CompiledKernelBundle, lower_kernel_bundle};

/// When to JIT-compile constraint evaluation.
#[cfg(feature = "jit")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JitMode {
    #[default]
    Auto,
    ForceJit,
    ForceInterpreted,
}

#[cfg(feature = "jit")]
#[derive(Clone, Debug)]
pub struct JITConfig {
    pub jit_threshold: usize,
    pub estimated_iterations: usize,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub mode: JitMode,
}

#[cfg(feature = "jit")]
impl Default for JITConfig {
    fn default() -> Self {
        Self { jit_threshold: 1000, estimated_iterations: 50, max_iterations: 200, tolerance: 1e-8, mode: JitMode::Auto }
    }
}

#[cfg(feature = "jit")]
impl JITConfig {
    pub fn always_jit() -> Self { Self { mode: JitMode::ForceJit, ..Default::default() } }
    pub fn always_interpreted() -> Self { Self { mode: JitMode::ForceInterpreted, ..Default::default() } }
    pub fn for_large_problems() -> Self { Self { jit_threshold: 500, max_iterations: 500, tolerance: 1e-10, ..Default::default() } }
}

#[cfg(feature = "jit")]
pub fn jit_available() -> bool { cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) }
