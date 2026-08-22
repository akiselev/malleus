//! Structured compilation contracts for finite-precision local kernels.
//!
//! Malleus begins at a local iteration domain and ends at a scheduled executable. It
//! deliberately has no model language, mesh, global assembly, or solver-policy surface.
//! Frontends lower their typed local mathematics into [`StructuredKernel`]; backends consume
//! the validated, schedule-bearing [`Executable`]. The built-in [`Interpreter`] is the
//! deterministic reference backend.
#![forbid(unsafe_code)]

mod campaign;
mod differentiate;
mod executable;
mod interpreter;
mod ir;
mod validate;

pub use campaign::{
    CampaignError, ComparisonTolerance, LocalCampaignReport, LocalCheckKind, LocalCheckResult,
    LocalDifferentialCase, LocalExecutableRunner, NumericPolicyMutation, OperandValues,
    check_numeric_policy_mutation, run_local_differential_campaign,
};
pub use differentiate::{DifferentiationError, differentiate};
pub use executable::{
    Executable, ExecutableError, ExecutableModule, KernelSchedule, ParallelMapping, TileDecision,
    VectorizationPlan,
};
pub use interpreter::{BufferBinding, ExecutionError, Interpreter};
pub use ir::{
    AccessMode, AxisId, BinaryOp, BufferRegion, CompareOp, DenseLayout, DerivativeMode,
    DerivativeOperand, DerivativeProduct, DerivativeRequest, FmaPolicy, IndexExpr, IndexTerm,
    IndexingMap, IterationDomain, IteratorKind, KernelOperand, KernelRegion, LocalId,
    NumericPolicy, OperandId, Predicate, Reassociation, ReductionOp, ReductionOrder, ScalarExpr,
    ScalarType, Statement, StructuredKernel, StructuredModule, UnaryOp,
};
pub use validate::{ValidatedKernel, ValidatedModule, ValidationError, validate, validate_module};
