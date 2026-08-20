//! Structured compilation contracts for finite-precision local kernels.
//!
//! Malleus begins at a local iteration domain and ends at a scheduled executable. It
//! deliberately has no model language, mesh, global assembly, or solver-policy surface.
//! Frontends lower their typed local mathematics into [`StructuredKernel`]; backends consume
//! the validated, schedule-bearing [`Executable`]. The built-in [`Interpreter`] is the
//! deterministic reference backend.
#![forbid(unsafe_code)]

mod executable;
mod interpreter;
mod ir;
mod validate;

pub use executable::{
    Executable, ExecutableError, ExecutableModule, KernelSchedule, ParallelMapping, TileDecision,
    VectorizationPlan,
};
pub use interpreter::{BufferBinding, ExecutionError, Interpreter};
pub use ir::{
    AccessMode, AxisId, BinaryOp, CompareOp, DerivativeMode, DerivativeRequest, FmaPolicy,
    IndexExpr, IndexTerm, IndexingMap, IterationDomain, IteratorKind, KernelOperand, KernelRegion,
    LocalId, NumericPolicy, OperandId, Predicate, Reassociation, ReductionOp, ReductionOrder,
    ScalarExpr, ScalarType, Statement, StructuredKernel, StructuredModule, UnaryOp,
};
pub use validate::{ValidatedKernel, ValidatedModule, ValidationError, validate, validate_module};
