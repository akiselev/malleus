//! Structured compilation contracts for finite-precision local kernels.
//!
//! Malleus begins at a local iteration domain and ends at a scheduled executable. It
//! deliberately has no model language, mesh, global assembly, or solver-policy surface.
//! Frontends lower their typed local mathematics into [`StructuredKernel`]; backends consume
//! the validated, schedule-bearing [`Executable`]. The built-in [`Interpreter`] is the
//! deterministic reference backend.
//!
//! Two structured objects sit beside single kernels without changing them: a
//! [`KernelComposition`] binds one kernel's output buffer to another kernel's input operand at
//! execution time (the kernel-level form of a bound model input), and a [`FacetPairKernel`]
//! gives a point kernel evaluated on an interior or interface facet a side role for every
//! operand. Both carry deterministic digests built from the verbatim kernel digests.
#![forbid(unsafe_code)]

mod campaign;
mod compose;
mod differentiate;
mod digest;
mod executable;
mod facet;
mod interpreter;
mod ir;
mod validate;

pub use campaign::{
    CampaignError, ComparisonTolerance, LocalCampaignReport, LocalCheckKind, LocalCheckResult,
    LocalDifferentialCase, LocalExecutableRunner, NumericPolicyMutation, OperandValues,
    check_numeric_policy_mutation, run_local_differential_campaign,
};
pub use compose::{
    CompositionBinding, CompositionDerivativeOperand, CompositionDerivativeProduct,
    CompositionDerivativeRequest, CompositionDerivativeStage, CompositionDifferentiationError,
    CompositionError, CompositionExecutionError, CompositionPrimalStage, CompositionTarget,
    ExecutableComposition, KERNEL_COMPOSITION_SCHEMA, KernelComposition, SharedBuffer,
    StageOperand, ValidatedComposition, composition_digest, differentiate_composition,
    validate_composition,
};
pub use differentiate::{DifferentiationError, differentiate};
pub use digest::{
    Digest, KERNEL_DIGEST_SCHEMA, MODULE_DIGEST_SCHEMA, kernel_digest, module_digest,
};
pub use executable::{
    Executable, ExecutableError, ExecutableModule, KernelSchedule, ParallelMapping, TileDecision,
    VectorizationPlan,
};
pub use facet::{
    FACET_NORMAL_CONVENTION, FACET_PAIR_KERNEL_SCHEMA, FacetOperandRole,
    FacetPairDerivativeProduct, FacetPairDifferentiationError, FacetPairError, FacetPairKernel,
    FacetSide, FacetSwapDeviation, FacetSwapError, FacetSwapReport, SwapParity,
    ValidatedFacetPairKernel, check_facet_swap_symmetry, differentiate_facet_pair,
    facet_pair_digest, validate_facet_pair,
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
