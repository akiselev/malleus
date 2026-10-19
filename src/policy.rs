//! Explicit numeric policy levels, the reference scalar operations, and FMA contraction
//! (`malleus::policy`; deliberately not re-exported at the crate root).
//!
//! Every [`NumericPolicy`] carries a [`NumericLevel`] that is part of kernel identity:
//!
//! | Level | Meaning |
//! |---|---|
//! | [`NumericLevel::StrictPortable`] (L0) | IEEE basic operations plus the Malleus-pinned software routines of [`PORTABLE_MATH`] for every transcendental and for `min`/`max`: identical bits on every host and backend |
//! | [`NumericLevel::StrictHost`] (L1, default) | IEEE basic operations plus the host's `std` math (glibc, msvcrt, ...): reproducible on one host, may differ in the last bits across hosts |
//! | [`NumericLevel::Contracted`] (L2) | L1 semantics for a kernel produced by [`contract_fma`]: multiply-add sites at fixed, documented positions are explicit [`ScalarExpr::Fma`] nodes |
//!
//! Backends never contract, reassociate or substitute operations on their own; the only fused
//! operation is the explicit IR node `Fma`, which every backend executes as one correctly
//! rounded IEEE `fusedMultiplyAdd` (`mul_add`, Cranelift `fma`). An explicit `Fma` is admitted at
//! every level, since its result is exactly specified; the `Contracted` level records that the
//! contraction pass produced the kernel. A fixed-tree reduction level (L3) is not implemented.
//!
//! The older permissions [`FmaPolicy::Allowed`] and [`Reassociation::Allowed`] carry no
//! semantics: no backend acts on them, and every backend executes such kernels strictly. They
//! remain accepted at L1 so that existing kernels keep their identity, and validation refuses
//! them at L0 and L2 (`ValidationError::ImpreciseNumericPolicy`), whose identity promises exact
//! semantics. [`contract_fma`] refuses them for the same reason.
//!
//! [`NumericPolicy`]: crate::NumericPolicy
//! [`FmaPolicy::Allowed`]: crate::FmaPolicy::Allowed
//! [`Reassociation::Allowed`]: crate::Reassociation::Allowed

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    BinaryOp, FmaPolicy, Predicate, Reassociation, ScalarExpr, ScalarType, Statement,
    StructuredKernel, UnaryOp, ValidationError,
};

/// Identity of the portable math routines used at [`NumericLevel::StrictPortable`]: the
/// pure-Rust `libm` crate at this exact version, with default features disabled.
pub const PORTABLE_MATH: &str = "libm =0.2.16 (default-features = false)";

/// A digest-visible numeric semantics level. The default, `StrictHost`, is omitted from the
/// serialized policy, so kernels that predate levels keep their bytes and digests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericLevel {
    /// L0: IEEE basic operations plus [`PORTABLE_MATH`] routines; host-independent bits.
    StrictPortable,
    /// L1: IEEE basic operations plus the host `std` math library.
    #[default]
    StrictHost,
    /// L2: L1 semantics with explicit `Fma` nodes inserted by [`contract_fma`].
    Contracted,
}

impl NumericLevel {
    /// Whether this is the default level (used to keep it out of serialized policies).
    pub fn is_strict_host(&self) -> bool {
        *self == Self::StrictHost
    }

    /// The math library this level evaluates transcendentals and `min`/`max` with.
    pub const fn math(self) -> MathLibrary {
        match self {
            Self::StrictPortable => MathLibrary::Portable,
            Self::StrictHost | Self::Contracted => MathLibrary::Host,
        }
    }
}

/// Implementation of the non-basic scalar operations (`exp`, `ln`, `sin`, `cos`, `tan`,
/// `pow`, `atan2`, `min`, `max`). Basic operations, `sqrt`, `abs`, `floor`, `ceil` and `fma`
/// are exactly specified by IEEE 754 and identical in both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MathLibrary {
    /// Rust `std` (the host C math library).
    Host,
    /// [`PORTABLE_MATH`].
    Portable,
}

/// Whether the reference result of `op` depends on the [`MathLibrary`].
pub const fn unary_uses_math(op: UnaryOp) -> bool {
    matches!(
        op,
        UnaryOp::Exp | UnaryOp::Ln | UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Tan
    )
}

/// Whether the reference result of `op` depends on the [`MathLibrary`].
pub const fn binary_uses_math(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Pow | BinaryOp::Min | BinaryOp::Max | BinaryOp::Atan2
    )
}

/// The reference unary operation on the f64 storage ABI: `f32` kernels round the argument
/// and the result to `f32`.
#[inline]
pub fn unary(op: UnaryOp, value: f64, scalar_type: ScalarType, math: MathLibrary) -> f64 {
    match scalar_type {
        ScalarType::F64 => match (op, math) {
            (UnaryOp::Neg, _) => -value,
            (UnaryOp::Abs, _) => value.abs(),
            (UnaryOp::Sqrt, _) => value.sqrt(),
            (UnaryOp::Floor, _) => value.floor(),
            (UnaryOp::Ceil, _) => value.ceil(),
            (UnaryOp::Exp, MathLibrary::Host) => value.exp(),
            (UnaryOp::Ln, MathLibrary::Host) => value.ln(),
            (UnaryOp::Sin, MathLibrary::Host) => value.sin(),
            (UnaryOp::Cos, MathLibrary::Host) => value.cos(),
            (UnaryOp::Tan, MathLibrary::Host) => value.tan(),
            (UnaryOp::Exp, MathLibrary::Portable) => libm::exp(value),
            (UnaryOp::Ln, MathLibrary::Portable) => libm::log(value),
            (UnaryOp::Sin, MathLibrary::Portable) => libm::sin(value),
            (UnaryOp::Cos, MathLibrary::Portable) => libm::cos(value),
            (UnaryOp::Tan, MathLibrary::Portable) => libm::tan(value),
        },
        ScalarType::F32 => {
            let value = value as f32;
            (match (op, math) {
                (UnaryOp::Neg, _) => -value,
                (UnaryOp::Abs, _) => value.abs(),
                (UnaryOp::Sqrt, _) => value.sqrt(),
                (UnaryOp::Floor, _) => value.floor(),
                (UnaryOp::Ceil, _) => value.ceil(),
                (UnaryOp::Exp, MathLibrary::Host) => value.exp(),
                (UnaryOp::Ln, MathLibrary::Host) => value.ln(),
                (UnaryOp::Sin, MathLibrary::Host) => value.sin(),
                (UnaryOp::Cos, MathLibrary::Host) => value.cos(),
                (UnaryOp::Tan, MathLibrary::Host) => value.tan(),
                (UnaryOp::Exp, MathLibrary::Portable) => libm::expf(value),
                (UnaryOp::Ln, MathLibrary::Portable) => libm::logf(value),
                (UnaryOp::Sin, MathLibrary::Portable) => libm::sinf(value),
                (UnaryOp::Cos, MathLibrary::Portable) => libm::cosf(value),
                (UnaryOp::Tan, MathLibrary::Portable) => libm::tanf(value),
            }) as f64
        }
    }
}

/// The reference binary operation on the f64 storage ABI (`atan2(lhs, rhs)` is
/// `lhs.atan2(rhs)`, i.e. `y = lhs`, `x = rhs`).
#[inline]
pub fn binary(op: BinaryOp, lhs: f64, rhs: f64, scalar_type: ScalarType, math: MathLibrary) -> f64 {
    match scalar_type {
        ScalarType::F64 => match (op, math) {
            (BinaryOp::Add, _) => lhs + rhs,
            (BinaryOp::Sub, _) => lhs - rhs,
            (BinaryOp::Mul, _) => lhs * rhs,
            (BinaryOp::Div, _) => lhs / rhs,
            (BinaryOp::Pow, MathLibrary::Host) => lhs.powf(rhs),
            (BinaryOp::Min, MathLibrary::Host) => lhs.min(rhs),
            (BinaryOp::Max, MathLibrary::Host) => lhs.max(rhs),
            (BinaryOp::Atan2, MathLibrary::Host) => lhs.atan2(rhs),
            (BinaryOp::Pow, MathLibrary::Portable) => libm::pow(lhs, rhs),
            (BinaryOp::Min, MathLibrary::Portable) => libm::fmin(lhs, rhs),
            (BinaryOp::Max, MathLibrary::Portable) => libm::fmax(lhs, rhs),
            (BinaryOp::Atan2, MathLibrary::Portable) => libm::atan2(lhs, rhs),
        },
        ScalarType::F32 => {
            let lhs = lhs as f32;
            let rhs = rhs as f32;
            (match (op, math) {
                (BinaryOp::Add, _) => lhs + rhs,
                (BinaryOp::Sub, _) => lhs - rhs,
                (BinaryOp::Mul, _) => lhs * rhs,
                (BinaryOp::Div, _) => lhs / rhs,
                (BinaryOp::Pow, MathLibrary::Host) => lhs.powf(rhs),
                (BinaryOp::Min, MathLibrary::Host) => lhs.min(rhs),
                (BinaryOp::Max, MathLibrary::Host) => lhs.max(rhs),
                (BinaryOp::Atan2, MathLibrary::Host) => lhs.atan2(rhs),
                (BinaryOp::Pow, MathLibrary::Portable) => libm::powf(lhs, rhs),
                (BinaryOp::Min, MathLibrary::Portable) => libm::fminf(lhs, rhs),
                (BinaryOp::Max, MathLibrary::Portable) => libm::fmaxf(lhs, rhs),
                (BinaryOp::Atan2, MathLibrary::Portable) => libm::atan2f(lhs, rhs),
            }) as f64
        }
    }
}

/// The reference fused multiply-add `a * b + c` with one rounding (in `f32` for `f32` kernels).
#[inline]
pub fn fma(a: f64, b: f64, c: f64, scalar_type: ScalarType) -> f64 {
    match scalar_type {
        ScalarType::F64 => a.mul_add(b, c),
        ScalarType::F32 => (a as f32).mul_add(b as f32, c as f32) as f64,
    }
}

/// A typed refusal of [`contract_fma`].
#[derive(Clone, Debug, PartialEq)]
pub enum ContractionError {
    /// Contraction starts from `StrictHost` kernels only (it would silently drop the
    /// portability promise of `StrictPortable`, and `Contracted` kernels are already contracted).
    Level(NumericLevel),
    /// `FmaPolicy::Allowed` or `Reassociation::Allowed` has no exact meaning to preserve.
    ImprecisePolicy,
    /// The contracted kernel failed structural validation.
    Invalid(ValidationError),
}

impl fmt::Display for ContractionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FMA contraction refused: {self:?}")
    }
}
impl Error for ContractionError {}

/// The result of [`contract_fma`]: a validated-shape kernel at [`NumericLevel::Contracted`]
/// and the number of multiply-add sites that became explicit `Fma` nodes.
#[derive(Clone, Debug, PartialEq)]
pub struct Contraction {
    pub kernel: StructuredKernel,
    pub sites: usize,
}

/// IR-to-IR contraction at fixed sites, bottom-up over every expression tree:
///
/// - `Add(Mul(a, b), c)` becomes `Fma(a, b, c)`; otherwise `Add(c, Mul(a, b))` becomes
///   `Fma(a, b, c)` (the left product wins when both operands are products);
/// - `Sub(Mul(a, b), c)` becomes `Fma(a, b, Neg(c))`; otherwise `Sub(c, Mul(a, b))` becomes
///   `Fma(Neg(a), b, c)` (negation is exact, so each is the single-rounding form of the
///   original expression).
///
/// Products reached through a `Let` local are not contracted (the product may have other
/// uses), and reducing stores are not contracted. The result has level `Contracted`, so its
/// digest differs from the input's; the input is validated first and the output again.
pub fn contract_fma(kernel: &StructuredKernel) -> Result<Contraction, ContractionError> {
    let policy = kernel.numeric_policy;
    if policy.level != NumericLevel::StrictHost {
        return Err(ContractionError::Level(policy.level));
    }
    if policy.fma != FmaPolicy::Forbidden || policy.reassociation != Reassociation::Forbidden {
        return Err(ContractionError::ImprecisePolicy);
    }
    crate::validate(kernel.clone()).map_err(ContractionError::Invalid)?;
    let mut sites = 0;
    let mut contracted = kernel.clone();
    contracted.numeric_policy.level = NumericLevel::Contracted;
    for statement in &mut contracted.body.statements {
        let (Statement::Let { value, .. } | Statement::Store { value, .. }) = statement;
        *value = contract(value, &mut sites);
    }
    crate::validate(contracted.clone()).map_err(ContractionError::Invalid)?;
    Ok(Contraction {
        kernel: contracted,
        sites,
    })
}

fn contract(expr: &ScalarExpr, sites: &mut usize) -> ScalarExpr {
    match expr {
        ScalarExpr::Unary { op, arg } => ScalarExpr::unary(*op, contract(arg, sites)),
        ScalarExpr::Binary { op, lhs, rhs } => {
            let lhs = contract(lhs, sites);
            let rhs = contract(rhs, sites);
            let product = |e: &ScalarExpr| match e {
                ScalarExpr::Binary {
                    op: BinaryOp::Mul,
                    lhs,
                    rhs,
                } => Some(((**lhs).clone(), (**rhs).clone())),
                _ => None,
            };
            let neg = |e: ScalarExpr| ScalarExpr::unary(UnaryOp::Neg, e);
            let fused = match op {
                BinaryOp::Add => match (product(&lhs), product(&rhs)) {
                    (Some((a, b)), _) => Some(ScalarExpr::fma(a, b, rhs.clone())),
                    (None, Some((a, b))) => Some(ScalarExpr::fma(a, b, lhs.clone())),
                    (None, None) => None,
                },
                BinaryOp::Sub => match (product(&lhs), product(&rhs)) {
                    (Some((a, b)), _) => Some(ScalarExpr::fma(a, b, neg(rhs.clone()))),
                    (None, Some((a, b))) => Some(ScalarExpr::fma(neg(a), b, lhs.clone())),
                    (None, None) => None,
                },
                _ => None,
            };
            match fused {
                Some(fused) => {
                    *sites += 1;
                    fused
                }
                None => ScalarExpr::binary(*op, lhs, rhs),
            }
        }
        ScalarExpr::Fma { lhs, rhs, addend } => ScalarExpr::fma(
            contract(lhs, sites),
            contract(rhs, sites),
            contract(addend, sites),
        ),
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => ScalarExpr::Select {
            condition: Box::new(contract_predicate(condition, sites)),
            if_true: Box::new(contract(if_true, sites)),
            if_false: Box::new(contract(if_false, sites)),
        },
        ScalarExpr::Constant(_)
        | ScalarExpr::Index(_)
        | ScalarExpr::Load(_)
        | ScalarExpr::Local(_) => expr.clone(),
    }
}

fn contract_predicate(predicate: &Predicate, sites: &mut usize) -> Predicate {
    match predicate {
        Predicate::Constant(value) => Predicate::Constant(*value),
        Predicate::Compare { op, lhs, rhs } => Predicate::Compare {
            op: *op,
            lhs: Box::new(contract(lhs, sites)),
            rhs: Box::new(contract(rhs, sites)),
        },
        Predicate::Not(value) => Predicate::Not(Box::new(contract_predicate(value, sites))),
        Predicate::And(lhs, rhs) => Predicate::And(
            Box::new(contract_predicate(lhs, sites)),
            Box::new(contract_predicate(rhs, sites)),
        ),
        Predicate::Or(lhs, rhs) => Predicate::Or(
            Box::new(contract_predicate(lhs, sites)),
            Box::new(contract_predicate(rhs, sites)),
        ),
    }
}
