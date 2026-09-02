//! Facet-pair (two-sided trace) kernels.
//!
//! A facet-pair kernel is an ordinary point kernel evaluated at one quadrature point of an
//! interior or interface facet, whose operands carry a *side role*: trace data owned by the
//! `Minus` cell, trace data owned by the `Plus` cell, or data native to the facet itself. The
//! only orientation convention Malleus fixes is that facet-native `Odd` data — the unit normal
//! above all — is oriented from the minus cell towards the plus cell, so relabelling the two
//! cells negates it. Everything else (which cell is minus, what a trace means, which function
//! space a datum lives in) belongs to the realization that gathers the traces.
//!
//! Because the minus/plus labelling is arbitrary, a well-formed facet-pair kernel is covariant
//! under swapping the sides: feeding it the plus data in the minus slots (and vice versa) with
//! odd facet data negated must reproduce the side outputs swapped and the facet outputs
//! multiplied by their declared parity. [`check_facet_swap_symmetry`] proves that at supplied
//! data by execution; the declaration alone proves nothing.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::digest::{Digest, kernel_digest};
use crate::{
    AccessMode, BufferBinding, DerivativeProduct, DerivativeRequest, DifferentiationError,
    Executable, ExecutionError, Interpreter, OperandId, ReductionOp, StructuredKernel,
    ValidatedKernel, ValidationError, differentiate, validate,
};

/// Schema of the payload behind [`facet_pair_digest`].
pub const FACET_PAIR_KERNEL_SCHEMA: &str = "malleus-facet-pair-kernel/1";
/// The orientation convention every `Odd` facet operand follows.
pub const FACET_NORMAL_CONVENTION: &str = "minus_to_plus";

/// The adjacent cell that owns a trace operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FacetSide {
    Minus,
    Plus,
}

impl FacetSide {
    pub const fn opposite(self) -> Self {
        match self {
            Self::Minus => Self::Plus,
            Self::Plus => Self::Minus,
        }
    }
}

/// How facet-native data transforms when the two sides are relabelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwapParity {
    Even,
    Odd,
}

/// The side role of one operand of a facet-pair kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FacetOperandRole {
    /// Data owned by one adjacent cell: a field or gradient trace, a side-owned residual
    /// contribution, or a side-owned facet datum such as an outward flux unknown. `partner`
    /// names the same datum owned by the opposite cell when the kernel carries both.
    Cell {
        side: FacetSide,
        partner: Option<OperandId>,
    },
    /// Data native to the facet: coordinates, measure, and even-parity outputs are `Even`; the
    /// unit normal oriented from minus to plus and every quantity that flips with it are `Odd`.
    Facet { parity: SwapParity },
}

/// A point kernel with a side role for every operand.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FacetPairKernel {
    pub kernel: StructuredKernel,
    pub roles: Vec<FacetOperandRole>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FacetPairError {
    Kernel(ValidationError),
    RoleCount {
        expected: usize,
        actual: usize,
    },
    MissingSide(FacetSide),
    InvalidPartner {
        operand: usize,
        partner: usize,
    },
    SelfPartner(usize),
    PartnerSide {
        operand: usize,
        partner: usize,
    },
    PartnerAsymmetric {
        operand: usize,
        partner: usize,
    },
    PartnerShape {
        operand: usize,
        partner: usize,
    },
    PartnerAccess {
        operand: usize,
        partner: usize,
    },
    /// An odd facet output reduced with anything but addition has no swap image.
    OddReduction(usize),
}

impl fmt::Display for FacetPairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid facet-pair kernel: {self:?}")
    }
}

impl Error for FacetPairError {}

/// A facet-pair kernel whose inner kernel and roles passed [`validate_facet_pair`].
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedFacetPairKernel {
    kernel: ValidatedKernel,
    roles: Vec<FacetOperandRole>,
}

impl ValidatedFacetPairKernel {
    pub fn kernel(&self) -> &ValidatedKernel {
        &self.kernel
    }

    pub fn roles(&self) -> &[FacetOperandRole] {
        &self.roles
    }

    pub fn into_parts(self) -> (ValidatedKernel, Vec<FacetOperandRole>) {
        (self.kernel, self.roles)
    }
}

pub fn validate_facet_pair(
    kernel: FacetPairKernel,
) -> Result<ValidatedFacetPairKernel, FacetPairError> {
    let FacetPairKernel { kernel, roles } = kernel;
    let validated = validate(kernel).map_err(FacetPairError::Kernel)?;
    validate_roles(validated.as_kernel(), &roles)?;
    Ok(ValidatedFacetPairKernel {
        kernel: validated,
        roles,
    })
}

fn validate_roles(
    kernel: &StructuredKernel,
    roles: &[FacetOperandRole],
) -> Result<(), FacetPairError> {
    if roles.len() != kernel.operands.len() {
        return Err(FacetPairError::RoleCount {
            expected: kernel.operands.len(),
            actual: roles.len(),
        });
    }
    for side in [FacetSide::Minus, FacetSide::Plus] {
        if !roles
            .iter()
            .any(|role| matches!(role, FacetOperandRole::Cell { side: own, .. } if *own == side))
        {
            return Err(FacetPairError::MissingSide(side));
        }
    }
    for (index, role) in roles.iter().enumerate() {
        match role {
            FacetOperandRole::Cell {
                side,
                partner: Some(partner),
            } => {
                let partner_index = partner.index();
                if partner_index == index {
                    return Err(FacetPairError::SelfPartner(index));
                }
                let Some(partner_role) = roles.get(partner_index) else {
                    return Err(FacetPairError::InvalidPartner {
                        operand: index,
                        partner: partner_index,
                    });
                };
                match partner_role {
                    FacetOperandRole::Cell {
                        side: partner_side,
                        partner: back,
                    } => {
                        if *partner_side != side.opposite() {
                            return Err(FacetPairError::PartnerSide {
                                operand: index,
                                partner: partner_index,
                            });
                        }
                        if *back != Some(OperandId::new(index)) {
                            return Err(FacetPairError::PartnerAsymmetric {
                                operand: index,
                                partner: partner_index,
                            });
                        }
                    }
                    FacetOperandRole::Facet { .. } => {
                        return Err(FacetPairError::PartnerSide {
                            operand: index,
                            partner: partner_index,
                        });
                    }
                }
                let own = &kernel.operands[index];
                let other = &kernel.operands[partner_index];
                if own.shape != other.shape
                    || own.layout != other.layout
                    || own.region != other.region
                {
                    return Err(FacetPairError::PartnerShape {
                        operand: index,
                        partner: partner_index,
                    });
                }
                if own.access != other.access {
                    return Err(FacetPairError::PartnerAccess {
                        operand: index,
                        partner: partner_index,
                    });
                }
            }
            FacetOperandRole::Cell { partner: None, .. } => {}
            FacetOperandRole::Facet {
                parity: SwapParity::Odd,
            } => {
                if matches!(
                    kernel.operands[index].access,
                    AccessMode::Reduce(ReductionOp::Multiply)
                        | AccessMode::Reduce(ReductionOp::Min)
                        | AccessMode::Reduce(ReductionOp::Max)
                ) {
                    return Err(FacetPairError::OddReduction(index));
                }
            }
            FacetOperandRole::Facet {
                parity: SwapParity::Even,
            } => {}
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct FacetPairPayload<'a> {
    schema: &'static str,
    normal_convention: &'static str,
    kernel: Digest,
    roles: &'a [FacetOperandRole],
}

/// Identity of a facet-pair kernel: the inner kernel by digest, the roles, and the normal
/// convention. The inner kernel's own digest is untouched by the wrapping.
pub fn facet_pair_digest(kernel: &FacetPairKernel) -> Digest {
    Digest::of_payload(&FacetPairPayload {
        schema: FACET_PAIR_KERNEL_SCHEMA,
        normal_convention: FACET_NORMAL_CONVENTION,
        kernel: kernel_digest(&kernel.kernel),
        roles: &kernel.roles,
    })
}

/// A derivative kernel of a facet-pair kernel with a side role for every operand.
///
/// Derivative operands inherit the role of their primal operand; a partner survives only when
/// the partner's derivative operand exists in the same product (both sides requested).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FacetPairDerivativeProduct {
    pub product: DerivativeProduct,
    pub roles: Vec<FacetOperandRole>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FacetPairDifferentiationError {
    InvalidPrimal(FacetPairError),
    Differentiation(DifferentiationError),
    InvalidDerivative(FacetPairError),
}

impl fmt::Display for FacetPairDifferentiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "facet-pair differentiation failed: {self:?}")
    }
}

impl Error for FacetPairDifferentiationError {}

pub fn differentiate_facet_pair(
    kernel: &FacetPairKernel,
    request: &DerivativeRequest,
) -> Result<FacetPairDerivativeProduct, FacetPairDifferentiationError> {
    validate_roles(&kernel.kernel, &kernel.roles)
        .map_err(FacetPairDifferentiationError::InvalidPrimal)?;
    let product = differentiate(&kernel.kernel, request)
        .map_err(FacetPairDifferentiationError::Differentiation)?;
    let mut roles = vec![None; product.kernel.operands.len()];
    for table in [
        &product.primal_operands,
        &product.independent_operands,
        &product.dependent_operands,
    ] {
        let remap = table
            .iter()
            .map(|pair| (pair.primal, pair.derivative))
            .collect::<BTreeMap<_, _>>();
        for pair in table {
            let role = match kernel.roles[pair.primal.index()] {
                FacetOperandRole::Cell { side, partner } => FacetOperandRole::Cell {
                    side,
                    partner: partner.and_then(|partner| remap.get(&partner).copied()),
                },
                facet @ FacetOperandRole::Facet { .. } => facet,
            };
            roles[pair.derivative.index()] = Some(role);
        }
    }
    let roles = roles
        .into_iter()
        .map(|role| role.expect("every derivative operand descends from one primal operand"))
        .collect::<Vec<_>>();
    validate_roles(&product.kernel, &roles)
        .map_err(FacetPairDifferentiationError::InvalidDerivative)?;
    Ok(FacetPairDerivativeProduct { product, roles })
}

impl FacetPairDerivativeProduct {
    pub fn into_facet_pair(self) -> FacetPairKernel {
        FacetPairKernel {
            kernel: self.product.kernel,
            roles: self.roles,
        }
    }
}

/// The largest deviation from swap covariance observed for one writable operand.
#[derive(Clone, Debug, PartialEq)]
pub struct FacetSwapDeviation {
    pub operand: OperandId,
    pub max_absolute: f64,
}

/// Outcome of [`check_facet_swap_symmetry`].
#[derive(Clone, Debug, PartialEq)]
pub struct FacetSwapReport {
    pub deviations: Vec<FacetSwapDeviation>,
    pub max_absolute: f64,
    pub within_tolerance: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FacetSwapError {
    /// Every cell operand needs a partner before the sides can be exchanged.
    UnpairedOperand(usize),
    BufferCount {
        expected: usize,
        actual: usize,
    },
    BufferLength(usize),
    NonFinite(usize),
    InvalidTolerance,
    Execution(ExecutionError),
}

impl fmt::Display for FacetSwapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "facet swap check failed: {self:?}")
    }
}

impl Error for FacetSwapError {}

/// Prove swap covariance at supplied data.
///
/// `buffers` holds one buffer per operand, in operand order, with the initial contents of
/// writable buffers included (reduce targets accumulate on them). The kernel runs once as
/// given and once with the sides exchanged: every cell operand receives its partner's buffer
/// and every odd facet operand is negated, for readable and writable buffers alike. The check
/// passes when the exchanged outputs equal the partner's original output for cell outputs and
/// the parity-signed original output for facet outputs, within an absolute `tolerance`.
pub fn check_facet_swap_symmetry(
    kernel: &ValidatedFacetPairKernel,
    buffers: &[Vec<f64>],
    tolerance: f64,
) -> Result<FacetSwapReport, FacetSwapError> {
    if !(tolerance.is_finite() && tolerance >= 0.0) {
        return Err(FacetSwapError::InvalidTolerance);
    }
    let definition = kernel.kernel.as_kernel();
    if buffers.len() != definition.operands.len() {
        return Err(FacetSwapError::BufferCount {
            expected: definition.operands.len(),
            actual: buffers.len(),
        });
    }
    for (index, (operand, buffer)) in definition.operands.iter().zip(buffers).enumerate() {
        if buffer.len() < operand.region.offset + operand.region.length {
            return Err(FacetSwapError::BufferLength(index));
        }
        if buffer.iter().any(|value| !value.is_finite()) {
            return Err(FacetSwapError::NonFinite(index));
        }
        if let FacetOperandRole::Cell { partner: None, .. } = kernel.roles[index] {
            return Err(FacetSwapError::UnpairedOperand(index));
        }
    }
    let executable = Executable::reference(kernel.kernel.clone());
    let mut original = buffers.to_vec();
    run(&executable, &mut original)?;
    let mut swapped = swap_buffers(buffers, &kernel.roles);
    run(&executable, &mut swapped)?;
    let expected = swap_buffers(&original, &kernel.roles);
    let mut deviations = Vec::new();
    let mut max_absolute: f64 = 0.0;
    for (index, operand) in definition.operands.iter().enumerate() {
        if !operand.access.can_write() {
            continue;
        }
        let deviation = swapped[index]
            .iter()
            .zip(&expected[index])
            .map(|(observed, expected)| (observed - expected).abs())
            .fold(0.0_f64, f64::max);
        max_absolute = max_absolute.max(deviation);
        deviations.push(FacetSwapDeviation {
            operand: OperandId::new(index),
            max_absolute: deviation,
        });
    }
    Ok(FacetSwapReport {
        deviations,
        within_tolerance: max_absolute.is_finite() && max_absolute <= tolerance,
        max_absolute,
    })
}

fn swap_buffers(buffers: &[Vec<f64>], roles: &[FacetOperandRole]) -> Vec<Vec<f64>> {
    buffers
        .iter()
        .enumerate()
        .map(|(index, buffer)| match roles[index] {
            FacetOperandRole::Cell {
                partner: Some(partner),
                ..
            } => buffers[partner.index()].clone(),
            FacetOperandRole::Cell { partner: None, .. } => buffer.clone(),
            FacetOperandRole::Facet {
                parity: SwapParity::Even,
            } => buffer.clone(),
            FacetOperandRole::Facet {
                parity: SwapParity::Odd,
            } => buffer.iter().map(|value| -value).collect(),
        })
        .collect()
}

fn run(executable: &Executable, buffers: &mut [Vec<f64>]) -> Result<(), FacetSwapError> {
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(index, values)| BufferBinding::new(OperandId::new(index), values))
        .collect::<Vec<_>>();
    Interpreter::run(executable, &mut bindings).map_err(FacetSwapError::Execution)
}
