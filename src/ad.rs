//! Derivative options beyond the backend-neutral [`DerivativeRequest`]:
//! the nonsmooth policy, opt-in IEEE-exact simplification of generated derivative kernels, and
//! the dense Jacobian (vector forward mode) layout.
//!
//! The request type itself is unchanged, so every existing request keeps its fields and serialized
//! bytes; [`DerivativeOptions`] travels beside it and serializes as `{}` at its default.
//!
//! # Nonsmooth sites
//!
//! A *kink site* is an operation whose derivative is a convention rather than a limit:
//! `abs`, binary `min`/`max`, `floor`/`ceil`, and stores into `min`/`max` reductions, whenever
//! the differentiated argument carries a nonzero structural tangent from a requested
//! independent operand. `select` is not a kink: the derivative of a select is the derivative of
//! the taken branch, and its condition is never differentiated. Floor and ceil arguments are not
//! descended into (their derivative is zero whatever the argument).
//!
//! Under [`NonsmoothPolicy::Conventional`] (the default) differentiation applies the existing
//! conventions: `d|x| = +dx` at `x = 0` (and for NaN `x`, `-dx`), `min`/`max` ties take the
//! left operand's tangent, floor/ceil have zero derivative, and a store into a `min`/`max`
//! reduction keeps the reduction of the tangent values. Under [`NonsmoothPolicy::Refuse`]
//! any kink site is a typed [`DifferentiationError::NonsmoothRefused`] naming every site.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    AccessMode, AxisId, DerivativeMode, DerivativeProduct, DerivativeRequest, DifferentiationError,
    OperandId, ReductionOp, ScalarExpr, Statement, StructuredKernel, UnaryOp,
};

/// How differentiation treats nonsmooth operations (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonsmoothPolicy {
    /// The existing tie conventions; derivative kernels are unchanged.
    #[default]
    Conventional,
    /// Refuse when any kink site is differentiated.
    Refuse,
}

impl NonsmoothPolicy {
    pub fn is_conventional(&self) -> bool {
        *self == Self::Conventional
    }
}

/// Post-differentiation simplification of the generated kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Simplification {
    /// The generated kernel exactly as `differentiate` builds it.
    #[default]
    None,
    /// [`crate::simplify::simplify_exact`]: rewrites that change no bit of any output for any
    /// input (NaN payloads aside).
    Exact,
}

impl Simplification {
    pub fn is_none(&self) -> bool {
        *self == Self::None
    }
}

/// Options for [`differentiate_with`]; the default reproduces [`crate::differentiate`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DerivativeOptions {
    #[serde(default, skip_serializing_if = "NonsmoothPolicy::is_conventional")]
    pub nonsmooth: NonsmoothPolicy,
    #[serde(default, skip_serializing_if = "Simplification::is_none")]
    pub simplification: Simplification,
}

/// The nonsmooth operation at a kink site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KinkKind {
    Abs,
    Floor,
    Ceil,
    Min,
    Max,
    MinReduction,
    MaxReduction,
}

/// One differentiated nonsmooth operation: `statement` indexes the primal kernel body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KinkSite {
    pub statement: usize,
    pub kind: KinkKind,
}

/// The columns of a dense Jacobian that belong to one independent operand: column
/// `offset + i` is the derivative with respect to element `i` of the operand in row-major
/// logical order (independent of its storage layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JacobianColumns {
    pub primal: OperandId,
    pub offset: usize,
    pub len: usize,
}

/// Largest Jacobian width: column indices are formed exactly in `f32` arithmetic.
pub const MAX_JACOBIAN_WIDTH: usize = 1 << 24;

/// A dense Jacobian kernel (vector forward mode).
///
/// `product.kernel` is the primal kernel with one extra trailing `Parallel` axis
/// `seed_axis` of extent `width`; at seed `s` it evaluates the JVP whose direction is the
/// identity seed `e_s`. Each dependent operand `y` of shape `S` becomes the output
/// `d_y` of shape `S ++ [width]` (row-major extension of `y`'s layout, region offset 0,
/// same access), holding `∂y/∂x` in the columns given by `columns`. Every element is
/// bitwise equal to the corresponding element of the JVP output for that identity seed, because
/// the seed axis is innermost and the identity seed is the constant `1` or `0` the JVP would
/// load. `product.independent_operands` is empty (there is no direction input);
/// `product.dependent_operands` pairs each dependent with its Jacobian output.
#[derive(Clone, Debug, PartialEq)]
pub struct JacobianProduct {
    pub product: DerivativeProduct,
    pub columns: Vec<JacobianColumns>,
    pub seed_axis: AxisId,
    pub width: usize,
}

/// [`crate::differentiate`] with explicit options.
pub fn differentiate_with(
    primal: &StructuredKernel,
    request: &DerivativeRequest,
    options: &DerivativeOptions,
) -> Result<DerivativeProduct, DifferentiationError> {
    if request.mode == DerivativeMode::Jacobian {
        return differentiate_jacobian(primal, request, options).map(|j| j.product);
    }
    check_nonsmooth(primal, request, options)?;
    let product = crate::differentiate(primal, request)?;
    finish(product, options)
}

/// The dense Jacobian of `primal` (`request.mode` must be `Jacobian`).
pub fn differentiate_jacobian(
    primal: &StructuredKernel,
    request: &DerivativeRequest,
    options: &DerivativeOptions,
) -> Result<JacobianProduct, DifferentiationError> {
    if request.mode != DerivativeMode::Jacobian {
        return Err(DifferentiationError::UnsupportedMode(request.mode));
    }
    check_nonsmooth(primal, request, options)?;
    let mut jacobian = crate::differentiate::jacobian(primal, request)?;
    jacobian.product = finish(jacobian.product, options)?;
    Ok(jacobian)
}

fn check_nonsmooth(
    primal: &StructuredKernel,
    request: &DerivativeRequest,
    options: &DerivativeOptions,
) -> Result<(), DifferentiationError> {
    if options.nonsmooth == NonsmoothPolicy::Refuse {
        let sites = kink_sites(primal, request)?;
        if !sites.is_empty() {
            return Err(DifferentiationError::NonsmoothRefused(sites));
        }
    }
    Ok(())
}

fn finish(
    mut product: DerivativeProduct,
    options: &DerivativeOptions,
) -> Result<DerivativeProduct, DifferentiationError> {
    if options.simplification == Simplification::Exact {
        product.kernel = crate::simplify::simplify_exact(&product.kernel).kernel;
        crate::validate(product.kernel.clone()).map_err(DifferentiationError::InvalidDerivative)?;
    }
    Ok(product)
}

/// Every kink site differentiated by `request` (module docs), in statement order.
pub fn kink_sites(
    primal: &StructuredKernel,
    request: &DerivativeRequest,
) -> Result<Vec<KinkSite>, DifferentiationError> {
    crate::validate(primal.clone()).map_err(DifferentiationError::InvalidPrimal)?;
    crate::differentiate::validate_request(primal, request)?;
    let independent: BTreeSet<OperandId> = request.independent_operands.iter().copied().collect();
    let dependent: BTreeSet<OperandId> = request.dependent_operands.iter().copied().collect();
    let statements = &primal.body.statements;

    // Locals whose value carries a structural tangent.
    let mut carries = Vec::new();
    let mut varies = Vec::new();
    for statement in statements {
        if let Statement::Let { value, .. } = statement {
            carries.push(tangent(value, &independent, &carries));
            varies.push(value_varies(value, &independent, &varies));
        }
    }

    // Statements that are differentiated: stores into dependents and the locals they reach
    // through differentiated positions.
    let mut live_locals = vec![false; carries.len()];
    let mut live = vec![false; statements.len()];
    let mut local_index = carries.len();
    for (index, statement) in statements.iter().enumerate().rev() {
        match statement {
            Statement::Let { value, .. } => {
                local_index -= 1;
                if live_locals[local_index] {
                    live[index] = true;
                    mark_locals(value, &mut live_locals);
                }
            }
            Statement::Store { operand, value } => {
                if dependent.contains(operand) {
                    live[index] = true;
                    mark_locals(value, &mut live_locals);
                }
            }
        }
    }

    let mut sites = Vec::new();
    for (index, statement) in statements.iter().enumerate() {
        if !live[index] {
            continue;
        }
        let value = match statement {
            Statement::Let { value, .. } => value,
            Statement::Store { operand, value } => {
                let kind = match primal.operands[operand.index()].access {
                    AccessMode::Reduce(ReductionOp::Min) => Some(KinkKind::MinReduction),
                    AccessMode::Reduce(ReductionOp::Max) => Some(KinkKind::MaxReduction),
                    _ => None,
                };
                if let Some(kind) = kind
                    && tangent(value, &independent, &carries)
                {
                    sites.push(KinkSite {
                        statement: index,
                        kind,
                    });
                }
                value
            }
        };
        collect_sites(value, index, &independent, (&carries, &varies), &mut sites);
    }
    Ok(sites)
}

/// Whether the derivative of `expr` is not structurally zero.
fn tangent(expr: &ScalarExpr, independent: &BTreeSet<OperandId>, carries: &[bool]) -> bool {
    match expr {
        ScalarExpr::Constant(_) | ScalarExpr::Index(_) => false,
        ScalarExpr::Load(operand) => independent.contains(operand),
        ScalarExpr::Local(local) => carries[local.index()],
        ScalarExpr::Unary {
            op: UnaryOp::Floor | UnaryOp::Ceil,
            ..
        } => false,
        ScalarExpr::Unary { arg, .. } => tangent(arg, independent, carries),
        ScalarExpr::Binary { lhs, rhs, .. } => {
            tangent(lhs, independent, carries) || tangent(rhs, independent, carries)
        }
        ScalarExpr::Fma { lhs, rhs, addend } => {
            tangent(lhs, independent, carries)
                || tangent(rhs, independent, carries)
                || tangent(addend, independent, carries)
        }
        ScalarExpr::Select {
            if_true, if_false, ..
        } => tangent(if_true, independent, carries) || tangent(if_false, independent, carries),
    }
}

fn mark_locals(expr: &ScalarExpr, live: &mut [bool]) {
    match expr {
        ScalarExpr::Local(local) => live[local.index()] = true,
        ScalarExpr::Constant(_) | ScalarExpr::Index(_) | ScalarExpr::Load(_) => {}
        ScalarExpr::Unary {
            op: UnaryOp::Floor | UnaryOp::Ceil,
            ..
        } => {}
        ScalarExpr::Unary { arg, .. } => mark_locals(arg, live),
        ScalarExpr::Binary { lhs, rhs, .. } => {
            mark_locals(lhs, live);
            mark_locals(rhs, live);
        }
        ScalarExpr::Fma { lhs, rhs, addend } => {
            mark_locals(lhs, live);
            mark_locals(rhs, live);
            mark_locals(addend, live);
        }
        ScalarExpr::Select {
            if_true, if_false, ..
        } => {
            mark_locals(if_true, live);
            mark_locals(if_false, live);
        }
    }
}

fn collect_sites(
    expr: &ScalarExpr,
    statement: usize,
    independent: &BTreeSet<OperandId>,
    tables: (&[bool], &[bool]),
    sites: &mut Vec<KinkSite>,
) {
    let (carries, varies) = tables;
    match expr {
        ScalarExpr::Constant(_)
        | ScalarExpr::Index(_)
        | ScalarExpr::Load(_)
        | ScalarExpr::Local(_) => {}
        ScalarExpr::Unary { op, arg } => match op {
            UnaryOp::Floor | UnaryOp::Ceil => {
                // The argument's value (not its tangent) decides the jump: report when the
                // argument varies with an independent operand through any path.
                if value_varies(arg, independent, varies) {
                    sites.push(KinkSite {
                        statement,
                        kind: if *op == UnaryOp::Floor {
                            KinkKind::Floor
                        } else {
                            KinkKind::Ceil
                        },
                    });
                }
            }
            UnaryOp::Abs => {
                if tangent(arg, independent, carries) {
                    sites.push(KinkSite {
                        statement,
                        kind: KinkKind::Abs,
                    });
                }
                collect_sites(arg, statement, independent, tables, sites);
            }
            _ => collect_sites(arg, statement, independent, tables, sites),
        },
        ScalarExpr::Binary { op, lhs, rhs } => {
            if matches!(op, crate::BinaryOp::Min | crate::BinaryOp::Max)
                && (tangent(lhs, independent, carries) || tangent(rhs, independent, carries))
            {
                sites.push(KinkSite {
                    statement,
                    kind: if *op == crate::BinaryOp::Min {
                        KinkKind::Min
                    } else {
                        KinkKind::Max
                    },
                });
            }
            collect_sites(lhs, statement, independent, tables, sites);
            collect_sites(rhs, statement, independent, tables, sites);
        }
        ScalarExpr::Fma { lhs, rhs, addend } => {
            collect_sites(lhs, statement, independent, tables, sites);
            collect_sites(rhs, statement, independent, tables, sites);
            collect_sites(addend, statement, independent, tables, sites);
        }
        ScalarExpr::Select {
            if_true, if_false, ..
        } => {
            collect_sites(if_true, statement, independent, tables, sites);
            collect_sites(if_false, statement, independent, tables, sites);
        }
    }
}

/// Whether the value of `expr` can vary with an independent operand: like [`tangent`], but
/// through floor/ceil arguments and locals too (a nested floor still moves with its input).
fn value_varies(expr: &ScalarExpr, independent: &BTreeSet<OperandId>, varies: &[bool]) -> bool {
    match expr {
        ScalarExpr::Constant(_) | ScalarExpr::Index(_) => false,
        ScalarExpr::Load(operand) => independent.contains(operand),
        ScalarExpr::Local(local) => varies[local.index()],
        ScalarExpr::Unary { arg, .. } => value_varies(arg, independent, varies),
        ScalarExpr::Binary { lhs, rhs, .. } => {
            value_varies(lhs, independent, varies) || value_varies(rhs, independent, varies)
        }
        ScalarExpr::Fma { lhs, rhs, addend } => {
            value_varies(lhs, independent, varies)
                || value_varies(rhs, independent, varies)
                || value_varies(addend, independent, varies)
        }
        ScalarExpr::Select {
            if_true, if_false, ..
        } => {
            value_varies(if_true, independent, varies)
                || value_varies(if_false, independent, varies)
        }
    }
}
