//! Structural validation for schedule-independent kernels.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use crate::{
    AccessMode, AxisId, IndexExpr, KernelRegion, LocalId, OperandId, Predicate, ScalarExpr,
    Statement, StructuredKernel, StructuredModule,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    EmptyName,
    IteratorRank {
        expected: usize,
        actual: usize,
    },
    DuplicateOperandName(String),
    InvalidOperand(usize),
    MissingIndexingMap(usize),
    DuplicateIndexingMap(usize),
    IndexRank {
        operand: usize,
        expected: usize,
        actual: usize,
    },
    InvalidAxis(usize),
    DuplicateIndexAxis {
        operand: usize,
        axis: usize,
    },
    InvalidLocal {
        expected: usize,
        actual: usize,
    },
    InvalidLoad(usize),
    InvalidStore(usize),
    NonFiniteConstant,
    DuplicateKernelName(String),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid structured kernel: {self:?}")
    }
}

impl Error for ValidationError {}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedKernel(StructuredKernel);

impl ValidatedKernel {
    pub fn as_kernel(&self) -> &StructuredKernel {
        &self.0
    }
    pub fn into_inner(self) -> StructuredKernel {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedModule {
    name: String,
    kernels: Vec<ValidatedKernel>,
}

impl ValidatedModule {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn kernels(&self) -> &[ValidatedKernel] {
        &self.kernels
    }
    pub fn into_parts(self) -> (String, Vec<ValidatedKernel>) {
        (self.name, self.kernels)
    }
}

pub fn validate(kernel: StructuredKernel) -> Result<ValidatedKernel, ValidationError> {
    if kernel.name.trim().is_empty() {
        return Err(ValidationError::EmptyName);
    }
    if kernel.iterators.len() != kernel.iteration_domain.rank() {
        return Err(ValidationError::IteratorRank {
            expected: kernel.iteration_domain.rank(),
            actual: kernel.iterators.len(),
        });
    }
    let mut names = BTreeSet::new();
    for operand in &kernel.operands {
        if operand.name.trim().is_empty() {
            return Err(ValidationError::EmptyName);
        }
        if !names.insert(operand.name.clone()) {
            return Err(ValidationError::DuplicateOperandName(operand.name.clone()));
        }
        operand
            .shape
            .iter()
            .try_fold(1usize, |n, extent| n.checked_mul(*extent))
            .ok_or(ValidationError::InvalidOperand(0))?;
    }
    let mut mapped = vec![false; kernel.operands.len()];
    for map in &kernel.indexing_maps {
        let operand = map.operand.index();
        let Some(definition) = kernel.operands.get(operand) else {
            return Err(ValidationError::InvalidOperand(operand));
        };
        if mapped[operand] {
            return Err(ValidationError::DuplicateIndexingMap(operand));
        }
        mapped[operand] = true;
        if map.results.len() != definition.shape.len() {
            return Err(ValidationError::IndexRank {
                operand,
                expected: definition.shape.len(),
                actual: map.results.len(),
            });
        }
        for expr in &map.results {
            validate_index(expr, operand, kernel.iteration_domain.rank())?;
        }
    }
    if let Some((operand, _)) = mapped.iter().enumerate().find(|(_, present)| !**present) {
        return Err(ValidationError::MissingIndexingMap(operand));
    }
    validate_region(&kernel.body, &kernel)?;
    Ok(ValidatedKernel(kernel))
}

pub fn validate_module(module: StructuredModule) -> Result<ValidatedModule, ValidationError> {
    if module.name.trim().is_empty() {
        return Err(ValidationError::EmptyName);
    }
    let mut names = BTreeSet::new();
    let mut kernels = Vec::with_capacity(module.kernels.len());
    for kernel in module.kernels {
        if !names.insert(kernel.name.clone()) {
            return Err(ValidationError::DuplicateKernelName(kernel.name));
        }
        kernels.push(validate(kernel)?);
    }
    Ok(ValidatedModule {
        name: module.name,
        kernels,
    })
}

fn validate_index(expr: &IndexExpr, operand: usize, rank: usize) -> Result<(), ValidationError> {
    let mut axes = BTreeSet::new();
    for term in &expr.terms {
        if term.axis.index() >= rank {
            return Err(ValidationError::InvalidAxis(term.axis.index()));
        }
        if !axes.insert(term.axis) {
            return Err(ValidationError::DuplicateIndexAxis {
                operand,
                axis: term.axis.index(),
            });
        }
    }
    Ok(())
}

fn validate_region(
    region: &KernelRegion,
    kernel: &StructuredKernel,
) -> Result<(), ValidationError> {
    let mut locals = 0;
    for statement in &region.statements {
        match statement {
            Statement::Let { local, value } => {
                if local.index() != locals {
                    return Err(ValidationError::InvalidLocal {
                        expected: locals,
                        actual: local.index(),
                    });
                }
                validate_expr(value, kernel, locals)?;
                locals += 1;
            }
            Statement::Store { operand, value } => {
                let Some(definition) = kernel.operands.get(operand.index()) else {
                    return Err(ValidationError::InvalidOperand(operand.index()));
                };
                if !definition.access.can_write() {
                    return Err(ValidationError::InvalidStore(operand.index()));
                }
                validate_expr(value, kernel, locals)?;
            }
        }
    }
    Ok(())
}

fn validate_expr(
    expr: &ScalarExpr,
    kernel: &StructuredKernel,
    locals: usize,
) -> Result<(), ValidationError> {
    match expr {
        ScalarExpr::Constant(value) if !value.is_finite() => {
            Err(ValidationError::NonFiniteConstant)
        }
        ScalarExpr::Constant(_) => Ok(()),
        ScalarExpr::Index(axis) if axis.index() >= kernel.iteration_domain.rank() => {
            Err(ValidationError::InvalidAxis(axis.index()))
        }
        ScalarExpr::Index(_) => Ok(()),
        ScalarExpr::Load(operand) => match kernel.operands.get(operand.index()) {
            Some(definition) if definition.access.can_read() => Ok(()),
            _ => Err(ValidationError::InvalidLoad(operand.index())),
        },
        ScalarExpr::Local(local) if local.index() >= locals => Err(ValidationError::InvalidLocal {
            expected: locals,
            actual: local.index(),
        }),
        ScalarExpr::Local(_) => Ok(()),
        ScalarExpr::Unary { arg, .. } => validate_expr(arg, kernel, locals),
        ScalarExpr::Binary { lhs, rhs, .. } => {
            validate_expr(lhs, kernel, locals)?;
            validate_expr(rhs, kernel, locals)
        }
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => {
            validate_predicate(condition, kernel, locals)?;
            validate_expr(if_true, kernel, locals)?;
            validate_expr(if_false, kernel, locals)
        }
    }
}

fn validate_predicate(
    predicate: &Predicate,
    kernel: &StructuredKernel,
    locals: usize,
) -> Result<(), ValidationError> {
    match predicate {
        Predicate::Constant(_) => Ok(()),
        Predicate::Compare { lhs, rhs, .. } => {
            validate_expr(lhs, kernel, locals)?;
            validate_expr(rhs, kernel, locals)
        }
        Predicate::Not(value) => validate_predicate(value, kernel, locals),
        Predicate::And(lhs, rhs) | Predicate::Or(lhs, rhs) => {
            validate_predicate(lhs, kernel, locals)?;
            validate_predicate(rhs, kernel, locals)
        }
    }
}

#[allow(dead_code)]
fn _ids_are_local(_: AxisId, _: LocalId, _: OperandId, _: AccessMode) {}
