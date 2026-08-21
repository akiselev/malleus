//! Structured forward- and reverse-mode differentiation.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::{
    AccessMode, BinaryOp, DerivativeMode, DerivativeOperand, DerivativeProduct, DerivativeRequest,
    IndexingMap, KernelOperand, KernelRegion, LocalId, OperandId, Predicate, ReductionOp,
    ScalarExpr, Statement, StructuredKernel, UnaryOp,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DifferentiationError {
    InvalidPrimal(crate::ValidationError),
    InvalidDerivative(crate::ValidationError),
    UnsupportedMode(DerivativeMode),
    EmptyDependentSet,
    DuplicateIndependent(usize),
    DuplicateDependent(usize),
    InvalidIndependent(usize),
    InvalidDependent(usize),
    IndependentNotReadable(usize),
    DependentNotWritten(usize),
    MissingIndexingMap(usize),
    UnsupportedReadWriteOperand(usize),
}

impl fmt::Display for DifferentiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "structured differentiation failed: {self:?}")
    }
}

impl Error for DifferentiationError {}

/// Differentiate a structurally valid primal kernel into another structured kernel.
///
/// The transformation is intentionally schedule-independent. Reduction stores remain
/// reductions, and reverse-mode cotangents always use additive reductions so aliasing is
/// explicit to validation and later schedulers.
pub fn differentiate(
    primal: &StructuredKernel,
    request: &DerivativeRequest,
) -> Result<DerivativeProduct, DifferentiationError> {
    crate::validate(primal.clone()).map_err(DifferentiationError::InvalidPrimal)?;
    validate_request(primal, request)?;
    let product = match request.mode {
        DerivativeMode::Jvp => jvp(primal, request),
        DerivativeMode::Vjp => vjp(primal, request),
        DerivativeMode::Jacobian => Err(DifferentiationError::UnsupportedMode(
            DerivativeMode::Jacobian,
        )),
    }?;
    crate::validate(product.kernel.clone()).map_err(DifferentiationError::InvalidDerivative)?;
    Ok(product)
}

fn validate_request(
    primal: &StructuredKernel,
    request: &DerivativeRequest,
) -> Result<(), DifferentiationError> {
    if request.dependent_operands.is_empty() {
        return Err(DifferentiationError::EmptyDependentSet);
    }
    let mut seen = BTreeSet::new();
    for operand in &request.independent_operands {
        if !seen.insert(*operand) {
            return Err(DifferentiationError::DuplicateIndependent(operand.index()));
        }
        let definition = primal
            .operands
            .get(operand.index())
            .ok_or(DifferentiationError::InvalidIndependent(operand.index()))?;
        if !definition.access.can_read() {
            return Err(DifferentiationError::IndependentNotReadable(
                operand.index(),
            ));
        }
    }
    seen.clear();
    for operand in &request.dependent_operands {
        if !seen.insert(*operand) {
            return Err(DifferentiationError::DuplicateDependent(operand.index()));
        }
        let definition = primal
            .operands
            .get(operand.index())
            .ok_or(DifferentiationError::InvalidDependent(operand.index()))?;
        if !definition.access.can_write() {
            return Err(DifferentiationError::DependentNotWritten(operand.index()));
        }
    }
    Ok(())
}

fn jvp(
    primal: &StructuredKernel,
    request: &DerivativeRequest,
) -> Result<DerivativeProduct, DifferentiationError> {
    let (mut operands, mut maps, primal_remap) = readable_primal_operands(primal)?;
    let mut tangent_by_primal = BTreeMap::new();
    let mut independent_operands = Vec::new();
    for primal_id in &request.independent_operands {
        let derivative = OperandId::new(operands.len());
        let definition = &primal.operands[primal_id.index()];
        operands.push(KernelOperand {
            name: format!("d_{}", definition.name),
            shape: definition.shape.clone(),
            region: definition.region,
            layout: definition.layout.clone(),
            access: AccessMode::Read,
        });
        maps.push(remap_map(primal, *primal_id, derivative)?);
        tangent_by_primal.insert(*primal_id, derivative);
        independent_operands.push(DerivativeOperand {
            primal: *primal_id,
            derivative,
        });
    }

    let mut dependent_by_primal = BTreeMap::new();
    let mut dependent_operands = Vec::new();
    for primal_id in &request.dependent_operands {
        let derivative = OperandId::new(operands.len());
        let definition = &primal.operands[primal_id.index()];
        operands.push(KernelOperand {
            name: format!("d_{}", definition.name),
            shape: definition.shape.clone(),
            region: definition.region,
            layout: definition.layout.clone(),
            access: definition.access,
        });
        maps.push(remap_map(primal, *primal_id, derivative)?);
        dependent_by_primal.insert(*primal_id, derivative);
        dependent_operands.push(DerivativeOperand {
            primal: *primal_id,
            derivative,
        });
    }

    let expanded = expanded_stores(primal)?;
    let mut statements = Vec::new();
    for (dependent, value) in expanded {
        let Some(output) = dependent_by_primal.get(&dependent).copied() else {
            continue;
        };
        let value = remap_expr(&value, &primal_remap);
        statements.push(Statement::Store {
            operand: output,
            value: simplify(directional(&value, &tangent_by_primal)),
        });
    }

    Ok(DerivativeProduct {
        mode: DerivativeMode::Jvp,
        kernel: StructuredKernel {
            name: format!("{}::jvp", primal.name),
            iteration_domain: primal.iteration_domain.clone(),
            iterators: primal.iterators.clone(),
            operands,
            indexing_maps: maps,
            body: KernelRegion { statements },
            numeric_policy: primal.numeric_policy,
        },
        independent_operands,
        dependent_operands,
    })
}

fn vjp(
    primal: &StructuredKernel,
    request: &DerivativeRequest,
) -> Result<DerivativeProduct, DifferentiationError> {
    let (mut operands, mut maps, primal_remap) = readable_primal_operands(primal)?;
    let mut seeds = BTreeMap::new();
    let mut dependent_operands = Vec::new();
    for primal_id in &request.dependent_operands {
        let derivative = OperandId::new(operands.len());
        let definition = &primal.operands[primal_id.index()];
        operands.push(KernelOperand {
            name: format!("bar_{}", definition.name),
            shape: definition.shape.clone(),
            region: definition.region,
            layout: definition.layout.clone(),
            access: AccessMode::Read,
        });
        maps.push(remap_map(primal, *primal_id, derivative)?);
        seeds.insert(*primal_id, derivative);
        dependent_operands.push(DerivativeOperand {
            primal: *primal_id,
            derivative,
        });
    }

    let mut cotangents = BTreeMap::new();
    let mut independent_operands = Vec::new();
    for primal_id in &request.independent_operands {
        let derivative = OperandId::new(operands.len());
        let definition = &primal.operands[primal_id.index()];
        operands.push(KernelOperand {
            name: format!("bar_{}", definition.name),
            shape: definition.shape.clone(),
            region: definition.region,
            layout: definition.layout.clone(),
            access: AccessMode::Reduce(ReductionOp::Add),
        });
        maps.push(remap_map(primal, *primal_id, derivative)?);
        cotangents.insert(*primal_id, derivative);
        independent_operands.push(DerivativeOperand {
            primal: *primal_id,
            derivative,
        });
    }

    let expanded = expanded_stores(primal)?;
    let mut statements = Vec::new();
    for (dependent, value) in expanded {
        let Some(seed) = seeds.get(&dependent).copied() else {
            continue;
        };
        let value = remap_expr(&value, &primal_remap);
        for independent in &request.independent_operands {
            statements.push(Statement::Store {
                operand: cotangents[independent],
                value: simplify(ScalarExpr::binary(
                    BinaryOp::Mul,
                    ScalarExpr::Load(seed),
                    partial(&value, primal_remap[independent]),
                )),
            });
        }
    }

    Ok(DerivativeProduct {
        mode: DerivativeMode::Vjp,
        kernel: StructuredKernel {
            name: format!("{}::vjp", primal.name),
            iteration_domain: primal.iteration_domain.clone(),
            iterators: primal.iterators.clone(),
            operands,
            indexing_maps: maps,
            body: KernelRegion { statements },
            numeric_policy: primal.numeric_policy,
        },
        independent_operands,
        dependent_operands,
    })
}

type ReadablePrimalOperands = (
    Vec<KernelOperand>,
    Vec<IndexingMap>,
    BTreeMap<OperandId, OperandId>,
);

fn readable_primal_operands(
    primal: &StructuredKernel,
) -> Result<ReadablePrimalOperands, DifferentiationError> {
    let mut operands = Vec::new();
    let mut maps = Vec::new();
    let mut remap = BTreeMap::new();
    for (index, definition) in primal.operands.iter().enumerate() {
        if !definition.access.can_read() {
            continue;
        }
        if definition.access == AccessMode::ReadWrite {
            return Err(DifferentiationError::UnsupportedReadWriteOperand(index));
        }
        let old = OperandId::new(index);
        let new = OperandId::new(operands.len());
        let mut definition = definition.clone();
        definition.access = AccessMode::Read;
        operands.push(definition);
        maps.push(remap_map(primal, old, new)?);
        remap.insert(old, new);
    }
    Ok((operands, maps, remap))
}

fn remap_map(
    primal: &StructuredKernel,
    old: OperandId,
    new: OperandId,
) -> Result<IndexingMap, DifferentiationError> {
    let mut map = primal
        .indexing_maps
        .iter()
        .find(|map| map.operand == old)
        .cloned()
        .ok_or(DifferentiationError::MissingIndexingMap(old.index()))?;
    map.operand = new;
    Ok(map)
}

fn expanded_stores(
    primal: &StructuredKernel,
) -> Result<Vec<(OperandId, ScalarExpr)>, DifferentiationError> {
    let mut locals = BTreeMap::<LocalId, ScalarExpr>::new();
    let mut stores = Vec::new();
    for statement in &primal.body.statements {
        match statement {
            Statement::Let { local, value } => {
                locals.insert(*local, expand_locals(value, &locals));
            }
            Statement::Store { operand, value } => {
                stores.push((*operand, expand_locals(value, &locals)));
            }
        }
    }
    Ok(stores)
}

fn expand_locals(expr: &ScalarExpr, locals: &BTreeMap<LocalId, ScalarExpr>) -> ScalarExpr {
    match expr {
        ScalarExpr::Local(local) => locals[local].clone(),
        ScalarExpr::Unary { op, arg } => ScalarExpr::unary(*op, expand_locals(arg, locals)),
        ScalarExpr::Binary { op, lhs, rhs } => {
            ScalarExpr::binary(*op, expand_locals(lhs, locals), expand_locals(rhs, locals))
        }
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => ScalarExpr::Select {
            condition: Box::new(expand_predicate(condition, locals)),
            if_true: Box::new(expand_locals(if_true, locals)),
            if_false: Box::new(expand_locals(if_false, locals)),
        },
        other => other.clone(),
    }
}

fn expand_predicate(predicate: &Predicate, locals: &BTreeMap<LocalId, ScalarExpr>) -> Predicate {
    match predicate {
        Predicate::Constant(value) => Predicate::Constant(*value),
        Predicate::Compare { op, lhs, rhs } => Predicate::Compare {
            op: *op,
            lhs: Box::new(expand_locals(lhs, locals)),
            rhs: Box::new(expand_locals(rhs, locals)),
        },
        Predicate::Not(value) => Predicate::Not(Box::new(expand_predicate(value, locals))),
        Predicate::And(lhs, rhs) => Predicate::And(
            Box::new(expand_predicate(lhs, locals)),
            Box::new(expand_predicate(rhs, locals)),
        ),
        Predicate::Or(lhs, rhs) => Predicate::Or(
            Box::new(expand_predicate(lhs, locals)),
            Box::new(expand_predicate(rhs, locals)),
        ),
    }
}

fn remap_expr(expr: &ScalarExpr, operands: &BTreeMap<OperandId, OperandId>) -> ScalarExpr {
    match expr {
        ScalarExpr::Load(operand) => ScalarExpr::Load(operands[operand]),
        ScalarExpr::Unary { op, arg } => ScalarExpr::unary(*op, remap_expr(arg, operands)),
        ScalarExpr::Binary { op, lhs, rhs } => {
            ScalarExpr::binary(*op, remap_expr(lhs, operands), remap_expr(rhs, operands))
        }
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => ScalarExpr::Select {
            condition: Box::new(remap_predicate(condition, operands)),
            if_true: Box::new(remap_expr(if_true, operands)),
            if_false: Box::new(remap_expr(if_false, operands)),
        },
        other => other.clone(),
    }
}

fn remap_predicate(predicate: &Predicate, operands: &BTreeMap<OperandId, OperandId>) -> Predicate {
    match predicate {
        Predicate::Constant(value) => Predicate::Constant(*value),
        Predicate::Compare { op, lhs, rhs } => Predicate::Compare {
            op: *op,
            lhs: Box::new(remap_expr(lhs, operands)),
            rhs: Box::new(remap_expr(rhs, operands)),
        },
        Predicate::Not(value) => Predicate::Not(Box::new(remap_predicate(value, operands))),
        Predicate::And(lhs, rhs) => Predicate::And(
            Box::new(remap_predicate(lhs, operands)),
            Box::new(remap_predicate(rhs, operands)),
        ),
        Predicate::Or(lhs, rhs) => Predicate::Or(
            Box::new(remap_predicate(lhs, operands)),
            Box::new(remap_predicate(rhs, operands)),
        ),
    }
}

fn directional(expr: &ScalarExpr, tangents: &BTreeMap<OperandId, OperandId>) -> ScalarExpr {
    derivative(expr, &mut |operand| {
        tangents
            .get(&operand)
            .copied()
            .map(ScalarExpr::Load)
            .unwrap_or(ScalarExpr::Constant(0.0))
    })
}

fn partial(expr: &ScalarExpr, independent: OperandId) -> ScalarExpr {
    derivative(expr, &mut |operand| {
        ScalarExpr::Constant(if operand == independent { 1.0 } else { 0.0 })
    })
}

fn derivative(
    expr: &ScalarExpr,
    load_derivative: &mut impl FnMut(OperandId) -> ScalarExpr,
) -> ScalarExpr {
    let d = |expr: &ScalarExpr, load_derivative: &mut _| derivative(expr, load_derivative);
    match expr {
        ScalarExpr::Constant(_) | ScalarExpr::Index(_) => ScalarExpr::Constant(0.0),
        ScalarExpr::Load(operand) => load_derivative(*operand),
        ScalarExpr::Local(_) => unreachable!("locals are expanded before differentiation"),
        ScalarExpr::Unary { op, arg } => {
            let da = d(arg, load_derivative);
            match op {
                UnaryOp::Neg => ScalarExpr::unary(UnaryOp::Neg, da),
                UnaryOp::Abs => ScalarExpr::Select {
                    condition: Box::new(Predicate::Compare {
                        op: crate::CompareOp::GreaterEqual,
                        lhs: arg.clone(),
                        rhs: Box::new(ScalarExpr::Constant(0.0)),
                    }),
                    if_true: Box::new(da.clone()),
                    if_false: Box::new(ScalarExpr::unary(UnaryOp::Neg, da)),
                },
                UnaryOp::Sqrt => ScalarExpr::binary(
                    BinaryOp::Div,
                    da,
                    ScalarExpr::binary(
                        BinaryOp::Mul,
                        ScalarExpr::Constant(2.0),
                        ScalarExpr::unary(UnaryOp::Sqrt, (**arg).clone()),
                    ),
                ),
                UnaryOp::Exp => ScalarExpr::binary(
                    BinaryOp::Mul,
                    ScalarExpr::unary(UnaryOp::Exp, (**arg).clone()),
                    da,
                ),
                UnaryOp::Ln => ScalarExpr::binary(BinaryOp::Div, da, (**arg).clone()),
                UnaryOp::Sin => ScalarExpr::binary(
                    BinaryOp::Mul,
                    ScalarExpr::unary(UnaryOp::Cos, (**arg).clone()),
                    da,
                ),
                UnaryOp::Cos => ScalarExpr::unary(
                    UnaryOp::Neg,
                    ScalarExpr::binary(
                        BinaryOp::Mul,
                        ScalarExpr::unary(UnaryOp::Sin, (**arg).clone()),
                        da,
                    ),
                ),
                UnaryOp::Tan => ScalarExpr::binary(
                    BinaryOp::Div,
                    da,
                    ScalarExpr::binary(
                        BinaryOp::Pow,
                        ScalarExpr::unary(UnaryOp::Cos, (**arg).clone()),
                        ScalarExpr::Constant(2.0),
                    ),
                ),
                UnaryOp::Floor | UnaryOp::Ceil => ScalarExpr::Constant(0.0),
            }
        }
        ScalarExpr::Binary { op, lhs, rhs } => {
            let dl = simplify(d(lhs, load_derivative));
            let dr = simplify(d(rhs, load_derivative));
            match op {
                BinaryOp::Add => ScalarExpr::binary(BinaryOp::Add, dl, dr),
                BinaryOp::Sub => ScalarExpr::binary(BinaryOp::Sub, dl, dr),
                BinaryOp::Mul => ScalarExpr::binary(
                    BinaryOp::Add,
                    ScalarExpr::binary(BinaryOp::Mul, dl, (**rhs).clone()),
                    ScalarExpr::binary(BinaryOp::Mul, (**lhs).clone(), dr),
                ),
                BinaryOp::Div => ScalarExpr::binary(
                    BinaryOp::Div,
                    ScalarExpr::binary(
                        BinaryOp::Sub,
                        ScalarExpr::binary(BinaryOp::Mul, dl, (**rhs).clone()),
                        ScalarExpr::binary(BinaryOp::Mul, (**lhs).clone(), dr),
                    ),
                    ScalarExpr::binary(BinaryOp::Mul, (**rhs).clone(), (**rhs).clone()),
                ),
                BinaryOp::Pow => {
                    let left = ScalarExpr::binary(
                        BinaryOp::Mul,
                        ScalarExpr::binary(
                            BinaryOp::Mul,
                            (**rhs).clone(),
                            ScalarExpr::binary(
                                BinaryOp::Pow,
                                (**lhs).clone(),
                                ScalarExpr::binary(
                                    BinaryOp::Sub,
                                    (**rhs).clone(),
                                    ScalarExpr::Constant(1.0),
                                ),
                            ),
                        ),
                        dl,
                    );
                    let right = ScalarExpr::binary(
                        BinaryOp::Mul,
                        ScalarExpr::binary(
                            BinaryOp::Mul,
                            ScalarExpr::binary(BinaryOp::Pow, (**lhs).clone(), (**rhs).clone()),
                            ScalarExpr::unary(UnaryOp::Ln, (**lhs).clone()),
                        ),
                        dr,
                    );
                    ScalarExpr::binary(BinaryOp::Add, left, right)
                }
                BinaryOp::Min | BinaryOp::Max => ScalarExpr::Select {
                    condition: Box::new(Predicate::Compare {
                        op: if *op == BinaryOp::Min {
                            crate::CompareOp::LessEqual
                        } else {
                            crate::CompareOp::GreaterEqual
                        },
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                    }),
                    if_true: Box::new(dl),
                    if_false: Box::new(dr),
                },
                BinaryOp::Atan2 => ScalarExpr::binary(
                    BinaryOp::Div,
                    ScalarExpr::binary(
                        BinaryOp::Sub,
                        ScalarExpr::binary(BinaryOp::Mul, (**rhs).clone(), dl),
                        ScalarExpr::binary(BinaryOp::Mul, (**lhs).clone(), dr),
                    ),
                    ScalarExpr::binary(
                        BinaryOp::Add,
                        ScalarExpr::binary(BinaryOp::Mul, (**lhs).clone(), (**lhs).clone()),
                        ScalarExpr::binary(BinaryOp::Mul, (**rhs).clone(), (**rhs).clone()),
                    ),
                ),
            }
        }
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => ScalarExpr::Select {
            condition: condition.clone(),
            if_true: Box::new(d(if_true, load_derivative)),
            if_false: Box::new(d(if_false, load_derivative)),
        },
    }
}

fn simplify(expr: ScalarExpr) -> ScalarExpr {
    match expr {
        ScalarExpr::Unary { op, arg } => {
            let arg = simplify(*arg);
            match (op, &arg) {
                (UnaryOp::Neg, ScalarExpr::Constant(0.0)) => ScalarExpr::Constant(0.0),
                _ => ScalarExpr::unary(op, arg),
            }
        }
        ScalarExpr::Binary { op, lhs, rhs } => {
            let lhs = simplify(*lhs);
            let rhs = simplify(*rhs);
            match (op, &lhs, &rhs) {
                (BinaryOp::Add | BinaryOp::Sub, _, ScalarExpr::Constant(0.0)) => lhs,
                (BinaryOp::Add, ScalarExpr::Constant(0.0), _) => rhs,
                (BinaryOp::Mul, ScalarExpr::Constant(0.0), _)
                | (BinaryOp::Mul, _, ScalarExpr::Constant(0.0)) => ScalarExpr::Constant(0.0),
                (BinaryOp::Mul, ScalarExpr::Constant(1.0), _) => rhs,
                (BinaryOp::Mul, _, ScalarExpr::Constant(1.0)) => lhs,
                (BinaryOp::Div, ScalarExpr::Constant(0.0), _) => ScalarExpr::Constant(0.0),
                (BinaryOp::Div, _, ScalarExpr::Constant(1.0)) => lhs,
                _ => ScalarExpr::binary(op, lhs, rhs),
            }
        }
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => ScalarExpr::Select {
            condition,
            if_true: Box::new(simplify(*if_true)),
            if_false: Box::new(simplify(*if_false)),
        },
        other => other,
    }
}
