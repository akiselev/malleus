//! Deterministic sequential reference execution.

use std::error::Error;
use std::fmt;

use crate::{
    AccessMode, BinaryOp, CompareOp, Executable, IndexExpr, OperandId, Predicate, ReductionOp,
    ReductionOrder, ScalarExpr, ScalarType, Statement, UnaryOp,
};

pub struct BufferBinding<'a> {
    pub operand: OperandId,
    pub values: &'a mut [f64],
}

impl<'a> BufferBinding<'a> {
    pub fn new(operand: OperandId, values: &'a mut [f64]) -> Self {
        Self { operand, values }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    MissingBinding(usize),
    DuplicateBinding(usize),
    InvalidBinding(usize),
    InvalidIndex(usize),
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kernel execution failed: {self:?}")
    }
}
impl Error for ExecutionError {}

#[derive(Clone, Copy, Debug, Default)]
pub struct Interpreter;

impl Interpreter {
    pub fn run(
        executable: &Executable,
        bindings: &mut [BufferBinding<'_>],
    ) -> Result<(), ExecutionError> {
        let kernel = executable.kernel().as_kernel();
        let mut seen = vec![false; kernel.operands.len()];
        for binding in bindings.iter() {
            let id = binding.operand.index();
            let Some(operand) = kernel.operands.get(id) else {
                return Err(ExecutionError::InvalidBinding(id));
            };
            if seen[id] {
                return Err(ExecutionError::DuplicateBinding(id));
            }
            seen[id] = true;
            let expected = operand
                .region
                .offset
                .checked_add(operand.region.length)
                .ok_or(ExecutionError::InvalidBinding(id))?;
            if binding.values.len() < expected {
                return Err(ExecutionError::InvalidBinding(id));
            }
        }
        if let Some((id, _)) = seen.iter().enumerate().find(|(_, present)| !**present) {
            return Err(ExecutionError::MissingBinding(id));
        }
        let mut coordinates = vec![0; kernel.iteration_domain.rank()];
        let canonical_order;
        let order = if kernel.numeric_policy.reduction_order == ReductionOrder::Canonical {
            canonical_order = (0..kernel.iteration_domain.rank())
                .map(crate::AxisId::new)
                .collect::<Vec<_>>();
            &canonical_order
        } else {
            &executable.schedule().loop_order
        };
        visit(
            &kernel.iteration_domain.extents,
            order,
            0,
            &mut coordinates,
            &mut |coordinates| execute_point(executable, bindings, coordinates),
        )
    }
}

fn visit(
    extents: &[usize],
    order: &[crate::AxisId],
    depth: usize,
    coordinates: &mut [usize],
    callback: &mut impl FnMut(&[usize]) -> Result<(), ExecutionError>,
) -> Result<(), ExecutionError> {
    if depth == order.len() {
        return callback(coordinates);
    }
    let axis = order[depth].index();
    for value in 0..extents[axis] {
        coordinates[axis] = value;
        visit(extents, order, depth + 1, coordinates, callback)?;
    }
    Ok(())
}

fn execute_point(
    executable: &Executable,
    bindings: &mut [BufferBinding<'_>],
    coordinates: &[usize],
) -> Result<(), ExecutionError> {
    let kernel = executable.kernel().as_kernel();
    let mut locals = Vec::new();
    for statement in &kernel.body.statements {
        match statement {
            Statement::Let { value, .. } => {
                locals.push(eval(value, executable, bindings, coordinates, &locals)?)
            }
            Statement::Store { operand, value } => {
                let value = eval(value, executable, bindings, coordinates, &locals)?;
                let map = kernel
                    .indexing_maps
                    .iter()
                    .find(|map| map.operand == *operand)
                    .expect("validated map");
                let offset = offset(
                    &map.results,
                    &kernel.operands[operand.index()].shape,
                    &kernel.operands[operand.index()].layout.minor_to_major,
                    coordinates,
                )
                .and_then(|offset| {
                    kernel.operands[operand.index()]
                        .region
                        .offset
                        .checked_add(offset)
                })
                .ok_or(ExecutionError::InvalidIndex(operand.index()))?;
                let binding = bindings
                    .iter_mut()
                    .find(|binding| binding.operand == *operand)
                    .expect("validated binding");
                match kernel.operands[operand.index()].access {
                    AccessMode::Write | AccessMode::ReadWrite => binding.values[offset] = value,
                    AccessMode::Reduce(ReductionOp::Add) => {
                        binding.values[offset] = binary(
                            BinaryOp::Add,
                            binding.values[offset],
                            value,
                            kernel.numeric_policy.scalar_type,
                        )
                    }
                    AccessMode::Reduce(ReductionOp::Multiply) => {
                        binding.values[offset] = binary(
                            BinaryOp::Mul,
                            binding.values[offset],
                            value,
                            kernel.numeric_policy.scalar_type,
                        )
                    }
                    AccessMode::Reduce(ReductionOp::Min) => {
                        binding.values[offset] = binary(
                            BinaryOp::Min,
                            binding.values[offset],
                            value,
                            kernel.numeric_policy.scalar_type,
                        )
                    }
                    AccessMode::Reduce(ReductionOp::Max) => {
                        binding.values[offset] = binary(
                            BinaryOp::Max,
                            binding.values[offset],
                            value,
                            kernel.numeric_policy.scalar_type,
                        )
                    }
                    AccessMode::Read => unreachable!("validated store"),
                }
            }
        }
    }
    Ok(())
}

fn eval(
    expr: &ScalarExpr,
    executable: &Executable,
    bindings: &[BufferBinding<'_>],
    coordinates: &[usize],
    locals: &[f64],
) -> Result<f64, ExecutionError> {
    let kernel = executable.kernel().as_kernel();
    Ok(match expr {
        ScalarExpr::Constant(value) => cast(*value, kernel.numeric_policy.scalar_type),
        ScalarExpr::Index(axis) => cast(
            coordinates[axis.index()] as f64,
            kernel.numeric_policy.scalar_type,
        ),
        ScalarExpr::Local(local) => locals[local.index()],
        ScalarExpr::Load(operand) => {
            let map = kernel
                .indexing_maps
                .iter()
                .find(|map| map.operand == *operand)
                .expect("validated map");
            let offset = offset(
                &map.results,
                &kernel.operands[operand.index()].shape,
                &kernel.operands[operand.index()].layout.minor_to_major,
                coordinates,
            )
            .and_then(|offset| {
                kernel.operands[operand.index()]
                    .region
                    .offset
                    .checked_add(offset)
            })
            .ok_or(ExecutionError::InvalidIndex(operand.index()))?;
            cast(
                bindings
                    .iter()
                    .find(|binding| binding.operand == *operand)
                    .expect("validated binding")
                    .values[offset],
                kernel.numeric_policy.scalar_type,
            )
        }
        ScalarExpr::Unary { op, arg } => unary(
            *op,
            eval(arg, executable, bindings, coordinates, locals)?,
            kernel.numeric_policy.scalar_type,
        ),
        ScalarExpr::Binary { op, lhs, rhs } => binary(
            *op,
            eval(lhs, executable, bindings, coordinates, locals)?,
            eval(rhs, executable, bindings, coordinates, locals)?,
            kernel.numeric_policy.scalar_type,
        ),
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => {
            if predicate(condition, executable, bindings, coordinates, locals)? {
                eval(if_true, executable, bindings, coordinates, locals)?
            } else {
                eval(if_false, executable, bindings, coordinates, locals)?
            }
        }
    })
}

fn predicate(
    value: &Predicate,
    executable: &Executable,
    bindings: &[BufferBinding<'_>],
    coordinates: &[usize],
    locals: &[f64],
) -> Result<bool, ExecutionError> {
    Ok(match value {
        Predicate::Constant(value) => *value,
        Predicate::Compare { op, lhs, rhs } => compare(
            *op,
            eval(lhs, executable, bindings, coordinates, locals)?,
            eval(rhs, executable, bindings, coordinates, locals)?,
        ),
        Predicate::Not(value) => !predicate(value, executable, bindings, coordinates, locals)?,
        Predicate::And(lhs, rhs) => {
            predicate(lhs, executable, bindings, coordinates, locals)?
                && predicate(rhs, executable, bindings, coordinates, locals)?
        }
        Predicate::Or(lhs, rhs) => {
            predicate(lhs, executable, bindings, coordinates, locals)?
                || predicate(rhs, executable, bindings, coordinates, locals)?
        }
    })
}

fn offset(
    indices: &[IndexExpr],
    shape: &[usize],
    minor_to_major: &[usize],
    coordinates: &[usize],
) -> Option<usize> {
    let mut values = Vec::with_capacity(indices.len());
    for (expr, extent) in indices.iter().zip(shape) {
        let index = expr.terms.iter().try_fold(expr.constant, |value, term| {
            value.checked_add(
                term.coefficient
                    .checked_mul(coordinates[term.axis.index()] as isize)?,
            )
        })?;
        let index = usize::try_from(index).ok()?;
        if index >= *extent {
            return None;
        }
        values.push(index);
    }
    let mut offset = 0usize;
    let mut stride = 1usize;
    for dimension in minor_to_major {
        offset = offset.checked_add(values[*dimension].checked_mul(stride)?)?;
        stride = stride.checked_mul(shape[*dimension])?;
    }
    Some(offset)
}

fn unary(op: UnaryOp, value: f64, scalar_type: ScalarType) -> f64 {
    match scalar_type {
        ScalarType::F64 => match op {
            UnaryOp::Neg => -value,
            UnaryOp::Abs => value.abs(),
            UnaryOp::Sqrt => value.sqrt(),
            UnaryOp::Exp => value.exp(),
            UnaryOp::Ln => value.ln(),
            UnaryOp::Sin => value.sin(),
            UnaryOp::Cos => value.cos(),
            UnaryOp::Tan => value.tan(),
            UnaryOp::Floor => value.floor(),
            UnaryOp::Ceil => value.ceil(),
        },
        ScalarType::F32 => {
            let value = value as f32;
            (match op {
                UnaryOp::Neg => -value,
                UnaryOp::Abs => value.abs(),
                UnaryOp::Sqrt => value.sqrt(),
                UnaryOp::Exp => value.exp(),
                UnaryOp::Ln => value.ln(),
                UnaryOp::Sin => value.sin(),
                UnaryOp::Cos => value.cos(),
                UnaryOp::Tan => value.tan(),
                UnaryOp::Floor => value.floor(),
                UnaryOp::Ceil => value.ceil(),
            }) as f64
        }
    }
}

fn binary(op: BinaryOp, lhs: f64, rhs: f64, scalar_type: ScalarType) -> f64 {
    match scalar_type {
        ScalarType::F64 => match op {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
            BinaryOp::Pow => lhs.powf(rhs),
            BinaryOp::Min => lhs.min(rhs),
            BinaryOp::Max => lhs.max(rhs),
            BinaryOp::Atan2 => lhs.atan2(rhs),
        },
        ScalarType::F32 => {
            let lhs = lhs as f32;
            let rhs = rhs as f32;
            (match op {
                BinaryOp::Add => lhs + rhs,
                BinaryOp::Sub => lhs - rhs,
                BinaryOp::Mul => lhs * rhs,
                BinaryOp::Div => lhs / rhs,
                BinaryOp::Pow => lhs.powf(rhs),
                BinaryOp::Min => lhs.min(rhs),
                BinaryOp::Max => lhs.max(rhs),
                BinaryOp::Atan2 => lhs.atan2(rhs),
            }) as f64
        }
    }
}

fn cast(value: f64, scalar_type: ScalarType) -> f64 {
    match scalar_type {
        ScalarType::F32 => (value as f32) as f64,
        ScalarType::F64 => value,
    }
}

fn compare(op: CompareOp, lhs: f64, rhs: f64) -> bool {
    match op {
        CompareOp::Eq => lhs == rhs,
        CompareOp::NotEq => lhs != rhs,
        CompareOp::Less => lhs < rhs,
        CompareOp::LessEqual => lhs <= rhs,
        CompareOp::Greater => lhs > rhs,
        CompareOp::GreaterEqual => lhs >= rhs,
    }
}
