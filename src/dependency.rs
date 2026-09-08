//! Conservative input dependence of validated primal outputs.

use crate::{
    AccessMode, OperandId, Predicate, ScalarExpr, Statement, StructuredKernel, ValidationError,
    validate,
};

/// Whether any final writable operand may depend on the requested readable operand.
///
/// This is a structural proof of non-use, not a derivative or algebraic simplifier.
/// Ordered local definitions and overwritten stores are followed; dead locals do not
/// contribute. Both select branches and their predicates contribute even when a predicate
/// is constant. A queried read-write input always conservatively reads itself because
/// stores may leave elements untouched or execute zero times. Read-write buffers are themselves
/// outputs, so a carried dependence is conservatively
/// reported even if a later iteration could overwrite it. Operand elements are not distinguished.
/// All indexing maps and iteration extents in this IR are static and cannot depend on input data.
/// A `false` result proves no input-to-output value path, not absence of execution-time errors
/// in dead expressions. The full kernel is validated before inspecting it.
pub fn primal_output_reads_input(
    kernel: &StructuredKernel,
    input: OperandId,
) -> Result<bool, ValidationError> {
    validate(kernel.clone())?;
    let operand = kernel
        .operands
        .get(input.index())
        .ok_or(ValidationError::InvalidOperand(input.index()))?;
    if !operand.access.can_read() {
        return Err(ValidationError::InvalidLoad(input.index()));
    }
    // Validation does not require stores to cover the whole buffer, or a nonempty
    // iteration domain. Untouched elements retain the queried initial value.
    if operand.access == AccessMode::ReadWrite {
        return Ok(true);
    }
    let mut values = vec![false; kernel.operands.len()];
    values[input.index()] = true;
    let mut locals = Vec::new();
    for statement in &kernel.body.statements {
        match statement {
            Statement::Let { value, .. } => locals.push(reads(value, &values, &locals)),
            Statement::Store { operand, value } => {
                let depends = reads(value, &values, &locals);
                if matches!(
                    kernel.operands[operand.index()].access,
                    AccessMode::Reduce(_)
                ) {
                    values[operand.index()] |= depends;
                } else {
                    values[operand.index()] = depends;
                }
            }
        }
    }
    // Every writable operand is an observable output, including ReadWrite buffers.
    // If none carries the input after one iteration, none can carry it into another.
    Ok(kernel
        .operands
        .iter()
        .zip(values)
        .any(|(operand, value)| operand.access.can_write() && value))
}

fn reads(expr: &ScalarExpr, operands: &[bool], locals: &[bool]) -> bool {
    match expr {
        ScalarExpr::Constant(_) | ScalarExpr::Index(_) => false,
        ScalarExpr::Load(id) => operands[id.index()],
        ScalarExpr::Local(id) => locals[id.index()],
        ScalarExpr::Unary { arg, .. } => reads(arg, operands, locals),
        ScalarExpr::Binary { lhs, rhs, .. } => {
            reads(lhs, operands, locals) || reads(rhs, operands, locals)
        }
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => {
            predicate_reads(condition, operands, locals)
                || reads(if_true, operands, locals)
                || reads(if_false, operands, locals)
        }
    }
}

fn predicate_reads(predicate: &Predicate, operands: &[bool], locals: &[bool]) -> bool {
    match predicate {
        Predicate::Constant(_) => false,
        Predicate::Compare { lhs, rhs, .. } => {
            reads(lhs, operands, locals) || reads(rhs, operands, locals)
        }
        Predicate::Not(value) => predicate_reads(value, operands, locals),
        Predicate::And(lhs, rhs) | Predicate::Or(lhs, rhs) => {
            predicate_reads(lhs, operands, locals) || predicate_reads(rhs, operands, locals)
        }
    }
}
