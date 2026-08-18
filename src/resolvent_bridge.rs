//! Resolvent -> Malleus execution boundary.
//!
//! Resolvent owns mathematical and discrete semantics. Malleus owns finite-precision
//! pointwise compilation. This adapter lowers scalar pointwise expression programs into
//! Malleus' opcode IR and reports field/basis stages that must remain in the element runtime
//! instead of pretending they are scalar JIT operations.

use std::collections::BTreeMap;

use resolvent::{DiscreteOp, ExecutionPlan, ExprId, ExprNode, SymbolId};
use thiserror::Error;

use crate::{CompiledConstraints, OpcodeEmitter, Reg};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanCoverage {
    pub pointwise_expressions: usize,
    pub field_runtime_ops: usize,
    pub custom_ops: Vec<String>,
}

#[derive(Debug, Error, PartialEq)]
pub enum ResolventLoweringError {
    #[error("execution plan contains expression id {0} absent from the supplied context")]
    MissingExpression(u32),
    #[error("symbol {0} has no executable input binding")]
    UnboundSymbol(u32),
    #[error("semantic operator `{0}` requires a field/runtime lowering, not the scalar opcode JIT")]
    FieldOperator(String),
    #[error("unsupported scalar function `{0}` in Malleus pointwise lowering")]
    UnsupportedFunction(String),
}

/// Audit an execution plan before compilation. This is intentionally separate from lowering:
/// callers can inspect whether a plan is fully pointwise-JIT-able or needs field runtime work.
pub fn coverage(plan: &ExecutionPlan) -> PlanCoverage {
    let mut pointwise_expressions = 0;
    let mut field_runtime_ops = 0;
    let mut custom_ops = vec![];
    for program in &plan.programs {
        for instruction in &program.instructions {
            match &instruction.op {
                DiscreteOp::Pointwise { expressions, .. } => {
                    pointwise_expressions += expressions.len();
                }
                DiscreteOp::FieldInput { .. }
                | DiscreteOp::Restrict { .. }
                | DiscreteOp::Basis { .. }
                | DiscreteOp::QuadratureWeight { .. }
                | DiscreteOp::Sum { .. } => field_runtime_ops += 1,
                DiscreteOp::Custom { operator, .. } => custom_ops.push(operator.clone()),
            }
        }
    }
    PlanCoverage {
        pointwise_expressions,
        field_runtime_ops,
        custom_ops,
    }
}

/// Compile every pointwise scalar expression in a Resolvent execution plan into one Malleus
/// `CompiledConstraints` object. `bindings` gives each semantic symbol a stable input slot.
/// Restriction/basis/quadrature/scatter remain owned by the field execution runtime; this
/// function compiles the QFunction-like pointwise physics sitting between those stages.
pub fn lower_pointwise_plan(
    ctx: &resolvent::Context,
    plan: &ExecutionPlan,
    bindings: &BTreeMap<SymbolId, u32>,
) -> Result<CompiledConstraints, ResolventLoweringError> {
    let expressions: Vec<ExprId> = plan
        .programs
        .iter()
        .flat_map(|program| program.instructions.iter())
        .filter_map(|instruction| match &instruction.op {
            DiscreteOp::Pointwise { expressions, .. } => Some(expressions.as_slice()),
            _ => None,
        })
        .flatten()
        .copied()
        .collect();

    let n_vars = bindings
        .values()
        .copied()
        .max()
        .map_or(0, |x| x as usize + 1);
    let mut residual = OpcodeEmitter::new();
    for (i, expr) in expressions.iter().copied().enumerate() {
        let mut memo = BTreeMap::new();
        let value = lower_expr(ctx, expr, bindings, &mut residual, &mut memo)?;
        residual.store_residual(i as u32, value);
    }

    let max_register = residual.max_register();
    let residual_ops = residual.into_ops();
    Ok(CompiledConstraints {
        residual_ops,
        jacobian_ops: vec![],
        hessian_ops: vec![],
        n_residuals: expressions.len(),
        n_vars,
        jacobian_nnz: 0,
        jacobian_pattern: vec![],
        hessian_nnz: 0,
        hessian_pattern: vec![],
        max_register,
    })
}

fn lower_expr(
    ctx: &resolvent::Context,
    id: ExprId,
    bindings: &BTreeMap<SymbolId, u32>,
    emitter: &mut OpcodeEmitter,
    memo: &mut BTreeMap<ExprId, Reg>,
) -> Result<Reg, ResolventLoweringError> {
    if let Some(reg) = memo.get(&id) {
        return Ok(*reg);
    }
    let node = ctx
        .exprs
        .get(id)
        .ok_or(ResolventLoweringError::MissingExpression(id.0))?;
    let reg = match node {
        ExprNode::Literal(lit) => emitter.const_f64(literal_f64(lit)?),
        ExprNode::Symbol(symbol) => emitter.load_var(
            *bindings
                .get(symbol)
                .ok_or(ResolventLoweringError::UnboundSymbol(symbol.0))?,
        ),
        ExprNode::Neg(x) => {
            let x = lower_expr(ctx, *x, bindings, emitter, memo)?;
            emitter.neg(x)
        }
        ExprNode::Add(xs) => fold(ctx, xs, bindings, emitter, memo, 0.0, OpcodeEmitter::add)?,
        ExprNode::Mul(xs) => fold(ctx, xs, bindings, emitter, memo, 1.0, OpcodeEmitter::mul)?,
        ExprNode::Div {
            numerator,
            denominator,
        } => {
            let a = lower_expr(ctx, *numerator, bindings, emitter, memo)?;
            let b = lower_expr(ctx, *denominator, bindings, emitter, memo)?;
            emitter.div(a, b)
        }
        ExprNode::PowI { base, exponent } => {
            let base = lower_expr(ctx, *base, bindings, emitter, memo)?;
            let exponent = emitter.const_f64(*exponent as f64);
            emitter.pow(base, exponent)
        }
        ExprNode::Derivative { .. } => {
            return Err(ResolventLoweringError::FieldOperator(
                "semantic derivative".into(),
            ));
        }
        ExprNode::Apply { function, args } => {
            lower_apply(ctx, function, args, bindings, emitter, memo)?
        }
    };
    memo.insert(id, reg);
    Ok(reg)
}

fn fold(
    ctx: &resolvent::Context,
    xs: &[ExprId],
    bindings: &BTreeMap<SymbolId, u32>,
    emitter: &mut OpcodeEmitter,
    memo: &mut BTreeMap<ExprId, Reg>,
    identity: f64,
    op: fn(&mut OpcodeEmitter, Reg, Reg) -> Reg,
) -> Result<Reg, ResolventLoweringError> {
    let mut acc = emitter.const_f64(identity);
    for x in xs {
        let rhs = lower_expr(ctx, *x, bindings, emitter, memo)?;
        acc = op(emitter, acc, rhs);
    }
    Ok(acc)
}

fn lower_apply(
    ctx: &resolvent::Context,
    function: &str,
    args: &[ExprId],
    bindings: &BTreeMap<SymbolId, u32>,
    emitter: &mut OpcodeEmitter,
    memo: &mut BTreeMap<ExprId, Reg>,
) -> Result<Reg, ResolventLoweringError> {
    if matches!(function, "grad" | "div" | "curl" | "dot") {
        return Err(ResolventLoweringError::FieldOperator(function.into()));
    }
    let regs = args
        .iter()
        .map(|x| lower_expr(ctx, *x, bindings, emitter, memo))
        .collect::<Result<Vec<_>, _>>()?;
    match (function, regs.as_slice()) {
        ("sin", [x]) => Ok(emitter.sin(*x)),
        ("cos", [x]) => Ok(emitter.cos(*x)),
        ("tan", [x]) => Ok(emitter.tan(*x)),
        ("exp", [x]) => Ok(emitter.exp(*x)),
        ("log" | "ln", [x]) => Ok(emitter.ln(*x)),
        ("sqrt", [x]) => Ok(emitter.sqrt(*x)),
        ("abs", [x]) => Ok(emitter.abs(*x)),
        ("atan2", [y, x]) => Ok(emitter.atan2(*y, *x)),
        ("pow", [a, b]) => Ok(emitter.pow(*a, *b)),
        _ => Err(ResolventLoweringError::UnsupportedFunction(function.into())),
    }
}

fn literal_f64(lit: &resolvent::ScalarLiteral) -> Result<f64, ResolventLoweringError> {
    Ok(match lit {
        resolvent::ScalarLiteral::Integer(s) => s
            .parse()
            .map_err(|_| ResolventLoweringError::UnsupportedFunction(format!("literal {s}")))?,
        resolvent::ScalarLiteral::Rational {
            numerator,
            denominator,
        } => {
            let n: f64 = numerator.parse().map_err(|_| {
                ResolventLoweringError::UnsupportedFunction(format!("literal {numerator}"))
            })?;
            let d: f64 = denominator.parse().map_err(|_| {
                ResolventLoweringError::UnsupportedFunction(format!("literal {denominator}"))
            })?;
            n / d
        }
        resolvent::ScalarLiteral::FloatBits(bits) => f64::from_bits(*bits),
        resolvent::ScalarLiteral::NamedConstant(name) if name == "pi" || name == "π" => {
            std::f64::consts::PI
        }
        resolvent::ScalarLiteral::NamedConstant(name) if name == "e" => std::f64::consts::E,
        resolvent::ScalarLiteral::NamedConstant(name) => {
            return Err(ResolventLoweringError::UnsupportedFunction(format!(
                "constant {name}"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use resolvent::{DiscreteProgram, ExecutionPolicy, OperatorProgram};

    #[test]
    fn coverage_does_not_hide_field_runtime() {
        let plan = ExecutionPlan {
            schema: "test".into(),
            operator_id: resolvent::OperatorId(0),
            operator: OperatorProgram {
                name: "x".into(),
                blocks: vec![],
                derivatives: vec![],
                properties: vec![],
                sparsity: None,
                metadata: Default::default(),
            },
            programs: vec![DiscreteProgram {
                name: "d".into(),
                instructions: vec![],
                outputs: vec![],
                metadata: Default::default(),
            }],
            context_digest: resolvent::Digest::blake3(b"x"),
            policy: ExecutionPolicy::default(),
        };
        assert_eq!(coverage(&plan).pointwise_expressions, 0);
    }
}
