//! R15 scientific property lowering and finite-precision execution.
//!
//! This IR is deliberately local: it knows scalar property inputs, guards, branches and tables,
//! but nothing about meshes, fields, materials, timestepping or global assembly.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKernel {
    Constant(f64),
    Input(String),
    Neg(Box<ExprKernel>),
    Add(Box<ExprKernel>, Box<ExprKernel>),
    Sub(Box<ExprKernel>, Box<ExprKernel>),
    Mul(Box<ExprKernel>, Box<ExprKernel>),
    Div(Box<ExprKernel>, Box<ExprKernel>),
    Pow(Box<ExprKernel>, Box<ExprKernel>),
    Call {
        function: String,
        args: Vec<ExprKernel>,
    },
    Compare {
        op: CompareKernel,
        lhs: Box<ExprKernel>,
        rhs: Box<ExprKernel>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareKernel {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Dual {
    value: f64,
    tangent: f64,
}

impl ExprKernel {
    pub fn evaluate(&self, inputs: &BTreeMap<String, f64>) -> Result<f64, ScientificPropertyError> {
        Ok(self.dual(inputs, None)?.value)
    }

    pub fn derivative(
        &self,
        input: &str,
        inputs: &BTreeMap<String, f64>,
    ) -> Result<f64, ScientificPropertyError> {
        Ok(self.dual(inputs, Some(input))?.tangent)
    }

    fn dual(
        &self,
        inputs: &BTreeMap<String, f64>,
        derivative_input: Option<&str>,
    ) -> Result<Dual, ScientificPropertyError> {
        let eval = |expr: &ExprKernel| expr.dual(inputs, derivative_input);
        Ok(match self {
            Self::Constant(value) => Dual {
                value: *value,
                tangent: 0.0,
            },
            Self::Input(name) => Dual {
                value: *inputs
                    .get(name)
                    .ok_or_else(|| ScientificPropertyError::MissingInput(name.clone()))?,
                tangent: if derivative_input == Some(name.as_str()) {
                    1.0
                } else {
                    0.0
                },
            },
            Self::Neg(arg) => {
                let x = eval(arg)?;
                Dual {
                    value: -x.value,
                    tangent: -x.tangent,
                }
            }
            Self::Add(lhs, rhs) => {
                let a = eval(lhs)?;
                let b = eval(rhs)?;
                Dual {
                    value: a.value + b.value,
                    tangent: a.tangent + b.tangent,
                }
            }
            Self::Sub(lhs, rhs) => {
                let a = eval(lhs)?;
                let b = eval(rhs)?;
                Dual {
                    value: a.value - b.value,
                    tangent: a.tangent - b.tangent,
                }
            }
            Self::Mul(lhs, rhs) => {
                let a = eval(lhs)?;
                let b = eval(rhs)?;
                Dual {
                    value: a.value * b.value,
                    tangent: a.tangent * b.value + a.value * b.tangent,
                }
            }
            Self::Div(lhs, rhs) => {
                let a = eval(lhs)?;
                let b = eval(rhs)?;
                Dual {
                    value: a.value / b.value,
                    tangent: (a.tangent * b.value - a.value * b.tangent) / (b.value * b.value),
                }
            }
            Self::Pow(lhs, rhs) => {
                let a = eval(lhs)?;
                let b = eval(rhs)?;
                let value = a.value.powf(b.value);
                let tangent = if a.value > 0.0 {
                    value * (b.tangent * a.value.ln() + b.value * a.tangent / a.value)
                } else if b.tangent == 0.0 {
                    b.value * a.value.powf(b.value - 1.0) * a.tangent
                } else {
                    return Err(ScientificPropertyError::NonDifferentiable(
                        "pow with non-positive base and variable exponent".into(),
                    ));
                };
                Dual { value, tangent }
            }
            Self::Compare { op, lhs, rhs } => {
                let a = eval(lhs)?.value;
                let b = eval(rhs)?.value;
                let value = match op {
                    CompareKernel::Eq => a == b,
                    CompareKernel::Lt => a < b,
                    CompareKernel::Le => a <= b,
                    CompareKernel::Gt => a > b,
                    CompareKernel::Ge => a >= b,
                };
                Dual {
                    value: if value { 1.0 } else { 0.0 },
                    tangent: 0.0,
                }
            }
            Self::Call { function, args } => {
                let xs = args.iter().map(eval).collect::<Result<Vec<_>, _>>()?;
                match (function.as_str(), xs.as_slice()) {
                    ("sin", [x]) => Dual {
                        value: x.value.sin(),
                        tangent: x.value.cos() * x.tangent,
                    },
                    ("cos", [x]) => Dual {
                        value: x.value.cos(),
                        tangent: -x.value.sin() * x.tangent,
                    },
                    ("exp", [x]) => {
                        let value = x.value.exp();
                        Dual {
                            value,
                            tangent: value * x.tangent,
                        }
                    }
                    ("log" | "ln", [x]) => Dual {
                        value: x.value.ln(),
                        tangent: x.tangent / x.value,
                    },
                    ("sqrt", [x]) => {
                        let value = x.value.sqrt();
                        Dual {
                            value,
                            tangent: x.tangent / (2.0 * value),
                        }
                    }
                    ("abs", [x]) if x.value != 0.0 => Dual {
                        value: x.value.abs(),
                        tangent: x.value.signum() * x.tangent,
                    },
                    ("abs", [_]) => {
                        return Err(ScientificPropertyError::NonDifferentiable(
                            "abs at zero".into(),
                        ));
                    }
                    ("min", [a, b]) if a.value < b.value => *a,
                    ("min", [a, b]) if b.value < a.value => *b,
                    ("max", [a, b]) if a.value > b.value => *a,
                    ("max", [a, b]) if b.value > a.value => *b,
                    ("min" | "max", [_, _]) => {
                        return Err(ScientificPropertyError::NonDifferentiable(format!(
                            "{function} at branch tie"
                        )));
                    }
                    _ => {
                        return Err(ScientificPropertyError::UnsupportedExpression(format!(
                            "call `{function}`"
                        )));
                    }
                }
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GuardKernel {
    pub input: String,
    pub physical_min: Option<f64>,
    pub physical_max: Option<f64>,
    pub validity_min: Option<f64>,
    pub validity_max: Option<f64>,
    pub validity_policy: ValidityPolicyKernel,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidityPolicyKernel {
    Error,
    Warn,
    ExplicitExtrapolation(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredicateKernel {
    pub input: String,
    pub op: CompareKernel,
    pub value: f64,
}
impl PredicateKernel {
    fn matches(&self, inputs: &BTreeMap<String, f64>) -> Result<bool, ScientificPropertyError> {
        let value = *inputs
            .get(&self.input)
            .ok_or_else(|| ScientificPropertyError::MissingInput(self.input.clone()))?;
        Ok(match self.op {
            CompareKernel::Eq => value == self.value,
            CompareKernel::Lt => value < self.value,
            CompareKernel::Le => value <= self.value,
            CompareKernel::Gt => value > self.value,
            CompareKernel::Ge => value >= self.value,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BranchKernel {
    pub predicate: Option<PredicateKernel>,
    pub value: ExprKernel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GridAxisKernel {
    pub input: String,
    pub points: Vec<f64>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct GridKernel {
    pub axes: Vec<GridAxisKernel>,
    pub values: Vec<f64>,
}

impl GridKernel {
    fn validate(&self) -> Result<(), ScientificPropertyError> {
        if self.axes.is_empty()
            || self.axes.len() > 2
            || self
                .axes
                .iter()
                .any(|axis| axis.points.len() < 2 || axis.points.windows(2).any(|w| w[1] <= w[0]))
        {
            return Err(ScientificPropertyError::TableShape);
        }
        let expected = self
            .axes
            .iter()
            .map(|axis| axis.points.len())
            .product::<usize>();
        if expected != self.values.len() {
            return Err(ScientificPropertyError::TableShape);
        }
        Ok(())
    }
    fn interval(points: &[f64], value: f64) -> (usize, f64) {
        let mut i = 0usize;
        while i + 1 < points.len() - 1 && value > points[i + 1] {
            i += 1;
        }
        let t = (value - points[i]) / (points[i + 1] - points[i]);
        (i, t)
    }
    pub fn evaluate(&self, inputs: &BTreeMap<String, f64>) -> Result<f64, ScientificPropertyError> {
        self.validate()?;
        match self.axes.as_slice() {
            [x] => {
                let xv = *inputs
                    .get(&x.input)
                    .ok_or_else(|| ScientificPropertyError::MissingInput(x.input.clone()))?;
                let (i, t) = Self::interval(&x.points, xv);
                Ok(self.values[i] * (1.0 - t) + self.values[i + 1] * t)
            }
            [x, y] => {
                let xv = *inputs
                    .get(&x.input)
                    .ok_or_else(|| ScientificPropertyError::MissingInput(x.input.clone()))?;
                let yv = *inputs
                    .get(&y.input)
                    .ok_or_else(|| ScientificPropertyError::MissingInput(y.input.clone()))?;
                let (i, tx) = Self::interval(&x.points, xv);
                let (j, ty) = Self::interval(&y.points, yv);
                let ny = y.points.len();
                let v00 = self.values[i * ny + j];
                let v01 = self.values[i * ny + j + 1];
                let v10 = self.values[(i + 1) * ny + j];
                let v11 = self.values[(i + 1) * ny + j + 1];
                Ok((1.0 - tx) * (1.0 - ty) * v00
                    + (1.0 - tx) * ty * v01
                    + tx * (1.0 - ty) * v10
                    + tx * ty * v11)
            }
            _ => Err(ScientificPropertyError::TableShape),
        }
    }
    pub fn derivative(
        &self,
        input: &str,
        inputs: &BTreeMap<String, f64>,
    ) -> Result<Option<f64>, ScientificPropertyError> {
        self.validate()?;
        match self.axes.as_slice() {
            [x] if x.input == input => {
                let xv = *inputs
                    .get(input)
                    .ok_or_else(|| ScientificPropertyError::MissingInput(input.into()))?;
                let (i, _) = Self::interval(&x.points, xv);
                Ok(Some(
                    (self.values[i + 1] - self.values[i]) / (x.points[i + 1] - x.points[i]),
                ))
            }
            [x, y] if x.input == input => {
                let xv = *inputs
                    .get(&x.input)
                    .ok_or_else(|| ScientificPropertyError::MissingInput(x.input.clone()))?;
                let yv = *inputs
                    .get(&y.input)
                    .ok_or_else(|| ScientificPropertyError::MissingInput(y.input.clone()))?;
                let (i, _) = Self::interval(&x.points, xv);
                let (j, ty) = Self::interval(&y.points, yv);
                let ny = y.points.len();
                let lo = (1.0 - ty) * self.values[i * ny + j] + ty * self.values[i * ny + j + 1];
                let hi = (1.0 - ty) * self.values[(i + 1) * ny + j]
                    + ty * self.values[(i + 1) * ny + j + 1];
                Ok(Some((hi - lo) / (x.points[i + 1] - x.points[i])))
            }
            [x, y] if y.input == input => {
                let xv = *inputs
                    .get(&x.input)
                    .ok_or_else(|| ScientificPropertyError::MissingInput(x.input.clone()))?;
                let yv = *inputs
                    .get(&y.input)
                    .ok_or_else(|| ScientificPropertyError::MissingInput(y.input.clone()))?;
                let (i, tx) = Self::interval(&x.points, xv);
                let (j, _) = Self::interval(&y.points, yv);
                let ny = y.points.len();
                let lo = (1.0 - tx) * self.values[i * ny + j] + tx * self.values[(i + 1) * ny + j];
                let hi = (1.0 - tx) * self.values[i * ny + j + 1]
                    + tx * self.values[(i + 1) * ny + j + 1];
                Ok(Some((hi - lo) / (y.points[j + 1] - y.points[j])))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScientificPropertyModelKernel {
    Expression(ExprKernel),
    Piecewise(Vec<BranchKernel>),
    Grid(GridKernel),
    External { provider: String, property: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScientificPropertyKernel {
    pub id: String,
    pub model: ScientificPropertyModelKernel,
    pub guards: Vec<GuardKernel>,
    pub derivative_inputs: BTreeSet<String>,
    pub evidence_digest: Option<String>,
}

impl ScientificPropertyKernel {
    fn guard(&self, inputs: &BTreeMap<String, f64>) -> Result<(), ScientificPropertyError> {
        for guard in &self.guards {
            let value = *inputs
                .get(&guard.input)
                .ok_or_else(|| ScientificPropertyError::MissingInput(guard.input.clone()))?;
            if guard.physical_min.is_some_and(|x| value < x)
                || guard.physical_max.is_some_and(|x| value > x)
            {
                return Err(ScientificPropertyError::PhysicalBound {
                    input: guard.input.clone(),
                    value,
                });
            }
            if (guard.validity_min.is_some_and(|x| value < x)
                || guard.validity_max.is_some_and(|x| value > x))
                && matches!(guard.validity_policy, ValidityPolicyKernel::Error)
            {
                return Err(ScientificPropertyError::ValidityBound {
                    input: guard.input.clone(),
                    value,
                });
            }
        }
        Ok(())
    }
    pub fn evaluate(&self, inputs: &BTreeMap<String, f64>) -> Result<f64, ScientificPropertyError> {
        self.guard(inputs)?;
        match &self.model {
            ScientificPropertyModelKernel::Expression(expr) => expr.evaluate(inputs),
            ScientificPropertyModelKernel::Piecewise(branches) => branches
                .iter()
                .find_map(|branch| match &branch.predicate {
                    Some(p) => p.matches(inputs).ok().filter(|x| *x).map(|_| &branch.value),
                    None => Some(&branch.value),
                })
                .ok_or(ScientificPropertyError::NoBranch)?
                .evaluate(inputs),
            ScientificPropertyModelKernel::Grid(table) => table.evaluate(inputs),
            ScientificPropertyModelKernel::External { provider, property } => {
                Err(ScientificPropertyError::ExternalProviderRequired {
                    provider: provider.clone(),
                    property: property.clone(),
                })
            }
        }
    }
    pub fn derivative(
        &self,
        input: &str,
        inputs: &BTreeMap<String, f64>,
    ) -> Result<Option<f64>, ScientificPropertyError> {
        self.guard(inputs)?;
        if !self.derivative_inputs.contains(input) {
            return Ok(None);
        }
        match &self.model {
            ScientificPropertyModelKernel::Expression(expr) => {
                Ok(Some(expr.derivative(input, inputs)?))
            }
            ScientificPropertyModelKernel::Piecewise(branches) => {
                let expr = branches
                    .iter()
                    .find_map(|branch| match &branch.predicate {
                        Some(p) => p.matches(inputs).ok().filter(|x| *x).map(|_| &branch.value),
                        None => Some(&branch.value),
                    })
                    .ok_or(ScientificPropertyError::NoBranch)?;
                Ok(Some(expr.derivative(input, inputs)?))
            }
            ScientificPropertyModelKernel::Grid(table) => table.derivative(input, inputs),
            ScientificPropertyModelKernel::External { .. } => Ok(None),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScientificPropertyError {
    MissingInput(String),
    PhysicalBound { input: String, value: f64 },
    ValidityBound { input: String, value: f64 },
    TableShape,
    NoBranch,
    ExternalProviderRequired { provider: String, property: String },
    UnsupportedExpression(String),
    NonDifferentiable(String),
}
impl std::fmt::Display for ScientificPropertyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ScientificPropertyError {}

#[cfg(feature = "resolvent")]
#[derive(Clone, Debug, PartialEq)]
pub enum ScientificPropertyLoweringError {
    Unsupported(String),
}
#[cfg(feature = "resolvent")]
impl std::fmt::Display for ScientificPropertyLoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
#[cfg(feature = "resolvent")]
impl std::error::Error for ScientificPropertyLoweringError {}

#[cfg(feature = "resolvent")]
pub fn lower_property_definition(
    def: &resolvent::scientific::PropertyDefinition,
) -> Result<ScientificPropertyKernel, ScientificPropertyLoweringError> {
    use resolvent::scientific::{CompareOp, OutOfValidityPolicy, PropertyModel};
    let expr = |value: &resolvent::scientific::Expr| lower_expr(value);
    let model = match &def.model {
        PropertyModel::Constant(value) | PropertyModel::Expression(value) => {
            ScientificPropertyModelKernel::Expression(expr(value)?)
        }
        PropertyModel::Piecewise(branches) => ScientificPropertyModelKernel::Piecewise(
            branches
                .iter()
                .map(|branch| {
                    Ok(BranchKernel {
                        predicate: branch.when.as_ref().map(|p| PredicateKernel {
                            input: p.variable.clone(),
                            op: match p.op {
                                CompareOp::Lt => CompareKernel::Lt,
                                CompareOp::Le => CompareKernel::Le,
                                CompareOp::Gt => CompareKernel::Gt,
                                CompareOp::Ge => CompareKernel::Ge,
                            },
                            value: p.value,
                        }),
                        value: expr(&branch.value)?,
                    })
                })
                .collect::<Result<Vec<_>, ScientificPropertyLoweringError>>()?,
        ),
        PropertyModel::Table(table) => ScientificPropertyModelKernel::Grid(GridKernel {
            axes: table
                .axes
                .iter()
                .map(|axis| GridAxisKernel {
                    input: axis.name.clone(),
                    points: axis.points.clone(),
                })
                .collect(),
            values: table.values.clone(),
        }),
        PropertyModel::External(reference) => ScientificPropertyModelKernel::External {
            provider: reference.provider.clone(),
            property: reference.property.clone(),
        },
    };
    let guards = def
        .signature
        .inputs
        .iter()
        .map(|input| {
            let physical = def
                .domain
                .physical_bounds
                .iter()
                .find(|b| b.input == input.name);
            let validity = def
                .domain
                .validity_bounds
                .iter()
                .find(|b| b.input == input.name);
            GuardKernel {
                input: input.name.clone(),
                physical_min: physical.and_then(|b| b.min).or(input.physical_min),
                physical_max: physical.and_then(|b| b.max).or(input.physical_max),
                validity_min: validity.and_then(|b| b.min),
                validity_max: validity.and_then(|b| b.max),
                validity_policy: match &def.domain.out_of_validity {
                    OutOfValidityPolicy::Error => ValidityPolicyKernel::Error,
                    OutOfValidityPolicy::Warn => ValidityPolicyKernel::Warn,
                    OutOfValidityPolicy::ExplicitExtrapolation(x) => {
                        ValidityPolicyKernel::ExplicitExtrapolation(x.clone())
                    }
                },
            }
        })
        .collect();
    let derivative_inputs = def
        .signature
        .inputs
        .iter()
        .map(|input| input.name.clone())
        .collect();
    let evidence_digest = def
        .evidence
        .dataset_digest
        .clone()
        .or_else(|| def.evidence.fit_digest.clone());
    Ok(ScientificPropertyKernel {
        id: def.signature.id.clone(),
        model,
        guards,
        derivative_inputs,
        evidence_digest,
    })
}

#[cfg(feature = "resolvent")]
fn lower_expr(
    expr: &resolvent::scientific::Expr,
) -> Result<ExprKernel, ScientificPropertyLoweringError> {
    use resolvent::scientific::{BinaryOp, Expr};
    let boxed = |x: &Expr| lower_expr(x).map(Box::new);
    Ok(match expr {
        Expr::Number { value, .. } => ExprKernel::Constant(*value),
        Expr::Name(name) => ExprKernel::Input(name.clone()),
        Expr::Unary { arg, .. } => ExprKernel::Neg(boxed(arg)?),
        Expr::Binary { op, lhs, rhs } => match op {
            BinaryOp::Add => ExprKernel::Add(boxed(lhs)?, boxed(rhs)?),
            BinaryOp::Sub => ExprKernel::Sub(boxed(lhs)?, boxed(rhs)?),
            BinaryOp::Mul => ExprKernel::Mul(boxed(lhs)?, boxed(rhs)?),
            BinaryOp::Div => ExprKernel::Div(boxed(lhs)?, boxed(rhs)?),
            BinaryOp::Pow => ExprKernel::Pow(boxed(lhs)?, boxed(rhs)?),
            BinaryOp::Eq => ExprKernel::Compare {
                op: CompareKernel::Eq,
                lhs: boxed(lhs)?,
                rhs: boxed(rhs)?,
            },
            BinaryOp::Lt => ExprKernel::Compare {
                op: CompareKernel::Lt,
                lhs: boxed(lhs)?,
                rhs: boxed(rhs)?,
            },
            BinaryOp::Le => ExprKernel::Compare {
                op: CompareKernel::Le,
                lhs: boxed(lhs)?,
                rhs: boxed(rhs)?,
            },
            BinaryOp::Gt => ExprKernel::Compare {
                op: CompareKernel::Gt,
                lhs: boxed(lhs)?,
                rhs: boxed(rhs)?,
            },
            BinaryOp::Ge => ExprKernel::Compare {
                op: CompareKernel::Ge,
                lhs: boxed(lhs)?,
                rhs: boxed(rhs)?,
            },
        },
        Expr::Call { function, args } => ExprKernel::Call {
            function: function.clone(),
            args: args.iter().map(lower_expr).collect::<Result<Vec<_>, _>>()?,
        },
        Expr::String(_) | Expr::Index { .. } | Expr::Vector(_) => {
            return Err(ScientificPropertyLoweringError::Unsupported(format!(
                "{expr:?}"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonlinear_expression_derivative_matches_finite_difference() {
        let expr = ExprKernel::Add(
            Box::new(ExprKernel::Mul(
                Box::new(ExprKernel::Constant(2.0)),
                Box::new(ExprKernel::Input("T".into())),
            )),
            Box::new(ExprKernel::Call {
                function: "exp".into(),
                args: vec![ExprKernel::Div(
                    Box::new(ExprKernel::Input("T".into())),
                    Box::new(ExprKernel::Constant(100.0)),
                )],
            }),
        );
        let env = BTreeMap::from([("T".into(), 350.0)]);
        let h = 1e-5;
        let analytic = expr.derivative("T", &env).unwrap();
        let mut plus = env.clone();
        plus.insert("T".into(), 350.0 + h);
        let mut minus = env.clone();
        minus.insert("T".into(), 350.0 - h);
        let fd = (expr.evaluate(&plus).unwrap() - expr.evaluate(&minus).unwrap()) / (2.0 * h);
        assert!((analytic - fd).abs() < 1e-7);
    }
    #[test]
    fn two_dimensional_table_has_bilinear_value_and_partials() {
        let table = GridKernel {
            axes: vec![
                GridAxisKernel {
                    input: "T".into(),
                    points: vec![0.0, 1.0],
                },
                GridAxisKernel {
                    input: "p".into(),
                    points: vec![0.0, 2.0],
                },
            ],
            values: vec![0.0, 4.0, 3.0, 7.0],
        };
        let env = BTreeMap::from([("T".into(), 0.25), ("p".into(), 0.5)]);
        assert!((table.evaluate(&env).unwrap() - 1.75).abs() < 1e-12);
        assert!((table.derivative("T", &env).unwrap().unwrap() - 3.0).abs() < 1e-12);
        assert!((table.derivative("p", &env).unwrap().unwrap() - 2.0).abs() < 1e-12);
    }
}
