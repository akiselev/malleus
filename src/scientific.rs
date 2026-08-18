//! R13-R18 finite-precision scientific kernel contracts.
//!
//! Malleus owns local/pointwise compilation only. Global topology, function spaces,
//! quadrature traversal, gather/scatter, state history, and solve strategy remain outside
//! this crate.

use std::collections::{BTreeMap, BTreeSet};

#[cfg(feature = "resolvent")]
use resolvent::{Context, ExecutionPlan, SymbolId};

#[cfg(feature = "resolvent")]
use crate::{CompiledConstraints, ResolventLoweringError, lower_pointwise_plan};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BindingLayout {
    pub inputs: BTreeMap<String, u32>,
    pub outputs: BTreeMap<String, u32>,
}

#[cfg(feature = "resolvent")]
pub struct CompiledKernelBundle {
    pub primal: CompiledConstraints,
    pub jvp: Option<CompiledConstraints>,
    pub vjp: Option<CompiledConstraints>,
    pub parameter_derivatives: Vec<CompiledConstraints>,
    pub bindings: BindingLayout,
}

#[cfg(feature = "resolvent")]
pub fn lower_kernel_bundle(
    ctx: &Context,
    primal: &ExecutionPlan,
    jvp: Option<&ExecutionPlan>,
    vjp: Option<&ExecutionPlan>,
    parameter_derivatives: &[ExecutionPlan],
    semantic_bindings: &BTreeMap<SymbolId, u32>,
    bindings: BindingLayout,
) -> Result<CompiledKernelBundle, ResolventLoweringError> {
    Ok(CompiledKernelBundle {
        primal: lower_pointwise_plan(ctx, primal, semantic_bindings)?,
        jvp: jvp
            .map(|plan| lower_pointwise_plan(ctx, plan, semantic_bindings))
            .transpose()?,
        vjp: vjp
            .map(|plan| lower_pointwise_plan(ctx, plan, semantic_bindings))
            .transpose()?,
        parameter_derivatives: parameter_derivatives
            .iter()
            .map(|plan| lower_pointwise_plan(ctx, plan, semantic_bindings))
            .collect::<Result<Vec<_>, _>>()?,
        bindings,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardPolicy {
    Error,
    Warn,
    ExplicitExtrapolation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidityGuard {
    pub input: String,
    pub physical_min: Option<f64>,
    pub physical_max: Option<f64>,
    pub validity_min: Option<f64>,
    pub validity_max: Option<f64>,
    pub policy: GuardPolicy,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearTable {
    pub input: String,
    pub points: Vec<f64>,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PropertyKernel {
    Constant(f64),
    Linear {
        intercept: f64,
        slopes: BTreeMap<String, f64>,
    },
    Table1d(LinearTable),
    External {
        provider: String,
        property: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PropertyKernelBundle {
    pub id: String,
    pub primal: PropertyKernel,
    pub guards: Vec<ValidityGuard>,
    pub derivative_inputs: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PropertyKernelError {
    MissingInput(String),
    PhysicalBound { input: String, value: f64 },
    ValidityBound { input: String, value: f64 },
    TableShape,
    ExternalProviderRequired { provider: String, property: String },
}

impl std::fmt::Display for PropertyKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingInput(x) => write!(f, "missing property input `{x}`"),
            Self::PhysicalBound { input, value } => {
                write!(f, "physical bound violated: {input}={value}")
            }
            Self::ValidityBound { input, value } => {
                write!(f, "validity bound violated: {input}={value}")
            }
            Self::TableShape => write!(f, "invalid property table shape"),
            Self::ExternalProviderRequired { provider, property } => {
                write!(f, "external provider required: {provider}:{property}")
            }
        }
    }
}
impl std::error::Error for PropertyKernelError {}

impl PropertyKernelBundle {
    pub fn evaluate(&self, inputs: &BTreeMap<String, f64>) -> Result<f64, PropertyKernelError> {
        for guard in &self.guards {
            let value = *inputs
                .get(&guard.input)
                .ok_or_else(|| PropertyKernelError::MissingInput(guard.input.clone()))?;
            if guard.physical_min.is_some_and(|x| value < x)
                || guard.physical_max.is_some_and(|x| value > x)
            {
                return Err(PropertyKernelError::PhysicalBound {
                    input: guard.input.clone(),
                    value,
                });
            }
            let outside_validity = guard.validity_min.is_some_and(|x| value < x)
                || guard.validity_max.is_some_and(|x| value > x);
            if outside_validity && guard.policy == GuardPolicy::Error {
                return Err(PropertyKernelError::ValidityBound {
                    input: guard.input.clone(),
                    value,
                });
            }
        }
        match &self.primal {
            PropertyKernel::Constant(value) => Ok(*value),
            PropertyKernel::Linear { intercept, slopes } => {
                let mut value = *intercept;
                for (name, slope) in slopes {
                    value += slope
                        * inputs
                            .get(name)
                            .ok_or_else(|| PropertyKernelError::MissingInput(name.clone()))?;
                }
                Ok(value)
            }
            PropertyKernel::Table1d(table) => table_value(table, inputs),
            PropertyKernel::External { provider, property } => {
                Err(PropertyKernelError::ExternalProviderRequired {
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
    ) -> Result<Option<f64>, PropertyKernelError> {
        if !self.derivative_inputs.contains(input) {
            return Ok(None);
        }
        match &self.primal {
            PropertyKernel::Constant(_) => Ok(Some(0.0)),
            PropertyKernel::Linear { slopes, .. } => Ok(Some(*slopes.get(input).unwrap_or(&0.0))),
            PropertyKernel::Table1d(table) if table.input == input => {
                let x = *inputs
                    .get(input)
                    .ok_or_else(|| PropertyKernelError::MissingInput(input.into()))?;
                Ok(Some(table_segment(table, x)?.1))
            }
            PropertyKernel::Table1d(_) | PropertyKernel::External { .. } => Ok(None),
        }
    }
}

fn table_value(
    table: &LinearTable,
    inputs: &BTreeMap<String, f64>,
) -> Result<f64, PropertyKernelError> {
    let x = *inputs
        .get(&table.input)
        .ok_or_else(|| PropertyKernelError::MissingInput(table.input.clone()))?;
    Ok(table_segment(table, x)?.0)
}
fn table_segment(table: &LinearTable, x: f64) -> Result<(f64, f64), PropertyKernelError> {
    if table.points.len() < 2 || table.points.len() != table.values.len() {
        return Err(PropertyKernelError::TableShape);
    }
    let mut i = 0usize;
    while i + 1 < table.points.len() - 1 && x > table.points[i + 1] {
        i += 1;
    }
    let dx = table.points[i + 1] - table.points[i];
    if dx <= 0.0 {
        return Err(PropertyKernelError::TableShape);
    }
    let slope = (table.values[i + 1] - table.values[i]) / dx;
    Ok((table.values[i] + slope * (x - table.points[i]), slope))
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KernelBlockId(pub String);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalInputKind {
    FieldValue,
    FieldGradient,
    GeometryFactor,
    QuadratureWeight,
    Property,
    ConstitutiveResponse,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalBinding {
    pub name: String,
    pub slot: u32,
    pub kind: LocalInputKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementKernelContract {
    pub block: KernelBlockId,
    pub inputs: Vec<LocalBinding>,
    pub outputs: Vec<String>,
    pub assembled: bool,
    pub matrix_free: bool,
    pub shared_evaluations: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstitutiveKernelContract {
    pub law: String,
    pub primal_outputs: Vec<String>,
    pub tangent_outputs: Vec<String>,
    pub parameter_derivatives: Vec<String>,
    pub stateful: bool,
}

/// CSE/share planning across coupled residual blocks. This does not execute or cache the
/// evaluations; it makes safe reuse explicit for the runtime that owns state/versioning.
pub fn shared_evaluations(
    block_dependencies: &BTreeMap<KernelBlockId, BTreeSet<String>>,
) -> BTreeMap<String, Vec<KernelBlockId>> {
    let mut users: BTreeMap<String, Vec<KernelBlockId>> = BTreeMap::new();
    for (block, dependencies) in block_dependencies {
        for dependency in dependencies {
            users
                .entry(dependency.clone())
                .or_default()
                .push(block.clone());
        }
    }
    users.retain(|_, blocks| blocks.len() > 1);
    users
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_property_has_piecewise_slope() {
        let bundle = PropertyKernelBundle {
            id: "k".into(),
            primal: PropertyKernel::Table1d(LinearTable {
                input: "T".into(),
                points: vec![300.0, 400.0, 500.0],
                values: vec![10.0, 20.0, 50.0],
            }),
            guards: vec![],
            derivative_inputs: BTreeSet::from(["T".into()]),
        };
        let env = BTreeMap::from([("T".into(), 350.0)]);
        assert_eq!(bundle.evaluate(&env).unwrap(), 15.0);
        assert_eq!(bundle.derivative("T", &env).unwrap(), Some(0.1));
    }

    #[test]
    fn cross_block_cse_only_reports_shared_values() {
        let thermal = KernelBlockId("thermal".into());
        let electrical = KernelBlockId("electrical".into());
        let deps = BTreeMap::from([
            (
                thermal.clone(),
                BTreeSet::from(["sigma(T)".into(), "k(T)".into()]),
            ),
            (electrical.clone(), BTreeSet::from(["sigma(T)".into()])),
        ]);
        let shared = shared_evaluations(&deps);
        assert_eq!(shared.get("sigma(T)"), Some(&vec![electrical, thermal]));
        assert!(!shared.contains_key("k(T)"));
    }
}
