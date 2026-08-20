//! FC0-FC1 audit boundary for Resolvent variational-form artifacts.
//!
//! Malleus deliberately does not lower `VariationalFormV2` directly into kernels here.
//! FC0-FC1 establish the stable semantic boundary, verify its digests and legality, and
//! inventory the local capabilities later TensorIR/QFunction lowering must provide.

use resolvent::{
    AdjointKindV2, ArtifactEnvelopeV2, ArtifactIdV2, FormExprV2, MeasureV2, ScalarKindV2,
    VariationalFormV2,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

pub const MALLEUS_FORM_AUDIT_V2_SCHEMA: &str = "malleus-variational-form-audit/2";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalKernelRequirementV2 {
    ScalarArithmetic,
    ArgumentLoad,
    CoefficientLoad,
    ConstantLoad,
    ScientificScalarCall,
    Gradient,
    TimeDerivative,
    Dot,
    SesquilinearInner,
    TypedContraction,
    Conjugation,
    Transpose,
    HermitianTranspose,
    TraceRestriction,
    FacetOrInterfaceAccess,
    ComplexArithmetic,
    TensorAxes,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StructuredKernelGenerationV2 {
    Deferred {
        first_stage: String,
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegralKernelAuditV2 {
    pub label: String,
    pub measure: String,
    pub requirements: Vec<LocalKernelRequirementV2>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariationalKernelAuditV2 {
    pub schema: String,
    pub artifact_id: ArtifactIdV2,
    pub semantic_digest: ArtifactIdV2,
    pub scalar_kind: ScalarKindV2,
    pub arity: u16,
    pub integrals: Vec<IntegralKernelAuditV2>,
    pub requirements: Vec<LocalKernelRequirementV2>,
    pub derivative_artifacts: usize,
    pub operator_claims: usize,
    pub structured_kernel_generation: StructuredKernelGenerationV2,
    pub assembly_level_in_form_identity: bool,
}

pub fn audit_variational_form_v2(
    artifact: &ArtifactEnvelopeV2<VariationalFormV2>,
) -> Result<VariationalKernelAuditV2, VariationalAuditError> {
    artifact.verify().map_err(|error| VariationalAuditError::InvalidArtifact {
        resolvent_code: "FORM-ARTIFACT-001".into(),
        detail: error.to_string(),
    })?;
    artifact
        .payload
        .validate()
        .map_err(|error| VariationalAuditError::InvalidArtifact {
            resolvent_code: error
                .diagnostics
                .first()
                .map(|diagnostic| diagnostic.code.clone())
                .unwrap_or_else(|| "FORM-VALIDATION-001".into()),
            detail: error.to_string(),
        })?;

    let semantic_digest = artifact
        .payload
        .semantic_digest()
        .map_err(|error| VariationalAuditError::InvalidArtifact {
            resolvent_code: "FORM-DIGEST-001".into(),
            detail: error.to_string(),
        })?;

    let mut all_requirements = BTreeSet::new();
    if artifact.payload.scalar_kind.is_complex() {
        all_requirements.insert(LocalKernelRequirementV2::ComplexArithmetic);
    }
    if artifact
        .payload
        .spaces
        .iter()
        .any(|space| !space.value_type.axes.is_empty())
    {
        all_requirements.insert(LocalKernelRequirementV2::TensorAxes);
    }

    let mut integrals = Vec::with_capacity(artifact.payload.integrals.len());
    for integral in &artifact.payload.integrals {
        let mut requirements = BTreeSet::new();
        collect_requirements(&integral.integrand, &mut requirements);
        if !matches!(&integral.measure, MeasureV2::Cell { .. }) {
            requirements.insert(LocalKernelRequirementV2::FacetOrInterfaceAccess);
        }
        all_requirements.extend(requirements.iter().cloned());
        integrals.push(IntegralKernelAuditV2 {
            label: integral.label.clone(),
            measure: measure_name(&integral.measure).into(),
            requirements: requirements.into_iter().collect(),
        });
    }
    integrals.sort_by(|left, right| left.label.cmp(&right.label));

    Ok(VariationalKernelAuditV2 {
        schema: MALLEUS_FORM_AUDIT_V2_SCHEMA.into(),
        artifact_id: artifact.artifact_id.clone(),
        semantic_digest,
        scalar_kind: artifact.payload.scalar_kind,
        arity: artifact.payload.arity(),
        integrals,
        requirements: all_requirements.into_iter().collect(),
        derivative_artifacts: artifact.payload.capabilities.derivative_artifacts.len(),
        operator_claims: artifact.payload.capabilities.operator_claims.len(),
        structured_kernel_generation: StructuredKernelGenerationV2::Deferred {
            first_stage: "FC4".into(),
            reason: "FC0-FC1 establish semantic identity and legality; TensorIR/QFunction lowering is introduced later".into(),
        },
        assembly_level_in_form_identity: false,
    })
}

fn measure_name(measure: &MeasureV2) -> &'static str {
    match measure {
        MeasureV2::Cell { .. } => "cell",
        MeasureV2::ExteriorFacet { .. } => "exterior_facet",
        MeasureV2::InteriorFacet { .. } => "interior_facet",
        MeasureV2::Interface { .. } => "interface",
        MeasureV2::Ridge { .. } => "ridge",
        MeasureV2::Vertex { .. } => "vertex",
    }
}

fn collect_requirements(
    expression: &FormExprV2,
    requirements: &mut BTreeSet<LocalKernelRequirementV2>,
) {
    match expression {
        FormExprV2::Literal { .. } => {}
        FormExprV2::Scientific { .. } => {
            requirements.insert(LocalKernelRequirementV2::ScientificScalarCall);
        }
        FormExprV2::Argument(_) => {
            requirements.insert(LocalKernelRequirementV2::ArgumentLoad);
        }
        FormExprV2::Coefficient(_) => {
            requirements.insert(LocalKernelRequirementV2::CoefficientLoad);
        }
        FormExprV2::Constant(_) => {
            requirements.insert(LocalKernelRequirementV2::ConstantLoad);
        }
        FormExprV2::Neg(value) => {
            requirements.insert(LocalKernelRequirementV2::ScalarArithmetic);
            collect_requirements(value, requirements);
        }
        FormExprV2::Add(values) | FormExprV2::Product(values) => {
            requirements.insert(LocalKernelRequirementV2::ScalarArithmetic);
            for value in values {
                collect_requirements(value, requirements);
            }
        }
        FormExprV2::TimeDerivative(value) => {
            requirements.insert(LocalKernelRequirementV2::TimeDerivative);
            collect_requirements(value, requirements);
        }
        FormExprV2::Gradient { value, .. } => {
            requirements.insert(LocalKernelRequirementV2::Gradient);
            collect_requirements(value, requirements);
        }
        FormExprV2::Dot { left, right } => {
            requirements.insert(LocalKernelRequirementV2::Dot);
            collect_requirements(left, requirements);
            collect_requirements(right, requirements);
        }
        FormExprV2::Inner { left, right } => {
            requirements.insert(LocalKernelRequirementV2::SesquilinearInner);
            collect_requirements(left, requirements);
            collect_requirements(right, requirements);
        }
        FormExprV2::Contract { left, right, .. } => {
            requirements.insert(LocalKernelRequirementV2::TypedContraction);
            collect_requirements(left, requirements);
            collect_requirements(right, requirements);
        }
        FormExprV2::Conjugate(value) => {
            requirements.insert(LocalKernelRequirementV2::Conjugation);
            collect_requirements(value, requirements);
        }
        FormExprV2::Adjoint { value, kind, .. } => {
            requirements.insert(match kind {
                AdjointKindV2::Transpose => LocalKernelRequirementV2::Transpose,
                AdjointKindV2::Hermitian => LocalKernelRequirementV2::HermitianTranspose,
            });
            collect_requirements(value, requirements);
        }
        FormExprV2::Trace { value, .. } => {
            requirements.insert(LocalKernelRequirementV2::TraceRestriction);
            collect_requirements(value, requirements);
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum VariationalAuditError {
    #[error("[MAL-FORM-001] invalid Resolvent V2 artifact ({resolvent_code}): {detail}")]
    InvalidArtifact {
        resolvent_code: String,
        detail: String,
    },
}

impl VariationalAuditError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidArtifact { .. } => "MAL-FORM-001",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resolvent::{Digest, adapt_scalar_h1_model_v2, parse_scientific_module};

    const HEAT: &str = r#"
module test.heat;
model Heat {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  property rho = density(T);
  property cp = specific_heat(T);
  property k = thermal_conductivity(T);
  source Q: VolumetricHeatSource;
  equation energy on Omega { rho * cp * dt(T) - div(k * grad(T)) = Q; }
}
"#;

    #[test]
    fn scalar_v2_artifact_is_audited_without_claiming_structured_codegen() {
        let module = parse_scientific_module(HEAT).unwrap();
        let bundle = adapt_scalar_h1_model_v2(&module.models[0]).unwrap();
        let artifact = &bundle.forms[0];
        let audit = audit_variational_form_v2(artifact).unwrap();
        assert_eq!(audit.artifact_id, artifact.artifact_id);
        assert_eq!(
            audit.semantic_digest,
            artifact.payload.semantic_digest().unwrap()
        );
        assert_eq!(audit.arity, 1);
        assert!(audit.requirements.contains(&LocalKernelRequirementV2::Gradient));
        assert!(
            audit
                .requirements
                .contains(&LocalKernelRequirementV2::TimeDerivative)
        );
        assert_eq!(audit.derivative_artifacts, 0);
        assert_eq!(audit.operator_claims, 0);
        assert!(matches!(
            audit.structured_kernel_generation,
            StructuredKernelGenerationV2::Deferred { ref first_stage, .. }
                if first_stage == "FC4"
        ));
        assert!(!audit.assembly_level_in_form_identity);
    }

    #[test]
    fn tampered_artifacts_fail_with_a_stable_malleus_diagnostic() {
        let module = parse_scientific_module(HEAT).unwrap();
        let bundle = adapt_scalar_h1_model_v2(&module.models[0]).unwrap();
        let mut artifact = bundle.forms[0].clone();
        artifact.artifact_id = ArtifactIdV2(Digest::blake3(b"tampered"));
        let error = audit_variational_form_v2(&artifact).unwrap_err();
        assert_eq!(error.code(), "MAL-FORM-001");
        assert!(matches!(
            error,
            VariationalAuditError::InvalidArtifact { ref resolvent_code, .. }
                if resolvent_code == "FORM-ARTIFACT-001"
        ));
    }
}
