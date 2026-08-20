//! FC0-FC1 audit boundary for Resolvent variational-form artifacts.
//!
//! This module deliberately does not lower a `VariationalFormV2` into a local kernel.
//! Structured TensorIR/QFunction lowering starts at FC4. The FC0-FC1 contract verifies
//! the semantic artifact, inventories the local features that future lowering must
//! support, and exposes the legacy scalar-H1 oracle without confusing it with generated
//! structured code.

use resolvent::{
    DerivativeArtifactsV2, Digest, FormExprV2, FormV2Error, MeasureV2, ScalarKindV2,
    VariationalFormArtifactV2,
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
    pub id: String,
    pub measure: String,
    pub requirements: Vec<LocalKernelRequirementV2>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariationalKernelAuditV2 {
    pub schema: String,
    pub artifact_digest: Digest,
    pub semantic_digest: Digest,
    pub scalar_kind: ScalarKindV2,
    pub arity: u16,
    pub integrals: Vec<IntegralKernelAuditV2>,
    pub requirements: Vec<LocalKernelRequirementV2>,
    pub compatibility_oracle_digest: Option<Digest>,
    pub derivative_artifacts: DerivativeArtifactsV2,
    pub operator_claims: usize,
    pub structured_kernel_generation: StructuredKernelGenerationV2,
    pub assembly_level_in_form_identity: bool,
}

pub fn audit_variational_form_v2(
    artifact: &VariationalFormArtifactV2,
) -> Result<VariationalKernelAuditV2, VariationalAuditError> {
    artifact.verify().map_err(VariationalAuditError::from_form)?;

    let compatibility_oracle_digest = match artifact.payload.scalar_h1_compatibility.as_ref() {
        Some(compatibility) => {
            artifact
                .scalar_h1_compatibility_program()
                .map_err(VariationalAuditError::from_form)?;
            Some(compatibility.source_digest.clone())
        }
        None => None,
    };

    let mut all_requirements = BTreeSet::new();
    if artifact.payload.form.scalar_kind.is_complex() {
        all_requirements.insert(LocalKernelRequirementV2::ComplexArithmetic);
    }
    if artifact
        .payload
        .form
        .arguments
        .iter()
        .any(|argument| !argument.value_type.axes.is_empty())
        || artifact
            .payload
            .form
            .coefficients
            .iter()
            .any(|coefficient| !coefficient.value_type.axes.is_empty())
    {
        all_requirements.insert(LocalKernelRequirementV2::TensorAxes);
    }

    let mut integrals = Vec::with_capacity(artifact.payload.form.integrals.len());
    for integral in &artifact.payload.form.integrals {
        let mut requirements = BTreeSet::new();
        collect_requirements(&integral.integrand, &mut requirements);
        if !matches!(&integral.measure, MeasureV2::Cell { .. }) {
            requirements.insert(LocalKernelRequirementV2::FacetOrInterfaceAccess);
        }
        all_requirements.extend(requirements.iter().cloned());
        integrals.push(IntegralKernelAuditV2 {
            id: integral.id.clone(),
            measure: integral.measure.kind_name().into(),
            requirements: requirements.into_iter().collect(),
        });
    }
    integrals.sort_by(|left, right| left.id.cmp(&right.id));

    Ok(VariationalKernelAuditV2 {
        schema: MALLEUS_FORM_AUDIT_V2_SCHEMA.into(),
        artifact_digest: artifact.artifact_digest.clone(),
        semantic_digest: artifact.semantic_digest.clone(),
        scalar_kind: artifact.payload.form.scalar_kind,
        arity: artifact.payload.form.arity(),
        integrals,
        requirements: all_requirements.into_iter().collect(),
        compatibility_oracle_digest,
        derivative_artifacts: artifact.payload.derivatives.clone(),
        operator_claims: artifact.payload.operator_claims.len(),
        structured_kernel_generation: StructuredKernelGenerationV2::Deferred {
            first_stage: "FC4".into(),
            reason: "FC0-FC1 establish semantic identity and legality; TensorIR/QFunction lowering is not yet implemented".into(),
        },
        assembly_level_in_form_identity: false,
    })
}

fn collect_requirements(
    expression: &FormExprV2,
    requirements: &mut BTreeSet<LocalKernelRequirementV2>,
) {
    match expression {
        FormExprV2::ScientificScalar { .. } => {
            requirements.insert(LocalKernelRequirementV2::ScientificScalarCall);
        }
        FormExprV2::Argument { .. } => {
            requirements.insert(LocalKernelRequirementV2::ArgumentLoad);
        }
        FormExprV2::Coefficient { .. } => {
            requirements.insert(LocalKernelRequirementV2::CoefficientLoad);
        }
        FormExprV2::Constant { .. } => {
            requirements.insert(LocalKernelRequirementV2::ConstantLoad);
        }
        FormExprV2::Neg { value } => {
            requirements.insert(LocalKernelRequirementV2::ScalarArithmetic);
            collect_requirements(value, requirements);
        }
        FormExprV2::Add { values } | FormExprV2::Product { values } => {
            requirements.insert(LocalKernelRequirementV2::ScalarArithmetic);
            for value in values {
                collect_requirements(value, requirements);
            }
        }
        FormExprV2::Apply { args, .. } => {
            requirements.insert(LocalKernelRequirementV2::ScientificScalarCall);
            for argument in args {
                collect_requirements(argument, requirements);
            }
        }
        FormExprV2::Gradient { value, .. } => {
            requirements.insert(LocalKernelRequirementV2::Gradient);
            collect_requirements(value, requirements);
        }
        FormExprV2::TimeDerivative { value } => {
            requirements.insert(LocalKernelRequirementV2::TimeDerivative);
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
        FormExprV2::Conjugate { value } => {
            requirements.insert(LocalKernelRequirementV2::Conjugation);
            collect_requirements(value, requirements);
        }
        FormExprV2::Transpose { value } => {
            requirements.insert(LocalKernelRequirementV2::Transpose);
            collect_requirements(value, requirements);
        }
        FormExprV2::HermitianTranspose { value } => {
            requirements.insert(LocalKernelRequirementV2::HermitianTranspose);
            collect_requirements(value, requirements);
        }
        FormExprV2::Restrict { value, .. } => {
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
    fn from_form(error: FormV2Error) -> Self {
        Self::InvalidArtifact {
            resolvent_code: error.code().into(),
            detail: error.to_string(),
        }
    }

    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidArtifact { .. } => "MAL-FORM-001",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resolvent::{DerivativeArtifactStatusV2, adapt_scalar_h1_model_v2, parse_scientific_module};

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
        let artifact = adapt_scalar_h1_model_v2(&module.models[0]).unwrap();
        let audit = audit_variational_form_v2(&artifact).unwrap();
        assert_eq!(audit.artifact_digest, artifact.artifact_digest);
        assert_eq!(audit.semantic_digest, artifact.semantic_digest);
        assert_eq!(audit.arity, 1);
        assert_eq!(
            audit.compatibility_oracle_digest,
            Some(artifact.payload.receipt.source_digest.clone())
        );
        assert!(audit.requirements.contains(&LocalKernelRequirementV2::Gradient));
        assert!(
            audit
                .requirements
                .contains(&LocalKernelRequirementV2::TimeDerivative)
        );
        assert_eq!(
            audit.derivative_artifacts.jvp,
            DerivativeArtifactStatusV2::NotGenerated
        );
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
        let mut artifact = adapt_scalar_h1_model_v2(&module.models[0]).unwrap();
        artifact.artifact_digest = Digest::blake3(b"tampered");
        let error = audit_variational_form_v2(&artifact).unwrap_err();
        assert_eq!(error.code(), "MAL-FORM-001");
        assert!(matches!(
            error,
            VariationalAuditError::InvalidArtifact { ref resolvent_code, .. }
                if resolvent_code == "FORM-DIGEST-001"
        ));
    }
}
