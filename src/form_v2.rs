//! FC0-FC1 verification boundary for Resolvent variational artifacts.
//!
//! This module deliberately does not compile a variational form directly. Resolvent owns
//! variational meaning; later compiler stages must first produce TensorIR/QFunctionIR. FC0-FC1
//! only verify and inventory that boundary so downstream code cannot mistake an inspectable form
//! for an executable local kernel.

use resolvent::{
    AdjointKindV2, ArtifactEnvelopeV2, ArtifactIdV2, ArtifactStageV2, ConjugatedOperandV2,
    FormExprV2, INNER_CONJUGATED_OPERAND_V2, MeasureV2, ScalarKindV2, TensorTypeV2,
    VARIATIONAL_FORM_V2_SCHEMA, VariationalFormV2,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

pub const MALLEUS_FORM_AUDIT_V2_SCHEMA: &str = "malleus-form-audit/2";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalKernelRequirementV2 {
    RealScalar,
    ComplexScalar,
    TensorAxes,
    Gradient,
    TimeDerivative,
    BilinearDot,
    SesquilinearInner,
    TensorContract,
    Conjugation,
    Transpose,
    HermitianAdjoint,
    TraceSides,
    CellIntegral,
    ExteriorFacetIntegral,
    InteriorFacetIntegral,
    InterfaceIntegral,
    RidgeIntegral,
    VertexIntegral,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegralKernelAuditV2 {
    pub label: String,
    pub requirements: Vec<LocalKernelRequirementV2>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StructuredKernelGenerationV2 {
    Deferred { first_stage: String, reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariationalKernelAuditV2 {
    pub schema: &'static str,
    pub form_artifact: ArtifactIdV2,
    pub semantic_digest: ArtifactIdV2,
    pub scalar_kind: ScalarKindV2,
    pub arity: u16,
    pub integral_count: usize,
    pub requirements: Vec<LocalKernelRequirementV2>,
    pub integrals: Vec<IntegralKernelAuditV2>,
    pub inner_conjugated_operand: ConjugatedOperandV2,
    pub derivative_artifact_count: usize,
    pub operator_claim_count: usize,
    pub assembly_level_in_form_identity: bool,
    pub structured_kernel_generation: StructuredKernelGenerationV2,
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("Resolvent V2 form rejected at `{path}` [{resolvent_code}]: {message}")]
pub struct VariationalAuditError {
    pub resolvent_code: String,
    pub path: String,
    pub message: String,
}

pub fn audit_variational_form_v2(
    artifact: &ArtifactEnvelopeV2<VariationalFormV2>,
) -> Result<VariationalKernelAuditV2, VariationalAuditError> {
    artifact.verify().map_err(|error| VariationalAuditError {
        resolvent_code: "FORM-V2-ARTIFACT-INTEGRITY".into(),
        path: "artifact".into(),
        message: error.to_string(),
    })?;

    if artifact.stage != ArtifactStageV2::VariationalForm
        || artifact.payload_schema != VARIATIONAL_FORM_V2_SCHEMA
        || artifact.payload.schema != VARIATIONAL_FORM_V2_SCHEMA
    {
        return Err(VariationalAuditError {
            resolvent_code: "FORM-V2-ARTIFACT-KIND".into(),
            path: "artifact".into(),
            message: format!(
                "expected stage {:?} and payload schema `{}`, got stage {:?}, envelope schema `{}`, payload schema `{}`",
                ArtifactStageV2::VariationalForm,
                VARIATIONAL_FORM_V2_SCHEMA,
                artifact.stage,
                artifact.payload_schema,
                artifact.payload.schema
            ),
        });
    }

    artifact
        .payload
        .validate()
        .map_err(|error| VariationalAuditError {
            resolvent_code: error
                .diagnostics
                .first()
                .map_or_else(|| "FORM-V2-INVALID".into(), |value| value.code.clone()),
            path: error
                .diagnostics
                .first()
                .map_or_else(|| "form".into(), |value| value.path.clone()),
            message: error.to_string(),
        })?;

    let semantic_digest =
        artifact
            .payload
            .semantic_digest()
            .map_err(|error| VariationalAuditError {
                resolvent_code: "FORM-V2-DIGEST".into(),
                path: "form".into(),
                message: error.to_string(),
            })?;

    let mut requirements = BTreeSet::new();
    requirements.insert(match artifact.payload.scalar_kind {
        ScalarKindV2::Real32 | ScalarKindV2::Real64 => LocalKernelRequirementV2::RealScalar,
        ScalarKindV2::Complex32 | ScalarKindV2::Complex64 => {
            LocalKernelRequirementV2::ComplexScalar
        }
    });
    if artifact
        .payload
        .spaces
        .iter()
        .any(|space| has_tensor_axes(&space.value_type))
        || artifact
            .payload
            .coefficients
            .iter()
            .any(|coefficient| has_tensor_axes(&coefficient.value_type))
        || artifact
            .payload
            .constants
            .iter()
            .any(|constant| has_tensor_axes(&constant.value_type))
    {
        requirements.insert(LocalKernelRequirementV2::TensorAxes);
    }

    let mut integrals = Vec::with_capacity(artifact.payload.integrals.len());
    for integral in &artifact.payload.integrals {
        let mut local = BTreeSet::new();
        collect_requirements(&integral.integrand, &mut local);
        local.insert(match integral.measure {
            MeasureV2::Cell { .. } => LocalKernelRequirementV2::CellIntegral,
            MeasureV2::ExteriorFacet { .. } => LocalKernelRequirementV2::ExteriorFacetIntegral,
            MeasureV2::InteriorFacet { .. } => LocalKernelRequirementV2::InteriorFacetIntegral,
            MeasureV2::Interface { .. } => LocalKernelRequirementV2::InterfaceIntegral,
            MeasureV2::Ridge { .. } => LocalKernelRequirementV2::RidgeIntegral,
            MeasureV2::Vertex { .. } => LocalKernelRequirementV2::VertexIntegral,
        });
        requirements.extend(local.iter().copied());
        integrals.push(IntegralKernelAuditV2 {
            label: integral.label.clone(),
            requirements: local.into_iter().collect(),
        });
    }

    Ok(VariationalKernelAuditV2 {
        schema: MALLEUS_FORM_AUDIT_V2_SCHEMA,
        form_artifact: artifact.artifact_id,
        semantic_digest,
        scalar_kind: artifact.payload.scalar_kind,
        arity: artifact.payload.arity(),
        integral_count: artifact.payload.integrals.len(),
        requirements: requirements.into_iter().collect(),
        integrals,
        inner_conjugated_operand: INNER_CONJUGATED_OPERAND_V2,
        derivative_artifact_count: artifact.payload.capabilities.derivative_artifacts.len(),
        operator_claim_count: artifact.payload.capabilities.operator_claims.len(),
        assembly_level_in_form_identity: false,
        structured_kernel_generation: StructuredKernelGenerationV2::Deferred {
            first_stage: "FC6".into(),
            reason: "Malleus consumes structured TensorIR/QFunctionIR after FC4-FC5 preprocessing; FC0-FC1 only verify and inventory VariationalFormV2".into(),
        },
    })
}

fn has_tensor_axes(value_type: &TensorTypeV2) -> bool {
    !value_type.axes.is_empty()
}

fn collect_requirements(
    expression: &FormExprV2,
    requirements: &mut BTreeSet<LocalKernelRequirementV2>,
) {
    match expression {
        FormExprV2::Literal { value_type, .. } | FormExprV2::Scientific { value_type, .. } => {
            if has_tensor_axes(value_type) {
                requirements.insert(LocalKernelRequirementV2::TensorAxes);
            }
        }
        FormExprV2::Argument(_) | FormExprV2::Coefficient(_) | FormExprV2::Constant(_) => {}
        FormExprV2::Neg(value) => collect_requirements(value, requirements),
        FormExprV2::Add(values) | FormExprV2::Product(values) => {
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
            requirements.insert(LocalKernelRequirementV2::TensorAxes);
            collect_requirements(value, requirements);
        }
        FormExprV2::Dot { left, right } => {
            requirements.insert(LocalKernelRequirementV2::BilinearDot);
            requirements.insert(LocalKernelRequirementV2::TensorAxes);
            collect_requirements(left, requirements);
            collect_requirements(right, requirements);
        }
        FormExprV2::Inner { left, right } => {
            requirements.insert(LocalKernelRequirementV2::SesquilinearInner);
            requirements.insert(LocalKernelRequirementV2::TensorAxes);
            collect_requirements(left, requirements);
            collect_requirements(right, requirements);
        }
        FormExprV2::Contract { left, right, .. } => {
            requirements.insert(LocalKernelRequirementV2::TensorContract);
            requirements.insert(LocalKernelRequirementV2::TensorAxes);
            collect_requirements(left, requirements);
            collect_requirements(right, requirements);
        }
        FormExprV2::Conjugate(value) => {
            requirements.insert(LocalKernelRequirementV2::Conjugation);
            collect_requirements(value, requirements);
        }
        FormExprV2::Adjoint { value, kind, .. } => {
            requirements.insert(LocalKernelRequirementV2::TensorAxes);
            requirements.insert(match kind {
                AdjointKindV2::Transpose => LocalKernelRequirementV2::Transpose,
                AdjointKindV2::Hermitian => LocalKernelRequirementV2::HermitianAdjoint,
            });
            collect_requirements(value, requirements);
        }
        FormExprV2::Trace { value, .. } => {
            requirements.insert(LocalKernelRequirementV2::TraceSides);
            collect_requirements(value, requirements);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resolvent::{
        AxisKindV2, FormConstantIdV2, FormConstantV2, FrameIdV2, VarianceV2,
        adapt_scalar_h1_model_v2, parse_scientific_module,
    };

    fn heat_artifact() -> ArtifactEnvelopeV2<VariationalFormV2> {
        let source = r#"
module test.malleus;
model Heat {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  property rho = density(T);
  property cp = heat_capacity(T);
  property k = conductivity(T);
  source Q: Source;
  equation energy on Omega { rho * cp * dt(T) - div(k * grad(T)) = Q; }
}
"#;
        let model = parse_scientific_module(source).unwrap().models.remove(0);
        adapt_scalar_h1_model_v2(&model).unwrap().forms.remove(0)
    }

    #[test]
    fn audits_scalar_form_without_claiming_a_kernel() {
        let artifact = heat_artifact();
        let audit = audit_variational_form_v2(&artifact).unwrap();
        assert_eq!(audit.schema, MALLEUS_FORM_AUDIT_V2_SCHEMA);
        assert_eq!(audit.form_artifact, artifact.artifact_id);
        assert_eq!(audit.arity, 1);
        assert!(
            audit
                .requirements
                .contains(&LocalKernelRequirementV2::Gradient)
        );
        assert!(
            audit
                .requirements
                .contains(&LocalKernelRequirementV2::TensorAxes)
        );
        assert_eq!(audit.inner_conjugated_operand, ConjugatedOperandV2::Right);
        assert_eq!(audit.derivative_artifact_count, 0);
        assert_eq!(audit.operator_claim_count, 0);
        assert!(!audit.assembly_level_in_form_identity);
        assert!(matches!(
            audit.structured_kernel_generation,
            StructuredKernelGenerationV2::Deferred { ref first_stage, .. }
                if first_stage == "FC6"
        ));
    }

    #[test]
    fn rejects_tampered_resolvent_artifact_with_stable_code() {
        let mut artifact = heat_artifact();
        artifact.artifact_id = ArtifactIdV2(resolvent::Digest::blake3(b"tampered"));
        let error = audit_variational_form_v2(&artifact).unwrap_err();
        assert_eq!(error.resolvent_code, "FORM-V2-ARTIFACT-INTEGRITY");
        assert_eq!(error.path, "artifact");
    }

    #[test]
    fn rejects_wrong_stage_and_payload_schema_even_when_rehashed() {
        let artifact = heat_artifact();
        let wrong_stage = ArtifactEnvelopeV2::new(
            VARIATIONAL_FORM_V2_SCHEMA,
            ArtifactStageV2::Executable,
            artifact.payload.clone(),
        )
        .unwrap();
        let error = audit_variational_form_v2(&wrong_stage).unwrap_err();
        assert_eq!(error.resolvent_code, "FORM-V2-ARTIFACT-KIND");

        let wrong_schema = ArtifactEnvelopeV2::new(
            "resolvent-variational-form/999",
            ArtifactStageV2::VariationalForm,
            artifact.payload,
        )
        .unwrap();
        let error = audit_variational_form_v2(&wrong_schema).unwrap_err();
        assert_eq!(error.resolvent_code, "FORM-V2-ARTIFACT-KIND");
    }

    #[test]
    fn inventories_tensor_axes_declared_by_constants() {
        let artifact = heat_artifact();
        let mut payload = artifact.payload;
        payload.constants.push(FormConstantV2 {
            id: FormConstantIdV2(99),
            name: "direction".into(),
            value_type: TensorTypeV2::vector(
                ScalarKindV2::Real64,
                AxisKindV2::Spatial {
                    frame: FrameIdV2::new("Omega::spatial"),
                    variance: VarianceV2::Contravariant,
                    extent: 2,
                },
            ),
        });
        let artifact = ArtifactEnvelopeV2::new(
            VARIATIONAL_FORM_V2_SCHEMA,
            ArtifactStageV2::VariationalForm,
            payload,
        )
        .unwrap();
        let audit = audit_variational_form_v2(&artifact).unwrap();
        assert!(
            audit
                .requirements
                .contains(&LocalKernelRequirementV2::TensorAxes)
        );
    }
}
