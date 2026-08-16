"""Public hardened artifact contracts for the research-only causal alpha V3 lane."""

from trade_rl.workflows.universal_causal_alpha_v3_admission import (
    CausalAlphaV3AdmissionEvidenceV3,
    CausalAlphaV3AdmissionRecordV2,
    evaluate_causal_alpha_v3_admission_gate,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3ExecutionIdentity,
    CausalAlphaV3RunManifestV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher_artifacts import (
    CausalAlphaV3TeacherBatchArtifact,
    UniversalCausalAlphaV3TeacherPackageV2,
)

__all__ = [
    "CausalAlphaV3AdmissionEvidenceV3",
    "CausalAlphaV3AdmissionRecordV2",
    "CausalAlphaV3ExecutionIdentity",
    "CausalAlphaV3RunManifestV2",
    "CausalAlphaV3TeacherBatchArtifact",
    "UniversalCausalAlphaV3TeacherPackageV2",
    "evaluate_causal_alpha_v3_admission_gate",
]
