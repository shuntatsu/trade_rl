"""Thin facade for the hardened artifact-bound causal alpha V3 workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_oracle_bc import (
    evaluate_episode_action_path,
    evaluate_episode_action_path_on_environment,
)
from trade_rl.workflows.universal_causal_alpha_v3_admission import (
    CausalAlphaV3AdmissionEvidenceV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3ResearchConfig
from trade_rl.workflows.universal_causal_alpha_v3_pipeline import (
    CausalAlphaV3AdmissionRejected,
    CausalAlphaV3SignalRejected,
    run_universal_causal_alpha_v3_research_pipeline,
)
from trade_rl.workflows.universal_causal_alpha_v3_replay import (
    evaluate_causal_alpha_v3_admission as _evaluate_admission,
    evaluate_causal_alpha_v3_selection as _evaluate_selection,
)
from trade_rl.workflows.universal_causal_alpha_v3_runtime import (
    CausalAlphaV3PreparedResearchData,
    causal_alpha_v3_source_tree_digest,
    prepare_causal_alpha_v3_research_data,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_v2 import (
    evaluate_causal_alpha_v3_signal_gate_clustered,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher import (
    build_causal_alpha_v3_contract_targets,
    build_causal_alpha_v3_episode_batch,
    build_causal_alpha_v3_signal_scope_metric,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher_artifacts import (
    UniversalCausalAlphaV3TeacherPackageV2,
)

evaluate_causal_alpha_v3_signal_gate = evaluate_causal_alpha_v3_signal_gate_clustered


def causal_alpha_v3_generator_code_digest() -> str:
    return content_digest(
        {
            "schema_version": "universal_causal_alpha_v3_generator_code_v2",
            "source_tree_digest": causal_alpha_v3_source_tree_digest(),
        }
    )


def evaluate_causal_alpha_v3_selection(**kwargs: Any) -> Any:
    return _evaluate_selection(
        **kwargs,
        build_targets=build_causal_alpha_v3_contract_targets,
        evaluate_path=evaluate_episode_action_path_on_environment,
    )


def evaluate_causal_alpha_v3_admission(
    **kwargs: Any,
) -> CausalAlphaV3AdmissionEvidenceV2:
    return _evaluate_admission(
        **kwargs,
        evaluate_path=evaluate_episode_action_path,
    )


def run_universal_causal_alpha_v3_research(
    *,
    config: CausalAlphaV3ResearchConfig,
    prepared: CausalAlphaV3PreparedResearchData,
    output_root: Path,
) -> UniversalCausalAlphaV3TeacherPackageV2:
    return run_universal_causal_alpha_v3_research_pipeline(
        config=config,
        prepared=prepared,
        output_root=output_root,
        signal_scope_builder=build_causal_alpha_v3_signal_scope_metric,
        signal_gate_evaluator=evaluate_causal_alpha_v3_signal_gate,
        selection_evaluator=evaluate_causal_alpha_v3_selection,
        episode_batch_builder=build_causal_alpha_v3_episode_batch,
        admission_evaluator=evaluate_causal_alpha_v3_admission,
    )


__all__ = [
    "CausalAlphaV3AdmissionRejected",
    "CausalAlphaV3PreparedResearchData",
    "CausalAlphaV3SignalRejected",
    "causal_alpha_v3_generator_code_digest",
    "evaluate_causal_alpha_v3_admission",
    "evaluate_causal_alpha_v3_selection",
    "prepare_causal_alpha_v3_research_data",
    "run_universal_causal_alpha_v3_research",
]
