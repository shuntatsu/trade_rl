"""Exact-once holdout admission for the research-only Causal Alpha V3."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaTeacherHoldoutMetric,
    evaluate_causal_alpha_teacher_admission,
)
from trade_rl.learning.episode_oracle_bc import evaluate_episode_action_path
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3SelectionEvidence,
    CausalAlphaV3TeacherAdmissionEvidence,
    UniversalCausalAlphaV3TeacherPackage,
)


def _persist(path: Path, payload: Mapping[str, object]) -> None:
    atomic_write_bytes(Path(path), canonical_json_bytes(payload) + b"\n")


def _require_durable_selection(
    path: Path, selection: CausalAlphaV3SelectionEvidence
) -> None:
    source = Path(path)
    if not source.is_file():
        raise ValueError("V3 selection evidence is not durable")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("V3 selection evidence is not durable") from error
    if raw.get("artifact_digest") != selection.digest:
        raise ValueError("V3 durable selection digest mismatch")


def _target_digest(symbol: str, values: np.ndarray) -> str:
    return content_and_arrays_digest(
        {"schema_version": "causal_alpha_v3_target_weights_v1", "symbol": symbol},
        (("target_weights", values),),
    )


class CausalAlphaV3TeacherAdmissionRejected(RuntimeError):
    def __init__(self, evidence: CausalAlphaV3TeacherAdmissionEvidence) -> None:
        self.evidence = evidence
        super().__init__("Causal Alpha V3 teacher admission failed")


def admit_causal_alpha_v3_teacher(
    *,
    selection: CausalAlphaV3SelectionEvidence,
    selection_evidence_path: Path,
    holdout_contracts: Mapping[str, OracleEpisodeContract],
    holdout_targets: Mapping[str, np.ndarray],
    environment_factories: Mapping[str, Any],
    episode_hours: float,
    teacher_config_digest: str,
    admission_evidence_path: Path,
    package_evidence_path: Path,
) -> UniversalCausalAlphaV3TeacherPackage:
    """Replay each frozen holdout once, persist admission, then build a package."""

    _require_durable_selection(selection_evidence_path, selection)
    symbols = tuple(selection.holdout_episode_digests)
    if (
        not symbols
        or set(holdout_contracts) != set(symbols)
        or set(holdout_targets) != set(symbols)
        or set(environment_factories) != set(symbols)
    ):
        raise ValueError("V3 admission scope must match frozen holdouts")
    if not math.isfinite(episode_hours) or episode_hours <= 0.0:
        raise ValueError("V3 admission episode_hours must be positive")
    if not isinstance(teacher_config_digest, str) or len(teacher_config_digest) != 64:
        raise ValueError("V3 teacher config digest is invalid")
    episode_days = episode_hours / 24.0
    metrics: list[CausalAlphaTeacherHoldoutMetric] = []
    frozen_targets: dict[str, np.ndarray] = {}
    target_digests: dict[str, str] = {}
    for symbol in symbols:
        contract = holdout_contracts[symbol]
        if contract.digest != selection.holdout_episode_digests[symbol]:
            raise ValueError("V3 admission holdout contract identity drifted")
        targets = np.asarray(holdout_targets[symbol], dtype=np.float32)
        if targets.ndim != 2 or targets.shape != (
            contract.stop - contract.start - 1,
            1,
        ):
            raise ValueError("V3 admission target path is not contract aligned")
        evaluation = evaluate_episode_action_path(
            environment_factories[symbol], contract, actions=targets
        )
        performance = evaluation.performance
        metrics.append(
            CausalAlphaTeacherHoldoutMetric(
                symbol=symbol,
                gross_return=float(performance.gross_return),
                net_return=float(performance.net_return),
                turnover_per_day=float(performance.turnover_total) / episode_days,
                total_execution_cost=float(performance.cost_total),
                trade_count=int(performance.trade_count),
                maximum_drawdown=float(performance.maximum_drawdown),
            )
        )
        vector = targets.reshape(-1).copy()
        frozen_targets[symbol] = vector
        target_digests[symbol] = _target_digest(symbol, vector)
    admission = CausalAlphaV3TeacherAdmissionEvidence(
        selection_digest=selection.digest,
        selected_candidate_digest=selection.selected_candidate_digest,
        holdout_episode_digests=selection.holdout_episode_digests,
        admission=evaluate_causal_alpha_teacher_admission(tuple(metrics)),
    )
    _persist(admission_evidence_path, admission.to_payload())
    if not admission.admission.passed:
        raise CausalAlphaV3TeacherAdmissionRejected(admission)
    package = UniversalCausalAlphaV3TeacherPackage(
        selection=selection,
        teacher_admission=admission,
        target_weights=frozen_targets,
        target_digests=target_digests,
        teacher_config_digest=teacher_config_digest,
    )
    _persist(package_evidence_path, package.to_payload())
    return package


__all__ = [
    "CausalAlphaV3TeacherAdmissionRejected",
    "admit_causal_alpha_v3_teacher",
]
