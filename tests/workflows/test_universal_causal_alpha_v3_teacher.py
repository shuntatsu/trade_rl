from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3CandidateConfig,
    CausalAlphaV3CandidateEvidence,
    CausalAlphaV3EpisodeMetric,
    CausalAlphaV3SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher import (
    CausalAlphaV3TeacherAdmissionRejected,
    admit_causal_alpha_v3_teacher,
)


def _selection() -> CausalAlphaV3SelectionEvidence:
    candidate = CausalAlphaV3CandidateConfig(
        name="selected",
        fit=CausalAlphaV3FitConfig(),
        target=CausalAlphaV3TargetConfig(
            target_magnitudes=(0.0, 0.1),
            uncertainty_multiplier=1.0,
            execution_cost_multiplier=1.0,
            edge_margin=0.0,
            alpha_rebalance_decisions=16,
            strong_reversal_threshold=2.0,
            max_target_delta=0.1,
        ),
    )
    metric = CausalAlphaV3EpisodeMetric(
        candidate_digest=candidate.digest,
        symbol="BTCUSDT",
        episode_index=0,
        contract_digest=content_digest("selection-contract"),
        gross_return=0.02,
        net_return=0.01,
        turnover_per_day=0.2,
        total_execution_cost=4.0,
        trade_count=3,
        hard_risk_violation=False,
        unexplained_execution_rejection_count=0,
    )
    evidence = CausalAlphaV3CandidateEvidence.from_episode_metrics(
        candidate=candidate,
        episode_metrics=(metric,),
        admissible=True,
        rejection_reasons=(),
    )
    return CausalAlphaV3SelectionEvidence(
        candidates=(evidence,),
        selected_candidate_digest=candidate.digest,
        grid_digest=content_digest("grid"),
        thresholds_digest=content_digest("thresholds"),
        generator_code_digest=content_digest("generator"),
        sample_scope_digest=content_digest("samples"),
        holdout_episode_digests={"BTCUSDT": _contract().digest},
    )


def _contract() -> OracleEpisodeContract:
    return OracleEpisodeContract(
        dataset_id=content_digest("dataset"),
        episode_index=1,
        start=10,
        stop=14,
        initial_state_mode="cash",
        initial_weights=np.zeros(1),
    )


def _persist_selection(path: Path, selection: CausalAlphaV3SelectionEvidence) -> None:
    atomic_write_bytes(path, canonical_json_bytes(selection.to_payload()) + b"\n")


def test_v3_admission_requires_durable_selection_and_replays_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection()
    selection_path = tmp_path / "selection.json"
    admission_path = tmp_path / "admission.json"
    package_path = tmp_path / "package.json"
    _persist_selection(selection_path, selection)
    calls: list[str] = []

    def replay(factory: object, contract: object, *, actions: object) -> object:
        calls.append("BTCUSDT")
        return SimpleNamespace(
            performance=SimpleNamespace(
                gross_return=0.02,
                net_return=0.01,
                turnover_total=0.2,
                cost_total=4.0,
                trade_count=3,
                maximum_drawdown=0.01,
            )
        )

    monkeypatch.setattr(
        "trade_rl.workflows.universal_causal_alpha_v3_teacher.evaluate_episode_action_path",
        replay,
    )
    package = admit_causal_alpha_v3_teacher(
        selection=selection,
        selection_evidence_path=selection_path,
        holdout_contracts={"BTCUSDT": _contract()},
        holdout_targets={"BTCUSDT": np.zeros((3, 1), dtype=np.float32)},
        environment_factories={"BTCUSDT": object()},
        episode_hours=1.0,
        teacher_config_digest=content_digest("teacher"),
        admission_evidence_path=admission_path,
        package_evidence_path=package_path,
    )

    assert calls == ["BTCUSDT"]
    assert admission_path.is_file()
    assert package_path.is_file()
    assert package.teacher_admission.admission.passed


def test_v3_failed_admission_is_persisted_without_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection()
    selection_path = tmp_path / "selection.json"
    admission_path = tmp_path / "admission.json"
    package_path = tmp_path / "package.json"
    _persist_selection(selection_path, selection)

    monkeypatch.setattr(
        "trade_rl.workflows.universal_causal_alpha_v3_teacher.evaluate_episode_action_path",
        lambda *args, **kwargs: SimpleNamespace(
            performance=SimpleNamespace(
                gross_return=-0.02,
                net_return=-0.03,
                turnover_total=0.2,
                cost_total=4.0,
                trade_count=3,
                maximum_drawdown=0.04,
            )
        ),
    )

    with pytest.raises(CausalAlphaV3TeacherAdmissionRejected):
        admit_causal_alpha_v3_teacher(
            selection=selection,
            selection_evidence_path=selection_path,
            holdout_contracts={"BTCUSDT": _contract()},
            holdout_targets={"BTCUSDT": np.zeros((3, 1), dtype=np.float32)},
            environment_factories={"BTCUSDT": object()},
            episode_hours=1.0,
            teacher_config_digest=content_digest("teacher"),
            admission_evidence_path=admission_path,
            package_evidence_path=package_path,
        )

    assert admission_path.is_file()
    assert not package_path.exists()


def test_v3_admission_refuses_missing_selection_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selection evidence is not durable"):
        admit_causal_alpha_v3_teacher(
            selection=_selection(),
            selection_evidence_path=tmp_path / "missing.json",
            holdout_contracts={"BTCUSDT": _contract()},
            holdout_targets={"BTCUSDT": np.zeros((3, 1), dtype=np.float32)},
            environment_factories={"BTCUSDT": object()},
            episode_hours=1.0,
            teacher_config_digest=content_digest("teacher"),
            admission_evidence_path=tmp_path / "admission.json",
            package_evidence_path=tmp_path / "package.json",
        )
