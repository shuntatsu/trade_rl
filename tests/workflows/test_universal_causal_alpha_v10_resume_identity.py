from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import trade_rl.workflows.universal_causal_alpha_v10_stage_entry as stage_entry
from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Candidate


def test_v10_resume_rejects_stale_hierarchy_policy_input_digest(monkeypatch) -> None:
    candidate = CausalAlphaV10Candidate.HIERARCHICAL_WAVE
    final_target_digest = "t" * 64
    metric = SimpleNamespace(
        candidate=stage_entry.V8_CANDIDATE_BY_V10[candidate],
        v8_target_path_digest=final_target_digest,
        v8_config_digest="c" * 64,
        v6_metric=SimpleNamespace(
            symbol="BTCUSDT",
            episode_index=8,
            contract_digest="d" * 64,
        ),
        calibration_fit_digest="f" * 64,
        digest="m" * 64,
    )

    class MetricFactory:
        @staticmethod
        def from_payload(_payload: object) -> object:
            return metric

    monkeypatch.setattr(stage_entry, "CausalAlphaV8ReplayMetric", MetricFactory)

    stale_input_digest = "o" * 64
    leaf = {
        "candidate": candidate.value,
        "candidate_input_digest": stale_input_digest,
        "replay": {},
        "replay_digest": metric.digest,
        "target_path_digest": final_target_digest,
        "target": {
            "artifact_digest": final_target_digest,
            "candidate": candidate.value,
            "hierarchy_input_digest": stale_input_digest,
        },
    }

    class Store:
        config_digest = "c" * 64

        def load_leaf(self, _path: Path, *, expected_schema: str) -> dict[str, object]:
            assert expected_schema == "causal_alpha_v10_replay_leaf_v3"
            return leaf

    with pytest.raises(ValueError, match="resumed replay identity drifted"):
        stage_entry._load(
            Store(),
            path=Path("selection/replays/08/BTCUSDT/hierarchical_wave.json"),
            candidate=candidate,
            candidate_input_digest="n" * 64,
            expected_fast_fit_digest="f" * 64,
            expected_target_digest=None,
            symbol="BTCUSDT",
            episode=8,
            contract_digest="d" * 64,
            expected_dual_run_binding_digest=None,
        )
