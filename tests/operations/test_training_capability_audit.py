from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from trade_rl.artifacts.hashing import content_digest
from trade_rl.operations import _training_capability_audit_impl as impl
from trade_rl.operations.training_capability_audit import run_training_capability_audit


class _TrainingResult:
    pass


def test_run_training_capability_audit_preserves_report_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "audit"
    root.mkdir()
    (root / "stale.txt").write_text("stale", encoding="utf-8")

    results = {name: _TrainingResult() for name in ("ppo", "sac", "td3", "tqc")}

    def train_algorithm(output_root: Path, algorithm: str):
        assert output_root == root
        return {"algorithm": algorithm, "status": "pass"}, results[algorithm]

    monkeypatch.setattr(impl, "_train_algorithm", train_algorithm)
    monkeypatch.setattr(
        impl, "_behavior_cloning_training", lambda _: {"status": "pass"}
    )
    monkeypatch.setattr(impl, "_export_ppo", lambda _: {"status": "pass"})
    monkeypatch.setattr(
        impl,
        "_resume_replay",
        lambda output_root, source: {
            "source_matches_sac": source is results["sac"],
            "status": "pass",
        },
    )
    monkeypatch.setattr(
        impl, "_residual_feature_training", lambda _: {"status": "pass"}
    )
    monkeypatch.setattr(impl, "_sequence_training", lambda _: {"status": "pass"})
    monkeypatch.setattr(impl, "_resume_ppo", lambda _: {"status": "pass"})

    report = run_training_capability_audit(root)

    assert not (root / "stale.txt").exists()
    assert report["schema_version"] == "full_training_capability_audit_v1"
    algorithms = cast(dict[str, object], report["algorithms"])
    replay_resume = cast(dict[str, object], report["replay_resume"])
    assert set(algorithms) == {"ppo", "sac", "td3", "tqc"}
    assert replay_resume["source_matches_sac"] is True

    unsigned = dict(report)
    digest = unsigned.pop("digest")
    assert digest == content_digest(unsigned)

    expected_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    assert (root / "audit-report.json").read_bytes() == expected_bytes
    assert json.loads(expected_bytes) == report


def _sequence_failure_diagnostics(root: Path) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    gate_path = root / "behavior-cloning-gates.json"
    if gate_path.is_file():
        gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
        causal_group = gate_payload.get("causal_non_collapse_gate", {})
        diagnostics["causal_gate_metrics"] = {
            metric["name"]: {
                "minimum_support": metric.get("minimum_support"),
                "observed": metric.get("observed"),
                "status": metric.get("status"),
                "support": metric.get("support"),
                "threshold": metric.get("threshold"),
            }
            for metric in causal_group.get("metrics", ())
        }
    holdout_path = root / "behavior-cloning-holdout.json"
    if holdout_path.is_file():
        holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        records = holdout.get("records", ())
        diagnostics["holdout"] = {
            "causal_net_return_lower_confidence_bound": holdout.get(
                "causal_net_return_lower_confidence_bound"
            ),
            "causal_regret_upper_confidence_bound": holdout.get(
                "causal_regret_upper_confidence_bound"
            ),
            "episode_count": holdout.get("episode_count"),
            "normalized_oracle_regret": holdout.get("normalized_oracle_regret"),
            "policy_net_returns": [
                record["causal_policy_performance"]["net_return"] for record in records
            ],
            "oracle_net_returns": [
                record["oracle_performance"]["net_return"] for record in records
            ],
        }
    return diagnostics


def test_sequence_training_exercises_real_hierarchical_behavior_cloning(
    tmp_path: Path,
) -> None:
    try:
        record = impl._sequence_training(tmp_path)
    except RuntimeError as exc:
        diagnostics = _sequence_failure_diagnostics(tmp_path / "structured-sequence")
        raise AssertionError(
            f"sequence capability audit failed: {exc}; "
            f"diagnostics={json.dumps(diagnostics, sort_keys=True)}"
        ) from exc

    assert record["status"] == "pass"
    assert record["observation_encoder"] == "hierarchical_sequence_v2"
    behavior_cloning = cast(dict[str, object], record["behavior_cloning"])
    assert int(behavior_cloning["sample_count"]) > 0
