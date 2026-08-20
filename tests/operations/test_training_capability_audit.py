from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

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


def _gate_metrics(group: object) -> dict[str, object]:
    if not isinstance(group, dict):
        return {}
    return {
        metric["name"]: {
            "minimum_support": metric.get("minimum_support"),
            "observed": metric.get("observed"),
            "status": metric.get("status"),
            "support": metric.get("support"),
            "threshold": metric.get("threshold"),
        }
        for metric in group.get("metrics", ())
    }


def _sequence_failure_diagnostics(root: Path) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    behavior_cloning_path = root / "behavior-cloning.json"
    if behavior_cloning_path.is_file():
        behavior_cloning = json.loads(behavior_cloning_path.read_text(encoding="utf-8"))
        diagnostics["behavior_cloning"] = {
            field: behavior_cloning.get(field)
            for field in (
                "best_epoch",
                "final_mse",
                "initial_mse",
                "quality_passed",
                "relative_improvement",
                "required_relative_improvement",
                "sample_count",
                "validation_mse",
                "validation_sample_count",
            )
        }
    gate_path = root / "behavior-cloning-gates.json"
    if gate_path.is_file():
        gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
        diagnostics["teacher_gate_metrics"] = _gate_metrics(
            gate_payload.get("teacher_reconstruction_gate")
        )
        diagnostics["causal_gate_metrics"] = _gate_metrics(
            gate_payload.get("causal_non_collapse_gate")
        )
    holdout_path = root / "behavior-cloning-holdout.json"
    if holdout_path.is_file():
        holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        records = holdout.get("records", ())
        action_fields = (
            "direction_agreement_rate",
            "pearson_correlation",
            "teacher_mean",
            "teacher_std",
            "teacher_positive_rate",
            "teacher_negative_rate",
            "teacher_change_count",
            "policy_mean",
            "policy_std",
            "policy_positive_rate",
            "policy_negative_rate",
            "policy_change_count",
        )
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
            "action_diagnostics": [
                {
                    field: record.get("action_diagnostics", {}).get(field)
                    for field in action_fields
                }
                for record in records
            ],
        }
    return diagnostics


def test_sequence_training_densifies_audit_episodes_without_changing_time_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: dict[str, float | int | None] = {}

    class _Backend:
        def __init__(self, factory) -> None:
            self.factory = factory

        def train(self, *, seed, config, output_path):
            del seed, config, output_path
            self.factory()
            raise RuntimeError("captured sequence environment config")

    def capture_environment(dataset, *, trend_strategy, action_spec, config):
        del dataset, trend_strategy, action_spec
        observed["episode_bars"] = config.episode_bars
        observed["episode_hours"] = config.episode_hours
        observed["decision_hours"] = config.decision_hours
        return object()

    monkeypatch.setattr(impl, "ResidualMarketEnv", capture_environment)
    monkeypatch.setattr(impl, "StableBaselines3Backend", _Backend)

    with pytest.raises(RuntimeError, match="captured sequence environment config"):
        impl._sequence_training(tmp_path)

    assert observed["episode_bars"] == 2
    assert observed["episode_hours"] == pytest.approx(2.0)
    assert observed["decision_hours"] == pytest.approx(0.25)


def test_sequence_training_uses_maintained_causal_regret_admission_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: dict[str, float] = {}

    def capture_train(self, *, seed, config, output_path):
        del self, seed, output_path
        observed["max_causal_holdout_regret"] = (
            config.behavior_cloning_max_causal_holdout_regret
        )
        raise RuntimeError("captured sequence training config")

    monkeypatch.setattr(impl.StableBaselines3Backend, "train", capture_train)

    with pytest.raises(RuntimeError, match="captured sequence training config"):
        impl._sequence_training(tmp_path)

    assert observed["max_causal_holdout_regret"] == pytest.approx(0.2)


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

    gate_payload = json.loads(
        (tmp_path / "structured-sequence" / "behavior-cloning-gates.json").read_text(
            encoding="utf-8"
        )
    )
    teacher_metrics = _gate_metrics(gate_payload.get("teacher_reconstruction_gate"))
    causal_metrics = _gate_metrics(gate_payload.get("causal_non_collapse_gate"))
    assert teacher_metrics["active_target_rmse"]["status"] == "passed"
    assert causal_metrics["cash_baseline_after_cost_regret"]["status"] == "passed"
    assert causal_metrics["causal_regret_upper_confidence_bound"]["status"] == "passed"
    assert causal_metrics["causal_regret_upper_confidence_bound"][
        "threshold"
    ] == pytest.approx(0.2)


def test_sequence_training_keeps_bc_epochs_above_historical_all_hold_collapse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: dict[str, int] = {}

    def capture_train(self, *, seed, config, output_path):
        del self, seed, output_path
        observed["behavior_cloning_epochs"] = config.behavior_cloning_epochs
        raise RuntimeError("captured sequence training config")

    monkeypatch.setattr(impl.StableBaselines3Backend, "train", capture_train)

    with pytest.raises(RuntimeError, match="captured sequence training config"):
        impl._sequence_training(tmp_path)

    # One epoch collapsed this audit probe to all-HOLD with zero positive support.
    assert observed["behavior_cloning_epochs"] == 45
