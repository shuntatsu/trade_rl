from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from trade_rl.rl.checkpointing import publish_checkpoint
from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig
from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"


def test_full_training_config_is_not_a_smoke_run() -> None:
    config = TrainingRunConfig.from_json(EXAMPLE_ROOT / "training-full.json")

    assert config.training.algorithm == "ppo"
    assert (
        config.training.device,
        config.training.n_envs,
        config.training.policy_net_arch,
        config.training.asset_embedding_dim,
        config.training.global_embedding_dim,
    ) == ("cuda", 4, (256, 256), 128, 128)
    assert config.training.seeds == (0, 1, 2)
    assert config.training.timesteps >= 262_144
    assert config.training.n_steps == 2_048
    assert config.training.batch_size == 64
    assert config.training.n_epochs == 10
    assert config.environment.decision_hours == 1.0
    assert config.environment.episode_hours >= 720.0
    assert not config.action.risk_tilt_enabled
    assert config.action.n_factors == 3
    assert config.factor_artifact == EXAMPLE_ROOT / "relative-factor-artifact"
    execution = config.environment.execution_cost
    assert execution.fee_rate > 0.0
    assert execution.spread_rate > 0.0
    assert execution.impact_rate > 0.0
    assert config.portfolio_risk.max_abs_weight is not None
    assert config.portfolio_risk.max_abs_weight <= 0.5


def test_full_walk_forward_config_has_two_material_folds() -> None:
    config = MarketWalkForwardConfig.from_json(
        EXAMPLE_ROOT / "walk-forward-full.json",
        n_bars=13_128,
    )

    folds = config.workflow.build_folds()
    assert len(folds) == 2
    assert len(config.candidates) == 1
    candidate = config.candidates[0].run
    assert (
        candidate.training.device,
        candidate.training.n_envs,
        candidate.training.policy_net_arch,
        candidate.training.asset_embedding_dim,
        candidate.training.global_embedding_dim,
    ) == ("cuda", 4, (256, 256), 128, 128)
    assert candidate.training.seeds == (0, 1, 2)
    assert candidate.training.timesteps >= 65_536
    assert candidate.environment.decision_hours == 1.0
    assert not candidate.action.risk_tilt_enabled
    assert candidate.action.n_factors == 3
    assert candidate.factor_artifact == EXAMPLE_ROOT / "relative-factor-artifact"


def test_full_runner_uses_three_assets_and_four_native_timeframes() -> None:
    content = (EXAMPLE_ROOT / "run_full_research.py").read_text(encoding="utf-8")

    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        assert symbol in content
    for timeframe in ("15m", "1h", "4h", "1d"):
        assert timeframe in content
    assert "2024-12-01T00:00:00Z" in content
    assert "2026-06-01T00:00:00Z" in content
    assert "13_128" in content
    assert "dataset_id" in content
    assert "artifact_digest" in content
    assert "binance_multitimeframe_feature_specs" in content
    assert "feature_count" in content
    assert "96" in content
    assert '"train", "run"' in content
    assert '"walk-forward", "run"' in content


def _runner_namespace() -> dict[str, Any]:
    return runpy.run_path(str(EXAMPLE_ROOT / "run_full_research.py"))


class _CheckpointPolicy:
    def save(self, target: str) -> None:
        Path(target).with_suffix(".zip").write_bytes(b"checkpoint")


def test_full_runner_accepts_canonical_nested_checkpoints(tmp_path: Path) -> None:
    verify_training = _runner_namespace()["_verify_training"]
    for name in ("run.json", "ensemble.json", "environment.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    for index in range(3):
        member = tmp_path / "members" / f"member-{index:03d}"
        member.mkdir(parents=True)
        (member / "policy.zip").write_bytes(b"policy")
        publish_checkpoint(
            model=_CheckpointPolicy(),
            checkpoint_root=member / "checkpoints",
            algorithm="ppo",
            seed=index,
            requested_timestep=262_144,
            observed_timestep=262_144,
            environment_digest="e" * 64,
            training_config_digest="c" * 64,
        )

    verify_training(tmp_path)


def _write_walk_forward(path: Path, *, selected: float, baseline: float) -> None:
    path.mkdir(parents=True)
    (path / "walk-forward.json").write_text(
        json.dumps(
            {
                "selected_independent_summary": {"mean_fold_return": selected},
                "baseline_independent_summary": {"mean_fold_return": baseline},
                "folds": [
                    {"selected_returns": [0.10, -0.20]},
                    {"selected_returns": [-0.05, 0.02]},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_full_runner_publishes_passing_research_gate_and_summary(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "artifacts" / "runs" / "wf"
    _write_walk_forward(walk_forward_path, selected=0.04, baseline=0.01)
    summary: dict[str, object] = {"production_status": "NO-GO"}

    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary=summary,
    )

    gate = json.loads((tmp_path / "research-gate.json").read_text(encoding="utf-8"))
    published_summary = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert gate["passed"] is True
    assert gate["observed"]["selected_mean_return"] == 0.04
    assert gate["observed"]["baseline_uplift"] == 0.03
    assert gate["observed"][
        "maximum_independently_reset_fold_drawdown"
    ] == pytest.approx(0.20)
    assert published_summary["research_gate"] == gate
    assert summary["research_gate"]["passed"] is True


def test_full_runner_preserves_failed_gate_and_summary_before_nonzero_exit(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "artifacts" / "runs" / "wf"
    _write_walk_forward(walk_forward_path, selected=0.01, baseline=0.02)

    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary={"production_status": "NO-GO"},
    )

    gate_path = tmp_path / "research-gate.json"
    summary_path = tmp_path / "summary.json"
    assert exit_code != 0
    assert gate_path.is_file()
    assert summary_path.is_file()
    assert json.loads(gate_path.read_text(encoding="utf-8"))["passed"] is False
    assert json.loads(summary_path.read_text(encoding="utf-8"))[
        "research_gate"
    ] == json.loads(gate_path.read_text(encoding="utf-8"))


def test_full_runner_fails_closed_and_publishes_malformed_evidence(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "artifacts" / "runs" / "wf"
    walk_forward_path.mkdir(parents=True)
    (walk_forward_path / "walk-forward.json").write_text(
        json.dumps({"folds": [{"selected_returns": [0.01, "invalid"]}]}),
        encoding="utf-8",
    )

    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary={"production_status": "NO-GO"},
    )

    gate = json.loads((tmp_path / "research-gate.json").read_text(encoding="utf-8"))
    assert exit_code != 0
    assert gate["conditions"]["evidence_valid"] is False
    assert gate["evidence_errors"]
    assert (tmp_path / "summary.json").is_file()


def test_full_runner_fails_closed_when_fold_compounding_overflows(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "artifacts" / "runs" / "wf"
    walk_forward_path.mkdir(parents=True)
    (walk_forward_path / "walk-forward.json").write_text(
        json.dumps(
            {
                "selected_independent_summary": {"mean_fold_return": 0.04},
                "baseline_independent_summary": {"mean_fold_return": 0.01},
                "folds": [{"selected_returns": [1e308, 1e308]}],
            }
        ),
        encoding="utf-8",
    )

    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary={"production_status": "NO-GO"},
    )

    gate_path = tmp_path / "research-gate.json"
    summary_path = tmp_path / "summary.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert gate["passed"] is False
    assert gate["conditions"]["evidence_valid"] is False
    assert gate["evidence_errors"]
    assert summary_path.is_file()
    published_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert published_summary["research_gate"] == gate


def test_full_runner_publishes_standard_json_when_baseline_uplift_overflows(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "artifacts" / "runs" / "wf"
    _write_walk_forward(walk_forward_path, selected=1e308, baseline=-1e308)

    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary={"production_status": "NO-GO"},
    )

    gate_path = tmp_path / "research-gate.json"
    summary_path = tmp_path / "summary.json"
    gate_text = gate_path.read_text(encoding="utf-8")
    summary_text = summary_path.read_text(encoding="utf-8")
    gate = json.loads(gate_text)
    assert exit_code != 0
    assert gate["passed"] is False
    assert gate["conditions"]["evidence_valid"] is False
    assert gate["observed"]["baseline_uplift"] is None
    assert "Infinity" not in gate_text
    assert "NaN" not in gate_text
    assert "Infinity" not in summary_text
    assert "NaN" not in summary_text
    assert json.loads(summary_text)["research_gate"] == gate


def test_full_runner_persists_failed_gate_for_oversized_summary_integer(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "artifacts" / "runs" / "wf"
    _write_walk_forward(walk_forward_path, selected=10**400, baseline=0.01)

    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary={"production_status": "NO-GO"},
    )

    gate_path = tmp_path / "research-gate.json"
    summary_path = tmp_path / "summary.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert gate["passed"] is False
    assert gate["conditions"]["evidence_valid"] is False
    assert gate["observed"]["selected_mean_return"] is None
    assert summary_path.is_file()
    published_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert published_summary["research_gate"] == gate


def test_full_runner_persists_failed_gate_for_oversized_fold_integer(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "artifacts" / "runs" / "wf"
    walk_forward_path.mkdir(parents=True)
    (walk_forward_path / "walk-forward.json").write_text(
        json.dumps(
            {
                "selected_independent_summary": {"mean_fold_return": 0.04},
                "baseline_independent_summary": {"mean_fold_return": 0.01},
                "folds": [{"selected_returns": [10**400]}],
            }
        ),
        encoding="utf-8",
    )

    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary={"production_status": "NO-GO"},
    )

    gate_path = tmp_path / "research-gate.json"
    summary_path = tmp_path / "summary.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert gate["passed"] is False
    assert gate["conditions"]["evidence_valid"] is False
    assert gate["observed"]["maximum_independently_reset_fold_drawdown"] is None
    assert summary_path.is_file()
    published_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert published_summary["research_gate"] == gate


def test_full_runner_persists_failed_gate_for_fold_return_below_total_loss(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "artifacts" / "runs" / "wf"
    walk_forward_path.mkdir(parents=True)
    (walk_forward_path / "walk-forward.json").write_text(
        json.dumps(
            {
                "selected_independent_summary": {"mean_fold_return": 0.04},
                "baseline_independent_summary": {"mean_fold_return": 0.01},
                "folds": [{"selected_returns": [-2.0]}],
            }
        ),
        encoding="utf-8",
    )

    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary={"production_status": "NO-GO"},
    )

    gate_path = tmp_path / "research-gate.json"
    summary_path = tmp_path / "summary.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert gate["passed"] is False
    assert gate["conditions"]["evidence_valid"] is False
    assert gate["observed"]["maximum_independently_reset_fold_drawdown"] is None
    assert summary_path.is_file()
    published_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert published_summary["research_gate"] == gate


def test_full_runner_accepts_total_loss_fold_return_as_valid_evidence(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "artifacts" / "runs" / "wf"
    walk_forward_path.mkdir(parents=True)
    (walk_forward_path / "walk-forward.json").write_text(
        json.dumps(
            {
                "selected_independent_summary": {"mean_fold_return": 0.04},
                "baseline_independent_summary": {"mean_fold_return": 0.01},
                "folds": [{"selected_returns": [-1.0]}],
            }
        ),
        encoding="utf-8",
    )

    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary={"production_status": "NO-GO"},
    )

    gate = json.loads((tmp_path / "research-gate.json").read_text(encoding="utf-8"))
    assert exit_code != 0
    assert gate["passed"] is False
    assert gate["conditions"]["evidence_valid"] is True
    assert gate["observed"]["maximum_independently_reset_fold_drawdown"] == 1.0
    assert (tmp_path / "summary.json").is_file()
