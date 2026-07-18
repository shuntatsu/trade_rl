from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
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
        config.training.policy,
        config.training.policy_net_arch,
        config.training.value_net_arch,
        config.training.sequence_encoder,
    ) == ("cuda", 4, "MultiInputPolicy", (384, 256, 128), (512, 384, 256), True)
    assert config.training.seeds == (0, 1, 2)
    assert config.training.timesteps >= 524_288
    assert config.training.n_steps == 128
    assert config.training.batch_size == 128
    assert config.training.n_epochs == 10
    assert config.training.gamma == pytest.approx(0.998969062762624)
    assert config.training.decision_hours == 0.25
    assert config.environment.decision_hours == 0.25
    assert config.training.behavior_cloning_epochs == 15
    assert config.risk.max_turnover is None
    assert config.environment.episode_hours >= 720.0
    assert not config.action.risk_tilt_enabled
    assert config.action.mode.value == "target_weight"
    assert config.action.target_weight_count == 3
    assert config.action.n_factors == 0
    assert config.factor_artifact is None
    assert config.risk.entry_threshold == 0.10
    assert config.risk.exit_threshold == 0.03
    assert config.risk.no_trade_band == 0.05
    assert config.environment.emergency_risk.stop_loss_return == 0.03
    execution = config.environment.execution_cost
    assert execution.fee_rate > 0.0
    assert execution.spread_rate > 0.0
    assert execution.impact_rate > 0.0
    assert config.portfolio_risk.max_abs_weight is not None
    assert config.portfolio_risk.max_abs_weight <= 0.5


def test_full_walk_forward_config_has_six_material_folds() -> None:
    config = MarketWalkForwardConfig.from_json(
        EXAMPLE_ROOT / "walk-forward-full.json",
        n_bars=55_392,
    )

    folds = config.workflow.build_folds()
    assert len(folds) == 6
    assert config.checkpoint_finalists_per_seed == 1
    assert all(fold.test.size == 2_880 for fold in folds)
    assert sum(fold.test.size for fold in folds) == 17_280
    assert (folds[0].test.start, folds[0].test.stop) == (26_336, 29_216)
    assert (folds[-1].test.start, folds[-1].test.stop) == (40_736, 43_616)
    assert [candidate.name for candidate in config.candidates] == [
        "snapshot-ppo-15m-target",
        "ppo-15m-target",
        "oracle-bc-ppo-15m-target",
    ]
    candidate = next(
        item.run for item in config.candidates if item.name == "ppo-15m-target"
    )
    assert (
        candidate.training.device,
        candidate.training.n_envs,
        candidate.training.policy,
        candidate.training.policy_net_arch,
        candidate.training.value_net_arch,
        candidate.training.sequence_encoder,
    ) == ("cuda", 4, "MultiInputPolicy", (384, 256, 128), (512, 384, 256), True)
    assert candidate.training.seeds == (0, 1, 2)
    assert candidate.training.timesteps >= 524_288
    assert candidate.training.gamma == pytest.approx(0.998969062762624)
    assert candidate.training.decision_hours == 0.25
    assert candidate.environment.decision_hours == 0.25
    assert candidate.environment.structured_sequence_observation
    assert candidate.environment.resolved_sequence_windows == (
        ("15m", 96),
        ("1h", 168),
        ("4h", 120),
        ("1d", 60),
    )
    assert candidate.risk.max_turnover is None
    assert not candidate.action.risk_tilt_enabled
    assert candidate.action.mode.value == "target_weight"
    assert candidate.action.target_weight_count == 3
    assert candidate.action.n_factors == 0
    assert candidate.factor_artifact is None
    assert candidate.training.behavior_cloning_epochs == 0
    snapshot = next(
        item.run for item in config.candidates if item.name == "snapshot-ppo-15m-target"
    )
    assert snapshot.training.n_envs == 4
    assert not snapshot.training.sequence_encoder
    assert not snapshot.environment.structured_sequence_observation
    oracle = next(
        item.run
        for item in config.candidates
        if item.name == "oracle-bc-ppo-15m-target"
    )
    assert oracle.training.behavior_cloning_epochs == 15
    assert config.minimum_seed_success_fraction == pytest.approx(2.0 / 3.0)
    assert config.minimum_worst_seed_uplift == pytest.approx(0.0)
    assert config.maximum_seed_score_std == pytest.approx(0.10)
    assert config.maximum_selection_turnover_per_day == pytest.approx(1.0)
    assert config.maximum_selection_cost_fraction == pytest.approx(0.03)
    assert config.minimum_selection_score == pytest.approx(0.0)
    assert config.maximum_selection_drawdown == pytest.approx(0.20)


def test_full_runner_uses_three_assets_and_four_native_timeframes() -> None:
    content = (EXAMPLE_ROOT / "full_research_pipeline.py").read_text(encoding="utf-8")

    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        assert symbol in content
    for timeframe in ("15m", "1h", "4h", "1d"):
        assert timeframe in content
    assert "2024-12-01T00:00:00Z" in content
    assert "2026-07-01T00:00:00Z" in content
    assert "55_392" in content
    assert "dataset_id" in content
    assert "artifact_digest" in content
    assert "binance_multitimeframe_feature_specs" in content
    assert "raw_feature_count" in content
    assert "policy_observation_count" in content
    assert "231_005" in content
    assert "226" in content
    assert '"train", "run"' in content
    assert '"walk-forward", "run"' in content


def test_full_runner_separates_selection_from_final_training() -> None:
    content = (EXAMPLE_ROOT / "run_full_research_state.py").read_text(encoding="utf-8")

    develop = content.index("def _develop(")
    proposal = content.index("SelectionProposal.create(", develop)
    train_selected = content.index("def _train_selected(", proposal)
    final_training = content.index("*pipeline._TRAIN_RUN_COMMAND", train_selected)
    assert develop < proposal < train_selected < final_training
    assert "SelectionAuthorization.authorize" not in content


def _runner_namespace() -> dict[str, Any]:
    sys.path.insert(0, str(EXAMPLE_ROOT))
    return vars(importlib.import_module("full_research_pipeline"))


def test_full_runner_rejects_existing_generation_without_deleting_it(
    tmp_path: Path,
) -> None:
    prepare_run_roots = _runner_namespace()["_prepare_run_roots"]
    work_root = tmp_path / "runs" / "generation-001"
    cache_root = tmp_path / "cache" / "binance-vision"
    work_root.mkdir(parents=True)
    evidence = work_root / "failed-evidence.json"
    evidence.write_text('{"preserved": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="run generation already exists"):
        prepare_run_roots(work_root=work_root, cache_root=cache_root)

    assert evidence.read_text(encoding="utf-8") == '{"preserved": true}\n'
    assert not cache_root.exists()


def test_full_runner_prepares_stable_cache_outside_new_generation(
    tmp_path: Path,
) -> None:
    prepare_run_roots = _runner_namespace()["_prepare_run_roots"]
    cache_root = tmp_path / "cache" / "binance-vision"
    cached_download = cache_root / "monthly-klines.zip"
    cache_root.mkdir(parents=True)
    cached_download.write_bytes(b"cached-market-data")
    work_root = tmp_path / "runs" / "generation-002"

    prepared_work_root, prepared_cache_root = prepare_run_roots(
        work_root=work_root,
        cache_root=cache_root,
    )

    assert prepared_work_root == work_root.resolve()
    assert prepared_cache_root == cache_root.resolve()
    assert work_root.is_dir()
    assert cached_download.read_bytes() == b"cached-market-data"
    assert cache_root not in work_root.parents
    assert work_root not in cache_root.parents


def test_full_runner_rejects_cache_inside_generation(tmp_path: Path) -> None:
    prepare_run_roots = _runner_namespace()["_prepare_run_roots"]
    work_root = tmp_path / "runs" / "generation-003"

    with pytest.raises(ValueError, match="cache root must be outside"):
        prepare_run_roots(
            work_root=work_root,
            cache_root=work_root / "vision-cache",
        )

    assert not work_root.exists()


def test_full_runner_rejects_generation_inside_cache(tmp_path: Path) -> None:
    prepare_run_roots = _runner_namespace()["_prepare_run_roots"]
    cache_root = tmp_path / "cache" / "binance-vision"
    work_root = cache_root / "runs" / "generation-004"

    with pytest.raises(ValueError, match="cache root must be outside"):
        prepare_run_roots(work_root=work_root, cache_root=cache_root)

    assert not work_root.exists()


def test_full_runner_injects_packaged_git_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRADE_RL_GIT_COMMIT", "b" * 40)
    monkeypatch.setenv("TRADE_RL_GIT_DIRTY", "true")
    write_run_config = _runner_namespace()["_write_run_config"]
    output = tmp_path / "walk-forward.json"

    write_run_config(
        template_path=EXAMPLE_ROOT / "walk-forward-full.json",
        output_path=output,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["git_commit"] == "b" * 40
    assert payload["git_dirty"] is True
    for candidate in payload["candidates"]:
        assert candidate["run"]["git_commit"] == "b" * 40
        assert candidate["run"]["git_dirty"] is True


def test_full_runner_fails_closed_without_packaged_git_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRADE_RL_GIT_COMMIT", raising=False)
    monkeypatch.delenv("TRADE_RL_GIT_DIRTY", raising=False)
    write_run_config = _runner_namespace()["_write_run_config"]

    with pytest.raises(ValueError, match="TRADE_RL_GIT_COMMIT"):
        write_run_config(
            template_path=EXAMPLE_ROOT / "training-full.json",
            output_path=tmp_path / "training.json",
        )


class _CheckpointPolicy:
    def save(self, target: str) -> None:
        Path(target).with_suffix(".zip").write_bytes(b"checkpoint")


def test_full_runner_accepts_canonical_nested_checkpoints(tmp_path: Path) -> None:
    verify_training = _runner_namespace()["_verify_training"]
    (tmp_path / "run.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "environment.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "ensemble.json").write_text(
        json.dumps(
            {
                "expected_members": 3,
                "members": [{"seed": index} for index in range(3)],
            }
        ),
        encoding="utf-8",
    )
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


def _write_walk_forward(
    path: Path,
    *,
    selected: float,
    baseline: float,
    selected_policy_digests: tuple[str | None, ...] = ("a" * 64, "b" * 64),
) -> None:
    path.mkdir(parents=True)
    returns_by_fold = ([0.10, -0.20], [-0.05, 0.02])
    folds = [
        {
            "selected_configuration": "oracle-bc-ppo-15m-target",
            "selected_policy_digest": digest,
            "selected_member_seeds": [0, 1, 2],
            "selected_member_policy_digests": [
                f"{index + 1:064x}",
                f"{index + 11:064x}",
                f"{index + 21:064x}",
            ],
            "selected_returns": returns_by_fold[index % len(returns_by_fold)],
            "selected_turnover_per_day": 0.5,
            "selected_cost_fraction": 0.01,
            "candidate_aggregates": [
                {
                    "configuration": "oracle-bc-ppo-15m-target",
                    "eligible": True,
                }
            ],
        }
        for index, digest in enumerate(selected_policy_digests)
    ]
    (path / "walk-forward.json").write_text(
        json.dumps(
            {
                "selected_independent_summary": {
                    "fold_count": 2,
                    "mean_fold_return": selected,
                },
                "baseline_independent_summary": {
                    "fold_count": 2,
                    "mean_fold_return": baseline,
                },
                "folds": folds,
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
    assert gate["observed"]["selected_policy_digests"] == ["a" * 64, "b" * 64]
    assert gate["conditions"]["rl_policy_selected_all_folds"] is True
    assert published_summary["research_gate"] == gate
    assert summary["research_gate"]["passed"] is True


@pytest.mark.parametrize(
    "selected_policy_digests",
    [
        (None, None),
        ("a" * 64, None),
        ("a" * 64,),
        ("a" * 64, "b" * 64, "c" * 64),
        ("rl-policy", "b" * 64),
        (f" {'a' * 64} ", "b" * 64),
        ("A" * 64, "b" * 64),
        ("g" * 64, "b" * 64),
        ("a" * 64, "a" * 64),
    ],
)
def test_full_runner_rejects_invalid_rl_policy_identity_set(
    tmp_path: Path,
    selected_policy_digests: tuple[str | None, ...],
) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "artifacts" / "runs" / "wf"
    _write_walk_forward(
        walk_forward_path,
        selected=0.04,
        baseline=0.01,
        selected_policy_digests=selected_policy_digests,
    )

    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary={"production_status": "NO-GO"},
    )

    gate = json.loads((tmp_path / "research-gate.json").read_text(encoding="utf-8"))
    assert exit_code != 0
    assert gate["conditions"]["rl_policy_selected_all_folds"] is False
    assert gate["conditions"]["evidence_valid"] is False
    assert gate["evidence_errors"]
    assert (tmp_path / "summary.json").is_file()


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
                "folds": [
                    {
                        "selected_configuration": "ppo-15m-target",
                        "selected_policy_digest": "a" * 64,
                        "selected_member_seeds": [0, 1, 2],
                        "selected_member_policy_digests": [
                            "1" * 64,
                            "2" * 64,
                            "3" * 64,
                        ],
                        "selected_returns": [-1.0],
                        "selected_turnover_per_day": 0.5,
                        "selected_cost_fraction": 0.01,
                        "candidate_aggregates": [
                            {
                                "configuration": "ppo-15m-target",
                                "eligible": True,
                            }
                        ],
                    },
                    {
                        "selected_configuration": "ppo-15m-target",
                        "selected_policy_digest": "b" * 64,
                        "selected_member_seeds": [0, 1, 2],
                        "selected_member_policy_digests": [
                            "4" * 64,
                            "5" * 64,
                            "6" * 64,
                        ],
                        "selected_returns": [0.0],
                        "selected_turnover_per_day": 0.5,
                        "selected_cost_fraction": 0.01,
                        "candidate_aggregates": [
                            {
                                "configuration": "ppo-15m-target",
                                "eligible": True,
                            }
                        ],
                    },
                ],
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


def test_full_runner_requires_one_stable_seed_ensemble_recipe(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    stability = namespace["_selection_stability_passed"]
    stable = [
        {
            "selected_configuration": "ppo-15m-target",
            "selected_member_seeds": [0, 1, 2],
            "candidate_aggregates": [
                {"configuration": "ppo-15m-target", "eligible": True}
            ],
        }
        for _ in range(2)
    ]
    changed = [dict(stable[0]), dict(stable[1])]
    changed[1]["selected_member_seeds"] = [0, 2, 3]

    assert stability(stable) is True
    assert stability(changed) is False


def test_selected_walk_forward_recipe_preserves_seed_ensemble(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    select_recipe = namespace["_selected_walk_forward_recipe"]
    walk_forward_path = tmp_path / "walk-forward"
    walk_forward_path.mkdir()
    (walk_forward_path / "walk-forward.json").write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "selected_configuration": "ppo-15m-target",
                        "selected_member_seeds": [0, 1, 2],
                    },
                    {
                        "selected_configuration": "ppo-15m-target",
                        "selected_member_seeds": [0, 1, 2],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "walk-forward-config.json"
    config_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "name": "ppo-15m-target",
                        "run": {"training": {"seeds": [0, 1, 2]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "selected.json"

    name, seeds, path = select_recipe(walk_forward_path, config_path, output)

    assert name == "ppo-15m-target"
    assert seeds == (0, 1, 2)
    assert path == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["training"]["seeds"] == [0, 1, 2]

    evidence = json.loads(
        (walk_forward_path / "walk-forward.json").read_text(encoding="utf-8")
    )
    evidence["folds"][1]["selected_member_seeds"] = [0, 2, 3]
    (walk_forward_path / "walk-forward.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="seed ensemble"):
        select_recipe(
            walk_forward_path,
            config_path,
            tmp_path / "invalid-selected.json",
        )


def test_full_runner_preserves_all_selected_recipe_seeds(tmp_path: Path) -> None:
    namespace = _runner_namespace()
    select_recipe = namespace["_selected_walk_forward_recipe"]
    walk_forward_path = tmp_path / "wf"
    _write_walk_forward(walk_forward_path, selected=0.04, baseline=0.01)
    selected_name, seeds, output = select_recipe(
        walk_forward_path,
        EXAMPLE_ROOT / "walk-forward-full.json",
        tmp_path / "selected.json",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert selected_name == "oracle-bc-ppo-15m-target"
    assert seeds == (0, 1, 2)
    assert payload["training"]["seeds"] == [0, 1, 2]


def test_full_runner_strict_gate_rejects_two_short_folds(tmp_path: Path) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "wf"
    _write_walk_forward(walk_forward_path, selected=0.04, baseline=0.01)
    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary={"production_status": "NO-GO"},
        strict=True,
    )
    gate = json.loads((tmp_path / "research-gate.json").read_text(encoding="utf-8"))
    assert exit_code != 0
    assert gate["conditions"]["minimum_fold_count_met"] is False
    assert gate["conditions"]["minimum_oos_days_met"] is False


def test_signed_rule_history_is_authoritative_and_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64
    from datetime import UTC, datetime, timedelta

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from trade_rl.release.offline_signing import public_key_bytes, sign_payload

    namespace = _runner_namespace()
    load_history = namespace["_load_rule_history"]
    symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
    payload = {
        "schema_version": "binance_instrument_rule_history_v4",
        "policy_version": "binance_metadata_modes_v2",
        "market": "usds-m",
        "symbol_order": list(symbols),
        "coverage": {
            "start_time": "2024-12-01T00:00:00+00:00",
            "end_time": "2026-07-01T00:00:00+00:00",
        },
        "issued_at": "2026-07-17T00:00:00+00:00",
        "source_uri": "operator://signed-binance-rules",
        "symbols": {
            symbol: {
                "listed_at": "2020-01-01T00:00:00+00:00",
                "tick_size": 0.01,
                "lot_size": 0.0001,
                "minimum_notional": 10.0,
                "execution_rules": [
                    {
                        "effective_at": "2024-01-01T00:00:00+00:00",
                        "tick_size": 0.1,
                        "lot_size": 0.001,
                        "minimum_notional": 5.0,
                    },
                    {
                        "effective_at": "2026-01-01T00:00:00+00:00",
                        "tick_size": 0.01,
                        "lot_size": 0.0001,
                        "minimum_notional": 10.0,
                    },
                ],
            }
            for symbol in symbols
        },
    }
    private_key = Ed25519PrivateKey.from_private_bytes(b"\x55" * 32)
    signed_at = datetime(2026, 7, 17, 1, tzinfo=UTC)
    envelope = sign_payload(
        payload,
        key_id="metadata-key",
        purpose="binance-rule-history",
        private_key=private_key,
        signed_at=signed_at,
    )
    path = tmp_path / "metadata-history.json"
    path.write_bytes(
        canonical_json_bytes({"payload": payload, "envelope": envelope.to_mapping()})
    )
    key_store = tmp_path / "metadata-public-keys.json"
    key_store.write_text(
        json.dumps(
            {
                "schema_version": "public_verification_key_store_v1",
                "keys": [
                    {
                        "key_id": "metadata-key",
                        "public_key": base64.b64encode(
                            public_key_bytes(private_key)
                        ).decode("ascii"),
                        "purpose": "binance-rule-history",
                        "valid_from": (signed_at - timedelta(days=1)).isoformat(),
                        "valid_until": (signed_at + timedelta(days=365)).isoformat(),
                        "algorithm": "ed25519",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADE_RL_BINANCE_RULE_HISTORY", str(path))
    monkeypatch.setenv("TRADE_RL_METADATA_PUBLIC_KEYS", str(key_store))

    verified = load_history(trusted_now=datetime(2026, 7, 18, tzinfo=UTC))

    assert len(verified.payload_digest) == 64
    assert verified.metadata["BTCUSDT"]["listed_at"].startswith("2020-01-01")
    assert verified.metadata["BTCUSDT"]["tick_size"] == pytest.approx(0.01)
    assert len(verified.execution_rule_histories["BTCUSDT"]) == 2
