from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from tests.strategies.test_ppo_interleaved_training import pooled_market
from trade_rl.evaluation import directional_candidates
from trade_rl.evaluation.experiments import ResolvedRunConfig


class _Policy:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"policy")


def _config() -> ResolvedRunConfig:
    return ResolvedRunConfig(
        signal_name="signal",
        signal_index=0,
        feature_names=("signal",),
        feature_indices=(0,),
        fit_symbol_names=("BTCUSDT", "ETHUSDT"),
        fit_symbol_indices=(0, 1),
        fit_cutoff="2026-01-01T03:00:00.000000000",
        rule_entry_threshold=0.1,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=256,
        ppo_seed=0,
        evaluation_start="2026-01-02T00:00:00.000000000",
        evaluation_stop_exclusive="2026-01-03T00:00:00.000000000",
        gross_budget=0.5,
        initial_capital=100_000.0,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
    )


def test_directional_ppo_training_uses_same_borrow_economics_as_evaluation(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_fit_ppo_strategy(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(policy=_Policy())

    monkeypatch.setattr(
        directional_candidates,
        "fit_ppo_strategy",
        fake_fit_ppo_strategy,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda threads: None),
    )

    directional_candidates.fit_directional_candidate(
        "ppo0",
        pooled_market(),
        _config(),
        tmp_path,
    )

    execution_cost = captured["execution_cost"]
    assert execution_cost.borrow_rate_multiplier == 1.0
    assert execution_cost.processing_bar_volume_capacity is False
