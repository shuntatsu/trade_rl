from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 10 anchor in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def add_tests() -> None:
    replace_once(
        "tests/rl/test_environment_time_config.py",
        '''from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
''',
        '''from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
''',
    )
    replace_once(
        "tests/rl/test_environment_time_config.py",
        '''    assert env.hybrid.total_cost > 0.0
    assert env.shadow.total_cost == pytest.approx(env.hybrid.total_cost)
    assert info["hybrid_liquidation"].fill_count == 2
''',
        '''    assert env.hybrid.total_cost > 0.0
    assert env.shadow.total_cost == pytest.approx(env.hybrid.total_cost)
    assert info["hybrid_liquidation"].fill_count == 2
    assert info["terminal_accounting_mode"] == "liquidate_at_close"
    assert info["terminal_liquidation_cost"] > 0.0
    assert info["pending_target_discarded"] is False
''',
    )
    append_once(
        "tests/rl/test_environment_time_config.py",
        "test_end_of_episode_mark_to_market_truncates_without_closing_positions",
        r'''
def test_end_of_episode_mark_to_market_truncates_without_closing_positions() -> None:
    dataset = market()
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        config=ResidualMarketEnvConfig(
            episode_bars=4,
            decision_every=4,
            initial_capital=1_000.0,
            liquidate_on_end=False,
            execution_cost=ExecutionCostConfig(
                fee_rate=0.001,
                spread_rate=0.0,
                impact_rate=0.0,
                max_participation_rate=1.0,
            ),
        ),
    )
    env.reset(options={"start_idx": 24})

    _, _, terminated, truncated, info = env.step(np.zeros(2))

    assert terminated is False
    assert truncated is True
    assert np.any(np.abs(env.hybrid.quantities) > 1e-12)
    assert "hybrid_liquidation" not in info
    assert info["terminal_accounting_mode"] == "mark_to_market"
    assert info["terminal_liquidation_cost"] == 0.0
    assert info["pending_target_discarded"] is False
    assert env.config.terminal_accounting_mode == "mark_to_market"


def test_final_delayed_target_is_reported_and_discarded_at_horizon() -> None:
    dataset = market()
    env = ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
        ),
        action_spec=ActionSpec(
            mode="target_weight",
            alpha_enabled=False,
            risk_tilt_enabled=False,
            target_weight_count=2,
        ),
        config=ResidualMarketEnvConfig(
            episode_bars=2,
            decision_every=1,
            signal_delay_decisions=1,
            initial_capital=1_000.0,
            liquidate_on_end=False,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )
    env.reset(options={"start_idx": 24})
    first = np.array([0.4, -0.2], dtype=np.float32)
    final = np.array([-0.3, 0.1], dtype=np.float32)
    env.step(first)

    _, _, terminated, truncated, info = env.step(final)

    assert terminated is False
    assert truncated is True
    assert info["pending_target_discarded"] is True
    np.testing.assert_allclose(info["discarded_pending_target"], final)
    np.testing.assert_allclose(info["executed_target"], first)
    assert env._pending_hybrid_target is None
    assert env._pending_shadow_target is None
''',
    )

    replace_once(
        "tests/workflows/test_training_run.py",
        '''    assert (published / "policy-loader.json").is_file()
    loader = json.loads((published / "policy-loader.json").read_text(encoding="utf-8"))
''',
        '''    assert (published / "policy-loader.json").is_file()
    environment = json.loads(
        (published / "environment.json").read_text(encoding="utf-8")
    )
    assert environment["terminal_accounting_mode"] == "mark_to_market"
    loader = json.loads((published / "policy-loader.json").read_text(encoding="utf-8"))
''',
    )


def add_implementation() -> None:
    replace_once(
        "trade_rl/rl/environment_config.py",
        '''    @property
    def resolved_sequence_windows(self) -> tuple[tuple[str, int], ...]:
''',
        '''    @property
    def terminal_accounting_mode(self) -> str:
        return "liquidate_at_close" if self.liquidate_on_end else "mark_to_market"

    @property
    def resolved_sequence_windows(self) -> tuple[tuple[str, int], ...]:
''',
    )

    replace_once(
        "trade_rl/rl/environment.py",
        '''        self.current_index = hybrid_execution.next_index
        self._decision_step_index += 1
        time_limit_reached = self.current_index >= self.end_index
        hybrid_log_return = hybrid_execution.interval_log_return
''',
        '''        self.current_index = hybrid_execution.next_index
        self._decision_step_index += 1
        time_limit_reached = self.current_index >= self.end_index
        pending_target_discarded = bool(
            time_limit_reached
            and self.config.signal_delay_decisions == 1
            and self._pending_hybrid_target is not None
        )
        discarded_pending_target = (
            None
            if not pending_target_discarded
            else self._pending_hybrid_target.copy()
        )
        if time_limit_reached:
            self._pending_hybrid_target = None
            self._pending_shadow_target = None
        hybrid_log_return = hybrid_execution.interval_log_return
''',
    )
    replace_once(
        "trade_rl/rl/environment.py",
        '''        termination_reason = economic_transition.reason
        info: dict[str, object] = {
''',
        '''        termination_reason = economic_transition.reason
        terminal_accounting_mode = (
            "liquidate_at_close"
            if liquidation_terminal
            else "mark_to_market"
            if time_limit_reached
            else "economic_termination"
        )
        terminal_liquidation_cost = (
            float(hybrid_liquidation.interval_cost)
            if liquidation_terminal and hybrid_liquidation is not None
            else 0.0
        )
        info: dict[str, object] = {
''',
    )
    replace_once(
        "trade_rl/rl/environment.py",
        '''            "shadow_terminated": shadow_terminated,
            "termination_reason": termination_reason,
        }
''',
        '''            "shadow_terminated": shadow_terminated,
            "termination_reason": termination_reason,
            "terminal_accounting_mode": terminal_accounting_mode,
            "terminal_liquidation_cost": terminal_liquidation_cost,
            "pending_target_discarded": pending_target_discarded,
        }
        if discarded_pending_target is not None:
            info["discarded_pending_target"] = discarded_pending_target
''',
    )

    replace_once(
        "trade_rl/workflows/training_run.py",
        '''                "schema_version": "training_environment_v2",
                "trend": asdict(config.trend),
''',
        '''                "schema_version": "training_environment_v2",
                "terminal_accounting_mode": config.environment.terminal_accounting_mode,
                "trend": asdict(config.trend),
''',
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task10_terminal_accounting.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
