from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing replacement anchor in {path}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def add_tests() -> None:
    append_once(
        "tests/rl/test_sequence_environment_config.py",
        "test_signal_delay_decisions_accepts_only_zero_or_one",
        '''

def test_signal_delay_decisions_accepts_only_zero_or_one() -> None:
    assert _config(signal_delay_decisions=0).signal_delay_decisions == 0
    assert _config(signal_delay_decisions=1).signal_delay_decisions == 1
    for invalid in (-1, 2, True, 0.5):
        with pytest.raises(ValueError, match="signal_delay_decisions"):
            _config(signal_delay_decisions=invalid)
''',
    )
    replace_once(
        "tests/rl/test_target_weight_action.py",
        "def environment(spec: ActionSpec) -> ResidualMarketEnv:\n",
        "def environment(\n    spec: ActionSpec, *, signal_delay_decisions: int = 0\n) -> ResidualMarketEnv:\n",
    )
    replace_once(
        "tests/rl/test_target_weight_action.py",
        "            decision_every=1,\n            reward=AbsoluteGrowthRewardConfig(),\n",
        "            decision_every=1,\n            signal_delay_decisions=signal_delay_decisions,\n            reward=AbsoluteGrowthRewardConfig(),\n",
    )
    append_once(
        "tests/rl/test_target_weight_action.py",
        "test_delayed_target_executes_on_the_following_decision",
        '''

def test_delayed_target_executes_on_the_following_decision() -> None:
    env = environment(target_spec(count=2), signal_delay_decisions=1)
    env.reset(options={"start_idx": 10, "initial_state_mode": "cash"})

    _, _, _, _, first = env.step(np.array([0.40, 0.0], dtype=np.float32))
    np.testing.assert_allclose(env.hybrid.weights, np.zeros(2), atol=1e-12)
    assert first["execution_delay_warmup"] is True
    np.testing.assert_allclose(first["submitted_target"], np.array([0.40, 0.0]))
    np.testing.assert_allclose(first["executed_target"], np.zeros(2))

    _, _, _, _, second = env.step(np.zeros(2, dtype=np.float32))
    assert env.hybrid.weights[0] > 0.30
    assert second["execution_delay_warmup"] is False
    np.testing.assert_allclose(second["executed_target"], np.array([0.40, 0.0]))


def test_reset_clears_delayed_target_queue() -> None:
    env = environment(target_spec(count=2), signal_delay_decisions=1)
    env.reset(options={"start_idx": 10, "initial_state_mode": "cash"})
    env.step(np.array([0.40, 0.0], dtype=np.float32))

    env.reset(options={"start_idx": 10, "initial_state_mode": "cash"})
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    np.testing.assert_allclose(env.hybrid.weights, np.zeros(2), atol=1e-12)
    assert info["execution_delay_warmup"] is True
    np.testing.assert_allclose(info["executed_target"], np.zeros(2))
''',
    )


def add_implementation() -> None:
    replace_once(
        "trade_rl/rl/environment_config.py",
        "    decision_every: int | None = None\n    reward_scale: float = 100.0\n",
        "    decision_every: int | None = None\n    signal_delay_decisions: int = 0\n    reward_scale: float = 100.0\n",
    )
    replace_once(
        "trade_rl/rl/environment_config.py",
        "        if self.episode_bars is not None and self.episode_hour_choices:\n",
        "        if (\n            isinstance(self.signal_delay_decisions, bool)\n            or not isinstance(self.signal_delay_decisions, int)\n            or self.signal_delay_decisions not in {0, 1}\n        ):\n            raise ValueError(\n                \"signal_delay_decisions must be exactly zero or one so the pending \"\n                \"target remains fully observable\"\n            )\n        if self.episode_bars is not None and self.episode_hour_choices:\n",
    )
    replace_once(
        "trade_rl/rl/environment.py",
        "                \"decision_hours\": self.config.decision_hours,\n                \"resolved_decision_hours\": self._resolved_decision_hours,\n",
        "                \"decision_hours\": self.config.decision_hours,\n                \"signal_delay_decisions\": self.config.signal_delay_decisions,\n                \"resolved_decision_hours\": self._resolved_decision_hours,\n",
    )
    replace_once(
        "trade_rl/rl/environment.py",
        "        self._previous_action = np.zeros(self.action_spec.size, dtype=np.float32)\n        self._position_age = np.zeros(dataset.n_symbols, dtype=np.float64)\n",
        "        self._previous_action = np.zeros(self.action_spec.size, dtype=np.float32)\n        self._pending_hybrid_target: np.ndarray | None = None\n        self._pending_shadow_target: np.ndarray | None = None\n        self._position_age = np.zeros(dataset.n_symbols, dtype=np.float64)\n",
    )
    replace_once(
        "trade_rl/rl/environment.py",
        "        self._previous_action = np.zeros(self.action_spec.size, dtype=np.float32)\n        self._position_age = np.zeros(self.dataset.n_symbols, dtype=np.float64)\n",
        "        self._previous_action = np.zeros(self.action_spec.size, dtype=np.float32)\n        self._pending_hybrid_target = None\n        self._pending_shadow_target = None\n        self._position_age = np.zeros(self.dataset.n_symbols, dtype=np.float64)\n",
    )
    replace_once(
        "trade_rl/rl/environment.py",
        "        cursor = history_start\n        returns: list[float] = []\n        while cursor < reward_start:\n            target = self.trend_strategy.targets(self.dataset, cursor).base\n            constrained = self.pre_trade_risk.constrain(\n                target,\n",
        "        cursor = history_start\n        pending_target: np.ndarray | None = None\n        returns: list[float] = []\n        while cursor < reward_start:\n            submitted_target = self.trend_strategy.targets(self.dataset, cursor).base\n            if self.config.signal_delay_decisions == 0:\n                target = submitted_target\n            else:\n                target = book.weights.copy() if pending_target is None else pending_target\n                pending_target = submitted_target.copy()\n            constrained = self.pre_trade_risk.constrain(\n                target,\n",
    )
    replace_once(
        "trade_rl/rl/environment.py",
        "        hybrid_risk = self._constrain_target(composition.proposal, self.hybrid)\n        shadow_risk = self._constrain_target(trends.base, self.shadow)\n        bars = self._decision_bar_count()\n",
        "        submitted_hybrid_target = np.asarray(\n            composition.proposal, dtype=np.float64\n        ).copy()\n        submitted_shadow_target = np.asarray(trends.base, dtype=np.float64).copy()\n        execution_delay_warmup = False\n        if self.config.signal_delay_decisions == 0:\n            executed_hybrid_target = submitted_hybrid_target\n            executed_shadow_target = submitted_shadow_target\n        else:\n            execution_delay_warmup = self._pending_hybrid_target is None\n            executed_hybrid_target = (\n                self.hybrid.weights.copy()\n                if self._pending_hybrid_target is None\n                else self._pending_hybrid_target.copy()\n            )\n            executed_shadow_target = (\n                self.shadow.weights.copy()\n                if self._pending_shadow_target is None\n                else self._pending_shadow_target.copy()\n            )\n            self._pending_hybrid_target = submitted_hybrid_target\n            self._pending_shadow_target = submitted_shadow_target\n        hybrid_risk = self._constrain_target(executed_hybrid_target, self.hybrid)\n        shadow_risk = self._constrain_target(executed_shadow_target, self.shadow)\n        bars = self._decision_bar_count()\n",
    )
    replace_once(
        "trade_rl/rl/environment.py",
        "            \"emergency_deleverage\": emergency_deleverage,\n            \"drawdown_after\": self._drawdown(self.hybrid),\n",
        "            \"emergency_deleverage\": emergency_deleverage,\n            \"execution_delay_warmup\": execution_delay_warmup,\n            \"submitted_target\": submitted_hybrid_target.copy(),\n            \"executed_target\": executed_hybrid_target.copy(),\n            \"drawdown_after\": self._drawdown(self.hybrid),\n",
    )

    for relative in (
        "examples/binance-multitimeframe/training-full.json",
        "examples/binance-multitimeframe/walk-forward-full.json",
    ):
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs: list[dict[str, object]] = []
        if "environment" in payload:
            runs.append(payload)
        for candidate in payload.get("candidates", []):
            if isinstance(candidate, dict) and isinstance(candidate.get("run"), dict):
                runs.append(candidate["run"])
        for run in runs:
            environment = run.get("environment")
            if isinstance(environment, dict):
                environment["signal_delay_decisions"] = 1
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task1_latency.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
