"""One-shot, local synthetic diagnostic for PPO conditional actions.

This opt-in tool is a bounded software diagnostic. It does not authorize market
research, production changes, live orders, or a learnability requirement in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from enum import Enum
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import numpy as np

import trade_rl
import trade_rl.strategies.rl.ppo as ppo_module
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.risk import PreTradeRisk
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.stateful.execution import StatefulExecutionObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import (
    PPO_OBSERVATION_SCHEMA_V3,
    PPO_TRAINING_LAYOUT_INTERLEAVED,
    PPOTradingEnv,
    expected_ppo_realized_timesteps,
    fit_ppo_strategy,
)
from trade_rl.strategies.rl.ppo_artifact import (
    load_ppo_inference_bundle,
    save_ppo_inference_bundle,
)

PROTOCOL_SCHEMA = "ppo_conditional_diagnostic_protocol_v1"
PREFIT_SCHEMA = "ppo_conditional_diagnostic_prefit_v1"
ATTEMPT_SCHEMA = "ppo_conditional_diagnostic_attempt_start_v1"
RESULT_SCHEMA = "ppo_conditional_diagnostic_result_v2"
FAILURE_SCHEMA = "ppo_conditional_diagnostic_failure_v2"
HELDOUT_TRACE_SCHEMA = "ppo_conditional_diagnostic_heldout_trace_v1"
EVIDENCE_MANIFEST_SCHEMA = "ppo_conditional_diagnostic_evidence_manifest_v1"
EVIDENCE_CONTRACT_VERSION = 2

INITIAL_CAPITAL = 1_000.0
GROSS_BUDGET = 0.5
FEE_RATE = 0.005
FIXED_SEED = 83
N_BARS = 64
START_INDEX = 0
STOP_INDEX = 63
TOTAL_TIMESTEPS = 8192
ROLLOUT_STEPS_PER_ENV = 512
_RUNTIME_PACKAGES = ("numpy", "gymnasium", "stable-baselines3", "torch")
_REPO_ROOT = Path(__file__).resolve().parents[1]
P3_STATUS = "NOT_ATTEMPTED"


class _PredictPolicy(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> tuple[object, object]: ...


class _TraceEntry(TypedDict):
    action: int
    decision_index: int
    intent: str
    interval_cost: float
    interval_net_return: float
    quantity_after: float
    reward: float
    resulting_index: int


class _Episode(TypedDict):
    active_order_count: int
    cash: float
    fill_count: int
    portfolio_value: float
    quantity: float
    reward_sum: float
    total_cost: float
    trace: list[_TraceEntry]
    evidence_trace: list[dict[str, Any]]
    final_book_state: dict[str, Any]


def fixed_execution_cost() -> ExecutionCostConfig:
    """Return the frozen positive-fee, otherwise cost-free execution profile."""

    return ExecutionCostConfig(
        fee_rate=FEE_RATE,
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=1.0,
        slippage_std=0.0,
        tail_slippage_probability=0.0,
        tail_slippage_multiplier=0.0,
        random_seed=FIXED_SEED,
        minimum_notional=0.0,
        lot_size=0.0,
        tick_size=0.0,
        borrow_rate_multiplier=0.0,
        order_latency_bars=1,
        limit_offset_rate=0.0,
    )


def _array_snapshot(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError("diagnostic evidence arrays must have numeric dtypes")
    if array.dtype.hasobject:
        raise TypeError("diagnostic evidence arrays may not contain Python objects")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _canonical_object_snapshot(value: object, *, field: str) -> dict[str, Any]:
    raw = canonical_json_bytes(value)
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise TypeError(f"{field} evidence must serialize as a JSON object")
    return decoded


def _execution_observation_snapshot(
    observation: StatefulExecutionObservation,
) -> dict[str, Any]:
    return _canonical_object_snapshot(asdict(observation), field="execution")


def _book_state_snapshot(book: Any, *, index: int) -> dict[str, Any]:
    reason = book.termination_reason
    if reason is None:
        termination_reason: str | None = None
    elif isinstance(reason, Enum):
        value = reason.value
        if not isinstance(value, str):
            raise TypeError("book termination reason must have a string value")
        termination_reason = value
    elif isinstance(reason, str):
        termination_reason = reason
    else:
        raise TypeError("book termination reason is unsupported")

    exact_quantities = book._exact_quantities
    if exact_quantities is None:
        raise RuntimeError("book exact quantity snapshot is unavailable")
    return {
        "index": index,
        "cash": float(book.cash),
        "quantities": _array_snapshot(book.quantities),
        "exact_quantities": list(exact_quantities),
        "mark_prices": _array_snapshot(book.mark_prices),
        "contract_multipliers": _array_snapshot(book.contract_multipliers),
        "peak_value": float(book.peak_value),
        "portfolio_value": float(book.portfolio_value),
        "max_drawdown": float(book.max_drawdown),
        "turnover_total": float(book.turnover_total),
        "total_cost": float(book.total_cost),
        "funding_pnl": float(book.funding_pnl),
        "fill_count": int(book.fill_count),
        "rebalance_events": int(book.rebalance_events),
        "returns_history": [float(value) for value in book.returns_history],
        "borrow_cost": float(book.borrow_cost),
        "margin_used": float(book.margin_used),
        "maintenance_margin": float(book.maintenance_margin),
        "maintenance_requirement": float(book.maintenance_requirement),
        "margin_deficit": float(book.margin_deficit),
        "insolvent": bool(book.insolvent),
        "termination_reason": termination_reason,
    }


def build_fixture_dataset(*, heldout: bool = False) -> MarketDataset:
    """Build the fixed 64-row fixture without deriving its label from prices."""

    roles = ("B", "A") if heldout else ("A", "B")
    symbols = ("heldout_0", "heldout_1") if heldout else ("train_0", "train_1")
    active_start = 12 if heldout else 8
    close = np.full((N_BARS, 2), 100.0, dtype=np.float64)
    feature_values = np.zeros((N_BARS, 2, 1), dtype=np.float32)
    for symbol_index, role in enumerate(roles):
        if role == "A":
            # The regime label is a literal constant, available from row zero.
            feature_values[:, symbol_index, 0] = 1.0
            for offset in range(8):
                close[active_start + offset, symbol_index] = 100.0 * (
                    1.01 ** (offset + 1)
                )
            final_rise = close[active_start + 7, symbol_index]
            close[active_start + 8 :, symbol_index] = final_rise

    open_price = np.concatenate((close[:1], close[:-1]), axis=0)
    high = np.maximum(open_price, close)
    low = np.minimum(open_price, close)
    shape = (N_BARS, len(symbols))
    zero_cost = np.zeros(shape, dtype=np.float64)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        N_BARS
    ) * np.timedelta64(1, "h")
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=symbols,
        timestamps=timestamps,
        features=feature_values,
        global_features=np.zeros((N_BARS, 1), dtype=np.float32),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=np.full(shape, 1_000_000.0, dtype=np.float64),
        funding_rate=zero_cost.copy(),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones(feature_values.shape, dtype=np.bool_),
        feature_names=("synthetic_regime_label",),
        global_feature_names=("synthetic_global_zero",),
        periods_per_year=8_760,
        fee_rate=zero_cost.copy(),
        spread_rate=zero_cost.copy(),
        borrow_rate=zero_cost.copy(),
        funding_due=np.zeros(shape, dtype=np.bool_),
        mark_price=close.copy(),
        dividend=zero_cost.copy(),
        cash_rate=np.zeros(N_BARS, dtype=np.float64),
        volume_units=(VolumeUnit.BASE_ASSET,) * 2,
    )
    return dataset.with_content_identity(
        {"fixture": "ppo_conditional_synthetic_v1", "heldout": heldout}
    )


def _make_environment(
    dataset: MarketDataset,
    symbol_index: int,
    *,
    execution_observations: list[dict[str, Any]] | None = None,
) -> PPOTradingEnv:
    execution_observer: Callable[[StatefulExecutionObservation], None] | None = None
    if execution_observations is not None:

        def capture_execution_observation(
            observation: StatefulExecutionObservation,
        ) -> None:
            execution_observations.append(_execution_observation_snapshot(observation))

        execution_observer = capture_execution_observation

    return PPOTradingEnv(
        dataset,
        feature_indices=(0,),
        symbol_indices=(symbol_index,),
        information_symbol_indices=tuple(range(dataset.n_symbols)),
        start_index=START_INDEX,
        stop_index=STOP_INDEX,
        gross_budget=GROSS_BUDGET,
        initial_capital=INITIAL_CAPITAL,
        execution_cost=fixed_execution_cost(),
        settle_terminal_position=True,
        minimum_hold_bars=0,
        observation_schema=PPO_OBSERVATION_SCHEMA_V3,
        execution_observer=execution_observer,
    )


def _action_value(action: object) -> int:
    values = np.asarray(action).reshape(-1)
    if values.size != 1:
        raise ValueError("policy action must contain exactly one value")
    raw = values[0]
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Integral):
        raise ValueError("policy action must be an integer in {0, 1, 2}")
    value = int(raw)
    if value not in {0, 1, 2}:
        raise ValueError("policy action must be an integer in {0, 1, 2}")
    return value


def _run_episode(
    env: PPOTradingEnv,
    action_for_step: Callable[[PPOTradingEnv, np.ndarray], object],
    *,
    capture_evidence: bool = False,
    execution_observations: list[dict[str, Any]] | None = None,
) -> _Episode:
    if capture_evidence and execution_observations is None:
        raise ValueError("execution observations are required for trace capture")
    observation, _ = env.reset(seed=FIXED_SEED)
    trace: list[_TraceEntry] = []
    evidence_trace: list[dict[str, Any]] = []
    execution_observation_cursor = 0
    terminated = False
    while not terminated:
        decision_index = env.index
        pre_action_observation = (
            _array_snapshot(observation) if capture_evidence else None
        )
        first_execution_observation = execution_observation_cursor
        action = _action_value(action_for_step(env, observation))
        observation, reward, terminated, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError("synthetic episode was unexpectedly truncated")
        symbol_index = env.active_symbol_index
        trace.append(
            {
                "action": action,
                "decision_index": decision_index,
                "intent": PositionIntent(action - 1).name,
                "interval_cost": float(cast(Any, info["interval_cost_amount"])),
                "interval_net_return": float(cast(Any, info["interval_net_return"])),
                "quantity_after": float(env.book.quantities[symbol_index]),
                "reward": float(reward),
                "resulting_index": env.index,
            }
        )
        if len(trace) > N_BARS:
            raise RuntimeError("synthetic episode exceeded its fixed bar budget")
        if capture_evidence:
            assert execution_observations is not None
            step_observations = execution_observations[first_execution_observation:]
            execution_observation_cursor = len(execution_observations)
            settlement_intervals = info.get("terminal_settlement_intervals", 0)
            if (
                isinstance(settlement_intervals, bool)
                or not isinstance(settlement_intervals, int)
                or settlement_intervals < 0
            ):
                raise RuntimeError("terminal settlement interval count is invalid")
            if len(step_observations) != 1 + settlement_intervals:
                raise RuntimeError(
                    "execution observer calls do not match the policy step"
                )
            settlement_start = len(step_observations) - settlement_intervals
            execution_calls = [
                {
                    "call_index": first_execution_observation + offset,
                    "phase": (
                        "terminal_settlement"
                        if offset >= settlement_start
                        else "policy_step_execution"
                    ),
                    "observation": payload,
                }
                for offset, payload in enumerate(step_observations)
            ]
            evidence_trace.append(
                {
                    "decision_index": decision_index,
                    "pre_action_observation": pre_action_observation,
                    "action": action,
                    "step_info": _canonical_object_snapshot(
                        info,
                        field="PPO step info",
                    ),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "resulting_index": env.index,
                    "post_step_book": _book_state_snapshot(env.book, index=env.index),
                    "execution_calls": execution_calls,
                }
            )

    if capture_evidence and execution_observation_cursor != len(
        execution_observations or ()
    ):
        raise RuntimeError("execution observer emitted an unassigned call")

    return {
        "active_order_count": len(env.executor.compatibility_order_book.active_orders),
        "cash": float(env.book.cash),
        "fill_count": int(env.book.fill_count),
        "portfolio_value": float(env.book.portfolio_value),
        "quantity": float(env.book.quantities[env.active_symbol_index]),
        "reward_sum": math.fsum(float(item["reward"]) for item in trace),
        "total_cost": float(env.book.total_cost),
        "trace": trace,
        "evidence_trace": evidence_trace,
        "final_book_state": _book_state_snapshot(env.book, index=env.index),
    }


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9)


def _fixture_inputs_are_valid(dataset: MarketDataset) -> bool:
    expected_a = np.full(N_BARS, 100.0, dtype=np.float64)
    for row in range(8, 16):
        expected_a[row] = 100.0 * (1.01 ** (row - 7))
    expected_a[16:] = expected_a[15]
    expected_close = np.column_stack((expected_a, np.full(N_BARS, 100.0)))
    expected_open = np.concatenate((expected_close[:1], expected_close[:-1]), axis=0)
    available_at = dataset.available_at
    if available_at is None:
        return False
    if (
        dataset.n_bars != N_BARS
        or dataset.features.shape != (N_BARS, 2, 1)
        or not np.all(dataset.features[:, 0, 0] == 1.0)
        or not np.all(dataset.features[:, 1, 0] == 0.0)
        or not np.all(dataset.feature_available)
        or not np.all(dataset.resolved_array("feature_staleness") == 0.0)
        or not np.all(dataset.global_features == 0.0)
        or not np.all(dataset.resolved_array("global_feature_available"))
        or not np.all(dataset.resolved_array("global_feature_staleness_hours") == 0.0)
        or not np.all(dataset.resolved_array("information_available"))
        or not np.all(available_at <= dataset.timestamps[:, None])
        or not np.array_equal(dataset.close, expected_close)
        or not np.array_equal(dataset.open, expected_open)
        or not np.array_equal(dataset.high, np.maximum(expected_open, expected_close))
        or not np.array_equal(dataset.low, np.minimum(expected_open, expected_close))
    ):
        return False
    for name in (
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "borrow_rate",
        "dividend",
    ):
        if not np.all(dataset.resolved_array(name) == 0.0):
            return False
    return bool(
        np.all(dataset.funding_rate == 0.0)
        and not np.any(dataset.resolved_array("funding_due"))
        and np.all(dataset.resolved_array("cash_rate") == 0.0)
        and np.all(dataset.resolved_array("max_participation_rate") == 1.0)
        and np.all(dataset.tradable)
        and np.all(dataset.volume > 0.0)
        and np.all(dataset.mark_price == dataset.close)
    )


def run_prefit_oracle() -> dict[str, Any]:
    """Run fixed-action accounting controls before any PPO fit is attempted."""

    dataset = build_fixture_dataset()
    feature_checks = _fixture_inputs_are_valid(dataset)
    initial_a, _ = _make_environment(dataset, 0).reset(seed=FIXED_SEED)
    initial_b, _ = _make_environment(dataset, 1).reset(seed=FIXED_SEED)
    observation_check = np.array_equal(
        initial_a[:3], (1.0, 1.0, 0.0)
    ) and np.array_equal(initial_b[:3], (0.0, 1.0, 0.0))

    long_a = _run_episode(
        _make_environment(dataset, 0),
        lambda env, observation: 2,
    )
    flat_b = _run_episode(
        _make_environment(dataset, 1),
        lambda env, observation: 1,
    )
    round_trip_env = _make_environment(dataset, 1)
    flat_after_fill = False

    def round_trip_action(env: PPOTradingEnv, observation: np.ndarray) -> int:
        nonlocal flat_after_fill
        if env.book.quantities[env.active_symbol_index] != 0.0:
            flat_after_fill = True
        return 1 if flat_after_fill else 2

    # The first decision that observes a fill requests FLAT; subsequent
    # decisions keep the flat target while the delayed close executes.
    round_trip = _run_episode(round_trip_env, round_trip_action)

    terminal_price = 100.0 * (1.01**8)
    expected_entry_fee = 5.0 * 100.0 * FEE_RATE
    expected_exit_fee = 5.0 * terminal_price * FEE_RATE
    expected_a_cash = (
        INITIAL_CAPITAL
        - 5.0 * 100.0
        - expected_entry_fee
        + 5.0 * terminal_price
        - expected_exit_fee
    )
    trace_a = long_a["trace"]
    first_fee_index = next(
        (
            int(item["decision_index"])
            for item in trace_a
            if float(item["interval_cost"]) > 0.0
        ),
        N_BARS,
    )
    first_positive_return_index = next(
        (
            int(item["decision_index"])
            for item in trace_a
            if float(item["interval_net_return"]) > 0.0
        ),
        N_BARS,
    )

    checks = {
        "training_fixture_is_causal_and_cost_overlays_are_zero": (
            feature_checks and observation_check and dataset.identity_verified
        ),
        "delayed_long_cash_and_fees_match_reference": (
            _close(float(long_a["cash"]), expected_a_cash)
            and _close(
                float(long_a["total_cost"]), expected_entry_fee + expected_exit_fee
            )
            and int(long_a["fill_count"]) == 2
            and _close(
                float(long_a["reward_sum"]), math.log(expected_a_cash / INITIAL_CAPITAL)
            )
        ),
        "flat_control_has_no_turnover_or_fees": (
            _close(float(flat_b["cash"]), INITIAL_CAPITAL)
            and _close(float(flat_b["total_cost"]), 0.0)
            and int(flat_b["fill_count"]) == 0
            and _close(float(flat_b["reward_sum"]), 0.0)
        ),
        "fee_only_round_trip_loses_both_fees": (
            _close(float(round_trip["cash"]), 995.0)
            and _close(float(round_trip["total_cost"]), 5.0)
            and int(round_trip["fill_count"]) == 2
            and _close(float(round_trip["reward_sum"]), math.log(0.995))
        ),
        "entry_fee_precedes_first_positive_return": first_fee_index
        < first_positive_return_index,
        "flat_control_has_no_price_reward": all(
            _close(float(item["interval_net_return"]), 0.0) for item in flat_b["trace"]
        ),
        "terminal_positions_and_orders_are_flat": all(
            float(run["quantity"]) == 0.0 and int(run["active_order_count"]) == 0
            for run in (long_a, flat_b, round_trip)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": PREFIT_SCHEMA,
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "metrics": {
            "delayed_long_final_cash": float(long_a["cash"]),
            "delayed_long_total_cost": float(long_a["total_cost"]),
            "delayed_long_fill_count": int(long_a["fill_count"]),
            "delayed_long_reward_sum": float(long_a["reward_sum"]),
            "flat_final_cash": float(flat_b["cash"]),
            "flat_total_cost": float(flat_b["total_cost"]),
            "flat_fill_count": int(flat_b["fill_count"]),
            "fee_only_round_trip_final_cash": float(round_trip["cash"]),
            "fee_only_round_trip_total_cost": float(round_trip["total_cost"]),
            "fee_only_round_trip_fill_count": int(round_trip["fill_count"]),
            "first_fee_decision_index": first_fee_index,
            "first_positive_return_decision_index": first_positive_return_index,
            "reference_delayed_long_final_cash": expected_a_cash,
            "reference_delayed_long_entry_fee": expected_entry_fee,
            "reference_delayed_long_exit_fee": expected_exit_fee,
        },
    }


def _recipe_payload() -> dict[str, object]:
    return {
        "fixture": {
            "bars": N_BARS,
            "cadence": "hourly_continuous",
            "training_gain_rows": list(range(8, 16)),
            "heldout_gain_rows": list(range(12, 20)),
            "gain_per_row": 0.01,
            "flat_price": 100.0,
            "feature_one_role": "A",
            "feature_zero_role": "B",
            "feature_assignment": "literal_constant_from_row_zero",
            "heldout_columns_swapped_and_renamed": True,
        },
        "training": {
            "feature_indices": [0],
            "fit_symbol_indices": [0, 1],
            "start_index": START_INDEX,
            "stop_index": STOP_INDEX,
            "training_layout": PPO_TRAINING_LAYOUT_INTERLEAVED,
            "rollout_steps_per_env": ROLLOUT_STEPS_PER_ENV,
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": FIXED_SEED,
            "gross_budget": GROSS_BUDGET,
            "initial_capital": INITIAL_CAPITAL,
            "normalize_features": False,
            "minimum_hold_bars": 0,
            "observation_schema": PPO_OBSERVATION_SCHEMA_V3,
            "settle_terminal_position": True,
        },
        "costs": asdict(fixed_execution_cost()),
        "risk_config_argument": None,
        "effective_default_risk_config": asdict(
            PreTradeRisk.default_for_execution(
                max_leverage=fixed_execution_cost().max_leverage
            ).config
        ),
    }


def _ppo_default_snapshot() -> dict[str, object]:
    return {
        "n_steps_per_env": ROLLOUT_STEPS_PER_ENV,
        "batch_size": ppo_module._PPO_BATCH_SIZE,
        "gamma": ppo_module._PPO_GAMMA,
        "gae_lambda": ppo_module._PPO_GAE_LAMBDA,
        "network": {
            "policy": "MlpPolicy",
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": "torch.nn.Tanh",
            "ortho_init": True,
            "features_extractor_class": "stable_baselines3.common.torch_layers.FlattenExtractor",
            "share_features_extractor": True,
            "optimizer_class": "torch.optim.Adam",
            "optimizer_kwargs": {"eps": 1e-5},
            "device": "cpu",
        },
    }


def _runtime_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for package in _RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(
                f"required PPO runtime package is missing: {package}"
            ) from None
    return versions


def _loaded_source_origins() -> dict[str, str]:
    package_root = _REPO_ROOT / "trade_rl"
    package_file = package_root / "__init__.py"
    ppo_file = package_root / "strategies" / "rl" / "ppo.py"
    try:
        resolved_root = package_root.resolve(strict=True)
        resolved_package = Path(trade_rl.__file__).resolve(strict=True)
        resolved_ppo = Path(ppo_module.__file__).resolve(strict=True)
    except (OSError, RuntimeError, TypeError):
        raise RuntimeError("loaded PPO source origin is unavailable") from None
    if (
        resolved_root != package_root
        or resolved_package != package_file
        or resolved_ppo != ppo_file
    ):
        raise RuntimeError("loaded PPO source origin differs from this checkout")
    return {
        "trade_rl": str(resolved_package),
        "ppo_module": str(resolved_ppo),
    }


def _source_manifest() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", "trade_rl", "pyproject.toml", "uv.lock"],
            cwd=_REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError(
            "tracked production-source inventory is unavailable"
        ) from None
    tracked_paths = {
        os.fsdecode(raw_path) for raw_path in result.stdout.split(b"\0") if raw_path
    }
    relative_paths = {
        relative
        for relative in tracked_paths
        if relative.startswith("trade_rl/") and relative.endswith(".py")
    }
    required_inputs = {"pyproject.toml", "uv.lock"}
    if not required_inputs.issubset(tracked_paths):
        raise RuntimeError("dependency inputs are missing from the tracked source set")
    relative_paths.update(required_inputs)
    relative_paths.add("tools/ppo_conditional_diagnostic.py")

    manifest: dict[str, str] = {}
    for relative in sorted(relative_paths):
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError("tracked source inventory contains an unsafe path")
        path = _REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"source identity is unavailable for {relative}")
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _protocol_payload(
    *,
    prefit_oracle: dict[str, object],
    runtime_versions: dict[str, str],
) -> dict[str, object]:
    training = build_fixture_dataset()
    heldout = build_fixture_dataset(heldout=True)
    recipe = _recipe_payload()
    source_manifest = _source_manifest()
    loaded_source_origins = _loaded_source_origins()
    expected_transitions = expected_ppo_realized_timesteps(
        TOTAL_TIMESTEPS,
        training_layout=PPO_TRAINING_LAYOUT_INTERLEAVED,
        rollout_steps_per_env=ROLLOUT_STEPS_PER_ENV,
        n_envs=2,
    )
    if expected_transitions != TOTAL_TIMESTEPS:
        raise RuntimeError("frozen PPO budget no longer resolves to 8192 transitions")
    return {
        "schema": PROTOCOL_SCHEMA,
        "scope": {
            "kind": "synthetic_software_diagnostic_only",
            "learnability_ci_gate": False,
            "p3_status": P3_STATUS,
            "research_authorization": False,
            "production_or_live_authorization": False,
            "shared_cash_claim": False,
        },
        "recipe": recipe,
        "recipe_digest": content_digest(recipe),
        "expected_global_transitions": expected_transitions,
        "ppo_defaults": _ppo_default_snapshot(),
        "runtime_versions": runtime_versions,
        "source_manifest": source_manifest,
        "source_digest": content_digest(source_manifest),
        "loaded_source_origins": loaded_source_origins,
        "dataset_identity": {
            "training": training.dataset_id,
            "heldout": heldout.dataset_id,
        },
        "prefit_oracle_digest": content_digest(prefit_oracle),
    }


def _write_exclusive_json(path: Path, value: object) -> None:
    raw = canonical_json_bytes(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _read_canonical_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required diagnostic file is missing: {path.name}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"diagnostic JSON is invalid: {path.name}") from None
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"diagnostic JSON is not canonical: {path.name}")
    return value


def _verified_policy_bundle(
    root: Path,
    *,
    expected_digest: str,
) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("published PPO inference bundle is unavailable")
    members = {path.name for path in root.iterdir()}
    if members != {"manifest.json", "policy.zip"}:
        raise ValueError("PPO inference bundle file closure is invalid")
    manifest_path = root / "manifest.json"
    policy_path = root / "policy.zip"
    if manifest_path.is_symlink() or policy_path.is_symlink():
        raise ValueError("PPO inference bundle contains a symbolic link")
    manifest = _read_canonical_json(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    policy_bytes = policy_path.read_bytes()
    manifest_digest = content_digest(manifest)
    policy_digest = hashlib.sha256(policy_bytes).hexdigest()
    if manifest_digest != expected_digest:
        raise ValueError("PPO inference bundle digest differs from the saver result")
    if manifest.get("policy_sha256") != policy_digest:
        raise ValueError("PPO policy bytes differ from the inference manifest")
    return {
        "directory": "inference-bundle",
        "schema": manifest.get("schema"),
        "manifest_digest": manifest_digest,
        "manifest_file": "inference-bundle/manifest.json",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_size_bytes": len(manifest_bytes),
        "policy_file": "inference-bundle/policy.zip",
        "policy_sha256": policy_digest,
        "policy_size_bytes": len(policy_bytes),
    }


def _final_verified_file_bytes(
    path: Path,
    *,
    expected_sha256: object,
    expected_size_bytes: object,
    field: str,
) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"final diagnostic artifact is unavailable: {field}")
    raw = path.read_bytes()
    if (
        not isinstance(expected_sha256, str)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
        or isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or len(raw) != expected_size_bytes
    ):
        raise ValueError(f"final diagnostic artifact differs from readback: {field}")
    return raw


def _final_verified_json(
    path: Path,
    *,
    metadata: dict[str, object],
    field: str,
) -> dict[str, Any]:
    raw = _final_verified_file_bytes(
        path,
        expected_sha256=metadata.get("sha256"),
        expected_size_bytes=metadata.get("size_bytes"),
        field=field,
    )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"final diagnostic JSON is invalid: {field}") from None
    if (
        not isinstance(value, dict)
        or canonical_json_bytes(value) != raw
        or content_digest(value) != metadata.get("digest")
    ):
        raise ValueError(f"final diagnostic JSON differs from readback: {field}")
    return value


def _final_verified_policy_bundle(
    root: Path,
    *,
    metadata: dict[str, object],
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("final PPO inference bundle is unavailable")
    members = {path.name for path in root.iterdir()}
    if members != {"manifest.json", "policy.zip"}:
        raise ValueError("final PPO inference bundle file closure differs")
    manifest = _final_verified_json(
        root / "manifest.json",
        metadata={
            "digest": metadata.get("manifest_digest"),
            "sha256": metadata.get("manifest_sha256"),
            "size_bytes": metadata.get("manifest_size_bytes"),
        },
        field="inference-bundle/manifest.json",
    )
    policy_bytes = _final_verified_file_bytes(
        root / "policy.zip",
        expected_sha256=metadata.get("policy_sha256"),
        expected_size_bytes=metadata.get("policy_size_bytes"),
        field="inference-bundle/policy.zip",
    )
    if (
        manifest.get("schema") != metadata.get("schema")
        or manifest.get("policy_sha256") != hashlib.sha256(policy_bytes).hexdigest()
    ):
        raise ValueError("final PPO inference bundle differs from readback")


def _write_verified_json_artifact(
    path: Path,
    payload: dict[str, Any],
) -> dict[str, object]:
    _write_exclusive_json(path, payload)
    readback = _read_canonical_json(path)
    raw = path.read_bytes()
    if readback != payload or raw != canonical_json_bytes(payload):
        raise ValueError(f"diagnostic artifact readback differs: {path.name}")
    return {
        "file": path.name,
        "digest": content_digest(readback),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _artifact_file_inventory(output: Path) -> dict[str, dict[str, object]]:
    relative_paths = (
        "inference-bundle/manifest.json",
        "inference-bundle/policy.zip",
        "heldout-trace.json",
        "evidence-manifest.json",
        "result.pending.json",
        "result.json",
    )
    inventory: dict[str, dict[str, object]] = {}
    bundle_root = output / "inference-bundle"
    if bundle_root.is_symlink():
        inventory["inference-bundle"] = {"status": "symbolic_link"}
        return inventory
    for relative in relative_paths:
        path = output.joinpath(*relative.split("/"))
        try:
            if path.is_symlink():
                inventory[relative] = {"status": "symbolic_link"}
                continue
            if not path.exists():
                continue
            if not path.is_file():
                inventory[relative] = {"status": "not_file"}
                continue
            raw = path.read_bytes()
        except OSError as error:
            inventory[relative] = {
                "status": "unavailable",
                "error_type": type(error).__name__,
            }
            continue
        inventory[relative] = {
            "status": "present",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    return inventory


def _publish_completed_result(output: Path, result: dict[str, object]) -> None:
    pending = output / "result.pending.json"
    destination = output / "result.json"
    if any(path.exists() or path.is_symlink() for path in (pending, destination)):
        raise FileExistsError("diagnostic result destination already exists")
    _write_exclusive_json(pending, result)
    readback = _read_canonical_json(pending)
    raw = pending.read_bytes()
    if readback != result or raw != canonical_json_bytes(result):
        raise ValueError("completed diagnostic result failed readback verification")
    os.rename(pending, destination)


def _assert_source_identity(protocol: dict[str, Any], *, stage: str) -> None:
    manifest = _source_manifest()
    runtime_versions = _runtime_versions()
    loaded_source_origins = _loaded_source_origins()
    if manifest != protocol.get("source_manifest") or content_digest(
        manifest
    ) != protocol.get("source_digest"):
        raise RuntimeError(f"tracked source identity changed at {stage}")
    if runtime_versions != protocol.get("runtime_versions"):
        raise RuntimeError(f"PPO runtime identity changed at {stage}")
    if loaded_source_origins != protocol.get("loaded_source_origins"):
        raise RuntimeError(f"loaded PPO source origin changed at {stage}")


def _new_output_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError("diagnostic output directory already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def prepare(output_dir: str | Path) -> dict[str, object]:
    """Write the frozen protocol and manual pre-fit oracle, without training."""

    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("diagnostic output directory already exists")
    prefit = run_prefit_oracle()
    if not prefit["passed"]:
        raise ValueError("fixed-action pre-fit oracle failed; no fit was attempted")
    payload = _protocol_payload(
        prefit_oracle=prefit,
        runtime_versions=_runtime_versions(),
    )
    protocol = {**payload, "protocol_digest": content_digest(payload)}
    output = _new_output_directory(output)
    _write_exclusive_json(output / "protocol.json", protocol)
    _write_exclusive_json(output / "prefit-oracle.json", prefit)
    return {
        "status": "prepared",
        "protocol_digest": protocol["protocol_digest"],
        "prefit_passed": True,
    }


def _validated_protocol(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_canonical_json(output / "protocol.json")
    digest = protocol.get("protocol_digest")
    payload = {
        key: value for key, value in protocol.items() if key != "protocol_digest"
    }
    if protocol.get("schema") != PROTOCOL_SCHEMA or digest != content_digest(payload):
        raise ValueError("diagnostic protocol identity is invalid")
    expected_recipe = _recipe_payload()
    canonical_recipe = json.loads(canonical_json_bytes(expected_recipe))
    if protocol.get("recipe") != canonical_recipe:
        raise ValueError("diagnostic recipe differs from the frozen recipe")
    if protocol.get("recipe_digest") != content_digest(expected_recipe):
        raise ValueError("diagnostic recipe digest is invalid")
    _assert_source_identity(protocol, stage="pre-fit validation")
    current_datasets = {
        "training": build_fixture_dataset().dataset_id,
        "heldout": build_fixture_dataset(heldout=True).dataset_id,
    }
    if protocol.get("dataset_identity") != current_datasets:
        raise ValueError("synthetic dataset identity changed after preparation")
    prefit = _read_canonical_json(output / "prefit-oracle.json")
    if (
        prefit.get("schema") != PREFIT_SCHEMA
        or prefit.get("passed") is not True
        or protocol.get("prefit_oracle_digest") != content_digest(prefit)
    ):
        raise ValueError("pre-fit oracle identity or result is invalid")
    return protocol, prefit


def _heldout_episode(
    policy: _PredictPolicy,
    dataset: MarketDataset,
    symbol_index: int,
) -> _Episode:
    execution_observations: list[dict[str, Any]] = []
    env = _make_environment(
        dataset,
        symbol_index,
        execution_observations=execution_observations,
    )

    def action_for_step(_env: PPOTradingEnv, observation: np.ndarray) -> object:
        action, _ = policy.predict(observation, deterministic=True)
        return action

    return _run_episode(
        env,
        action_for_step,
        capture_evidence=True,
        execution_observations=execution_observations,
    )


def _heldout_clauses(
    policy: _PredictPolicy,
    dataset: MarketDataset,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    feature_one = int(np.flatnonzero(dataset.features[0, :, 0] == 1.0)[0])
    feature_zero = int(np.flatnonzero(dataset.features[0, :, 0] == 0.0)[0])
    episode_one = _heldout_episode(policy, dataset, feature_one)
    episode_zero = _heldout_episode(policy, dataset, feature_zero)
    trace_one = episode_one["trace"]
    trace_zero = episode_zero["trace"]
    first_one_action = int(trace_one[0]["action"])
    first_zero_action = int(trace_zero[0]["action"])
    positive_rows = [
        int(item["resulting_index"])
        for item in trace_one
        if float(item["quantity_after"]) > 0.0
    ]
    first_positive_row = min(positive_rows, default=N_BARS)
    quantities_through_move = [
        float(item["quantity_after"])
        for item in trace_one
        if first_positive_row <= int(item["resulting_index"]) <= 19
    ]
    exposure_through_move = (
        first_positive_row < 12
        and bool(quantities_through_move)
        and all(quantity > 0.0 for quantity in quantities_through_move)
        and max(int(item["resulting_index"]) for item in trace_one) >= 19
    )
    clauses = {
        "feature_one_initial_long": first_one_action == 2,
        "feature_zero_initial_flat": first_zero_action == 1,
        "feature_one_filled_before_heldout_move": first_positive_row < 12,
        "feature_one_exposure_through_heldout_move": exposure_through_move,
        "feature_zero_never_fills": (
            int(episode_zero["fill_count"]) == 0
            and float(episode_zero["quantity"]) == 0.0
        ),
        "feature_one_terminal_cash_above_initial": float(episode_one["cash"])
        > INITIAL_CAPITAL,
        "feature_zero_terminal_cash_unchanged": _close(
            float(episode_zero["cash"]), INITIAL_CAPITAL
        ),
        "both_heldout_accounts_terminal_flat": (
            float(episode_one["quantity"]) == 0.0
            and float(episode_zero["quantity"]) == 0.0
        ),
        "both_heldout_accounts_have_no_active_orders": (
            int(episode_one["active_order_count"]) == 0
            and int(episode_zero["active_order_count"]) == 0
        ),
    }
    metrics = {
        "feature_one_symbol_index": feature_one,
        "feature_zero_symbol_index": feature_zero,
        "feature_one_first_action": first_one_action,
        "feature_one_first_intent": str(trace_one[0]["intent"]),
        "feature_zero_first_action": first_zero_action,
        "feature_zero_first_intent": str(trace_zero[0]["intent"]),
        "feature_one_first_positive_quantity_row": first_positive_row,
        "feature_one_final_cash": float(episode_one["cash"]),
        "feature_one_total_cost": float(episode_one["total_cost"]),
        "feature_one_fill_count": int(episode_one["fill_count"]),
        "feature_zero_final_cash": float(episode_zero["cash"]),
        "feature_zero_total_cost": float(episode_zero["total_cost"]),
        "feature_zero_fill_count": int(episode_zero["fill_count"]),
    }
    episodes = {
        "episodes": [
            {
                "role": "feature_one",
                "symbol_index": feature_one,
                "symbol": dataset.symbols[feature_one],
                "final_book_state": episode_one["final_book_state"],
                "steps": episode_one["evidence_trace"],
            },
            {
                "role": "feature_zero",
                "symbol_index": feature_zero,
                "symbol": dataset.symbols[feature_zero],
                "final_book_state": episode_zero["final_book_state"],
                "steps": episode_zero["evidence_trace"],
            },
        ]
    }
    return clauses, metrics, episodes


def run_diagnostic(output_dir: str | Path) -> dict[str, Any]:
    """Run exactly one fit in an already prepared, one-shot local directory."""

    output = Path(output_dir)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("prepared diagnostic directory is missing")
    start_path = output / "attempt-start.json"
    result_path = output / "result.json"
    failure_path = output / "failure.json"
    pending_result_path = output / "result.pending.json"
    inference_bundle_path = output / "inference-bundle"
    heldout_trace_path = output / "heldout-trace.json"
    evidence_manifest_path = output / "evidence-manifest.json"
    if any(
        path.exists() or path.is_symlink()
        for path in (
            start_path,
            result_path,
            failure_path,
            pending_result_path,
            inference_bundle_path,
            heldout_trace_path,
            evidence_manifest_path,
        )
    ):
        raise FileExistsError("diagnostic directory already contains an attempt")
    protocol, prefit = _validated_protocol(output)
    attempt_started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    start_payload = {
        "schema": ATTEMPT_SCHEMA,
        "protocol_digest": protocol["protocol_digest"],
        "source_digest": protocol["source_digest"],
        "started_at": attempt_started_at,
        "fit_call_budget": 1,
        "p3_status": P3_STATUS,
        "scope": "synthetic_software_diagnostic_only",
    }
    _write_exclusive_json(start_path, start_payload)

    fit_call_count = 0
    source_identity_checks = {
        "before_fit": False,
        "after_fit": False,
        "after_bundle": False,
        "after_heldout": False,
    }
    artifact_statuses = {
        "inference_bundle": "not_started",
        "heldout_trace": "not_started",
        "evidence_manifest": "not_started",
        "result": "not_started",
    }
    active_artifact: str | None = None
    execution_phase = "before_fit"
    try:
        training_dataset = build_fixture_dataset()
        execution_cost = fixed_execution_cost()
        if training_dataset.dataset_id != protocol["dataset_identity"]["training"]:
            raise RuntimeError("synthetic training dataset identity changed")
        _assert_source_identity(protocol, stage="before_fit")
        source_identity_checks["before_fit"] = True
        execution_phase = "fit"
        fit_call_count = 1
        strategy = fit_ppo_strategy(
            training_dataset,
            feature_indices=(0,),
            fit_symbol_indices=(0, 1),
            start_index=START_INDEX,
            stop_index=STOP_INDEX,
            training_layout=PPO_TRAINING_LAYOUT_INTERLEAVED,
            rollout_steps_per_env=ROLLOUT_STEPS_PER_ENV,
            total_timesteps=TOTAL_TIMESTEPS,
            seed=FIXED_SEED,
            gross_budget=GROSS_BUDGET,
            initial_capital=INITIAL_CAPITAL,
            execution_cost=execution_cost,
            normalize_features=False,
            minimum_hold_bars=0,
            observation_schema=PPO_OBSERVATION_SCHEMA_V3,
            settle_terminal_position=True,
        )
        execution_phase = "after_fit"
        _assert_source_identity(protocol, stage="after_fit")
        source_identity_checks["after_fit"] = True

        policy_value = getattr(strategy, "policy", None)
        if policy_value is None or not callable(getattr(policy_value, "predict", None)):
            raise TypeError("fitted PPO strategy does not expose its policy")
        policy = cast(_PredictPolicy, policy_value)
        actual_timesteps = getattr(policy_value, "num_timesteps", None)
        if (
            isinstance(actual_timesteps, bool)
            or not isinstance(actual_timesteps, Integral)
            or int(actual_timesteps) != protocol["expected_global_transitions"]
        ):
            raise RuntimeError(
                "fitted PPO realized an unexpected global transition count"
            )
        actual_timesteps = int(actual_timesteps)

        execution_phase = "policy_bundle"
        active_artifact = "inference_bundle"
        artifact_statuses[active_artifact] = "writing"
        feed_feature_names = tuple(training_dataset.feature_names)
        bundle_digest = save_ppo_inference_bundle(
            inference_bundle_path,
            strategy,
            feature_names=feed_feature_names,
        )
        policy_bundle = _verified_policy_bundle(
            inference_bundle_path,
            expected_digest=bundle_digest,
        )
        artifact_statuses[active_artifact] = "verified"
        active_artifact = None

        execution_phase = "policy_reload"
        active_artifact = "inference_bundle"
        artifact_statuses[active_artifact] = "verifying"
        loaded_strategy = load_ppo_inference_bundle(
            inference_bundle_path,
            expected_digest=bundle_digest,
            feature_names=feed_feature_names,
        )
        if (
            tuple(loaded_strategy.feature_indices) != tuple(strategy.feature_indices)
            or tuple(loaded_strategy.feature_names or ())
            != tuple(strategy.feature_names or ())
            or loaded_strategy.observation_schema != strategy.observation_schema
            or loaded_strategy.minimum_hold_bars != strategy.minimum_hold_bars
        ):
            raise ValueError("reloaded PPO strategy differs from the fitted contract")
        loaded_policy = getattr(loaded_strategy, "policy", None)
        if loaded_policy is None or not callable(
            getattr(loaded_policy, "predict", None)
        ):
            raise TypeError("reloaded PPO strategy does not expose its policy")
        policy = cast(_PredictPolicy, loaded_policy)
        artifact_statuses[active_artifact] = "verified"
        active_artifact = None

        execution_phase = "after_bundle"
        _assert_source_identity(protocol, stage="after_bundle")
        source_identity_checks["after_bundle"] = True

        heldout_dataset = build_fixture_dataset(heldout=True)
        if heldout_dataset.dataset_id != protocol["dataset_identity"]["heldout"]:
            raise RuntimeError("synthetic heldout dataset identity changed")
        execution_phase = "heldout"
        clauses, metrics, episodes = _heldout_clauses(policy, heldout_dataset)

        heldout_trace_payload: dict[str, Any] = {
            "schema": HELDOUT_TRACE_SCHEMA,
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "protocol_digest": protocol["protocol_digest"],
            "source_digest": protocol["source_digest"],
            "recipe_digest": protocol["recipe_digest"],
            "runtime_versions": protocol["runtime_versions"],
            "heldout_dataset_id": heldout_dataset.dataset_id,
            "inference_mode": "deterministic",
            "episodes": episodes["episodes"],
        }
        execution_phase = "heldout_trace"
        active_artifact = "heldout_trace"
        artifact_statuses[active_artifact] = "writing"
        trace_metadata = _write_verified_json_artifact(
            heldout_trace_path,
            heldout_trace_payload,
        )
        artifact_statuses[active_artifact] = "verified"
        active_artifact = None

        execution_phase = "after_heldout"
        _assert_source_identity(protocol, stage="after_heldout")
        source_identity_checks["after_heldout"] = True

        evidence_manifest_payload: dict[str, Any] = {
            "schema": EVIDENCE_MANIFEST_SCHEMA,
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "protocol_digest": protocol["protocol_digest"],
            "source_digest": protocol["source_digest"],
            "recipe_digest": protocol["recipe_digest"],
            "runtime_versions": protocol["runtime_versions"],
            "dataset_identity": protocol["dataset_identity"],
            "policy_bundle": policy_bundle,
            "trace": {
                "file": trace_metadata["file"],
                "digest": trace_metadata["digest"],
                "sha256": trace_metadata["sha256"],
                "size_bytes": trace_metadata["size_bytes"],
            },
        }
        execution_phase = "evidence_manifest"
        active_artifact = "evidence_manifest"
        artifact_statuses[active_artifact] = "writing"
        evidence_manifest_metadata = _write_verified_json_artifact(
            evidence_manifest_path,
            evidence_manifest_payload,
        )
        artifact_statuses[active_artifact] = "verified"
        active_artifact = None

        failed = [name for name, passed in clauses.items() if not passed]
        result: dict[str, object] = {
            "schema": RESULT_SCHEMA,
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "status": "completed",
            "diagnostic_verdict": "pass" if not failed else "fail",
            "scope": "synthetic_software_diagnostic_only",
            "p3_status": P3_STATUS,
            "protocol_digest": protocol["protocol_digest"],
            "prefit_oracle_digest": protocol["prefit_oracle_digest"],
            "source_identity_checks": source_identity_checks,
            "fit_call_count": fit_call_count,
            "actual_fit_timesteps": actual_timesteps,
            "clauses": clauses,
            "failed_clauses": failed,
            "policy_bundle_manifest_digest": policy_bundle["manifest_digest"],
            "heldout_trace_digest": trace_metadata["digest"],
            "heldout_trace_sha256": trace_metadata["sha256"],
            "evidence_manifest_digest": evidence_manifest_metadata["digest"],
            "evidence_manifest_sha256": evidence_manifest_metadata["sha256"],
            "evidence_manifest_size_bytes": evidence_manifest_metadata["size_bytes"],
            "prefit_metrics": prefit["metrics"],
            "heldout_metrics": metrics,
            "started_at": attempt_started_at,
            "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        execution_phase = "final_artifact_closure"
        active_artifact = "inference_bundle"
        artifact_statuses[active_artifact] = "verifying"
        _final_verified_policy_bundle(
            inference_bundle_path,
            metadata=policy_bundle,
        )
        artifact_statuses[active_artifact] = "verified"

        active_artifact = "heldout_trace"
        artifact_statuses[active_artifact] = "verifying"
        _final_verified_json(
            heldout_trace_path,
            metadata=trace_metadata,
            field="heldout-trace.json",
        )
        artifact_statuses[active_artifact] = "verified"

        active_artifact = "evidence_manifest"
        artifact_statuses[active_artifact] = "verifying"
        _final_verified_json(
            evidence_manifest_path,
            metadata=evidence_manifest_metadata,
            field="evidence-manifest.json",
        )
        artifact_statuses[active_artifact] = "verified"

        execution_phase = "result"
        active_artifact = "result"
        artifact_statuses[active_artifact] = "writing"
        _publish_completed_result(output, result)
        artifact_statuses[active_artifact] = "published"
        active_artifact = None
        return result
    except Exception as error:
        if active_artifact is not None:
            artifact_statuses[active_artifact] = "failed"
        message = " ".join(str(error).split())[:400]
        failure: dict[str, object] = {
            "schema": FAILURE_SCHEMA,
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "status": "execution_failed",
            "scope": "synthetic_software_diagnostic_only",
            "p3_status": P3_STATUS,
            "protocol_digest": protocol["protocol_digest"],
            "execution_phase": execution_phase,
            "source_identity_stage": (
                execution_phase if execution_phase in source_identity_checks else None
            ),
            "source_identity_checks": source_identity_checks,
            "fit_call_count": fit_call_count,
            "artifact_statuses": artifact_statuses,
            "artifact_files": _artifact_file_inventory(output),
            "exception_type": type(error).__name__,
            "message": message,
            "started_at": attempt_started_at,
            "failed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        _write_exclusive_json(failure_path, failure)
        return failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or run a one-shot local synthetic PPO conditional-action "
            "software diagnostic. It is not market research or a CI learnability gate."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="write the frozen recipe and pre-fit accounting oracle only"
    )
    prepare_parser.add_argument("output_dir", type=Path)
    run_parser = commands.add_parser(
        "run", help="perform one fit in an already prepared output directory"
    )
    run_parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            report = prepare(args.output_dir)
        else:
            report = run_diagnostic(args.output_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report.get("status") == "prepared":
            return 0
        if report.get("status") == "completed":
            return 0 if report.get("diagnostic_verdict") == "pass" else 3
        return 1 if report.get("status") == "execution_failed" else 2
    except Exception as error:
        print(
            f"conditional PPO diagnostic: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
