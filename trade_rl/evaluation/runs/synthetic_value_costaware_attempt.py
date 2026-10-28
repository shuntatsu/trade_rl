"""Synthetic-only four-arm cost-aware comparison built over the immutable v1 attempt.

The v1 three-arm artifact/readers are not changed. A nested v1 complete attempt
is compared to one additional frozen-Ridge cost-aware arm on the same canonical
native portfolio replay. No empirical forecast fit or Study is authorized.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.runs.synthetic_value_attempt import (
    SyntheticValueAttemptConfig,
    _arm_row,
    _preflight,
    _read_record,
    _source_fingerprint,
    _write_record,
    load_synthetic_value_attempt,
    run_synthetic_value_attempt,
)
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.decision_value import LinearDecisionValueModel
from trade_rl.strategies.forecasts.controller import (
    CostAwareForecastIntentController,
    ForecastIntentConfig,
)
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.position_intent import PositionIntent

_FAMILIES = ("cash", "fixed_long", "linear_value", "cost_aware_ridge")
_STARTED = "synthetic_cost_aware_attempt_started_v2"
_ARM = "synthetic_cost_aware_attempt_arm_v2"
_RESULT = "synthetic_cost_aware_attempt_result_v2"
_COMPLETE = "synthetic_cost_aware_attempt_complete_v2"
_FAILURE = "synthetic_cost_aware_attempt_failure_v2"
_FORECAST = "synthetic_ridge_forecast_fixture_v1"


@dataclass(frozen=True, slots=True)
class SyntheticCostAwareAttemptConfig:
    """Explicitly frozen artificial forecast and corresponding native experiment."""

    value_attempt: SyntheticValueAttemptConfig
    forecast_model: RidgeForecastModel
    expected_forecast_sha256: str
    entry_threshold: float
    exit_threshold: float
    one_way_switch_cost: float


@dataclass(frozen=True, slots=True)
class PublishedSyntheticCostAwareAttempt:
    root: Path
    result_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedSyntheticCostAwareAttempt:
    root: Path
    result_sha256: str
    forecast_sha256: str
    dataset_id: str
    arms: tuple[dict[str, object], ...]


def _ridge_payload(model: RidgeForecastModel) -> dict[str, object]:
    if type(model) is not RidgeForecastModel:
        raise TypeError("forecast_model must be a RidgeForecastModel")
    return {
        "schema_version": _FORECAST,
        "feature_indices": model.feature_indices,
        "feature_mean_hex": tuple(float(v).hex() for v in model.feature_mean),
        "feature_scale_hex": tuple(float(v).hex() for v in model.feature_scale),
        "coefficients_hex": tuple(float(v).hex() for v in model.coefficients),
        "intercept_hex": float(model.intercept).hex(),
        "horizon_hours": model.horizon_hours,
        "alpha_hex": float(model.alpha).hex(),
        "n_samples": model.n_samples,
        "fit_cutoff_ns": int(model.fit_cutoff.astype(np.int64)),
    }


def synthetic_ridge_fixture_digest(model: RidgeForecastModel) -> str:
    """Canonical digest of one *caller-declared* artificial frozen forecast."""
    return hashlib.sha256(canonical_json_bytes(_ridge_payload(model))).hexdigest()


def _decode_hex(value: object, *, name: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{name} must be a canonical finite float.hex")
    try:
        decoded = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical finite float.hex") from error
    if not math.isfinite(decoded) or decoded.hex() != value:
        raise ValueError(f"{name} must be a canonical finite float.hex")
    return decoded


def _decode_ridge_payload(payload: dict[str, object]) -> RidgeForecastModel:
    expected_fields = {
        "schema_version",
        "feature_indices",
        "feature_mean_hex",
        "feature_scale_hex",
        "coefficients_hex",
        "intercept_hex",
        "horizon_hours",
        "alpha_hex",
        "n_samples",
        "fit_cutoff_ns",
    }
    if set(payload) != expected_fields or payload.get("schema_version") != _FORECAST:
        raise ValueError("synthetic frozen forecast schema differs from v1")
    try:
        indices = payload["feature_indices"]
        if not isinstance(indices, list) or not all(type(i) is int for i in indices):
            raise ValueError("forecast feature_indices must be a list of integers")
        arrays: list[np.ndarray] = []
        for name in ("feature_mean_hex", "feature_scale_hex", "coefficients_hex"):
            values = payload[name]
            if not isinstance(values, list):
                raise ValueError(f"forecast {name} must be a list")
            arrays.append(
                np.asarray(
                    [_decode_hex(v, name=name) for v in values], dtype=np.float64
                )
            )
        cutoff = payload["fit_cutoff_ns"]
        if type(cutoff) is not int:
            raise ValueError("forecast fit_cutoff_ns must be an integer")
        horizon = payload["horizon_hours"]
        samples = payload["n_samples"]
        if type(horizon) is not int or type(samples) is not int:
            raise ValueError("forecast horizon and sample count must be integers")
        model = RidgeForecastModel(
            feature_indices=tuple(indices),
            feature_mean=arrays[0],
            feature_scale=arrays[1],
            coefficients=arrays[2],
            intercept=_decode_hex(payload["intercept_hex"], name="intercept_hex"),
            horizon_hours=horizon,
            alpha=_decode_hex(payload["alpha_hex"], name="alpha_hex"),
            n_samples=samples,
            fit_cutoff=np.datetime64(cutoff, "ns"),
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "synthetic frozen forecast has invalid typed values"
        ) from error
    if canonical_json_bytes(_ridge_payload(model)) != canonical_json_bytes(payload):
        raise ValueError("synthetic forecast has noncanonical typed values")
    return model


def _preflight_cost_aware(
    config: SyntheticCostAwareAttemptConfig,
    observer: Callable[[str], None] | None,
) -> tuple[LinearDecisionValueModel, str, dict[str, object]]:
    if type(config) is not SyntheticCostAwareAttemptConfig:
        raise TypeError("TD4b config must be SyntheticCostAwareAttemptConfig")
    base = config.value_attempt
    value_model, fingerprint, parent_started = _preflight(base, observer)
    require_sha256(config.expected_forecast_sha256, field="expected_forecast_sha256")
    if synthetic_ridge_fixture_digest(config.forecast_model) != (
        config.expected_forecast_sha256
    ):
        raise ValueError("forecast digest differs from the expected frozen fixture")
    model = config.forecast_model
    dataset = base.dataset
    if max(model.feature_indices) >= len(dataset.feature_names):
        raise ValueError("forecast feature index exceeds the Dataset roster")
    start_timestamp = dataset.timestamps[base.start_index]
    if model.fit_cutoff >= start_timestamp:
        raise ValueError(
            "forecast fit_cutoff cannot be from the future evaluation window"
        )
    thresholds = ForecastIntentConfig(
        entry_threshold=config.entry_threshold, exit_threshold=config.exit_threshold
    )
    CostAwareForecastIntentController(
        thresholds, one_way_switch_cost=config.one_way_switch_cost
    )
    cost: ExecutionCostConfig = base.execution_cost
    if (
        cost.fee_rate <= 0.0
        or cost.order_type != "market"
        or cost.order_latency_bars != 0
        or cost.impact_rate != 0.0
        or cost.maker_fee_rate != 0.0
        or cost.taker_fee_rate != 0.0
        or cost.multiplier != 1.0
        or cost.borrow_rate_multiplier != 0.0
        or not cost.allow_short
        or cost.max_participation_rate != 1.0
    ):
        raise ValueError("cost proxy v2 requires a simple deterministic market recipe")
    if not math.isclose(
        config.one_way_switch_cost,
        cost.fee_rate + cost.spread_rate,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("cost proxy differs from fee plus spread")
    for field in (
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "borrow_rate",
        "funding_rate",
        "dividend",
        "cash_rate",
    ):
        values = (
            dataset.funding_rate
            if field == "funding_rate"
            else (dataset.resolved_array(field))
        )
        if np.any(values != 0):
            raise ValueError(f"cost proxy v2 rejects unsupported Dataset {field}")
    return (
        value_model,
        fingerprint,
        {
            "schema_version": _STARTED,
            "source_scope": "synthetic_only",
            "dataset_id": base.expected_dataset_id,
            "source_fingerprint": fingerprint,
            "value_model_sha256": base.expected_model_sha256,
            "forecast_sha256": config.expected_forecast_sha256,
            "base_started_sha256": hashlib.sha256(
                canonical_json_bytes(parent_started)
            ).hexdigest(),
            "execution_policy_digest": cost.execution_policy_digest,
            "entry_threshold_hex": float(config.entry_threshold).hex(),
            "exit_threshold_hex": float(config.exit_threshold).hex(),
            "one_way_switch_cost_hex": float(config.one_way_switch_cost).hex(),
            "families": _FAMILIES,
        },
    )


class _ExactQuantityCostAwareRidgePolicy:
    """Portfolio bridge for the existing Ridge+cost-aware controller.

    The legacy generic adapter intentionally rejects even a one-ULP difference
    between exact fractional quantity and its native float reporting view.
    Here we verify a bounded reporting projection without changing that adapter
    or the native BookState. Target intent semantics remain the adapter's:
    LONG/SHORT -> signed budget, FLAT -> zero, with hard risk downstream.
    """

    __slots__ = (
        "_symbols",
        "_feature_names",
        "_global_feature_names",
        "_budgets",
        "_strategies",
        "_last_index",
        "_last_timestamp_ns",
        "_agent_stop_index",
        "_poisoned",
    )

    def __init__(
        self,
        dataset: MarketDataset,
        config: SyntheticCostAwareAttemptConfig,
        target_abs_weights: tuple[float, ...],
    ) -> None:
        self._symbols = tuple(dataset.symbols)
        self._feature_names = tuple(dataset.feature_names)
        self._global_feature_names = tuple(dataset.global_feature_names)
        self._budgets = target_abs_weights
        self._strategies = tuple(
            RidgeForecastStrategy(
                config.forecast_model,
                entry_threshold=config.entry_threshold,
                exit_threshold=config.exit_threshold,
                one_way_switch_cost=config.one_way_switch_cost,
            )
            for _ in self._symbols
        )
        self._last_index: int | None = None
        self._last_timestamp_ns: int | None = None
        self._agent_stop_index: int | None = None
        self._poisoned = False

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if self._poisoned:
            raise RuntimeError("cost-aware policy is poisoned after a decision failure")
        try:
            if type(snapshot) is not PortfolioDecisionSnapshot:
                raise TypeError("cost-aware policy requires a portfolio snapshot")
            if (
                snapshot.symbols != self._symbols
                or snapshot.feature_names != self._feature_names
                or snapshot.global_feature_names != self._global_feature_names
            ):
                raise ValueError("cost-aware policy symbol/feature roster mismatch")
            if (
                snapshot.account.insolvent
                or snapshot.account.termination_reason is not None
            ):
                raise ValueError("cost-aware policy cannot run after termination")
            if snapshot.active_orders:
                raise ValueError("pending orders are unsupported by cost-aware v2")
            if self._last_index is not None and (
                snapshot.index != self._last_index + 1
                or self._last_timestamp_ns is None
                or snapshot.timestamp_ns <= self._last_timestamp_ns
                or snapshot.agent_stop_index != self._agent_stop_index
            ):
                raise ValueError("cost-aware policy decision clock is not contiguous")
            timestamp = np.datetime64(snapshot.timestamp_ns, "ns")
            if np.isnat(timestamp):
                raise ValueError("cost-aware decision timestamp must not be NaT")
            targets: list[float] = []
            for i, (strategy, budget) in enumerate(
                zip(self._strategies, self._budgets, strict=True)
            ):
                raw_exact = snapshot.account.exact_quantities[i]
                try:
                    exact = Fraction(raw_exact)
                    if str(exact) != raw_exact:
                        raise ValueError("noncanonical native exact position quantity")
                    quantity = float(exact)
                except (ValueError, ZeroDivisionError, OverflowError) as error:
                    raise ValueError(
                        "cost-aware exact position quantity is invalid"
                    ) from error
                if (
                    not math.isfinite(quantity)
                    or (exact != 0 and quantity == 0.0)
                    or not math.isclose(
                        quantity,
                        snapshot.account.quantities[i],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        "native exact and reporting quantities materially disagree"
                    )
                current = (
                    PositionIntent.LONG
                    if exact > 0
                    else PositionIntent.SHORT
                    if exact < 0
                    else PositionIntent.FLAT
                )
                if not all(snapshot.feature_available[i]) or not all(
                    snapshot.global_feature_available
                ):
                    raise ValueError("cost-aware prediction feature is unavailable")
                obs = StrategyObservation(
                    index=snapshot.index,
                    timestamp=timestamp,
                    symbol=self._symbols[i],
                    features=np.asarray(snapshot.features[i], dtype=np.float64),
                    feature_available=np.asarray(
                        snapshot.feature_available[i], dtype=np.bool_
                    ),
                    feature_staleness=np.asarray(
                        snapshot.feature_staleness[i], dtype=np.float32
                    ),
                    global_features=np.asarray(
                        snapshot.global_features, dtype=np.float64
                    ),
                    global_feature_available=np.asarray(
                        snapshot.global_feature_available, dtype=np.bool_
                    ),
                    current_intent=current,
                    current_weight=snapshot.account.weights[i],
                    current_position_quantity=quantity,
                    intent_gross_budget=budget,
                )
                intent = strategy.decide(obs)
                if not isinstance(intent, PositionIntent):
                    raise TypeError("cost-aware Ridge returned a non-intent action")
                targets.append(float(int(intent)) * budget)
            proposal = PortfolioTargetProposal(target_weights=tuple(targets))
        except BaseException:
            self._poisoned = True
            raise
        self._last_index = snapshot.index
        self._last_timestamp_ns = snapshot.timestamp_ns
        self._agent_stop_index = snapshot.agent_stop_index
        return proposal


def _cost_aware_policy(
    dataset: MarketDataset,
    config: SyntheticCostAwareAttemptConfig,
    target_abs_weights: tuple[float, ...],
) -> _ExactQuantityCostAwareRidgePolicy:
    return _ExactQuantityCostAwareRidgePolicy(dataset, config, target_abs_weights)


def _v2_rows(
    root: Path,
    *,
    expected_result_sha256: str,
    expected_model_sha256: str,
    expected_forecast_sha256: str,
    expected_dataset_id: str,
) -> tuple[dict[str, object], ...]:
    result, result_digest = _read_record(root, "result/record.json")
    if result_digest != expected_result_sha256:
        raise ValueError("synthetic cost-aware result digest mismatch")
    started, started_digest = _read_record(root, "started/record.json")
    if (
        result.get("schema_version") != _RESULT
        or started.get("schema_version") != _STARTED
        or result.get("started_sha256") != started_digest
        or result.get("dataset_id") != expected_dataset_id
        or started.get("dataset_id") != expected_dataset_id
        or result.get("value_model_sha256") != expected_model_sha256
        or started.get("value_model_sha256") != expected_model_sha256
        or result.get("forecast_sha256") != expected_forecast_sha256
        or started.get("forecast_sha256") != expected_forecast_sha256
        or result.get("source_fingerprint") != started.get("source_fingerprint")
        or started.get("source_scope") != "synthetic_only"
        or started.get("families") != list(_FAMILIES)
    ):
        raise ValueError("synthetic cost-aware source, model, or schema mismatch")
    base_started, base_started_digest = _read_record(
        root / "three_arm_v1", "started/record.json"
    )
    if (
        started.get("base_started_sha256") != base_started_digest
        or base_started.get("source_fingerprint") != started.get("source_fingerprint")
        or base_started.get("model_sha256") != expected_model_sha256
        or base_started.get("dataset_id") != expected_dataset_id
        or base_started.get("execution_policy_digest")
        != started.get("execution_policy_digest")
    ):
        raise ValueError("nested v1 source/economic identity is inconsistent")
    raw_forecast, forecast_digest = _read_record(root, "input/forecast.json")
    if forecast_digest != expected_forecast_sha256:
        raise ValueError("forecast model fixture digest mismatch")
    forecast_model = _decode_ridge_payload(raw_forecast)
    if synthetic_ridge_fixture_digest(forecast_model) != expected_forecast_sha256:
        raise ValueError("forecast model decoded digest differs")
    base_digest = result.get("three_arm_result_sha256")
    if type(base_digest) is not str:
        raise ValueError("nested v1 result digest is missing")
    legacy = load_synthetic_value_attempt(
        root / "three_arm_v1",
        expected_result_sha256=base_digest,
        expected_model_sha256=expected_model_sha256,
        expected_dataset_id=expected_dataset_id,
    )
    if tuple(item["family"] for item in legacy.arms) != _FAMILIES[:3]:
        raise ValueError("nested v1 family roster differs")
    cost_row, digest = _read_record(root, "arms/cost_aware_ridge/record.json")
    ledger = cost_row.get("ledger")
    if not isinstance(ledger, dict):
        raise ValueError("cost-aware native ledger is missing")
    if (
        result.get("cost_aware_row_sha256") != digest
        or cost_row.get("schema_version") != _ARM
        or cost_row.get("family") != _FAMILIES[3]
        or cost_row.get("valid_endpoint") is not True
        or ledger.get("dataset_id") != expected_dataset_id
        or ledger.get("execution_policy_digest")
        != started.get("execution_policy_digest")
        or ledger.get("final_portfolio_value") != cost_row.get("terminal_equity")
        or ledger.get("final_total_cost") != cost_row.get("total_cost")
        or ledger.get("terminal_exact_quantities") != ["0", "0"]
        or ledger.get("active_order_remainders") != []
        or result.get("cost_aware_terminal_equity") != cost_row.get("terminal_equity")
    ):
        raise ValueError("cost-aware native row digest/endpoint identity mismatch")
    intervals = ledger.get("intervals")
    start_index = base_started.get("start_index")
    stop_index = base_started.get("stop_index")
    if (
        type(start_index) is not int
        or type(stop_index) is not int
        or not isinstance(intervals, list)
        or len(intervals) != stop_index - start_index
    ):
        raise ValueError("cost-aware native interval count mismatch")
    return (*legacy.arms, cost_row)


def run_synthetic_cost_aware_attempt(
    attempt_root: str | Path,
    config: SyntheticCostAwareAttemptConfig,
    *,
    on_arm_completed: Callable[[str], None] | None = None,
) -> PublishedSyntheticCostAwareAttempt:
    """Write a nested v1 attempt, then one true cost-aware native replay arm."""
    value_model, fingerprint, started = _preflight_cost_aware(config, on_arm_completed)
    base = config.value_attempt
    root = Path(attempt_root).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError("attempt root already exists")
    if any(parent.is_symlink() for parent in root.parents):
        raise ValueError("attempt parent must not be a symlink")
    if not root.parent.is_dir():
        raise ValueError("attempt parent must already exist")
    root.mkdir()
    completed: list[str] = []
    phase = "started"
    result_sha256: str | None = None
    try:
        started_sha256 = _write_record(root, "started/record.json", started)
        _write_record(
            root, "input/forecast.json", _ridge_payload(config.forecast_model)
        )
        phase = "three_arm_v1"

        def persist_inner(family: str) -> None:
            if family != _FAMILIES[len(completed)]:
                raise RuntimeError("nested v1 arm completion order differs")
            completed.append(family)
            if on_arm_completed is not None:
                on_arm_completed(family)

        parent = run_synthetic_value_attempt(
            root / "three_arm_v1",
            base,
            on_arm_completed=persist_inner,
        )
        phase = _FAMILIES[3]
        if _source_fingerprint(base.dataset) != fingerprint:
            raise RuntimeError("synthetic Dataset drifted before cost-aware arm")
        if synthetic_ridge_fixture_digest(config.forecast_model) != (
            config.expected_forecast_sha256
        ):
            raise RuntimeError("frozen forecast changed before cost-aware arm")
        native = run_shared_portfolio_replay(
            base.dataset,
            _cost_aware_policy(base.dataset, config, value_model.target_abs_weights),
            start_index=base.start_index,
            stop_index=base.stop_index,
            initial_capital=float(cast(float, base.initial_capital)),
            execution_cost=base.execution_cost,
            risk=PreTradeRisk(base.risk_config),
            settle_terminal_position=True,
            capture_ledger_evidence=True,
        )
        row = _arm_row(base, _FAMILIES[3], native)
        row["schema_version"] = _ARM
        row_sha = _write_record(root, "arms/cost_aware_ridge/record.json", row)
        completed.append(_FAMILIES[3])
        if on_arm_completed is not None:
            on_arm_completed(_FAMILIES[3])
        if not row["valid_endpoint"]:
            raise RuntimeError("cost-aware native endpoint is incomplete")
        if _source_fingerprint(base.dataset) != fingerprint:
            raise RuntimeError("synthetic Dataset drifted after cost-aware replay")
        if synthetic_ridge_fixture_digest(config.forecast_model) != (
            config.expected_forecast_sha256
        ):
            raise RuntimeError("frozen forecast drifted after replay")
        phase = "result"
        payload = {
            "schema_version": _RESULT,
            "dataset_id": base.expected_dataset_id,
            "source_fingerprint": fingerprint,
            "value_model_sha256": base.expected_model_sha256,
            "forecast_sha256": config.expected_forecast_sha256,
            "started_sha256": started_sha256,
            "three_arm_result_sha256": parent.result_sha256,
            "cost_aware_row_sha256": row_sha,
            "cost_aware_terminal_equity": row["terminal_equity"],
        }
        result_sha256 = _write_record(root, "result/record.json", payload)
        _v2_rows(
            root,
            expected_result_sha256=result_sha256,
            expected_model_sha256=base.expected_model_sha256,
            expected_forecast_sha256=config.expected_forecast_sha256,
            expected_dataset_id=base.expected_dataset_id,
        )
        phase = "complete"
        _write_record(
            root,
            "complete/record.json",
            {"schema_version": _COMPLETE, "result_sha256": result_sha256},
        )
        load_synthetic_cost_aware_attempt(
            root,
            expected_result_sha256=result_sha256,
            expected_model_sha256=base.expected_model_sha256,
            expected_forecast_sha256=config.expected_forecast_sha256,
            expected_dataset_id=base.expected_dataset_id,
        )
        return PublishedSyntheticCostAwareAttempt(
            root=root, result_sha256=result_sha256
        )
    except BaseException as error:
        failure = {
            "schema_version": _FAILURE,
            "phase": phase,
            "completed_families": completed,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "result_sha256": result_sha256,
        }
        try:
            _write_record(root, "failure/record.json", failure)
        except BaseException as secondary:
            error.add_note(f"failure journal also failed: {secondary}")
        raise


def load_synthetic_cost_aware_attempt(
    attempt_root: str | Path,
    *,
    expected_result_sha256: str,
    expected_model_sha256: str,
    expected_forecast_sha256: str,
    expected_dataset_id: str,
) -> LoadedSyntheticCostAwareAttempt:
    """Verify the complete v2 marker, nested v1 result, all rows and model pins."""
    for name, expected in (
        ("result SHA", expected_result_sha256),
        ("value model SHA", expected_model_sha256),
        ("forecast SHA", expected_forecast_sha256),
        ("Dataset ID", expected_dataset_id),
    ):
        require_sha256(expected, field=name)
    root = Path(attempt_root).expanduser().absolute()
    if root.is_symlink() or (root / "failure").exists():
        raise ValueError("cost-aware attempt has failure evidence or unsafe root")
    if not (root / "complete/record.json").is_file():
        raise ValueError("cost-aware attempt has no complete marker")
    complete, _ = _read_record(root, "complete/record.json")
    if (
        complete.get("schema_version") != _COMPLETE
        or complete.get("result_sha256") != expected_result_sha256
    ):
        raise ValueError("cost-aware complete marker differs")
    rows = _v2_rows(
        root,
        expected_result_sha256=expected_result_sha256,
        expected_model_sha256=expected_model_sha256,
        expected_forecast_sha256=expected_forecast_sha256,
        expected_dataset_id=expected_dataset_id,
    )
    return LoadedSyntheticCostAwareAttempt(
        root=root,
        result_sha256=expected_result_sha256,
        forecast_sha256=expected_forecast_sha256,
        dataset_id=expected_dataset_id,
        arms=rows,
    )


__all__ = [
    "SyntheticCostAwareAttemptConfig",
    "PublishedSyntheticCostAwareAttempt",
    "LoadedSyntheticCostAwareAttempt",
    "synthetic_ridge_fixture_digest",
    "run_synthetic_cost_aware_attempt",
    "load_synthetic_cost_aware_attempt",
]
