"""Opt-in, fit-scope-bound observations for the shared-portfolio Gym adapter."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from numbers import Real
from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.accounting import EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import OrderStatus, OrderType, TimeInForce
from trade_rl.strategies.rl.portfolio_env import (
    ACTIVE_ORDER_FIELDS,
    PORTFOLIO_GYM_OBSERVATION_KEYS,
    PORTFOLIO_GYM_OBSERVATION_SCHEMA,
    PORTFOLIO_STATE_FIELDS,
    RISK_STATE_FIELDS,
    SharedPortfolioGymEnv,
)
from trade_rl.strategies.rl.ppo_normalization import (
    PPOFeatureNormalizer,
    fit_ppo_feature_normalizer,
)

PORTFOLIO_OBSERVATION_TRANSFORM_SCHEMA = "shared_portfolio_observation_transform_v1"
PORTFOLIO_TRANSFORMED_OBSERVATION_SCHEMA = "shared_portfolio_dict_transform_v1"
PORTFOLIO_TRANSFORMED_OBSERVATION_KEYS = PORTFOLIO_GYM_OBSERVATION_KEYS

_FLOAT_KEYS = (
    "features",
    "feature_staleness",
    "global_features",
    "account",
    "portfolio",
    "risk",
    "active_orders",
)
_BINARY_KEYS = ("feature_available", "global_feature_available", "terminal")
_REASON_NAMES = ("none", *(reason.value for reason in EconomicTerminationReason))
_ORDER_TYPES = tuple(item.value for item in OrderType)
_TIME_IN_FORCE = tuple(item.value for item in TimeInForce)
_ORDER_STATUSES = tuple(item.value for item in OrderStatus)
_ORDER_WIDTH = len(ACTIVE_ORDER_FIELDS)
_ACTIVE_ORDER_OUTPUT_FIELDS = (
    "present",
    *(f"order_type_{name}" for name in _ORDER_TYPES),
    *(f"time_in_force_{name}" for name in _TIME_IN_FORCE),
    *(f"status_{name}" for name in _ORDER_STATUSES),
    "requested_marked_notional_over_capital",
    "remaining_marked_notional_over_capital",
    "reduce_only",
    "submit_offset_fraction",
    "eligible_offset_fraction",
    "has_last_processed_index",
    "last_processed_offset_fraction",
    "has_trigger_index",
    "trigger_offset_fraction",
    "has_expiry_index",
    "expiry_offset_fraction",
    "has_limit_price",
    "limit_log_ratio_to_submission",
    "has_stop_price",
    "stop_log_ratio_to_submission",
    "has_submission_reference_price",
    "submission_log_ratio_to_mark",
)
_PORTFOLIO_OUTPUT_FIELDS = (
    "cash_over_initial_capital",
    "equity_over_initial_capital",
    "peak_equity_over_initial_capital",
    "max_drawdown",
    "turnover_total",
    "total_cost_over_initial_capital",
    "funding_pnl_over_initial_capital",
    "borrow_cost_over_initial_capital",
    "margin_used_over_initial_capital",
    "maintenance_margin_rate",
    "maintenance_requirement_over_initial_capital",
    "margin_deficit_over_initial_capital",
    "insolvent",
    *(f"termination_{name}" for name in _REASON_NAMES),
)
_ACCOUNT_OUTPUT_FIELDS = (
    "signed_marked_notional_over_initial_capital",
    "weight",
    "log_mark_relative_to_fit_geometric_mean",
    "log_contract_multiplier",
)
_CLOCK_OUTPUT_FIELDS = (
    "elapsed_actor_fraction",
    "remaining_actor_fraction",
    "elapsed_utc_horizon_fraction",
)
_OUTPUT_LAYOUT: dict[str, object] = {
    "account": list(_ACCOUNT_OUTPUT_FIELDS),
    "portfolio": list(_PORTFOLIO_OUTPUT_FIELDS),
    "risk": list(RISK_STATE_FIELDS),
    "active_orders": list(_ACTIVE_ORDER_OUTPUT_FIELDS),
    "clock": list(_CLOCK_OUTPUT_FIELDS),
    "order_type_codes": {name: index + 1 for index, name in enumerate(_ORDER_TYPES)},
    "time_in_force_codes": {
        name: index + 1 for index, name in enumerate(_TIME_IN_FORCE)
    },
    "order_status_codes": {
        name: index + 1 for index, name in enumerate(_ORDER_STATUSES)
    },
    "termination_reason_codes": {
        name: index for index, name in enumerate(_REASON_NAMES)
    },
}
_RAW_DTYPES = {
    "features": np.dtype(np.float64),
    "feature_available": np.dtype(np.int8),
    "feature_staleness": np.dtype(np.float64),
    "global_features": np.dtype(np.float64),
    "global_feature_available": np.dtype(np.int8),
    "account": np.dtype(np.float64),
    "portfolio": np.dtype(np.float64),
    "risk": np.dtype(np.float64),
    "active_orders": np.dtype(np.float64),
    "clock": np.dtype(np.int64),
    "terminal": np.dtype(np.int8),
}
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _finite_real(value: object, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real non-boolean number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _native_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a native integer")
    return value


def _datetime_ns(value: datetime) -> int:
    normalized = value.astimezone(UTC)
    elapsed = normalized - _UTC_EPOCH
    return (
        elapsed.days * 86_400 + elapsed.seconds
    ) * 1_000_000_000 + elapsed.microseconds * 1_000


def _digest(value: object) -> str:
    return content_digest(value)


def _float32(values: np.ndarray, *, field: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{field} exceeds finite float32 range")
    return result


def _log_marked_notional(
    quantity: float, price: float, multiplier: float, capital: float, *, field: str
) -> float:
    if quantity == 0.0:
        return 0.0
    log_magnitude = (
        math.log(abs(quantity))
        + math.log(price)
        + math.log(multiplier)
        - math.log(capital)
    )
    if log_magnitude > math.log(_FLOAT32_MAX):
        raise ValueError(f"{field} exceeds finite float32 range")
    magnitude = math.exp(log_magnitude)
    if not math.isfinite(magnitude):
        raise ValueError(f"{field} is not finite")
    return math.copysign(magnitude, quantity)


def _risk_payload(config: PreTradeRiskConfig) -> dict[str, object]:
    return asdict(config)


def _execution_payload(config: ExecutionCostConfig) -> dict[str, object]:
    return asdict(config)


def _matches_output_layout(value: object) -> bool:
    try:
        return canonical_json_bytes(value) == canonical_json_bytes(_OUTPUT_LAYOUT)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class PortfolioObservationTransform:
    """Frozen preprocessing parameters bound to one Gym input context."""

    source_dataset_id: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    start_index: int
    stop_index: int
    agent_stop_index: int
    initial_capital: float
    bound_digest: str
    objective_digest: str
    clock_digest: str
    objective_start_ns: int
    economic_horizon_seconds: int
    native_interval_seconds: int
    risk_config_digest: str
    execution_config_digest: str
    market_order_profile_digest: str | None
    local_normalizer: PPOFeatureNormalizer
    global_mean: tuple[float, ...]
    global_scale: tuple[float, ...]
    global_usable_counts: tuple[int, ...]
    reference_log_price_mean: tuple[float, ...]
    reference_price_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        symbols = tuple(self.symbols)
        features = tuple(self.feature_names)
        globals_ = tuple(self.global_feature_names)
        if (
            not symbols
            or len(set(symbols)) != len(symbols)
            or any(not isinstance(value, str) or not value for value in symbols)
        ):
            raise ValueError("symbols must be an ordered unique non-empty tuple")
        if len(set(features)) != len(features) or any(
            not isinstance(value, str) or not value for value in features
        ):
            raise ValueError("feature_names must be ordered unique names")
        if len(set(globals_)) != len(globals_) or any(
            not isinstance(value, str) or not value for value in globals_
        ):
            raise ValueError("global_feature_names must be ordered unique names")
        for name in ("start_index", "stop_index", "agent_stop_index"):
            _native_int(getattr(self, name), field=name)
        for name in (
            "objective_start_ns",
            "economic_horizon_seconds",
            "native_interval_seconds",
        ):
            _native_int(getattr(self, name), field=name)
        if not 0 <= self.start_index < self.agent_stop_index <= self.stop_index:
            raise ValueError("training and decision scopes are inconsistent")
        if self.economic_horizon_seconds <= 0 or self.native_interval_seconds <= 0:
            raise ValueError("objective horizon and native interval must be positive")
        if (
            self.economic_horizon_seconds
            != (self.stop_index - self.start_index) * self.native_interval_seconds
        ):
            raise ValueError("objective horizon must match the bound index window")
        capital = _finite_real(self.initial_capital, field="initial_capital")
        if capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        if not isinstance(self.local_normalizer, PPOFeatureNormalizer):
            raise TypeError("local_normalizer must be a PPOFeatureNormalizer")
        if self.local_normalizer.source_dataset_id != self.source_dataset_id:
            raise ValueError("local normalizer Dataset binding differs")
        if self.local_normalizer.start_index != self.start_index or (
            self.local_normalizer.stop_index != self.agent_stop_index
        ):
            raise ValueError("local normalizer fit scope differs")
        if (
            self.local_normalizer.feature_names != features
            or self.local_normalizer.feature_indices != tuple(range(len(features)))
            or self.local_normalizer.fit_symbol_indices != tuple(range(len(symbols)))
        ):
            raise ValueError("local normalizer feature or symbol roster differs")
        n_globals = len(globals_)
        means = tuple(
            _finite_real(value, field="global_mean") for value in self.global_mean
        )
        scales = tuple(
            _finite_real(value, field="global_scale") for value in self.global_scale
        )
        counts = tuple(self.global_usable_counts)
        if not (len(means) == len(scales) == len(counts) == n_globals):
            raise ValueError("global fit statistics must match global feature names")
        if any(scale <= 0.0 for scale in scales):
            raise ValueError("global scales must be positive")
        if any(
            type(count) is not int
            or not 0 < count <= (self.agent_stop_index - self.start_index)
            for count in counts
        ):
            raise ValueError("global usable counts must be within the fit scope")
        reference_means = tuple(
            _finite_real(value, field="reference_log_price_mean")
            for value in self.reference_log_price_mean
        )
        reference_counts = tuple(self.reference_price_counts)
        if len(reference_means) != len(symbols) or len(reference_counts) != len(
            symbols
        ):
            raise ValueError("reference price statistics must match symbols")
        fit_count = self.agent_stop_index - self.start_index
        if any(
            type(count) is not int or count != fit_count for count in reference_counts
        ):
            raise ValueError("reference price counts must cover actor training rows")
        for name in (
            "bound_digest",
            "objective_digest",
            "clock_digest",
            "risk_config_digest",
            "execution_config_digest",
        ):
            require_sha256(getattr(self, name), field=name)
        require_sha256(self.source_dataset_id, field="source_dataset_id")
        if self.market_order_profile_digest is not None:
            require_sha256(
                self.market_order_profile_digest,
                field="market_order_profile_digest",
            )
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "feature_names", features)
        object.__setattr__(self, "global_feature_names", globals_)
        object.__setattr__(self, "initial_capital", capital)
        object.__setattr__(self, "global_mean", means)
        object.__setattr__(self, "global_scale", scales)
        object.__setattr__(self, "global_usable_counts", counts)
        object.__setattr__(self, "reference_log_price_mean", reference_means)
        object.__setattr__(self, "reference_price_counts", reference_counts)

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return PORTFOLIO_TRANSFORMED_OBSERVATION_KEYS

    @property
    def observation_space(self) -> spaces.Dict:
        n_symbols = len(self.symbols)
        n_features = len(self.feature_names)
        n_globals = len(self.global_feature_names)
        return spaces.Dict(
            OrderedDict(
                (
                    (
                        "features",
                        spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(n_symbols, n_features),
                            dtype=np.float32,
                        ),
                    ),
                    (
                        "feature_available",
                        spaces.MultiBinary((n_symbols, n_features)),
                    ),
                    (
                        "feature_staleness",
                        spaces.Box(
                            0.0, 1.0, shape=(n_symbols, n_features), dtype=np.float32
                        ),
                    ),
                    (
                        "global_features",
                        spaces.Box(
                            -np.inf, np.inf, shape=(n_globals,), dtype=np.float32
                        ),
                    ),
                    (
                        "global_feature_available",
                        spaces.MultiBinary((n_globals,)),
                    ),
                    (
                        "account",
                        spaces.Box(
                            -np.inf, np.inf, shape=(n_symbols, 4), dtype=np.float32
                        ),
                    ),
                    (
                        "portfolio",
                        spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(len(_PORTFOLIO_OUTPUT_FIELDS),),
                            dtype=np.float32,
                        ),
                    ),
                    (
                        "risk",
                        spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(len(RISK_STATE_FIELDS),),
                            dtype=np.float32,
                        ),
                    ),
                    (
                        "active_orders",
                        spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(n_symbols, len(_ACTIVE_ORDER_OUTPUT_FIELDS)),
                            dtype=np.float32,
                        ),
                    ),
                    ("clock", spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)),
                    ("terminal", spaces.MultiBinary(1)),
                )
            )
        )

    @property
    def digest(self) -> str:
        """Bind the canonical serialized preprocessing parameters."""

        return content_digest(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": PORTFOLIO_OBSERVATION_TRANSFORM_SCHEMA,
            "input_schema": PORTFOLIO_GYM_OBSERVATION_SCHEMA,
            "output_schema": PORTFOLIO_TRANSFORMED_OBSERVATION_SCHEMA,
            "output_keys": list(self.observation_keys),
            "output_layout": deepcopy(_OUTPUT_LAYOUT),
            "source_dataset_id": self.source_dataset_id,
            "symbols": list(self.symbols),
            "feature_names": list(self.feature_names),
            "global_feature_names": list(self.global_feature_names),
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "agent_stop_index": self.agent_stop_index,
            "initial_capital": self.initial_capital,
            "bound_digest": self.bound_digest,
            "objective_digest": self.objective_digest,
            "clock_digest": self.clock_digest,
            "objective_start_ns": self.objective_start_ns,
            "economic_horizon_seconds": self.economic_horizon_seconds,
            "native_interval_seconds": self.native_interval_seconds,
            "risk_config_digest": self.risk_config_digest,
            "execution_config_digest": self.execution_config_digest,
            "market_order_profile_digest": self.market_order_profile_digest,
            "local_normalizer": self.local_normalizer.to_payload(),
            "global_mean": list(self.global_mean),
            "global_scale": list(self.global_scale),
            "global_usable_counts": list(self.global_usable_counts),
            "reference_log_price_mean": list(self.reference_log_price_mean),
            "reference_price_counts": list(self.reference_price_counts),
        }

    @classmethod
    def from_payload(
        cls, value: object, *, expected_digest: str
    ) -> PortfolioObservationTransform:
        keys = {
            "schema",
            "input_schema",
            "output_schema",
            "output_keys",
            "output_layout",
            "source_dataset_id",
            "symbols",
            "feature_names",
            "global_feature_names",
            "start_index",
            "stop_index",
            "agent_stop_index",
            "initial_capital",
            "bound_digest",
            "objective_digest",
            "clock_digest",
            "objective_start_ns",
            "economic_horizon_seconds",
            "native_interval_seconds",
            "risk_config_digest",
            "execution_config_digest",
            "market_order_profile_digest",
            "local_normalizer",
            "global_mean",
            "global_scale",
            "global_usable_counts",
            "reference_log_price_mean",
            "reference_price_counts",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("transform payload has missing or extra fields")
        require_sha256(expected_digest, field="expected_digest")
        try:
            payload_digest = content_digest(value)
        except (TypeError, ValueError) as error:
            raise ValueError("transform payload is not canonical JSON") from error
        if payload_digest != expected_digest:
            raise ValueError("transform payload digest differs from expected_digest")
        if (
            value["schema"] != PORTFOLIO_OBSERVATION_TRANSFORM_SCHEMA
            or value["input_schema"] != PORTFOLIO_GYM_OBSERVATION_SCHEMA
            or value["output_schema"] != PORTFOLIO_TRANSFORMED_OBSERVATION_SCHEMA
            or value["output_keys"] != list(PORTFOLIO_TRANSFORMED_OBSERVATION_KEYS)
            or not _matches_output_layout(value["output_layout"])
        ):
            raise ValueError("transform payload schema or field layout differs")
        sequence_keys = (
            "symbols",
            "feature_names",
            "global_feature_names",
            "global_mean",
            "global_scale",
            "global_usable_counts",
            "reference_log_price_mean",
            "reference_price_counts",
        )
        if any(type(value[key]) is not list for key in sequence_keys):
            raise ValueError("transform sequences must be JSON arrays")
        try:
            normalizer = PPOFeatureNormalizer.from_payload(value["local_normalizer"])
            return cls(
                source_dataset_id=value["source_dataset_id"],
                symbols=tuple(value["symbols"]),
                feature_names=tuple(value["feature_names"]),
                global_feature_names=tuple(value["global_feature_names"]),
                start_index=value["start_index"],
                stop_index=value["stop_index"],
                agent_stop_index=value["agent_stop_index"],
                initial_capital=value["initial_capital"],
                bound_digest=value["bound_digest"],
                objective_digest=value["objective_digest"],
                clock_digest=value["clock_digest"],
                objective_start_ns=value["objective_start_ns"],
                economic_horizon_seconds=value["economic_horizon_seconds"],
                native_interval_seconds=value["native_interval_seconds"],
                risk_config_digest=value["risk_config_digest"],
                execution_config_digest=value["execution_config_digest"],
                market_order_profile_digest=value["market_order_profile_digest"],
                local_normalizer=normalizer,
                global_mean=tuple(value["global_mean"]),
                global_scale=tuple(value["global_scale"]),
                global_usable_counts=tuple(value["global_usable_counts"]),
                reference_log_price_mean=tuple(value["reference_log_price_mean"]),
                reference_price_counts=tuple(value["reference_price_counts"]),
            )
        except (TypeError, OverflowError) as error:
            raise ValueError("invalid transform metadata") from error

    def validate_environment(self, env: SharedPortfolioGymEnv) -> None:
        if not isinstance(env, SharedPortfolioGymEnv):
            raise TypeError("environment must be SharedPortfolioGymEnv")
        expected_agent_stop = env.stop_index - env.execution_cost.order_latency_bars - 1
        profile_digest = (
            None
            if env.market_order_profile is None
            else env.market_order_profile.digest
        )
        if (
            env.dataset.dataset_id != self.source_dataset_id
            or tuple(env.dataset.symbols) != self.symbols
            or tuple(env.dataset.feature_names) != self.feature_names
            or tuple(env.dataset.global_feature_names) != self.global_feature_names
            or env.start_index != self.start_index
            or env.stop_index != self.stop_index
            or expected_agent_stop != self.agent_stop_index
            or env.initial_capital != self.initial_capital
            or env.bound.digest != self.bound_digest
            or env.bound.objective.digest != self.objective_digest
            or env.bound.clock.digest != self.clock_digest
            or _datetime_ns(env.bound.objective.evaluation_start)
            != self.objective_start_ns
            or env.bound.clock.economic_horizon_seconds != self.economic_horizon_seconds
            or env.bound.clock.decision_interval_seconds != self.native_interval_seconds
            or _digest(_risk_payload(env.risk_config)) != self.risk_config_digest
            or _digest(_execution_payload(env.execution_cost))
            != self.execution_config_digest
            or profile_digest != self.market_order_profile_digest
        ):
            raise ValueError("transform context differs from the Gym environment")
        self.local_normalizer.validate_training_scope(
            env.dataset,
            tuple(range(env.dataset.n_symbols)),
            self.start_index,
            self.agent_stop_index,
        )

    def transform_observation(self, raw: object) -> dict[str, np.ndarray]:
        values, current_index, is_terminal = self._validated_raw(raw)
        n_symbols = len(self.symbols)
        n_features = len(self.feature_names)
        n_globals = len(self.global_feature_names)
        if is_terminal:
            return self._terminal_padding(n_symbols, n_features, n_globals)

        local = np.zeros((n_symbols, n_features), dtype=np.float64)
        local_mask = values["feature_available"].astype(np.bool_, copy=False)
        for symbol_index in range(n_symbols):
            local[symbol_index] = self.local_normalizer.transform(
                values["features"][symbol_index], local_mask[symbol_index]
            )

        global_values = values["global_features"]
        global_mask = values["global_feature_available"].astype(np.bool_, copy=False)
        normalized_global = np.zeros(n_globals, dtype=np.float64)
        for index, usable in enumerate(global_mask):
            if usable:
                normalized_global[index] = (
                    global_values[index] - self.global_mean[index]
                ) / self.global_scale[index]

        account = self._transform_account(values["account"])
        portfolio = self._transform_portfolio(values["portfolio"])
        risk = values["risk"]
        orders = self._transform_orders(
            values["active_orders"], values["account"], current_index
        )
        clock = self._transform_clock(values["clock"])
        output = {
            "features": _float32(local, field="features"),
            "feature_available": values["feature_available"].copy(),
            "feature_staleness": _float32(
                values["feature_staleness"], field="feature_staleness"
            ),
            "global_features": _float32(normalized_global, field="global_features"),
            "global_feature_available": values["global_feature_available"].copy(),
            "account": _float32(account, field="account"),
            "portfolio": _float32(portfolio, field="portfolio"),
            "risk": _float32(risk.copy(), field="risk"),
            "active_orders": _float32(orders, field="active_orders"),
            "clock": _float32(clock, field="clock"),
            "terminal": np.zeros(1, dtype=np.int8),
        }
        if tuple(
            output
        ) != self.observation_keys or not self.observation_space.contains(output):
            raise RuntimeError("transformed observation violates its declared space")
        return output

    def _validated_raw(self, raw: object) -> tuple[dict[str, np.ndarray], int, bool]:
        if not isinstance(raw, Mapping) or tuple(raw) != PORTFOLIO_GYM_OBSERVATION_KEYS:
            raise ValueError("raw observation keys or ordering differ from schema")
        n_symbols = len(self.symbols)
        n_features = len(self.feature_names)
        n_globals = len(self.global_feature_names)
        expected_shapes = {
            "features": (n_symbols, n_features),
            "feature_available": (n_symbols, n_features),
            "feature_staleness": (n_symbols, n_features),
            "global_features": (n_globals,),
            "global_feature_available": (n_globals,),
            "account": (n_symbols, 4),
            "portfolio": (len(PORTFOLIO_STATE_FIELDS),),
            "risk": (len(RISK_STATE_FIELDS),),
            "active_orders": (n_symbols, _ORDER_WIDTH),
            "clock": (4,),
            "terminal": (1,),
        }
        checked: dict[str, np.ndarray] = {}
        for key in PORTFOLIO_GYM_OBSERVATION_KEYS:
            item = raw[key]
            if not isinstance(item, np.ndarray):
                raise TypeError(f"raw {key} must be a numpy array")
            if item.dtype != _RAW_DTYPES[key] or item.shape != expected_shapes[key]:
                raise ValueError(f"raw {key} dtype or shape differs from schema")
            if key in _FLOAT_KEYS and not np.isfinite(item).all():
                raise ValueError(f"raw {key} must be finite")
            if key in _BINARY_KEYS and np.any((item != 0) & (item != 1)):
                raise ValueError(f"raw {key} must contain binary values")
            checked[key] = item

        terminal = bool(checked["terminal"][0])
        if terminal:
            if any(
                np.any(checked[key] != 0)
                for key in PORTFOLIO_GYM_OBSERVATION_KEYS
                if key != "terminal"
            ):
                raise ValueError("terminal observations must use zero padding")
            return checked, -1, True

        clock = checked["clock"]
        current_index, timestamp_ns, agent_stop, remaining = (
            int(value) for value in clock
        )
        if (
            current_index < self.start_index
            or current_index >= self.agent_stop_index
            or agent_stop != self.agent_stop_index
            or remaining != agent_stop - current_index
        ):
            raise ValueError("raw clock is outside the bound actor window")
        expected_timestamp = (
            self.objective_start_ns
            + (current_index - self.start_index)
            * self.native_interval_seconds
            * 1_000_000_000
        )
        if timestamp_ns != expected_timestamp:
            raise ValueError("raw timestamp differs from the bound native clock")
        account = checked["account"]
        if np.any(account[:, 2] <= 0.0) or np.any(account[:, 3] <= 0.0):
            raise ValueError("account mark prices and multipliers must be positive")
        risk = checked["risk"]
        if any(risk[index] not in (0.0, 1.0) for index in (3, 6)):
            raise ValueError("risk flags must be binary")
        if risk[3] == 0.0 and risk[2] != 0.0:
            raise ValueError("max_turnover presence does not match its value")
        portfolio = checked["portfolio"]
        if portfolio[12] not in (0.0, 1.0):
            raise ValueError("insolvency flag must be binary")
        reason_code = portfolio[13]
        if reason_code != math.floor(reason_code) or not 0 <= reason_code < len(
            _REASON_NAMES
        ):
            raise ValueError("termination reason code is unsupported")
        staleness = checked["feature_staleness"]
        if np.any((staleness < 0.0) | (staleness > 1.0)):
            raise ValueError("feature staleness must be within [0, 1]")
        self._validate_active_orders(checked["active_orders"])
        return checked, current_index, False

    @staticmethod
    def _validate_active_orders(values: np.ndarray) -> None:
        for row in values:
            if row[0] == 0.0:
                if np.any(row != 0.0):
                    raise ValueError("absent order rows must be all zero")
                continue
            if row[0] != 1.0:
                raise ValueError("active order presence must be binary")
            order_type, tif, status = row[1:4]
            if any(
                value != math.floor(value) or value < 1 or value > limit
                for value, limit in (
                    (order_type, len(_ORDER_TYPES)),
                    (tif, len(_TIME_IN_FORCE)),
                    (status, len(_ORDER_STATUSES)),
                )
            ):
                raise ValueError("active order contains an unknown enum code")
            for flag_index in (6, 9, 11, 13, 15, 17, 19):
                if row[flag_index] not in (0.0, 1.0):
                    raise ValueError("active order presence flags must be binary")
            if row[19] != 1.0 or row[20] <= 0.0:
                raise ValueError("active order price context is incomplete")
            if row[15] == 0.0 and row[16] != 0.0:
                raise ValueError("absent limit price must be zero")
            if row[17] == 0.0 and row[18] != 0.0:
                raise ValueError("absent stop price must be zero")
            if row[9] == 0.0 and row[10] != 0.0:
                raise ValueError("absent last-processed index must be zero")
            if row[11] == 0.0 and row[12] != 0.0:
                raise ValueError("absent trigger index must be zero")
            if row[13] == 0.0 and row[14] != 0.0:
                raise ValueError("absent expiry index must be zero")
            if row[1] == 1.0 and (row[15] != 0.0 or row[17] != 0.0):
                raise ValueError("market order may not have limit or stop prices")
            if row[1] == 2.0 and (row[15] != 1.0 or row[17] != 0.0):
                raise ValueError("limit order price context is invalid")
            if row[1] == 3.0 and (row[17] != 1.0 or row[15] != 0.0):
                raise ValueError("stop-market order price context is invalid")
            if row[15] == 1.0 and row[16] <= 0.0:
                raise ValueError("limit price must be positive")
            if row[17] == 1.0 and row[18] <= 0.0:
                raise ValueError("stop price must be positive")
            index_fields = [(7, True), (8, True)]
            index_fields.extend(
                (
                    (10, row[9] == 1.0),
                    (12, row[11] == 1.0),
                    (14, row[13] == 1.0),
                )
            )
            if any(
                present and row[index] != math.floor(row[index])
                for index, present in index_fields
            ):
                raise ValueError("order indices must be integers")

    def _transform_account(self, raw: np.ndarray) -> np.ndarray:
        output = np.zeros((len(self.symbols), len(_ACCOUNT_OUTPUT_FIELDS)))
        for index, (quantity, weight, mark, multiplier) in enumerate(raw):
            output[index] = (
                _log_marked_notional(
                    float(quantity),
                    float(mark),
                    float(multiplier),
                    self.initial_capital,
                    field="account marked notional",
                ),
                float(weight),
                math.log(float(mark)) - self.reference_log_price_mean[index],
                math.log(float(multiplier)),
            )
        return output

    def _transform_portfolio(self, raw: np.ndarray) -> np.ndarray:
        capital = self.initial_capital
        output = np.empty(len(_PORTFOLIO_OUTPUT_FIELDS), dtype=np.float64)
        currency_indexes = (0, 1, 2, 5, 6, 7, 8, 10, 11)
        for index in currency_indexes:
            output[index] = float(raw[index]) / capital
        output[3] = raw[3]
        output[4] = raw[4]
        output[9] = raw[9]
        output[12] = raw[12]
        reason_index = int(raw[13])
        output[13:] = 0.0
        output[13 + reason_index] = 1.0
        return output

    def _transform_orders(
        self, raw: np.ndarray, account: np.ndarray, current_index: int
    ) -> np.ndarray:
        output = np.zeros((len(self.symbols), len(_ACTIVE_ORDER_OUTPUT_FIELDS)))
        decision_count = self.agent_stop_index - self.start_index
        type_start = 1
        tif_start = type_start + len(_ORDER_TYPES)
        status_start = tif_start + len(_TIME_IN_FORCE)
        numeric_start = status_start + len(_ORDER_STATUSES)
        for symbol_index, row in enumerate(raw):
            if row[0] == 0.0:
                continue
            output_row = output[symbol_index]
            output_row[0] = 1.0
            output_row[type_start + int(row[1]) - 1] = 1.0
            output_row[tif_start + int(row[2]) - 1] = 1.0
            output_row[status_start + int(row[3]) - 1] = 1.0
            requested = _log_marked_notional(
                float(row[4]),
                float(account[symbol_index, 2]),
                float(account[symbol_index, 3]),
                self.initial_capital,
                field="active order requested notional",
            )
            remaining = _log_marked_notional(
                float(row[5]),
                float(account[symbol_index, 2]),
                float(account[symbol_index, 3]),
                self.initial_capital,
                field="active order remaining notional",
            )
            output_row[numeric_start : numeric_start + 3] = (
                requested,
                remaining,
                row[6],
            )
            output_row[numeric_start + 3] = (row[7] - current_index) / decision_count
            output_row[numeric_start + 4] = (row[8] - current_index) / decision_count
            output_row[numeric_start + 5] = row[9]
            output_row[numeric_start + 6] = (
                0.0 if row[9] == 0.0 else (row[10] - current_index) / decision_count
            )
            output_row[numeric_start + 7] = row[11]
            output_row[numeric_start + 8] = (
                0.0 if row[11] == 0.0 else (row[12] - current_index) / decision_count
            )
            output_row[numeric_start + 9] = row[13]
            output_row[numeric_start + 10] = (
                0.0 if row[13] == 0.0 else (row[14] - current_index) / decision_count
            )
            output_row[numeric_start + 11] = row[15]
            output_row[numeric_start + 12] = (
                0.0 if row[15] == 0.0 else math.log(row[16]) - math.log(row[20])
            )
            output_row[numeric_start + 13] = row[17]
            output_row[numeric_start + 14] = (
                0.0 if row[17] == 0.0 else math.log(row[18]) - math.log(row[20])
            )
            output_row[numeric_start + 15] = row[19]
            output_row[numeric_start + 16] = math.log(row[20]) - math.log(
                float(account[symbol_index, 2])
            )
        return output

    def _transform_clock(self, raw: np.ndarray) -> np.ndarray:
        current_index = int(raw[0])
        timestamp_ns = int(raw[1])
        actor_count = self.agent_stop_index - self.start_index
        elapsed_actor = (current_index - self.start_index) / actor_count
        remaining_actor = (self.agent_stop_index - current_index) / actor_count
        start_ns = self.objective_start_ns
        horizon_ns = self.economic_horizon_seconds * 1_000_000_000
        elapsed_utc = (timestamp_ns - start_ns) / horizon_ns
        return np.asarray((elapsed_actor, remaining_actor, elapsed_utc))

    @staticmethod
    def _terminal_padding(
        n_symbols: int, n_features: int, n_globals: int
    ) -> dict[str, np.ndarray]:
        return {
            "features": np.zeros((n_symbols, n_features), dtype=np.float32),
            "feature_available": np.zeros((n_symbols, n_features), dtype=np.int8),
            "feature_staleness": np.zeros((n_symbols, n_features), dtype=np.float32),
            "global_features": np.zeros(n_globals, dtype=np.float32),
            "global_feature_available": np.zeros(n_globals, dtype=np.int8),
            "account": np.zeros((n_symbols, 4), dtype=np.float32),
            "portfolio": np.zeros(len(_PORTFOLIO_OUTPUT_FIELDS), dtype=np.float32),
            "risk": np.zeros(len(RISK_STATE_FIELDS), dtype=np.float32),
            "active_orders": np.zeros(
                (n_symbols, len(_ACTIVE_ORDER_OUTPUT_FIELDS)), dtype=np.float32
            ),
            "clock": np.zeros(3, dtype=np.float32),
            "terminal": np.ones(1, dtype=np.int8),
        }


def fit_portfolio_observation_transform(
    env: SharedPortfolioGymEnv,
) -> PortfolioObservationTransform:
    """Fit opt-in statistics only on the actor rows for one Gym environment."""

    if not isinstance(env, SharedPortfolioGymEnv):
        raise TypeError("env must be a SharedPortfolioGymEnv")
    dataset = env.dataset
    agent_stop = env.stop_index - env.execution_cost.order_latency_bars - 1
    start = env.start_index
    if agent_stop <= start:
        raise ValueError("environment has no actor fit rows")
    symbols = tuple(range(dataset.n_symbols))
    feature_indices = tuple(range(len(dataset.feature_names)))
    local = fit_ppo_feature_normalizer(
        dataset,
        feature_indices=feature_indices,
        fit_symbol_indices=symbols,
        start_index=start,
        stop_index=agent_stop,
    )
    global_values = np.asarray(
        dataset.global_features[start:agent_stop], dtype=np.float64
    )
    global_available = np.asarray(
        dataset.resolved_array("global_feature_available")[start:agent_stop],
        dtype=np.bool_,
    )
    means: list[float] = []
    scales: list[float] = []
    counts: list[int] = []
    for feature_index in range(global_values.shape[1]):
        values = global_values[:, feature_index]
        usable = global_available[:, feature_index] & np.isfinite(values)
        selected = values[usable]
        if selected.size == 0:
            raise ValueError("every global feature needs usable actor fit values")
        mean = float(np.mean(selected, dtype=np.float64))
        variance = float(np.mean((selected - mean) ** 2, dtype=np.float64))
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale > 1e-12 else 1.0)
        counts.append(int(selected.size))

    marks = np.asarray(dataset.resolved_array("mark_price"), dtype=np.float64)
    log_means = tuple(
        math.fsum(math.log(float(mark)) for mark in marks[start:agent_stop, symbol])
        / (agent_stop - start)
        for symbol in symbols
    )
    profile = env.market_order_profile
    return PortfolioObservationTransform(
        source_dataset_id=dataset.dataset_id,
        symbols=tuple(dataset.symbols),
        feature_names=tuple(dataset.feature_names),
        global_feature_names=tuple(dataset.global_feature_names),
        start_index=start,
        stop_index=env.stop_index,
        agent_stop_index=agent_stop,
        initial_capital=env.initial_capital,
        bound_digest=env.bound.digest,
        objective_digest=env.bound.objective.digest,
        clock_digest=env.bound.clock.digest,
        objective_start_ns=_datetime_ns(env.bound.objective.evaluation_start),
        economic_horizon_seconds=env.bound.clock.economic_horizon_seconds,
        native_interval_seconds=env.bound.clock.decision_interval_seconds,
        risk_config_digest=_digest(_risk_payload(env.risk_config)),
        execution_config_digest=_digest(_execution_payload(env.execution_cost)),
        market_order_profile_digest=None if profile is None else profile.digest,
        local_normalizer=local,
        global_mean=tuple(means),
        global_scale=tuple(scales),
        global_usable_counts=tuple(counts),
        reference_log_price_mean=log_means,
        reference_price_counts=(agent_stop - start,) * dataset.n_symbols,
    )


class PortfolioGymObservationWrapper(gym.ObservationWrapper):
    """Apply a frozen transform while preserving the native Gym transition."""

    def __init__(
        self,
        env: SharedPortfolioGymEnv,
        transform: PortfolioObservationTransform,
    ) -> None:
        if not isinstance(transform, PortfolioObservationTransform):
            raise TypeError("transform must be a PortfolioObservationTransform")
        transform.validate_environment(env)
        super().__init__(env)
        self.transform = transform
        self.observation_space = transform.observation_space
        self._poisoned = False
        self._last_failure: str | None = None

    @property
    def last_failure(self) -> str | None:
        """Return the last failure fact until a fresh reset succeeds."""

        return self._last_failure

    def observation(self, observation: object) -> dict[str, np.ndarray]:
        return self.transform.transform_observation(observation)

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        try:
            raw, info = cast(SharedPortfolioGymEnv, self.env).reset(
                seed=seed, options=options
            )
            transformed = self.observation(raw)
        except BaseException as error:
            self._poison(error)
            raise
        self._poisoned = False
        self._last_failure = None
        return transformed, info

    def step(
        self, action: object
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        if self._poisoned:
            raise RuntimeError("transform wrapper is unusable; reset is required")
        try:
            raw, reward, terminated, truncated, info = cast(
                SharedPortfolioGymEnv, self.env
            ).step(action)
            transformed = self.observation(raw)
        except BaseException as error:
            self._poison(error)
            raise
        return transformed, reward, terminated, truncated, info

    def _poison(self, error: BaseException) -> None:
        self._poisoned = True
        self._last_failure = f"{type(error).__name__}: {error}"


__all__ = [
    "PORTFOLIO_OBSERVATION_TRANSFORM_SCHEMA",
    "PORTFOLIO_TRANSFORMED_OBSERVATION_SCHEMA",
    "PortfolioGymObservationWrapper",
    "PortfolioObservationTransform",
    "fit_portfolio_observation_transform",
]
