from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/learning/causal_alpha_v4.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one match, got {count}: {old[:120]!r}")
    text = text.replace(old, new, 1)


replace_once(
    "from __future__ import annotations\n\nfrom collections.abc import Mapping\n",
    "from __future__ import annotations\n\nimport math\nfrom collections.abc import Mapping\n",
)
replace_once(
    "from dataclasses import field as dataclass_field\nfrom types import MappingProxyType\n",
    "from dataclasses import field as dataclass_field\nfrom enum import Enum\nfrom types import MappingProxyType\n",
)
replace_once(
    'CAUSAL_ALPHA_V4_HORIZONS: Final = ("4h", "24h", "72h")\n',
    'CAUSAL_ALPHA_V4_HORIZONS: Final = ("4h", "24h", "72h")\n'
    'CAUSAL_ALPHA_V4_UNCERTAINTY_SCHEMA: Final = "causal_alpha_v4_uncertainty_v1"\n'
    '_V4_MINIMUM_STATE_ESS: Final = 30.0\n',
)

addition = r'''

class V4ForecastState(str, Enum):
    NORMAL = "normal"
    HIGH_REALIZED_VOLATILITY = "high_realized_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    BASIS_POSITIONING_STRESS = "basis_positioning_stress"


_V4_FORECAST_STATES: Final = (
    V4ForecastState.NORMAL,
    V4ForecastState.HIGH_REALIZED_VOLATILITY,
    V4ForecastState.LOW_LIQUIDITY,
    V4ForecastState.BASIS_POSITIONING_STRESS,
)


@dataclass(frozen=True, slots=True)
class CausalAlphaV4UncertaintyCell:
    state: V4ForecastState
    support: int
    effective_sample_size: float
    global_rmse: float
    state_rmse: float | None
    selected_uncertainty: float
    fallback_reason: str | None

    def __post_init__(self) -> None:
        state = V4ForecastState(self.state)
        if isinstance(self.support, bool) or not isinstance(self.support, int) or self.support < 0:
            raise ValueError("V4 uncertainty support must be a non-negative integer")
        for field_name in (
            "effective_sample_size",
            "global_rmse",
            "selected_uncertainty",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"V4 uncertainty {field_name} must be non-negative")
        if self.state_rmse is not None and (
            not math.isfinite(self.state_rmse) or self.state_rmse < 0.0
        ):
            raise ValueError("V4 uncertainty state_rmse must be null or non-negative")
        if self.effective_sample_size < _V4_MINIMUM_STATE_ESS:
            if (
                self.fallback_reason != "insufficient_state_ess"
                or self.selected_uncertainty != self.global_rmse
            ):
                raise ValueError("V4 low-ESS state must fall back to global RMSE")
        else:
            if (
                self.fallback_reason is not None
                or self.state_rmse is None
                or self.selected_uncertainty != self.state_rmse
            ):
                raise ValueError("V4 supported state must use its state RMSE")
        object.__setattr__(self, "state", state)

    def to_payload(self) -> dict[str, object]:
        return {
            "effective_sample_size": self.effective_sample_size,
            "fallback_reason": self.fallback_reason,
            "global_rmse": self.global_rmse,
            "selected_uncertainty": self.selected_uncertainty,
            "state": self.state.value,
            "state_rmse": self.state_rmse,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaV4UncertaintyModel:
    high_realized_volatility_threshold: float
    low_liquidity_threshold: float
    basis_positioning_stress_threshold: float
    threshold_digest: str
    global_rmse: Mapping[str, float]
    cells: Mapping[str, Mapping[V4ForecastState, CausalAlphaV4UncertaintyCell]]
    minimum_state_effective_sample_size: float = _V4_MINIMUM_STATE_ESS
    schema_version: str = CAUSAL_ALPHA_V4_UNCERTAINTY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "high_realized_volatility_threshold",
            "low_liquidity_threshold",
            "basis_positioning_stress_threshold",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"V4 uncertainty {field_name} must be finite")
        if self.basis_positioning_stress_threshold < 0.0:
            raise ValueError("V4 stress threshold must be non-negative")
        require_sha256(self.threshold_digest, field="V4 uncertainty threshold_digest")
        if self.minimum_state_effective_sample_size != _V4_MINIMUM_STATE_ESS:
            raise ValueError("V4 minimum state ESS must remain 30.0")
        if self.schema_version != CAUSAL_ALPHA_V4_UNCERTAINTY_SCHEMA:
            raise ValueError("unsupported V4 uncertainty schema")

        global_rmse = dict(self.global_rmse)
        if tuple(global_rmse) != CAUSAL_ALPHA_V4_HORIZONS:
            raise ValueError("V4 uncertainty global RMSE horizon order drifted")
        for horizon, value in global_rmse.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"V4 uncertainty global RMSE {horizon} is invalid")

        raw_cells = dict(self.cells)
        if tuple(raw_cells) != CAUSAL_ALPHA_V4_HORIZONS:
            raise ValueError("V4 uncertainty cell horizon order drifted")
        cells: dict[str, Mapping[V4ForecastState, CausalAlphaV4UncertaintyCell]] = {}
        for horizon in CAUSAL_ALPHA_V4_HORIZONS:
            horizon_cells = dict(raw_cells[horizon])
            if tuple(horizon_cells) != _V4_FORECAST_STATES:
                raise ValueError("V4 uncertainty state order drifted")
            for state, cell in horizon_cells.items():
                if not isinstance(cell, CausalAlphaV4UncertaintyCell) or cell.state is not state:
                    raise TypeError("V4 uncertainty cell identity drifted")
                if cell.global_rmse != global_rmse[horizon]:
                    raise ValueError("V4 uncertainty cell/global RMSE mismatch")
            cells[horizon] = MappingProxyType(horizon_cells)

        payload = {
            "basis_positioning_stress_threshold": self.basis_positioning_stress_threshold,
            "cells": tuple(
                (
                    horizon,
                    tuple(cells[horizon][state].to_payload() for state in _V4_FORECAST_STATES),
                )
                for horizon in CAUSAL_ALPHA_V4_HORIZONS
            ),
            "global_rmse": tuple(global_rmse.items()),
            "high_realized_volatility_threshold": self.high_realized_volatility_threshold,
            "low_liquidity_threshold": self.low_liquidity_threshold,
            "minimum_state_effective_sample_size": self.minimum_state_effective_sample_size,
            "schema_version": self.schema_version,
            "threshold_digest": self.threshold_digest,
        }
        expected = content_digest(payload)
        if self.digest and self.digest != expected:
            raise ValueError("V4 uncertainty model digest mismatch")
        object.__setattr__(self, "global_rmse", MappingProxyType(global_rmse))
        object.__setattr__(self, "cells", MappingProxyType(cells))
        object.__setattr__(self, "digest", expected)

    def resolve_states(
        self,
        *,
        realized_volatility: object,
        liquidity: object,
        basis_positioning_stress: object,
    ) -> np.ndarray:
        volatility = np.asarray(realized_volatility, dtype=np.float64).reshape(-1)
        liquidity_values = np.asarray(liquidity, dtype=np.float64).reshape(-1)
        stress = np.asarray(basis_positioning_stress, dtype=np.float64).reshape(-1)
        if (
            volatility.size == 0
            or liquidity_values.shape != volatility.shape
            or stress.shape != volatility.shape
            or not np.isfinite(volatility).all()
            or not np.isfinite(liquidity_values).all()
            or not np.isfinite(stress).all()
        ):
            raise ValueError("V4 uncertainty state inputs must be aligned and finite")
        states = np.full(volatility.shape, V4ForecastState.NORMAL, dtype=object)
        high_volatility = volatility >= self.high_realized_volatility_threshold
        low_liquidity = liquidity_values <= self.low_liquidity_threshold
        positioning_stress = (
            np.abs(stress) >= self.basis_positioning_stress_threshold
        )
        states[high_volatility] = V4ForecastState.HIGH_REALIZED_VOLATILITY
        states[low_liquidity] = V4ForecastState.LOW_LIQUIDITY
        states[positioning_stress] = V4ForecastState.BASIS_POSITIONING_STRESS
        return states

    def resolve_uncertainty(
        self,
        *,
        horizon: str,
        realized_volatility: object,
        liquidity: object,
        basis_positioning_stress: object,
    ) -> np.ndarray:
        if horizon not in CAUSAL_ALPHA_V4_HORIZONS:
            raise ValueError("unsupported V4 uncertainty horizon")
        states = self.resolve_states(
            realized_volatility=realized_volatility,
            liquidity=liquidity,
            basis_positioning_stress=basis_positioning_stress,
        )
        return np.asarray(
            [self.cells[horizon][state].selected_uncertainty for state in states],
            dtype=np.float64,
        )


def _uncertainty_effective_sample_size(weights: np.ndarray) -> float:
    total = float(np.sum(weights, dtype=np.float64))
    squared = float(np.sum(np.square(weights), dtype=np.float64))
    if total <= 0.0 or squared <= 0.0:
        return 0.0
    return float(total * total / squared)


def _uncertainty_weighted_rmse(
    prediction: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    total = float(np.sum(weights, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("V4 uncertainty RMSE requires positive weight")
    residual = labels - prediction
    value = math.sqrt(
        float(np.sum(weights * np.square(residual), dtype=np.float64) / total)
    )
    if not math.isfinite(value):
        raise ValueError("V4 uncertainty RMSE became non-finite")
    return value


def _uncertainty_horizon_map(
    value: Mapping[str, object], *, rows: int, field_name: str, weights: bool = False
) -> dict[str, np.ndarray]:
    if tuple(value) != CAUSAL_ALPHA_V4_HORIZONS:
        raise ValueError(f"V4 uncertainty {field_name} horizon order drifted")
    result: dict[str, np.ndarray] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        array = np.asarray(value[horizon], dtype=np.float64).reshape(-1).copy(order="C")
        if array.shape != (rows,):
            raise ValueError(f"V4 uncertainty {field_name}[{horizon}] is misaligned")
        if weights and (not np.isfinite(array).all() or np.any(array < 0.0)):
            raise ValueError(f"V4 uncertainty {field_name}[{horizon}] is invalid")
        array.setflags(write=False)
        result[horizon] = array
    return result


def fit_causal_alpha_v4_uncertainty(
    *,
    final_predictions: Mapping[str, object],
    labels: Mapping[str, object],
    weights: Mapping[str, object],
    state_eligible: object,
    realized_volatility: object,
    liquidity: object,
    basis_positioning_stress: object,
) -> CausalAlphaV4UncertaintyModel:
    """Fit train-prefix state RMSE from final hierarchical forecast residuals."""

    eligible = np.asarray(state_eligible, dtype=np.bool_).reshape(-1)
    rows = int(eligible.size)
    if rows == 0 or not np.any(eligible):
        raise ValueError("V4 uncertainty requires eligible train-prefix state rows")
    volatility = np.asarray(realized_volatility, dtype=np.float64).reshape(-1)
    liquidity_values = np.asarray(liquidity, dtype=np.float64).reshape(-1)
    stress = np.asarray(basis_positioning_stress, dtype=np.float64).reshape(-1)
    if (
        volatility.shape != (rows,)
        or liquidity_values.shape != (rows,)
        or stress.shape != (rows,)
        or not np.isfinite(volatility[eligible]).all()
        or not np.isfinite(liquidity_values[eligible]).all()
        or not np.isfinite(stress[eligible]).all()
    ):
        raise ValueError("V4 uncertainty eligible state variables must be finite")

    high_volatility_threshold = float(np.quantile(volatility[eligible], 0.80))
    low_liquidity_threshold = float(np.quantile(liquidity_values[eligible], 0.20))
    stress_threshold = float(np.quantile(np.abs(stress[eligible]), 0.80))
    threshold_digest = content_digest(
        {
            "basis_positioning_stress_quantile": 0.80,
            "basis_positioning_stress_threshold": stress_threshold,
            "high_realized_volatility_quantile": 0.80,
            "high_realized_volatility_threshold": high_volatility_threshold,
            "low_liquidity_quantile": 0.20,
            "low_liquidity_threshold": low_liquidity_threshold,
            "schema_version": "causal_alpha_v4_uncertainty_thresholds_v1",
        }
    )

    prediction_map = _uncertainty_horizon_map(
        final_predictions, rows=rows, field_name="final_predictions"
    )
    label_map = _uncertainty_horizon_map(labels, rows=rows, field_name="labels")
    weight_map = _uncertainty_horizon_map(
        weights, rows=rows, field_name="weights", weights=True
    )

    state_model = CausalAlphaV4UncertaintyModel.__new__(CausalAlphaV4UncertaintyModel)
    object.__setattr__(state_model, "high_realized_volatility_threshold", high_volatility_threshold)
    object.__setattr__(state_model, "low_liquidity_threshold", low_liquidity_threshold)
    object.__setattr__(state_model, "basis_positioning_stress_threshold", stress_threshold)
    states = CausalAlphaV4UncertaintyModel.resolve_states(
        state_model,
        realized_volatility=np.where(eligible, volatility, high_volatility_threshold),
        liquidity=np.where(eligible, liquidity_values, low_liquidity_threshold + 1.0),
        basis_positioning_stress=np.where(eligible, stress, 0.0),
    )

    global_rmse: dict[str, float] = {}
    cells: dict[str, dict[V4ForecastState, CausalAlphaV4UncertaintyCell]] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        prediction = prediction_map[horizon]
        target = label_map[horizon]
        horizon_weights = weight_map[horizon]
        if np.any((horizon_weights > 0.0) & ~eligible):
            raise ValueError("V4 uncertainty weights cannot cross the train-prefix state scope")
        positive = horizon_weights > 0.0
        if not np.any(positive):
            raise ValueError(f"V4 uncertainty {horizon} has no positive weight")
        if not np.isfinite(prediction[positive]).all() or not np.isfinite(target[positive]).all():
            raise ValueError(f"V4 uncertainty {horizon} weighted rows must be finite")
        global_value = _uncertainty_weighted_rmse(
            prediction[positive], target[positive], horizon_weights[positive]
        )
        global_rmse[horizon] = global_value
        horizon_cells: dict[V4ForecastState, CausalAlphaV4UncertaintyCell] = {}
        for state in _V4_FORECAST_STATES:
            mask = positive & (states == state)
            selected_weights = horizon_weights[mask]
            support = int(np.count_nonzero(mask))
            ess = _uncertainty_effective_sample_size(selected_weights)
            state_rmse = (
                None
                if support == 0
                else _uncertainty_weighted_rmse(
                    prediction[mask], target[mask], selected_weights
                )
            )
            fallback = ess < _V4_MINIMUM_STATE_ESS
            horizon_cells[state] = CausalAlphaV4UncertaintyCell(
                state=state,
                support=support,
                effective_sample_size=ess,
                global_rmse=global_value,
                state_rmse=state_rmse,
                selected_uncertainty=(
                    global_value if fallback else float(state_rmse)
                ),
                fallback_reason=("insufficient_state_ess" if fallback else None),
            )
        cells[horizon] = horizon_cells

    return CausalAlphaV4UncertaintyModel(
        high_realized_volatility_threshold=high_volatility_threshold,
        low_liquidity_threshold=low_liquidity_threshold,
        basis_positioning_stress_threshold=stress_threshold,
        threshold_digest=threshold_digest,
        global_rmse=global_rmse,
        cells=cells,
    )
'''

replace_once("\n\n__all__ = [\n", addition + "\n\n__all__ = [\n")
replace_once(
    '''    "CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA",\n    "CausalAlphaV4FitConfig",\n''',
    '''    "CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA",\n    "CAUSAL_ALPHA_V4_UNCERTAINTY_SCHEMA",\n    "CausalAlphaV4FitConfig",\n''',
)
replace_once(
    '''    "CausalAlphaV4SymbolSamples",\n    "build_causal_alpha_v4_forecast",\n''',
    '''    "CausalAlphaV4SymbolSamples",\n    "CausalAlphaV4UncertaintyCell",\n    "CausalAlphaV4UncertaintyModel",\n    "V4ForecastState",\n    "build_causal_alpha_v4_forecast",\n''',
)
replace_once(
    '''    "build_causal_alpha_v4_residual_labels",\n]\n''',
    '''    "build_causal_alpha_v4_residual_labels",\n    "fit_causal_alpha_v4_uncertainty",\n]\n''',
)
path.write_text(text, encoding="utf-8")
