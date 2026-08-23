from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/learning/causal_alpha_v4.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one match, got {count}: {old[:100]!r}")
    text = text.replace(old, new, 1)


replace_once(
    "from dataclasses import dataclass\nfrom typing import Any, Final\n",
    "from collections.abc import Mapping\n"
    "from dataclasses import dataclass, field\n"
    "from types import MappingProxyType\n"
    "from typing import Any, Final\n",
)
replace_once(
    "import numpy as np\n\nfrom trade_rl.data.identity import content_and_arrays_digest\n",
    "import numpy as np\n\n"
    "from trade_rl.artifacts.hashing import content_digest\n"
    "from trade_rl.data.identity import content_and_arrays_digest\n",
)
constants = (
    'CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA: Final = "causal_alpha_v4_symbol_samples_v1"\n'
    'CAUSAL_ALPHA_V4_RESIDUAL_LABELS_SCHEMA: Final = "causal_alpha_v4_residual_labels_v1"\n'
)
replace_once(
    constants,
    constants
    + 'CAUSAL_ALPHA_V4_FIT_CONFIG_SCHEMA: Final = "causal_alpha_v4_fit_config_v1"\n'
    + 'CAUSAL_ALPHA_V4_FORECAST_SCHEMA: Final = "causal_alpha_v4_forecast_v1"\n'
    + 'CAUSAL_ALPHA_V4_HORIZONS: Final = ("4h", "24h", "72h")\n',
)

addition = r'''

@dataclass(frozen=True, slots=True)
class CausalAlphaV4FitConfig:
    """The single predeclared hierarchical ridge hypothesis for V4."""

    market_ridge_strength: float = 1.0
    residual_ridge_strength: float = 0.1
    direction_ridge_strength: float = 0.1
    schema_version: str = CAUSAL_ALPHA_V4_FIT_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAUSAL_ALPHA_V4_FIT_CONFIG_SCHEMA:
            raise ValueError("unsupported V4 fit config schema")
        if self.market_ridge_strength != 1.0:
            raise ValueError("V4 market ridge strength must remain 1.0")
        if self.residual_ridge_strength != 0.1:
            raise ValueError("V4 residual ridge strength must remain 0.1")
        if self.direction_ridge_strength != 0.1:
            raise ValueError("V4 direction ridge strength must remain 0.1")

    @property
    def digest(self) -> str:
        return content_digest(self)


def _canonical_horizon_arrays(
    value: Mapping[str, object],
    *,
    field_name: str,
    rows: int,
) -> dict[str, np.ndarray]:
    if set(value) != set(CAUSAL_ALPHA_V4_HORIZONS):
        raise ValueError(f"V4 forecast {field_name} horizon set is invalid")
    resolved: dict[str, np.ndarray] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        array = (
            np.asarray(value[horizon], dtype=np.float64)
            .reshape(-1)
            .copy(order="C")
        )
        if array.shape != (rows,) or not np.isfinite(array).all():
            raise ValueError(
                f"V4 forecast {field_name}[{horizon}] must be aligned and finite"
            )
        array.setflags(write=False)
        resolved[horizon] = array
    return resolved


def _canonical_horizon_digests(
    value: Mapping[str, str],
    *,
    field_name: str,
) -> dict[str, str]:
    if set(value) != set(CAUSAL_ALPHA_V4_HORIZONS):
        raise ValueError(f"V4 forecast {field_name} horizon set is invalid")
    resolved: dict[str, str] = {}
    for horizon in CAUSAL_ALPHA_V4_HORIZONS:
        digest = str(value[horizon])
        require_sha256(digest, field=f"V4 forecast {field_name}[{horizon}]")
        resolved[horizon] = digest
    return resolved


@dataclass(frozen=True, slots=True)
class CausalAlphaV4Forecast:
    """One symbol's immutable three-horizon hierarchical forecast."""

    symbol: str
    decision_indices: np.ndarray
    beta: np.ndarray
    beta_available: np.ndarray
    market_predictions: Mapping[str, np.ndarray]
    residual_predictions: Mapping[str, np.ndarray]
    direction_scores: Mapping[str, np.ndarray]
    market_model_digests: Mapping[str, str]
    residual_model_digests: Mapping[str, str]
    direction_model_digests: Mapping[str, str]
    fit_digest: str
    beta_scaled_market_contributions: Mapping[str, np.ndarray] = field(
        init=False, default_factory=dict
    )
    final_predictions: Mapping[str, np.ndarray] = field(
        init=False, default_factory=dict
    )
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("V4 forecast symbol must be non-empty")
        decisions = _readonly(self.decision_indices, dtype=np.int64).reshape(-1)
        rows = int(decisions.size)
        if rows == 0 or np.any(decisions < 0) or np.any(np.diff(decisions) <= 0):
            raise ValueError("V4 forecast decisions must be strictly increasing")
        beta = _readonly(self.beta, dtype=np.float64).reshape(-1)
        beta_available = _readonly(self.beta_available, dtype=np.bool_).reshape(-1)
        if beta.shape != (rows,) or beta_available.shape != (rows,):
            raise ValueError("V4 forecast beta arrays must be aligned")
        if not np.isfinite(beta).all():
            raise ValueError("V4 forecast beta must be finite")
        if np.any(beta[beta_available] < -3.0) or np.any(
            beta[beta_available] > 3.0
        ):
            raise ValueError("V4 forecast beta exceeds authored bounds")

        market = _canonical_horizon_arrays(
            self.market_predictions,
            field_name="market_predictions",
            rows=rows,
        )
        residual = _canonical_horizon_arrays(
            self.residual_predictions,
            field_name="residual_predictions",
            rows=rows,
        )
        direction = _canonical_horizon_arrays(
            self.direction_scores,
            field_name="direction_scores",
            rows=rows,
        )
        market_digests = _canonical_horizon_digests(
            self.market_model_digests,
            field_name="market_model_digests",
        )
        residual_digests = _canonical_horizon_digests(
            self.residual_model_digests,
            field_name="residual_model_digests",
        )
        direction_digests = _canonical_horizon_digests(
            self.direction_model_digests,
            field_name="direction_model_digests",
        )
        require_sha256(self.fit_digest, field="V4 forecast fit_digest")

        beta_scaled: dict[str, np.ndarray] = {}
        final: dict[str, np.ndarray] = {}
        for horizon in CAUSAL_ALPHA_V4_HORIZONS:
            contribution = np.asarray(beta * market[horizon], dtype=np.float64)
            composed = np.asarray(
                contribution + residual[horizon], dtype=np.float64
            )
            contribution.setflags(write=False)
            composed.setflags(write=False)
            beta_scaled[horizon] = contribution
            final[horizon] = composed

        expected = content_and_arrays_digest(
            {
                "direction_model_digests": tuple(direction_digests.items()),
                "fit_digest": self.fit_digest,
                "market_model_digests": tuple(market_digests.items()),
                "residual_model_digests": tuple(residual_digests.items()),
                "schema_version": CAUSAL_ALPHA_V4_FORECAST_SCHEMA,
                "symbol": self.symbol,
            },
            (
                ("decision_indices", decisions),
                ("beta", beta),
                ("beta_available", beta_available),
                *tuple(
                    (f"market_prediction:{horizon}", market[horizon])
                    for horizon in CAUSAL_ALPHA_V4_HORIZONS
                ),
                *tuple(
                    (f"residual_prediction:{horizon}", residual[horizon])
                    for horizon in CAUSAL_ALPHA_V4_HORIZONS
                ),
                *tuple(
                    (f"direction_score:{horizon}", direction[horizon])
                    for horizon in CAUSAL_ALPHA_V4_HORIZONS
                ),
                *tuple(
                    (f"beta_market:{horizon}", beta_scaled[horizon])
                    for horizon in CAUSAL_ALPHA_V4_HORIZONS
                ),
                *tuple(
                    (f"final_prediction:{horizon}", final[horizon])
                    for horizon in CAUSAL_ALPHA_V4_HORIZONS
                ),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 forecast digest mismatch")
        object.__setattr__(self, "decision_indices", decisions)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "beta_available", beta_available)
        object.__setattr__(
            self, "market_predictions", MappingProxyType(market)
        )
        object.__setattr__(
            self, "residual_predictions", MappingProxyType(residual)
        )
        object.__setattr__(
            self, "direction_scores", MappingProxyType(direction)
        )
        object.__setattr__(
            self, "market_model_digests", MappingProxyType(market_digests)
        )
        object.__setattr__(
            self, "residual_model_digests", MappingProxyType(residual_digests)
        )
        object.__setattr__(
            self, "direction_model_digests", MappingProxyType(direction_digests)
        )
        object.__setattr__(
            self,
            "beta_scaled_market_contributions",
            MappingProxyType(beta_scaled),
        )
        object.__setattr__(
            self, "final_predictions", MappingProxyType(final)
        )
        object.__setattr__(self, "digest", expected)


def build_causal_alpha_v4_forecast(
    *,
    symbol: str,
    decision_indices: object,
    beta: object,
    beta_available: object,
    market_predictions: Mapping[str, object],
    residual_predictions: Mapping[str, object],
    direction_scores: Mapping[str, object],
    market_model_digests: Mapping[str, str],
    residual_model_digests: Mapping[str, str],
    direction_model_digests: Mapping[str, str],
    fit_digest: str,
) -> CausalAlphaV4Forecast:
    """Compose persisted-beta market and shared-residual V4 forecasts."""

    return CausalAlphaV4Forecast(
        symbol=symbol,
        decision_indices=np.asarray(decision_indices, dtype=np.int64),
        beta=np.asarray(beta, dtype=np.float64),
        beta_available=np.asarray(beta_available, dtype=np.bool_),
        market_predictions={
            horizon: np.asarray(value, dtype=np.float64)
            for horizon, value in market_predictions.items()
        },
        residual_predictions={
            horizon: np.asarray(value, dtype=np.float64)
            for horizon, value in residual_predictions.items()
        },
        direction_scores={
            horizon: np.asarray(value, dtype=np.float64)
            for horizon, value in direction_scores.items()
        },
        market_model_digests=dict(market_model_digests),
        residual_model_digests=dict(residual_model_digests),
        direction_model_digests=dict(direction_model_digests),
        fit_digest=fit_digest,
    )
'''
replace_once("\n\n__all__ = [\n", addition + "\n\n__all__ = [\n")
replace_once(
    '''__all__ = [
    "CAUSAL_ALPHA_V4_RESIDUAL_LABELS_SCHEMA",
    "CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA",
    "CausalAlphaV4ResidualLabels",
    "CausalAlphaV4SymbolSamples",
    "build_causal_alpha_v4_residual_labels",
]
''',
    '''__all__ = [
    "CAUSAL_ALPHA_V4_FIT_CONFIG_SCHEMA",
    "CAUSAL_ALPHA_V4_FORECAST_SCHEMA",
    "CAUSAL_ALPHA_V4_HORIZONS",
    "CAUSAL_ALPHA_V4_RESIDUAL_LABELS_SCHEMA",
    "CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA",
    "CausalAlphaV4FitConfig",
    "CausalAlphaV4Forecast",
    "CausalAlphaV4ResidualLabels",
    "CausalAlphaV4SymbolSamples",
    "build_causal_alpha_v4_forecast",
    "build_causal_alpha_v4_residual_labels",
]
''',
)
path.write_text(text, encoding="utf-8")
