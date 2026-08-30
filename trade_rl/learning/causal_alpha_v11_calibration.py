"""Causal pooled long/short edge calibration for Causal Alpha V11."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v9 import CausalAlphaV9Config
from trade_rl.learning.causal_alpha_v9_wave import (
    CausalAlphaV9TrainingRows,
    CausalAlphaV9WaveFit,
)
from trade_rl.learning.causal_alpha_v11 import CausalAlphaV11Config

CAUSAL_ALPHA_V11_SIGN_CALIBRATION_SCHEMA: Final = "causal_alpha_v11_sign_calibration_v1"
_DECISIONS_PER_HOUR: Final = 4


@dataclass(frozen=True, slots=True)
class CausalAlphaV11SignCalibration:
    """One pooled, symbol-free ridge calibration split by forecast direction."""

    calibration_start: int
    outer_cutoff: int
    maximum_label_end_index: int
    long_coefficients: tuple[float, float]
    short_coefficients: tuple[float, float]
    long_support: int
    short_support: int
    source_fit_digest: str
    pooled_rows_digest: str
    config_digest: str
    schema_version: str = CAUSAL_ALPHA_V11_SIGN_CALIBRATION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if (
            self.calibration_start >= self.outer_cutoff
            or self.maximum_label_end_index >= self.outer_cutoff
            or self.long_support <= 0
            or self.short_support <= 0
        ):
            raise ValueError("V11 sign calibration causal support is invalid")
        coefficients = self.long_coefficients + self.short_coefficients
        if len(coefficients) != 4 or not all(math.isfinite(x) for x in coefficients):
            raise ValueError("V11 sign calibration coefficients are invalid")
        for field_name in ("source_fit_digest", "pooled_rows_digest", "config_digest"):
            require_sha256(getattr(self, field_name), field=f"V11 {field_name}")
        if self.schema_version != CAUSAL_ALPHA_V11_SIGN_CALIBRATION_SCHEMA:
            raise ValueError("unsupported V11 sign calibration schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V11 sign calibration digest mismatch")
        object.__setattr__(self, "digest", expected)

    def calibrated_edge(self, *, direction: int, raw_edge: float) -> float:
        """Predict signed realized return for one qualified directional edge."""

        if direction not in (-1, 1) or not math.isfinite(raw_edge):
            raise ValueError("V11 calibrated edge input is invalid")
        intercept, slope = (
            self.long_coefficients if direction > 0 else self.short_coefficients
        )
        return float(intercept + slope * raw_edge)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "calibration_start": self.calibration_start,
            "config_digest": self.config_digest,
            "long_coefficients": self.long_coefficients,
            "long_support": self.long_support,
            "maximum_label_end_index": self.maximum_label_end_index,
            "outer_cutoff": self.outer_cutoff,
            "pooled_rows_digest": self.pooled_rows_digest,
            "schema_version": self.schema_version,
            "short_coefficients": self.short_coefficients,
            "short_support": self.short_support,
            "source_fit_digest": self.source_fit_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _ridge_coefficients(
    raw_edges: np.ndarray, signed_labels: np.ndarray, *, strength: float
) -> tuple[float, float]:
    design = np.column_stack((np.ones(len(raw_edges)), raw_edges))
    gram = design.T @ design
    penalty = strength * np.eye(2, dtype=np.float64)
    coefficients = np.linalg.solve(gram + penalty, design.T @ signed_labels)
    return float(coefficients[0]), float(coefficients[1])


def fit_causal_alpha_v11_sign_calibration(
    rows: Mapping[str, CausalAlphaV9TrainingRows],
    *,
    source_fit: CausalAlphaV9WaveFit,
    outer_cutoff: int,
    config: CausalAlphaV11Config,
) -> CausalAlphaV11SignCalibration:
    """Fit one week of pooled OOS predictions without crossing outer cutoff."""

    calibration_start = outer_cutoff - config.calibration_hours * _DECISIONS_PER_HOUR
    if source_fit.knowledge_cutoff != calibration_start:
        raise ValueError("V11 source fit cutoff must equal calibration start")
    v9_config = CausalAlphaV9Config()
    if source_fit.config_digest != v9_config.digest:
        raise ValueError("V11 source fit must use the frozen V9 config")
    records = dict(rows)
    if not records or any(key != record.symbol for key, record in records.items()):
        raise ValueError("V11 calibration row symbols are invalid")

    edge_blocks: list[np.ndarray] = []
    direction_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    decision_blocks: list[np.ndarray] = []
    end_blocks: list[np.ndarray] = []
    symbol_counts: list[tuple[str, int]] = []
    for symbol in sorted(records):
        record = records[symbol]
        if record.feature_names != source_fit.feature_names:
            raise ValueError("V11 calibration feature schema drifted")
        selected = (
            (record.decision_indices >= calibration_start)
            & (record.decision_indices < outer_cutoff)
            & (record.label_end_indices >= 0)
            & (record.label_end_indices < outer_cutoff)
            & (
                (outer_cutoff - record.decision_indices) % v9_config.horizon_decisions
                == 0
            )
            & np.all(record.feature_available, axis=1)
        )
        if not np.any(selected):
            raise ValueError(f"V11 {symbol} has no causal calibration rows")
        predictions = source_fit.predict_heads(record.features[selected])
        means = np.mean(predictions, axis=0, dtype=np.float64)
        uncertainties = np.std(predictions, axis=0, dtype=np.float64)
        signs = np.sign(predictions)
        agreed = np.all(signs == signs[0], axis=0) & (signs[0] != 0.0)
        raw_edges = np.abs(means) - uncertainties - v9_config.edge_margin
        directions = np.where(agreed & (raw_edges > 0.0), np.sign(means), 0.0).astype(
            np.int8
        )
        qualified = directions != 0
        if not np.any(qualified):
            raise ValueError(f"V11 {symbol} has no qualified calibration rows")
        edge_blocks.append(raw_edges[qualified])
        direction_blocks.append(directions[qualified])
        label_blocks.append(record.labels[selected][qualified])
        decision_blocks.append(record.decision_indices[selected][qualified])
        end_blocks.append(record.label_end_indices[selected][qualified])
        symbol_counts.append((symbol, int(np.count_nonzero(qualified))))

    edges = np.concatenate(edge_blocks)
    directions = np.concatenate(direction_blocks)
    labels = np.concatenate(label_blocks)
    decisions = np.concatenate(decision_blocks)
    ends = np.concatenate(end_blocks)
    signed_labels = directions.astype(np.float64) * labels
    long_mask = directions > 0
    short_mask = directions < 0
    if not np.any(long_mask) or not np.any(short_mask):
        raise ValueError("V11 calibration requires pooled long and short support")
    pooled_rows_digest = content_and_arrays_digest(
        {
            "calibration_start": calibration_start,
            "outer_cutoff": outer_cutoff,
            "schema_version": "causal_alpha_v11_calibration_rows_v1",
            "source_fit_digest": source_fit.digest,
            "symbol_counts": tuple(symbol_counts),
        },
        (
            ("decision_indices", decisions),
            ("label_end_indices", ends),
            ("raw_edges", edges),
            ("directions", directions),
            ("signed_labels", signed_labels),
        ),
    )
    return CausalAlphaV11SignCalibration(
        calibration_start=calibration_start,
        outer_cutoff=outer_cutoff,
        maximum_label_end_index=int(np.max(ends)),
        long_coefficients=_ridge_coefficients(
            edges[long_mask],
            signed_labels[long_mask],
            strength=config.calibration_ridge_strength,
        ),
        short_coefficients=_ridge_coefficients(
            edges[short_mask],
            signed_labels[short_mask],
            strength=config.calibration_ridge_strength,
        ),
        long_support=int(np.count_nonzero(long_mask)),
        short_support=int(np.count_nonzero(short_mask)),
        source_fit_digest=source_fit.digest,
        pooled_rows_digest=pooled_rows_digest,
        config_digest=config.digest,
    )


__all__ = [
    "CAUSAL_ALPHA_V11_SIGN_CALIBRATION_SCHEMA",
    "CausalAlphaV11SignCalibration",
    "fit_causal_alpha_v11_sign_calibration",
]
