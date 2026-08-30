from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v9 import CausalAlphaV9Config
from trade_rl.learning.causal_alpha_v9_wave import (
    CausalAlphaV9TrainingRows,
    CausalAlphaV9WaveFit,
)
from trade_rl.learning.causal_alpha_v11 import CausalAlphaV11Config
from trade_rl.learning.causal_alpha_v11_calibration import (
    fit_causal_alpha_v11_sign_calibration,
)


def _source_fit(*, cutoff: int) -> CausalAlphaV9WaveFit:
    return CausalAlphaV9WaveFit(
        knowledge_cutoff=cutoff,
        maximum_label_end_index=cutoff - 1,
        feature_names=("signal",),
        feature_mean=np.zeros(1),
        feature_scale=np.ones(1),
        hidden_weights=np.zeros((3, 1, 1)),
        hidden_bias=np.zeros((3, 1)),
        coefficients=np.asarray([[1.0, 0.0]] * 3),
        training_row_count=10,
        config_digest=CausalAlphaV9Config().digest,
    )


def _rows(symbol: str, *, outer_cutoff: int) -> CausalAlphaV9TrainingRows:
    start = outer_cutoff - 168 * 4
    decisions = np.arange(start, outer_cutoff, 16, dtype=np.int64)
    signal = np.where(np.arange(len(decisions)) % 2 == 0, 0.01, -0.02)
    labels = np.where(signal > 0.0, 0.03, -0.05)
    return CausalAlphaV9TrainingRows(
        symbol=symbol,
        decision_indices=decisions,
        label_end_indices=decisions + 15,
        features=signal.reshape(-1, 1),
        feature_available=np.ones((len(decisions), 1), dtype=np.bool_),
        labels=labels,
        feature_names=("signal",),
    )


def test_v11_sign_calibration_is_pooled_causal_and_symbol_free() -> None:
    outer_cutoff = 10_000
    fit = fit_causal_alpha_v11_sign_calibration(
        rows={
            "A": _rows("A", outer_cutoff=outer_cutoff),
            "B": _rows("B", outer_cutoff=outer_cutoff),
        },
        source_fit=_source_fit(cutoff=outer_cutoff - 168 * 4),
        outer_cutoff=outer_cutoff,
        config=CausalAlphaV11Config(),
    )

    assert fit.maximum_label_end_index < outer_cutoff
    assert fit.long_support > 0 and fit.short_support > 0
    assert not hasattr(fit, "symbol_coefficients")
    assert fit.calibrated_edge(direction=1, raw_edge=0.02) > 0.0
    assert fit.calibrated_edge(direction=-1, raw_edge=0.02) > 0.0


def test_v11_sign_calibration_digest_changes_with_coefficients() -> None:
    outer_cutoff = 10_000
    first = fit_causal_alpha_v11_sign_calibration(
        rows={"A": _rows("A", outer_cutoff=outer_cutoff)},
        source_fit=_source_fit(cutoff=outer_cutoff - 168 * 4),
        outer_cutoff=outer_cutoff,
        config=CausalAlphaV11Config(),
    )

    changed = replace(first, long_coefficients=(9.0, 9.0), digest="")

    assert first.digest != changed.digest


def test_v11_sign_calibration_rejects_source_fit_at_wrong_cutoff() -> None:
    outer_cutoff = 10_000

    with pytest.raises(ValueError, match="calibration start"):
        fit_causal_alpha_v11_sign_calibration(
            rows={"A": _rows("A", outer_cutoff=outer_cutoff)},
            source_fit=_source_fit(cutoff=outer_cutoff - 100),
            outer_cutoff=outer_cutoff,
            config=CausalAlphaV11Config(),
        )


def test_v11_sign_calibration_excludes_labels_crossing_outer_cutoff() -> None:
    outer_cutoff = 10_000
    rows = _rows("A", outer_cutoff=outer_cutoff)
    bad_ends = rows.label_end_indices.copy()
    bad_ends[-1] = outer_cutoff
    bad = replace(rows, label_end_indices=bad_ends)

    fit = fit_causal_alpha_v11_sign_calibration(
        rows={"A": bad},
        source_fit=_source_fit(cutoff=outer_cutoff - 168 * 4),
        outer_cutoff=outer_cutoff,
        config=CausalAlphaV11Config(),
    )

    assert fit.maximum_label_end_index < outer_cutoff
