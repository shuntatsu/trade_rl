from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.universal_features import UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
from trade_rl.data.v4_context import V4ContextBlock
from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4FitConfig,
    CausalAlphaV4SymbolSamples,
)
from trade_rl.learning.causal_alpha_v5 import CausalAlphaV5CalibrationConfig
from trade_rl.workflows.universal_causal_alpha_v4_fitting import fit_causal_alpha_v4
from trade_rl.workflows.universal_causal_alpha_v5_calibration import (
    CausalAlphaV5CalibrationSplit,
    calibrate_causal_alpha_v5_forecast,
    fit_causal_alpha_v5_calibration,
)

_SYMBOLS = (
    "APTUSDT",
    "ARBUSDT",
    "BCHUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "SOLUSDT",
    "XRPUSDT",
)


def _digest(char: str) -> str:
    return char * 64


def _block(
    *, decisions: np.ndarray, name: str, values: np.ndarray, source: str
) -> V4ContextBlock:
    matrix = values[:, None].astype(np.float64)
    return V4ContextBlock(
        feature_names=(name,),
        decision_indices=decisions,
        values=matrix,
        available=np.ones(matrix.shape, dtype=np.bool_),
        staleness_hours=np.zeros(matrix.shape, dtype=np.float64),
        source_digest=_digest(source),
    )


def _sample(*, symbol: str, rows: int = 1_100) -> CausalAlphaV4SymbolSamples:
    decisions = np.arange(rows, dtype=np.int64)
    phase = float(sum(map(ord, symbol)) % 17) / 17.0
    global_x = np.sin(decisions * 0.013) + 0.2 * np.cos(decisions * 0.003)
    local_x = np.cos(decisions * 0.021 + phase)
    target_x = (np.sin(decisions * 0.017 + phase) + 0.1 * global_x)[:, None]
    descriptors = np.column_stack(
        tuple(
            0.01 * float(index + 1)
            + 0.001 * phase
            + decisions.astype(np.float64) * 1e-7
            for index in range(len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES))
        )
    )
    beta = np.ones(rows, dtype=np.float64)
    if symbol != "BTCUSDT":
        beta[:] = 1.0 + 0.2 * phase
    market = 0.008 * global_x + 0.001
    residual = 0.004 * local_x + 0.0015 * target_x[:, 0]
    labels_4h = beta * market + residual
    labels_24h = 2.0 * labels_4h + 0.001 * np.sin(decisions * 0.031)
    labels_72h = 3.0 * labels_24h + 0.002 * np.cos(decisions * 0.011)
    return CausalAlphaV4SymbolSamples(
        symbol=symbol,
        dataset_id=_digest("a"),
        target_local_feature_names=("target_x",),
        target_local_feature_schema_digest=_digest("b"),
        source_sample_digest=_digest("c"),
        source_context_digest=_digest("d"),
        decision_indices=decisions,
        target_local_features=target_x,
        target_local_available=np.ones(target_x.shape, dtype=np.bool_),
        instrument_descriptor_names=UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
        instrument_descriptors=descriptors,
        instrument_descriptor_available=np.ones(descriptors.shape, dtype=np.bool_),
        local_context=_block(
            decisions=decisions, name="local_x", values=local_x, source="1"
        ),
        global_context=_block(
            decisions=decisions, name="global_x", values=global_x, source="2"
        ),
        beta=beta,
        beta_available=np.ones(rows, dtype=np.bool_),
        labels_4h=labels_4h,
        label_end_indices_4h=decisions + 1,
        labels_24h=labels_24h,
        label_end_indices_24h=decisions + 2,
        labels_72h=labels_72h,
        label_end_indices_72h=decisions + 4,
    )


def _samples(*, rows: int = 1_100) -> dict[str, CausalAlphaV4SymbolSamples]:
    return {symbol: _sample(symbol=symbol, rows=rows) for symbol in _SYMBOLS}


def _split(
    samples: dict[str, CausalAlphaV4SymbolSamples],
    *,
    train_stop: int = 1_000,
    config: CausalAlphaV5CalibrationConfig | None = None,
) -> CausalAlphaV5CalibrationSplit:
    return CausalAlphaV5CalibrationSplit.from_samples(
        train_symbols=_SYMBOLS,
        samples=samples,
        train_stop=train_stop,
        config=CausalAlphaV5CalibrationConfig() if config is None else config,
    )


def _v4_fit(
    samples: dict[str, CausalAlphaV4SymbolSamples],
    split: CausalAlphaV5CalibrationSplit,
):
    return fit_causal_alpha_v4(
        train_symbols=_SYMBOLS,
        samples=samples,
        knowledge_cutoff=split.calibration_start,
        config=CausalAlphaV4FitConfig(),
    )


def _uncertainty(
    samples: dict[str, CausalAlphaV4SymbolSamples],
) -> dict[str, np.ndarray]:
    return {
        symbol: np.full(len(sample.decision_indices), 0.01, dtype=np.float64)
        for symbol, sample in samples.items()
    }


def _fit(
    samples: dict[str, CausalAlphaV4SymbolSamples],
    *,
    train_stop: int = 1_000,
    config: CausalAlphaV5CalibrationConfig | None = None,
):
    resolved = CausalAlphaV5CalibrationConfig() if config is None else config
    split = _split(samples, train_stop=train_stop, config=resolved)
    v4_fit = _v4_fit(samples, split)
    fit = fit_causal_alpha_v5_calibration(
        train_symbols=_SYMBOLS,
        samples=samples,
        v4_fit=v4_fit,
        slow_uncertainty=_uncertainty(samples),
        train_stop=train_stop,
        config=resolved,
    )
    return split, v4_fit, fit


def test_v5_split_is_chronological_purged_and_has_no_mask_escape_hatch() -> None:
    samples = _samples()
    split = _split(samples)

    assert split.calibration_start == 498
    assert split.train_stop == 1_000
    assert split.block_boundaries == (498, 623, 748, 872, 1_000)
    assert tuple(split.train_symbols) == _SYMBOLS
    parameters = inspect.signature(
        CausalAlphaV5CalibrationSplit.from_samples
    ).parameters
    assert "mask" not in parameters
    reference = samples["BTCUSDT"]
    base = reference.label_end_indices_72h < split.calibration_start
    calibration = (reference.decision_indices >= split.calibration_start) & (
        reference.label_end_indices_72h < split.train_stop
    )
    assert np.all(reference.label_end_indices_72h[base] < split.calibration_start)
    assert np.all(reference.label_end_indices_72h[calibration] < split.train_stop)


def test_v5_fit_persists_support_forward_evidence_and_v4_identity() -> None:
    samples = _samples()
    split, v4_fit, fit = _fit(samples)

    assert fit.v4_fit_digest == v4_fit.digest
    assert fit.v4_fit_config_digest == v4_fit.config.digest
    assert fit.v4_sample_scope_digest == v4_fit.sample_scope_digest
    assert fit.calibration_start == split.calibration_start
    assert fit.train_stop == split.train_stop
    assert fit.model.knowledge_cutoff == split.train_stop
    assert fit.model.sample_count == fit.pooled_support
    assert len(fit.forward_model_digests) == 3
    assert len(fit.forward_residual_digests) == 3
    assert len(fit.forward_weight_digests) == 3
    assert fit.calibration_block_support == (1_125, 1_125, 1_116, 1_116)
    assert fit.forward_block_symbol_counts == (9, 9, 9)
    assert dict(fit.per_symbol_support) == {symbol: 498 for symbol in _SYMBOLS}
    assert fit.calibration_residual_rmse >= 0.0
    assert fit.direction_score_rmse >= 0.0


def test_v5_forward_sequence_does_not_train_b1_model_on_b2_labels() -> None:
    samples = _samples()
    split, v4_fit, first = _fit(samples)
    changed: dict[str, CausalAlphaV4SymbolSamples] = {}
    for symbol, sample in samples.items():
        in_b2 = (sample.decision_indices >= split.block_boundaries[1]) & (
            sample.decision_indices < split.block_boundaries[2]
        )
        labels24 = sample.labels_24h.copy()
        labels72 = sample.labels_72h.copy()
        labels24[in_b2] += 0.5
        labels72[in_b2] += 1.5
        changed[symbol] = replace(
            sample, labels_24h=labels24, labels_72h=labels72, digest=""
        )

    second = fit_causal_alpha_v5_calibration(
        train_symbols=_SYMBOLS,
        samples=changed,
        v4_fit=v4_fit,
        slow_uncertainty=_uncertainty(changed),
        train_stop=1_000,
        config=CausalAlphaV5CalibrationConfig(),
    )

    assert first.forward_model_digests[0] == second.forward_model_digests[0]
    assert first.forward_model_digests[1:] != second.forward_model_digests[1:]


def test_v5_fit_ignores_signal_labels_and_post_cutoff_features() -> None:
    samples = _samples()
    _, v4_fit, first = _fit(samples)
    changed: dict[str, CausalAlphaV4SymbolSamples] = {}
    for symbol, sample in samples.items():
        outside = sample.label_end_indices_72h >= 1_000
        post_cutoff = sample.decision_indices >= 1_000
        labels24 = sample.labels_24h.copy()
        labels72 = sample.labels_72h.copy()
        descriptors = sample.instrument_descriptors.copy()
        labels24[outside] += 100.0
        labels72[outside] -= 300.0
        descriptors[post_cutoff] += 1_000.0
        changed[symbol] = replace(
            sample,
            labels_24h=labels24,
            labels_72h=labels72,
            instrument_descriptors=descriptors,
            digest="",
        )

    second = fit_causal_alpha_v5_calibration(
        train_symbols=_SYMBOLS,
        samples=changed,
        v4_fit=v4_fit,
        slow_uncertainty=_uncertainty(changed),
        train_stop=1_000,
        config=CausalAlphaV5CalibrationConfig(),
    )

    assert first.digest == second.digest


def test_v5_fit_rejects_v4_cutoff_drift() -> None:
    samples = _samples()
    split = _split(samples)
    wrong = fit_causal_alpha_v4(
        train_symbols=_SYMBOLS,
        samples=samples,
        knowledge_cutoff=split.calibration_start + 1,
        config=CausalAlphaV4FitConfig(),
    )
    with pytest.raises(ValueError, match="cutoff"):
        fit_causal_alpha_v5_calibration(
            train_symbols=_SYMBOLS,
            samples=samples,
            v4_fit=wrong,
            slow_uncertainty=_uncertainty(samples),
            train_stop=1_000,
            config=CausalAlphaV5CalibrationConfig(),
        )


def test_v5_fit_fails_closed_on_insufficient_symbol_support() -> None:
    samples = _samples(rows=70)
    split = _split(samples, train_stop=34)
    v4_fit = _v4_fit(samples, split)
    with pytest.raises(ValueError, match="symbol support"):
        fit_causal_alpha_v5_calibration(
            train_symbols=_SYMBOLS,
            samples=samples,
            v4_fit=v4_fit,
            slow_uncertainty=_uncertainty(samples),
            train_stop=34,
            config=CausalAlphaV5CalibrationConfig(),
        )


def test_v5_calibration_uses_overlap_weighted_rows_for_long_horizons() -> None:
    samples = {
        symbol: replace(
            sample,
            label_end_indices_72h=sample.decision_indices + 100,
            digest="",
        )
        for symbol, sample in _samples(rows=1_050).items()
    }

    split, _v4_fit_value, fit = _fit(samples, train_stop=950)

    assert split.calibration_start == 425
    assert split.block_boundaries == (425, 532, 638, 744, 950)
    assert dict(fit.per_symbol_support) == {symbol: 425 for symbol in _SYMBOLS}


def test_v5_fit_rejects_missing_calibration_descriptor_availability() -> None:
    samples = _samples()
    split = _split(samples)
    sample = samples["APTUSDT"]
    available = sample.instrument_descriptor_available.copy()
    available[split.calibration_start, 0] = False
    samples["APTUSDT"] = replace(
        sample, instrument_descriptor_available=available, digest=""
    )
    v4_fit = _v4_fit(samples, split)
    with pytest.raises(ValueError, match="descriptor availability"):
        fit_causal_alpha_v5_calibration(
            train_symbols=_SYMBOLS,
            samples=samples,
            v4_fit=v4_fit,
            slow_uncertainty=_uncertainty(samples),
            train_stop=1_000,
            config=CausalAlphaV5CalibrationConfig(),
        )


def test_v5_calibrates_unseen_symbol_without_symbol_identity() -> None:
    samples = _samples()
    _, v4_fit, fit = _fit(samples)
    unseen = _sample(symbol="ETHUSDT")
    forecast = v4_fit.predict(unseen)
    rows = len(unseen.decision_indices)

    selective = calibrate_causal_alpha_v5_forecast(
        v4_forecast=forecast,
        sample=unseen,
        slow_uncertainty=np.full(rows, 0.01),
        one_way_cost_rates=np.zeros(rows),
        actionable_mask=np.ones(rows, dtype=np.bool_),
        calibration_fit=fit,
    )

    assert selective.symbol == "ETHUSDT"
    assert selective.calibration_fit_digest == fit.digest
    assert selective.decision_indices.shape == unseen.decision_indices.shape
