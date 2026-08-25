from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.learning.causal_alpha_v7 import (
    CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    CausalAlphaV7CalibrationConfig,
    CausalAlphaV7CalibrationRange,
    CausalAlphaV7Candidate,
)


def _digest(char: str) -> str:
    return char * 64


def _range() -> CausalAlphaV7CalibrationRange:
    return CausalAlphaV7CalibrationRange(
        base_fit_cutoff=500,
        calibration_start=500,
        train_stop=1_000,
        block_boundaries=(500, 625, 750, 875, 1_000),
        split_digest=_digest("a"),
        feature_names=CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    )


def test_v7_candidates_are_fixed_and_canonically_ordered() -> None:
    assert tuple(candidate.value for candidate in CausalAlphaV7Candidate) == (
        "v6_control",
        "symmetric_contrarian",
        "causal_calibrated",
    )


def test_v7_config_reuses_exact_v5_split_and_bounds_working_memory() -> None:
    config = CausalAlphaV7CalibrationConfig()

    assert config.calibration_fraction == 0.50
    assert config.forward_block_count == 4
    assert config.ridge_strength == 1.0
    assert config.minimum_pooled_support == 256
    assert config.minimum_symbol_support == 16
    assert config.working_memory_rows == 4_096
    assert len(config.digest) == 64

    with pytest.raises(ValueError, match="calibration fraction"):
        replace(config, calibration_fraction=0.40)
    with pytest.raises(ValueError, match="forward block count"):
        replace(config, forward_block_count=3)
    with pytest.raises(ValueError, match="working memory"):
        replace(config, working_memory_rows=4_097)


def test_v7_range_binds_base_fit_to_four_chronological_blocks() -> None:
    value = _range()

    assert value.base_fit_cutoff == value.calibration_start
    assert value.block_boundaries[0] == value.calibration_start
    assert value.block_boundaries[-1] == value.train_stop
    assert len(value.block_boundaries) == 5
    assert len(value.digest) == 64

    with pytest.raises(ValueError, match="base fit cutoff"):
        replace(value, base_fit_cutoff=501, digest="")
    with pytest.raises(ValueError, match="block boundaries"):
        replace(value, block_boundaries=(500, 625, 625, 875, 1_000), digest="")
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(value, digest=_digest("f"))


@pytest.mark.parametrize(
    "feature_names",
    (
        (*CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES, "symbol"),
        (*CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES, "symbol_id"),
        (*CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES, "instrument_symbol_hash"),
    ),
)
def test_v7_range_rejects_symbol_identity_features(
    feature_names: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="symbol identity"):
        replace(_range(), feature_names=feature_names, digest="")
