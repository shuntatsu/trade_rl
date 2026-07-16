from __future__ import annotations

import pytest

from trade_rl.rl.environment_config import ResidualMarketEnvConfig


def _config(**kwargs: object) -> ResidualMarketEnvConfig:
    return ResidualMarketEnvConfig(initial_capital=100_000.0, **kwargs)


def test_sequence_observation_resolves_maintained_native_windows() -> None:
    config = _config(structured_sequence_observation=True)
    assert config.resolved_sequence_windows == (
        ("15m", 96),
        ("1h", 168),
        ("4h", 120),
        ("1d", 60),
    )


def test_sequence_windows_are_normalized_and_reject_duplicate_clocks() -> None:
    config = _config(
        structured_sequence_observation=True,
        sequence_windows=(("15m", 32), ("1h", 24)),
    )
    assert config.resolved_sequence_windows == (("15m", 32), ("1h", 24))

    with pytest.raises(ValueError, match="unique"):
        _config(
            structured_sequence_observation=True,
            sequence_windows=(("15m", 32), ("15m", 64)),
        )


def test_signal_delay_decisions_accepts_only_zero_or_one() -> None:
    assert _config(signal_delay_decisions=0).signal_delay_decisions == 0
    assert _config(signal_delay_decisions=1).signal_delay_decisions == 1
    for invalid in (-1, 2, True, 0.5):
        with pytest.raises(ValueError, match="signal_delay_decisions"):
            _config(signal_delay_decisions=invalid)
