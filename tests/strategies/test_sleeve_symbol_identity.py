"""Public sleeve symbol identity cannot be redirected by foreign string hooks."""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.strategies.allocation.sleeves import (
    SleeveCombinerConfig,
    SleeveCombinerError,
    SleeveMethod,
    SleeveTarget,
    combine_sleeves,
)


class _EqualToEverything(str):
    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = str.__hash__


class _UntrustedHash(str):
    def __hash__(self) -> int:
        raise AssertionError("untrusted symbol hash executed")


def _combine(symbols: tuple[str, ...]) -> object:
    return combine_sleeves(
        [SleeveTarget("s", ("ETHUSDT",), (0.4,))],
        symbols=symbols,
        history=np.array([[-1.0], [1.0]]),
        config=SleeveCombinerConfig(
            method=SleeveMethod.FIXED,
            lookback=2,
            ddof=0,
            shrinkage=0.0,
            fixed_weights=(1.0,),
        ),
    )


def test_rejects_deceptive_sleeve_name() -> None:
    with pytest.raises(TypeError, match="native"):
        SleeveTarget(_EqualToEverything("s"), ("ETHUSDT",), (1.0,))


def test_rejects_deceptive_sleeve_symbol_before_mapping() -> None:
    deceptive = _EqualToEverything("BTCUSDT")
    assert deceptive == "ETHUSDT" and str(deceptive) != "ETHUSDT"
    with pytest.raises(TypeError, match="native"):
        SleeveTarget("s", (deceptive,), (1.0,))


def test_rejects_deceptive_direct_symbol_before_comparing() -> None:
    with pytest.raises(TypeError, match="symbols"):
        _combine((_EqualToEverything("BTCUSDT"),))


def test_rejects_symbol_hash_hook_before_invocation() -> None:
    with pytest.raises(TypeError, match="symbols"):
        _combine((_UntrustedHash("ETHUSDT"),))


def test_rejects_unhashable_symbol_with_controlled_type_error() -> None:
    with pytest.raises(TypeError, match="symbols"):
        _combine((["ETHUSDT"],))  # type: ignore[arg-type]


def test_native_string_identity_and_output_preserved() -> None:
    target = SleeveTarget("s", ("ETHUSDT",), (0.4,))
    assert target.symbols == ("ETHUSDT",)
    result = _combine(("ETHUSDT",))
    assert result.symbols == ("ETHUSDT",)
    assert result.sleeve_names == ("s",)
    assert result.sleeve_weights == (1.0,)
    assert result.combined_weights == (0.4,)


def test_duplicate_native_symbols_preserve_reason_code() -> None:
    with pytest.raises(SleeveCombinerError) as caught:
        _combine(("ETHUSDT", "ETHUSDT"))
    assert caught.value.reason == "symbols"
