from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.binance_signed_helpers import (
    END,
    START,
    SYMBOLS,
    TRUSTED_KEYS,
    TRUSTED_NOW,
    signed_rule_history_document,
)
from trade_rl.workflows.binance_metadata_modes import (
    load_verified_binance_rule_history,
    resolution_from_historical_signed,
)


def _load(document: dict[str, object]):
    return load_verified_binance_rule_history(
        document,
        trusted_keys=TRUSTED_KEYS,
        trusted_now=TRUSTED_NOW,
    )


def test_signed_history_requires_verified_exact_scope() -> None:
    verified = _load(signed_rule_history_document())

    result = resolution_from_historical_signed(
        verified,
        start_time=START,
        end_time=END,
    )

    assert result.identity_evidence["market"] == "usds-m"
    assert result.identity_evidence["symbols"] == SYMBOLS
    assert result.identity_evidence["authentication"] == "ed25519"
    assert result.identity_evidence["point_in_time"] is True


def test_signed_history_preserves_explicit_symbol_order() -> None:
    reversed_symbols = tuple(reversed(SYMBOLS))
    verified = _load(signed_rule_history_document(symbol_order=reversed_symbols))

    assert verified.symbols == reversed_symbols
    assert tuple(verified.metadata) == reversed_symbols
    assert tuple(verified.execution_rule_histories) == reversed_symbols


def test_signed_history_rejects_non_usdm_market() -> None:
    with pytest.raises(ValueError, match="market"):
        _load(signed_rule_history_document(market="spot"))


@pytest.mark.parametrize(
    ("start_time", "end_time"),
    [
        (datetime(2024, 12, 2, tzinfo=UTC), END),
        (START, datetime(2026, 6, 30, tzinfo=UTC)),
    ],
)
def test_signed_resolution_rejects_coverage_mismatch(
    start_time: datetime,
    end_time: datetime,
) -> None:
    verified = _load(signed_rule_history_document())

    with pytest.raises(ValueError, match="coverage"):
        resolution_from_historical_signed(
            verified,
            start_time=start_time,
            end_time=end_time,
        )


def test_signed_history_rejects_zero_execution_rules() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _load(signed_rule_history_document(tick_size=0.0))


def test_signed_history_rejects_rule_after_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        _load(
            signed_rule_history_document(
                extra_rule_effective_at=datetime(2026, 7, 2, tzinfo=UTC),
            )
        )


def test_signed_history_rejects_metadata_that_differs_from_final_rule() -> None:
    with pytest.raises(ValueError, match="final execution rule"):
        _load(signed_rule_history_document(final_tick_size=0.2))


def test_signed_history_rejects_payload_tampering() -> None:
    document = signed_rule_history_document()
    payload = document["payload"]
    assert isinstance(payload, dict)
    payload["source_uri"] = "operator://tampered"

    with pytest.raises(ValueError, match="payload digest"):
        _load(document)


def test_resolution_rejects_unverified_mapping() -> None:
    with pytest.raises(TypeError, match="load_verified_binance_rule_history"):
        resolution_from_historical_signed(  # type: ignore[arg-type]
            signed_rule_history_document(),
            start_time=START,
            end_time=END,
        )
