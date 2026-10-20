from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest

import tools.market_b4_oracle.frozen_forecast_index as forecast_index_module
from tools.market_b4_oracle.allocation_choice import AllocationAssumptions
from tools.market_b4_oracle.frozen_forecast import (
    OwnedDecisionSnapshot,
    bind_frozen_forecast_assumptions,
)
from tools.market_b4_oracle.frozen_forecast_index import (
    ForecastIndexLimits,
    FrozenForecastIndex,
)
from trade_rl.strategies.forecasts.simple_return import (
    SimpleReturnRidgeModel,
    SimpleReturnTrainingSet,
)
from trade_rl.strategies.forecasts.simple_stream import (
    FrozenSimpleReturnStream,
    SimpleReturnPacket,
    SimpleReturnVintage,
)
from trade_rl.strategies.forecasts.stream import ForecastBlock
from trade_rl.strategies.forecasts.supervised import CausalForecastTrainingSet
from trade_rl.strategies.forecasts.training_trace import ForecastTrainingTrace


def _ns(hour: int) -> int:
    return int(
        (np.datetime64("2026-01-01T00:00:00", "ns") + np.timedelta64(hour, "h")).astype(
            np.int64
        )
    )


def _fixture():
    trace = ForecastTrainingTrace(
        row_symbols=("ALPHA", "ALPHA"),
        start_times=np.array([_ns(0), _ns(1)], dtype="datetime64[ns]"),
        end_times=np.array([_ns(1), _ns(2)], dtype="datetime64[ns]"),
        start_available_at=np.array([_ns(0), _ns(1)], dtype="datetime64[ns]"),
        end_available_at=np.array([_ns(1), _ns(2)], dtype="datetime64[ns]"),
        start_close=np.array([100.0, 100.0]),
        end_close=np.array([110.0, 90.0]),
    )
    log_training = CausalForecastTrainingSet(
        feature_indices=(2, 4),
        feature_names=("feature_2", "feature_4"),
        features=np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float64),
        labels=np.array(
            [math.log(110.0 / 100.0), math.log(90.0 / 100.0)],
            dtype=np.float64,
        ),
        sample_weights=np.ones(2, dtype=np.float64),
        label_end_times=trace.end_times,
        fit_cutoff=np.datetime64(_ns(3), "ns"),
        horizon_hours=1,
        trace=trace,
    )
    training = SimpleReturnTrainingSet(log_training)
    model = SimpleReturnRidgeModel(
        feature_indices=(2, 4),
        feature_mean=np.array([0.0, 0.0]),
        feature_scale=np.array([1.0, 1.0]),
        coefficients=np.array([0.25, 0.0]),
        intercept=0.0,
        horizon_hours=1,
        alpha=1.0,
        n_samples=2,
        fit_cutoff=np.datetime64(_ns(3), "ns"),
        fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
    )
    owner = SimpleReturnVintage(
        ForecastBlock(
            np.datetime64(_ns(3), "ns"),
            np.datetime64(_ns(3), "ns"),
            np.datetime64(_ns(4), "ns"),
            np.datetime64(_ns(8), "ns"),
            inference_delay_seconds=0,
        ),
        training,
        model,
        ("ALPHA", "BETA"),
    )
    stream = FrozenSimpleReturnStream(
        dataset_id="a" * 64,
        vintages=(owner,),
        packets=(
            SimpleReturnPacket(
                symbol="ALPHA",
                as_of=np.datetime64(_ns(4), "ns"),
                source_available_at=np.datetime64(_ns(4), "ns"),
                forecast_available_at=np.datetime64(_ns(4), "ns"),
                horizon_end=np.datetime64(_ns(5), "ns"),
                horizon_seconds=3600,
                expected_simple_return=0.125,
                fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
                vintage_digest=owner.digest,
                feature_values=(0.5, 0.7),
                decision_close=100.0,
            ),
            SimpleReturnPacket(
                symbol="ALPHA",
                as_of=np.datetime64(_ns(5), "ns"),
                source_available_at=np.datetime64(_ns(5), "ns"),
                forecast_available_at=np.datetime64(_ns(5), "ns"),
                horizon_end=np.datetime64(_ns(6), "ns"),
                horizon_seconds=3600,
                expected_simple_return=0.15,
                fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
                vintage_digest=owner.digest,
                feature_values=(0.6, 0.8),
                decision_close=101.0,
            ),
            SimpleReturnPacket(
                symbol="BETA",
                as_of=np.datetime64(_ns(4), "ns"),
                source_available_at=np.datetime64(_ns(4), "ns"),
                forecast_available_at=np.datetime64(_ns(4), "ns"),
                horizon_end=np.datetime64(_ns(5), "ns"),
                horizon_seconds=3600,
                expected_simple_return=0.2,
                fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
                vintage_digest=owner.digest,
                feature_values=(0.8, 0.9),
                decision_close=102.0,
            ),
        ),
    )
    source = OwnedDecisionSnapshot(
        dataset_id="a" * 64,
        symbol="ALPHA",
        decision_time_ns=_ns(4),
        source_available_at_ns=_ns(4),
        information_available=True,
        feature_names=(
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
        ),
        feature_values=(0.1, 0.2, 0.5, 0.4, 0.7, 0.6),
        feature_available=(True, True, True, True, True, True),
        dataset_close=100.0,
        book_mark=100.0,
        expected_horizon_seconds=3600,
    )
    costs = AllocationAssumptions(
        expected_simple_return=9.0,
        return_variance=9.0,
        buy_cost=0.001,
        sell_cost=0.002,
        exit_cost=0.003,
        funding_return=0.004,
        borrow_return=0.005,
        cash_return=0.006,
    )
    return stream.payload(), source, costs


def make_multivintage_payload():
    payload, source, costs = _fixture()
    payload = copy.deepcopy(payload)
    second = copy.deepcopy(payload["vintages"][0])
    second_digest = "b" * 64
    second["digest"] = second_digest
    second["block"]["model_fit_time"] = _ns(7)
    second["block"]["prediction_start"] = _ns(8)
    second["block"]["prediction_stop"] = _ns(12)
    payload["vintages"].append(second)

    for symbol, expected, values, close in (
        ("ALPHA", 0.2, (0.25, 0.35), 101.0),
        ("BETA", 0.175, (0.45, 0.55), 103.0),
    ):
        packet = copy.deepcopy(payload["packets"][0])
        packet.update(
            {
                "symbol": symbol,
                "as_of": _ns(9),
                "source_available_at": _ns(9),
                "forecast_available_at": _ns(9),
                "horizon_end": _ns(10),
                "expected_simple_return": expected,
                "vintage_digest": second_digest,
                "feature_values": list(values),
                "decision_close": close,
                "digest": hashlib.sha256(
                    f"synthetic:{symbol}:9".encode("ascii")
                ).hexdigest(),
                "snapshot_digest": hashlib.sha256(
                    f"snapshot:{symbol}:9".encode("ascii")
                ).hexdigest(),
            }
        )
        payload["packets"].append(packet)
    return payload, source, costs


def encode(payload) -> bytes:
    return json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def make_index(raw: bytes, *, limits: ForecastIndexLimits | None = None):
    return FrozenForecastIndex(
        raw,
        pinned_bytes=len(raw),
        pinned_sha256=hashlib.sha256(raw).hexdigest(),
        limits=limits,
    )


def owned_source(source, *, symbol, hour, values, close):
    full_values = list(source.feature_values)
    full_values[2], full_values[4] = values
    return replace(
        source,
        symbol=symbol,
        decision_time_ns=_ns(hour),
        source_available_at_ns=_ns(hour),
        feature_values=tuple(full_values),
        dataset_close=close,
        book_mark=close,
    )


def canonical_assumptions(value: AllocationAssumptions) -> bytes:
    return json.dumps(
        asdict(value), allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def cost_estimates_payload(source, *, buy_cost: float = 0.001):
    decision = source.decision_time_ns
    return {
        "schema": "horizon_cost_estimates_v1",
        "valuation_basis": "same_close_price_return",
        "cost_basis": "declared_initial_notional_rates",
        "symbol": source.symbol,
        "decision_time": decision,
        "available_at": decision,
        "horizon_end": decision + source.expected_horizon_seconds * 1_000_000_000,
        "source_identity": "f" * 64,
        "buy_cost": buy_cost,
        "sell_cost": 0.002,
        "exit_cost": 0.003,
        "funding_return": 0.004,
        "borrow_return": 0.005,
        "cash_return": 0.006,
    }


def test_index_matches_full_legacy_stream_for_multiple_symbols_and_vintages():
    payload, source, costs = make_multivintage_payload()
    raw = encode(payload)
    index = make_index(raw)
    cases = (
        ("ALPHA", 4, (0.5, 0.7), 100.0, 0.125),
        ("BETA", 4, (0.8, 0.9), 102.0, 0.2),
        ("ALPHA", 9, (0.25, 0.35), 101.0, 0.2),
        ("BETA", 9, (0.45, 0.55), 103.0, 0.175),
    )
    for symbol, hour, values, close, expected_return in cases:
        selected_source = owned_source(
            source, symbol=symbol, hour=hour, values=values, close=close
        )
        legacy = bind_frozen_forecast_assumptions(
            payload, source=selected_source, cost_assumptions=costs
        )
        indexed = index.lookup(selected_source, cost_assumptions=costs)
        assert type(indexed) is AllocationAssumptions
        assert indexed == legacy
        assert canonical_assumptions(indexed) == canonical_assumptions(legacy)
        assert indexed.expected_simple_return == expected_return


def test_allocation_source_identity_matches_the_closed_expected_digest():
    payload, source, _ = _fixture()
    index = make_index(encode(payload))
    estimates = cost_estimates_payload(source)

    actual = index.allocation_source_identity(source, cost_estimates_payload=estimates)
    packet = payload["packets"][0]
    expected_payload = {
        "schema": "forecast_allocation_source_v1",
        "dataset_id": source.dataset_id,
        "packet": packet["digest"],
        "vintage": packet["vintage_digest"],
        "estimates": estimates,
    }
    expected = hashlib.sha256(
        json.dumps(
            expected_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert actual == expected
    assert len(actual) == 64


def test_allocation_source_identity_rejects_foreign_dataset_snapshot():
    payload, source, _ = _fixture()
    index = make_index(encode(payload))
    foreign = replace(source, dataset_id="b" * 64)

    with pytest.raises(ValueError, match="Dataset identities differ"):
        index.allocation_source_identity(
            foreign, cost_estimates_payload=cost_estimates_payload(foreign)
        )


def test_allocation_source_identity_rejects_mutated_owned_forecast_inputs():
    payload, source, _ = _fixture()
    index = make_index(encode(payload))
    changed_features = list(source.feature_values)
    changed_features[2] += 0.01
    changed = replace(source, feature_values=tuple(changed_features))

    with pytest.raises(ValueError, match="packet features differ"):
        index.allocation_source_identity(
            changed, cost_estimates_payload=cost_estimates_payload(changed)
        )


def test_allocation_source_identity_changes_when_a_cost_input_changes():
    payload, source, _ = _fixture()
    index = make_index(encode(payload))
    original = index.allocation_source_identity(
        source, cost_estimates_payload=cost_estimates_payload(source)
    )
    changed = index.allocation_source_identity(
        source, cost_estimates_payload=cost_estimates_payload(source, buy_cost=0.002)
    )

    assert changed != original


def test_index_metadata_is_immutable_and_does_not_expose_payload_or_rows():
    payload, _, _ = make_multivintage_payload()
    raw = encode(payload)
    index = make_index(raw)
    metadata = index.metadata
    assert metadata.scope == "FROZEN_FORECAST_INDEX_LOOKUP_ONLY"
    assert metadata.pinned_bytes == len(raw)
    assert metadata.pinned_sha256 == hashlib.sha256(raw).hexdigest()
    assert metadata.packet_count == len(payload["packets"])
    assert metadata.vintage_count == len(payload["vintages"])
    with pytest.raises(FrozenInstanceError):
        metadata.packet_count = 0
    assert not hasattr(index, "payload")
    assert not hasattr(index, "packets")
    assert not hasattr(index, "vintages")


def test_stream_json_is_parsed_once_and_never_during_lookup(monkeypatch):
    payload, source, costs = make_multivintage_payload()
    raw = encode(payload)
    original_loads = forecast_index_module.json.loads
    parse_calls = 0

    def count_parse(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_loads(*args, **kwargs)

    monkeypatch.setattr(forecast_index_module.json, "loads", count_parse)
    index = make_index(raw)
    assert parse_calls == 1

    selected_source = owned_source(
        source, symbol="ALPHA", hour=9, values=(0.25, 0.35), close=101.0
    )

    def forbidden_parse(*args, **kwargs):
        pytest.fail("lookup reparsed the pinned full stream")

    monkeypatch.setattr(forecast_index_module.json, "loads", forbidden_parse)
    assert (
        index.lookup(selected_source, cost_assumptions=costs).expected_simple_return
        == 0.2
    )


@pytest.mark.parametrize("bad_raw", [bytearray(b"{}"), memoryview(b"{}")])
def test_rejects_mutable_or_non_bytes_input_even_with_matching_pin(bad_raw):
    with pytest.raises(ValueError):
        FrozenForecastIndex(
            bad_raw,
            pinned_bytes=len(bad_raw),
            pinned_sha256=hashlib.sha256(bytes(bad_raw)).hexdigest(),
        )


def test_rejects_mismatched_external_size_or_digest_pin():
    raw = encode(make_multivintage_payload()[0])
    with pytest.raises(ValueError):
        FrozenForecastIndex(
            raw,
            pinned_bytes=len(raw) + 1,
            pinned_sha256=hashlib.sha256(raw).hexdigest(),
        )
    with pytest.raises(ValueError):
        FrozenForecastIndex(raw, pinned_bytes=len(raw), pinned_sha256="0" * 64)


def test_rejects_invalid_utf8_before_json_parse():
    raw = b'{"bad":"\xff"}'
    with pytest.raises(ValueError):
        FrozenForecastIndex(
            raw,
            pinned_bytes=len(raw),
            pinned_sha256=hashlib.sha256(raw).hexdigest(),
        )


def _replace_once(raw: bytes, old: bytes, new: bytes) -> bytes:
    assert raw.count(old) == 1
    return raw.replace(old, new, 1)


def test_rejects_nested_duplicate_json_keys_after_escape_decoding():
    raw = encode(make_multivintage_payload()[0])
    packet_start = raw.index(b'"packets":')
    key_start = raw.index(b'"symbol":"ALPHA"', packet_start)
    original = b'"symbol":"ALPHA"'
    replacement = rb'"symbol":"placeholder","\u0073ymbol":"ALPHA"'
    duplicate = raw[:key_start] + replacement + raw[key_start + len(original) :]
    with pytest.raises(ValueError):
        make_index(duplicate)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (b'"expected_simple_return":0.125', b'"expected_simple_return":NaN'),
        (b'"expected_simple_return":0.125', b'"expected_simple_return":1e999'),
        pytest.param(
            b'"expected_simple_return":0.125',
            b'"expected_simple_return":0.' + b"1" * 4095,
            id="decimal-token-over-4096-characters",
        ),
        pytest.param(
            b'"expected_simple_return":0.125',
            b'"expected_simple_return":1e' + b"9" * 4095,
            id="exponent-token-over-4096-characters",
        ),
        pytest.param(
            b'"expected_simple_return":0.125',
            b'"expected_simple_return":' + b"1" * 4097,
            id="integer-token-over-4096-characters",
        ),
    ],
)
def test_rejects_nonfinite_or_overflowing_json_number_tokens(old, new):
    raw = encode(make_multivintage_payload()[0])
    malformed = _replace_once(raw, old, new)
    match = "4096-character limit" if len(new) > 4096 else None
    with pytest.raises(ValueError, match=match):
        make_index(malformed)


def test_accepts_a_finite_float_token_at_the_exact_4096_character_limit():
    raw = encode(make_multivintage_payload()[0])
    token = b"0." + b"1" * 4094
    assert len(token) == 4096
    bounded = _replace_once(
        raw,
        b'"expected_simple_return":0.125',
        b'"expected_simple_return":' + token,
    )
    assert make_index(bounded).metadata.pinned_bytes == len(bounded)


def test_rejects_json_nesting_above_configured_depth():
    raw = encode(make_multivintage_payload()[0])
    limits = replace(ForecastIndexLimits(), max_depth=2)
    with pytest.raises(ValueError):
        make_index(raw, limits=limits)


def test_default_limits_match_the_frozen_input_caps():
    limits = ForecastIndexLimits()
    assert limits.max_bytes == 64 * 1024 * 1024
    assert limits.max_packets == 20_000
    assert limits.max_vintages == 1_024


def test_enforces_packet_vintage_and_raw_byte_limits():
    raw = encode(make_multivintage_payload()[0])
    with pytest.raises(ValueError):
        make_index(raw, limits=replace(ForecastIndexLimits(), max_packets=4))
    with pytest.raises(ValueError):
        make_index(raw, limits=replace(ForecastIndexLimits(), max_vintages=1))
    with pytest.raises(ValueError):
        make_index(raw, limits=replace(ForecastIndexLimits(), max_bytes=len(raw) - 1))
    with pytest.raises(ValueError):
        replace(ForecastIndexLimits(), max_bytes=64 * 1024 * 1024 + 1)


def test_rejects_duplicate_packet_index_key_and_duplicate_vintage_digest():
    payload, _, _ = make_multivintage_payload()
    duplicate_packet = copy.deepcopy(payload)
    duplicate_packet["packets"].append(copy.deepcopy(duplicate_packet["packets"][0]))
    with pytest.raises(ValueError):
        make_index(encode(duplicate_packet))

    duplicate_vintage = copy.deepcopy(payload)
    duplicate_vintage["vintages"].append(
        copy.deepcopy(duplicate_vintage["vintages"][0])
    )
    with pytest.raises(ValueError):
        make_index(encode(duplicate_vintage))


def test_rejects_packets_that_reference_a_missing_vintage():
    payload, _, _ = make_multivintage_payload()
    payload["packets"][0]["vintage_digest"] = "c" * 64
    with pytest.raises(ValueError):
        make_index(encode(payload))


def test_selected_source_clock_corruption_still_fails_adapter_semantics():
    payload, source, costs = make_multivintage_payload()
    index = make_index(encode(payload))
    bad_source = replace(source, source_available_at_ns=_ns(3))
    with pytest.raises(ValueError):
        index.lookup(bad_source, cost_assumptions=costs)
