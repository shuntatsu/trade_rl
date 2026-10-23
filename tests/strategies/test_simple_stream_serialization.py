"""Public serialization contracts for the bounded simple-stream refactor."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from tests.strategies.test_simple_return_stream import fit, load_api
from trade_rl.artifacts import canonical_json_bytes, content_digest


def _timestamp_ns(value: np.datetime64) -> int:
    return int(value.astype("datetime64[ns]").astype(np.int64))


def _packet_payload(
    packet: object,
    *,
    snapshot_digest: str | None = None,
    digest: str | None = None,
) -> dict[str, object]:
    snapshot = {
        "symbol": packet.symbol,
        "as_of": _timestamp_ns(packet.as_of),
        "source_available_at": _timestamp_ns(packet.source_available_at),
        "feature_values": list(packet.feature_values),
        "decision_close": packet.decision_close,
    }
    packet_snapshot_digest = (
        content_digest(snapshot) if snapshot_digest is None else snapshot_digest
    )
    content = {
        "schema": "simple_return_packet_v1",
        **snapshot,
        "snapshot_digest": packet_snapshot_digest,
        "forecast_available_at": _timestamp_ns(packet.forecast_available_at),
        "horizon_end": _timestamp_ns(packet.horizon_end),
        "horizon_seconds": packet.horizon_seconds,
        "expected_simple_return": packet.expected_simple_return,
        "fit_prefix_marginal_variance": packet.fit_prefix_marginal_variance,
        "vintage_digest": packet.vintage_digest,
        "return_unit": packet.return_unit,
        "valuation_basis": packet.valuation_basis,
        "variance_kind": packet.variance_kind,
        "forecast_kind": packet.forecast_kind,
    }
    return {**content, "digest": content_digest(content) if digest is None else digest}


def _stream_payload(stream: object) -> dict[str, object]:
    causal_content = {
        "vintages": [vintage.payload() for vintage in stream.vintages],
        "packets": [_packet_payload(packet) for packet in stream.packets],
    }
    return {
        "schema": "simple_return_ridge_stream_v1",
        "availability_semantics": "declared_simulation_v1",
        "dataset_id": stream.dataset_id,
        **causal_content,
        "causal_scope_digest": content_digest(causal_content),
    }


def _clone_as(record_type: type, record: object):
    return record_type(
        **{field.name: getattr(record, field.name) for field in fields(record)}
    )


def test_public_packet_and_stream_outputs_match_independent_canonical_payloads() -> (
    None
):
    stream = fit(load_api())
    packet = stream.packets[0]
    expected_packet = _packet_payload(packet)
    expected_stream = _stream_payload(stream)

    assert packet.payload() == expected_packet
    assert packet.snapshot_digest == content_digest(
        {
            "symbol": packet.symbol,
            "as_of": _timestamp_ns(packet.as_of),
            "source_available_at": _timestamp_ns(packet.source_available_at),
            "feature_values": list(packet.feature_values),
            "decision_close": packet.decision_close,
        }
    )
    assert packet.digest == content_digest(
        {key: value for key, value in expected_packet.items() if key != "digest"}
    )
    assert canonical_json_bytes(packet.payload()) == canonical_json_bytes(
        expected_packet
    )

    assert stream.payload() == expected_stream
    assert stream.causal_scope_digest == expected_stream["causal_scope_digest"]
    assert stream.digest == content_digest(expected_stream)
    assert canonical_json_bytes(stream.payload()) == canonical_json_bytes(
        expected_stream
    )


def test_each_payload_access_returns_detached_nested_values() -> None:
    stream = fit(load_api())
    expected = stream.payload()
    returned = stream.payload()
    returned["packets"][0]["feature_values"][0] = -123.0
    returned["vintages"][0]["model"]["coefficients"][0] = 987.0
    returned["vintages"][0]["training"]["log_training"]["features"][0][0] = -9.0

    assert stream.payload() == expected
    assert stream.digest == content_digest(expected)


@pytest.mark.parametrize(
    "component", ("future_packet", "model", "trace", "shape", "dtype")
)
def test_nested_mutation_between_accesses_changes_identity_and_reader_rejects(
    component: str,
) -> None:
    api = load_api()
    stream = fit(api)
    before = stream.digest

    if component == "future_packet":
        object.__setattr__(stream.packets[-1], "expected_simple_return", 0.3)
    elif component == "model":
        model = stream.vintages[0].model
        object.__setattr__(model, "intercept", model.intercept + 0.1)
    elif component == "trace":
        trace = stream.vintages[0].training.trace
        changed_prices = trace.end_close.copy()
        changed_prices[0] += 1.0
        object.__setattr__(trace, "end_close", changed_prices)
    elif component == "shape":
        model = stream.vintages[0].model
        object.__setattr__(model, "coefficients", model.coefficients.reshape(1, -1))
    else:
        model = stream.vintages[0].model
        object.__setattr__(model, "coefficients", model.coefficients.astype(np.int64))

    payload = stream.payload()
    changed_digest = stream.digest
    assert changed_digest != before
    assert changed_digest == content_digest(payload)
    with pytest.raises(ValueError):
        api.FrozenSimpleReturnStream.from_payload(
            payload, expected_digest=changed_digest
        )


def test_subclass_digest_overrides_remain_visible_in_public_payloads() -> None:
    api = load_api()
    stream = fit(api)
    packet = stream.packets[0]
    overridden_snapshot = "a" * 64
    SnapshotOverridePacket = type(
        "SnapshotOverridePacket",
        (type(packet),),
        {"snapshot_digest": property(lambda self: overridden_snapshot)},
    )
    snapshot_packet = _clone_as(SnapshotOverridePacket, packet)
    assert snapshot_packet.payload() == _packet_payload(
        snapshot_packet, snapshot_digest=overridden_snapshot
    )

    overridden_digest = "b" * 64
    DigestOverridePacket = type(
        "DigestOverridePacket",
        (type(packet),),
        {"digest": property(lambda self: overridden_digest)},
    )
    digest_packet = _clone_as(DigestOverridePacket, packet)
    assert digest_packet.payload() == _packet_payload(
        digest_packet, digest=overridden_digest
    )

    overridden_scope = "c" * 64
    ScopeOverrideStream = type(
        "ScopeOverrideStream",
        (type(stream),),
        {"causal_scope_digest": property(lambda self: overridden_scope)},
    )
    scope_stream = _clone_as(ScopeOverrideStream, stream)
    expected_stream = _stream_payload(stream)
    expected_stream["causal_scope_digest"] = overridden_scope
    assert scope_stream.payload() == expected_stream
