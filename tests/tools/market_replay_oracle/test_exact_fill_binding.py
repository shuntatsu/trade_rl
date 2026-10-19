"""H-source event contexts use pinned raw rows, never observed event fields."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from tools.market_replay_oracle import DatasetContentPin, NativeSourcePin, ProtocolPin
from tools.market_replay_oracle.artifact import VerifiedDatasetContent
from tools.market_replay_oracle.identity import canonical
from tools.market_replay_oracle.profiles import effective_profile

from .test_profiles import RISK
from .test_protocol import _content

H_SOURCE_HEAD = "d361b6e3c3cdc83a8f5f05da9f82e1f72b66e3e5"


def _api() -> Any:
    from tools.market_replay_oracle import exact_fill_binding

    return exact_fill_binding


def _binding() -> Any:
    from .test_protocol_v4 import _document

    owner = _api()
    from tools.market_replay_oracle.protocol_v4 import bind_schedule_v4

    protocol = _document()
    content = _content(protocol)
    arrays = dict(content.arrays)
    lots = np.full((9504, 2), 0.01)
    lots[2903, 0] = 9.0  # A decision-row lot must never qualify the next fill.
    lots[2904, 0] = 0.125
    lots[2904, 1] = 0.5
    lots[9431, 0] = 0.0
    lots[9502, 0] = 0.25
    arrays["lot_size"] = np.frombuffer(lots.tobytes(), dtype="<f8").reshape(9504, 2)
    arrays["timestamps"] = np.frombuffer(
        arrays["timestamps"].tobytes(), dtype="datetime64[ns]"
    )
    content = VerifiedDatasetContent(
        content.dataset_id, MappingProxyType(arrays), content.dataset
    )
    payload = {
        "schema": "independent_exact_fill_attempt_binding_v1",
        "scope": "STATIC_INPUT_COMPARISON_ONLY",
        "protocol": {"source_head": H_SOURCE_HEAD},
        "dataset": {"dataset_id": content.dataset_id},
        "exact_fill_evidence": protocol["exact_fill_evidence"],
        "clock": {"actor_start": 2903, "actor_stop": 9430, "settlement_stop": 9502},
        "schedule": bind_schedule_v4(protocol, content),
        "profiles": {
            scenario: effective_profile(scenario, 6527, RISK)
            for scenario in protocol["scenarios"]
        },
    }
    return owner.ExactFillAttemptBinding(canonical(payload), content)


def test_exact_fill_binding_api_is_available() -> None:
    assert importlib.util.find_spec("tools.market_replay_oracle.exact_fill_binding")


@pytest.mark.parametrize(
    "count,processing,lot", [(1, 2904, 0.125), (6528, 9431, 0.0), (6599, 9502, 0.25)]
)
def test_event_context_derives_both_clocks_and_processing_row_lot(
    count: int, processing: int, lot: float
) -> None:
    binding = _binding()
    context = binding.event_context("H24_RIDGE_A1_BTCUSDT", "BASEL0", "nonrl", count)
    assert context.dataset_id == binding.content.dataset_id
    assert context.processing_index == processing
    assert context.decision_index == processing - 1
    assert context.symbol_index == 0
    assert context.effective_lot_size == lot
    assert context.decision_time_ns == int(
        binding.content.arrays["timestamps"][processing - 1].astype("int64")
    )
    assert context.processing_time_ns - context.decision_time_ns == 3_600_000_000_000
    assert (
        context.execution_policy_digest
        == effective_profile("BASEL0", 6527, RISK)["execution_policy_digest"]
    )


def test_context_binds_selected_symbol_scenario_and_cash_role() -> None:
    binding = _binding()
    context = binding.event_context("H24_RIDGE_A1_ETHUSDT", "COST2L0", "cash", 1)
    assert context.symbol_index == 1 and context.effective_lot_size == 0.5
    assert (
        context.execution_policy_digest
        == effective_profile("COST2L0", 6527, RISK)["execution_policy_digest"]
    )
    detached = binding.payload()
    detached["profiles"]["COST2L0"]["cost_policy"]["lot_size"] = 100.0
    assert (
        binding.event_context("H24_RIDGE_A1_ETHUSDT", "COST2L0", "cash", 1) == context
    )


@pytest.mark.parametrize("count", [True, 0, -1, 6600, 1.0])
def test_context_rejects_initial_outside_or_noninteger_progress(count: object) -> None:
    with pytest.raises(ValueError):
        _binding().event_context("H24_RIDGE_A1_BTCUSDT", "BASEL0", "nonrl", count)


@pytest.mark.parametrize(
    "account,scenario,role",
    [
        ("H24_RIDGE_A1_BTCUSDT", "BASEL0", "invented"),
        ("H24_RIDGE_A1_BTCUSDT", "COST4L0", "nonrl"),
        ("H24_RIDGE_A1_OTHER", "BASEL0", "nonrl"),
    ],
)
def test_context_rejects_foreign_arm(account: str, scenario: str, role: str) -> None:
    with pytest.raises(ValueError):
        _binding().event_context(account, scenario, role, 1)


@pytest.mark.parametrize(
    "mismatch", ["source", "root", "identity", "manifest", "arrays"]
)
def test_h_loader_rejects_foreign_inputs_before_source_or_dataset_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    from .test_protocol_v4 import _document

    owner = _api()
    protocol = _document()
    raw = canonical(protocol)
    path = tmp_path / "protocol.json"
    path.write_bytes(raw)
    protocol_pin = ProtocolPin(hashlib.sha256(raw).hexdigest(), len(raw), H_SOURCE_HEAD)
    dataset = protocol["dataset"]
    dataset_pin = DatasetContentPin(
        dataset["dataset_id"] if mismatch != "identity" else "0" * 64,
        dataset["input_sha256"]["manifest.json"]
        if mismatch != "manifest"
        else "0" * 64,
        dataset["input_sha256"]["arrays.npz"] if mismatch != "arrays" else "0" * 64,
        7453,
        5864784,
    )

    def forbidden(*args: object) -> None:
        pytest.fail("foreign pins reached source or Dataset access")

    monkeypatch.setattr(owner, "verify_native_source", forbidden)
    monkeypatch.setattr(owner, "load_pinned_v4", forbidden)
    source_pin = NativeSourcePin(
        H_SOURCE_HEAD if mismatch != "source" else "1" * 40, "2" * 64
    )
    root = tmp_path if mismatch == "root" else Path(dataset["root"])
    with pytest.raises(ValueError, match="pins differ|root/pins differ"):
        owner.load_pinned_exact_fill_attempt(
            path, protocol_pin, tmp_path, source_pin, root, dataset_pin
        )
