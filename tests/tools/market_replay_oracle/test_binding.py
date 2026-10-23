"""Static projections cannot replace their bound symbol, clock or full policy."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import MappingProxyType

import pytest

from tools.market_replay_oracle.artifact import VerifiedDatasetContent
from tools.market_replay_oracle.binding import FrozenAttemptBinding, load_pinned_attempt
from tools.market_replay_oracle.identity import canonical
from tools.market_replay_oracle.profiles import effective_profile

from .test_profiles import RISK
from .test_protocol import SOURCE_HEAD, _document, _wire


def _binding() -> FrozenAttemptBinding:
    payload = {
        "clock": {"actor_start": 2903, "actor_stop": 9430, "settlement_stop": 9502},
        "schedule": {
            "arms": [
                {
                    "account": "H24_RIDGE_A1_BTCUSDT",
                    "scenario": "BASEL0",
                    "role": "nonrl",
                    "symbol_index": 0,
                    "forecast_horizon_seconds": 86400,
                }
            ]
        },
        "profiles": {"BASEL0": effective_profile("BASEL0", 6527, RISK)},
    }
    content = VerifiedDatasetContent(
        "a" * 64, MappingProxyType({}), MappingProxyType({})
    )
    return FrozenAttemptBinding(canonical(payload), content)


@pytest.mark.parametrize(
    "count,index,phase",
    [
        (1, 2904, "actor"),
        (6527, 9430, "actor"),
        (6528, 9431, "reserve"),
        (6599, 9502, "reserve"),
    ],
)
def test_actual_processing_boundary(count: int, index: int, phase: str) -> None:
    assert _binding().progress(count) == {
        "count": count,
        "processing_index": index,
        "phase": phase,
    }


@pytest.mark.parametrize("count", [True, 0, -1, 6600, 1.0])
def test_no_initial_or_beyond_terminal_progress(count: object) -> None:
    with pytest.raises(ValueError):
        _binding().progress(count)


def test_payload_mutation_does_not_rewrite_bound_facts() -> None:
    binding = _binding()
    initial = binding.digest
    returned = binding.payload()
    returned["profiles"]["BASEL0"]["cost_policy"]["multiplier"] = 7.0
    assert binding.digest == initial
    assert binding.payload()["profiles"]["BASEL0"]["cost_policy"]["multiplier"] == 1.0


def _projection(binding: FrozenAttemptBinding) -> dict:
    profile = binding.payload()["profiles"]["BASEL0"]
    return {
        "symbol_index": 0,
        "forecast_horizon_seconds": 86400,
        "cost_policy": profile["cost_policy"],
        "execution_policy_digest": profile["execution_policy_digest"],
        "runtime_profile": profile["runtime_profile"],
    }


def test_coherently_rehashed_foreign_policy_and_wrong_horizon_reject() -> None:
    binding = _binding()
    original = _projection(binding)
    binding.assert_arm_profile("H24_RIDGE_A1_BTCUSDT", "BASEL0", "nonrl", original)
    tampered = copy.deepcopy(original)
    tampered["cost_policy"]["max_leverage"] = 2.0
    base = hashlib.sha256(canonical(tampered["cost_policy"])).hexdigest()
    digest = hashlib.sha256(
        canonical(
            {
                "base_policy_digest": base,
                "schema_version": "insolvency_execution_policy_v1",
                "insolvency_valuation": "retain_debt",
            }
        )
    ).hexdigest()
    tampered["execution_policy_digest"] = digest
    tampered["runtime_profile"]["economics_digest"] = digest
    with pytest.raises(ValueError):
        binding.assert_arm_profile("H24_RIDGE_A1_BTCUSDT", "BASEL0", "nonrl", tampered)
    for field, replacement in (
        ("symbol_index", 1),
        ("forecast_horizon_seconds", 259200),
    ):
        with pytest.raises(ValueError):
            binding.assert_arm_profile(
                "H24_RIDGE_A1_BTCUSDT",
                "BASEL0",
                "nonrl",
                {**original, field: replacement},
            )
    with pytest.raises(ValueError):
        binding.assert_arm_profile("H24_RIDGE_A1_BTCUSDT", "BASEL0", "cash", original)
    with pytest.raises(ValueError):
        binding.assert_arm_profile(
            "H24_RIDGE_A1_BTCUSDT",
            "BASEL0",
            "nonrl",
            {"execution_policy_digest": original["execution_policy_digest"]},
        )


@pytest.mark.parametrize(
    "mismatch", ["source", "root", "identity", "manifest", "arrays"]
)
def test_entry_rejects_foreign_inputs_before_source_or_dataset_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    """Exercise actual protocol admission, then fail before lower-level I/O."""
    from tools.market_replay_oracle import (
        DatasetContentPin,
        NativeSourcePin,
        ProtocolPin,
    )
    from tools.market_replay_oracle import binding as owner

    protocol = _document()
    raw = _wire(protocol)
    path = tmp_path / "protocol.json"
    path.write_bytes(raw)
    document_pin = ProtocolPin(hashlib.sha256(raw).hexdigest(), len(raw), SOURCE_HEAD)
    dataset = protocol["dataset"]
    pin = DatasetContentPin(
        dataset["dataset_id"] if mismatch != "identity" else "0" * 64,
        dataset["input_sha256"]["manifest.json"]
        if mismatch != "manifest"
        else "0" * 64,
        dataset["input_sha256"]["arrays.npz"] if mismatch != "arrays" else "0" * 64,
        7453,
        5864784,
    )

    def forbidden(*args: object) -> None:
        pytest.fail("mismatched pins reached native-source or Dataset I/O")

    monkeypatch.setattr(owner, "verify_native_source", forbidden)
    monkeypatch.setattr(owner, "load_pinned_v4", forbidden)
    source_pin = NativeSourcePin(
        SOURCE_HEAD if mismatch != "source" else "1" * 40, "2" * 64
    )
    # The frozen consumed protocol names a Windows path. Other platforms must
    # also reject it as unavailable; this test does not remap that source path.
    root = tmp_path if mismatch == "root" else Path(dataset["root"])
    with pytest.raises(ValueError, match="pins differ|root/pins differ"):
        load_pinned_attempt(path, document_pin, tmp_path, source_pin, root, pin)


@pytest.mark.parametrize("extra", ["result.json", "unrelated"])
def test_postflight_rejects_added_dataset_outputs(tmp_path: Path, extra: str) -> None:
    """The source/input directory cannot acquire result files during binding."""
    from tools.market_replay_oracle import DatasetContentPin
    from tools.market_replay_oracle.binding import _dataset_still_pinned

    manifest, arrays = b"manifest bytes", b"array bytes"
    (tmp_path / "manifest.json").write_bytes(manifest)
    (tmp_path / "arrays.npz").write_bytes(arrays)
    pin = DatasetContentPin(
        "1" * 64,
        hashlib.sha256(manifest).hexdigest(),
        hashlib.sha256(arrays).hexdigest(),
        len(manifest),
        len(arrays),
    )
    _dataset_still_pinned(tmp_path, pin)
    (tmp_path / extra).write_text("must reject", encoding="utf-8")
    with pytest.raises(ValueError, match="roster changed"):
        _dataset_still_pinned(tmp_path, pin)


def test_postflight_rejects_same_size_dataset_replacement(tmp_path: Path) -> None:
    from tools.market_replay_oracle import DatasetContentPin
    from tools.market_replay_oracle.binding import _dataset_still_pinned

    for name in ("manifest.json", "arrays.npz"):
        (tmp_path / name).write_bytes(b"before")
    digest = hashlib.sha256(b"before").hexdigest()
    pin = DatasetContentPin("1" * 64, digest, digest, 6, 6)
    _dataset_still_pinned(tmp_path, pin)
    (tmp_path / "arrays.npz").write_bytes(b"after!")
    with pytest.raises(ValueError, match="SHA-256"):
        _dataset_still_pinned(tmp_path, pin)
