from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.tools.market_replay_oracle.test_protocol import (
    _document as _native_v3_document,
)
from tests.tools.market_replay_oracle.test_protocol import _wire
from tests.tools.market_replay_oracle.test_protocol_v4 import (
    _document as _comparison_v4_document,
)
from tools.market_replay_oracle import B631_CANDIDATE_PROTOCOL_SOURCE_PROFILE
from tools.market_replay_oracle.protocol import ProtocolPin

B631_SOURCE_HEAD = "b6318664c24da5a486baf0179d7a52fadaaa502d"
B631_SOURCE_PROFILE = B631_CANDIDATE_PROTOCOL_SOURCE_PROFILE
FA36_SOURCE_HEAD = "fa36f7d5c2fcf4e7401d62ca19f6ec3df114fa00"
FA36_PROFILE = "fa36-native-v3-v4"
FA36_NATIVE_SHA256 = "36570820c67c44b2f825a136e860b55b3ffb7a8e4ac8b9e114b7d47e6ee4d174"
FA36_COMPARISON_SHA256 = (
    "3823a80c617acd185ae3c8ea9b2d835df4688eb4de3c48e0a027697e3c94fce4"
)


def _candidate_pair_files(
    root: Path,
    *,
    source_head: str = B631_SOURCE_HEAD,
    name: str = "b631",
) -> tuple[Path, ProtocolPin, Path, ProtocolPin, dict[str, Any], dict[str, Any]]:
    native = copy.deepcopy(_native_v3_document())
    native["source_head"] = source_head
    comparison = copy.deepcopy(_comparison_v4_document())
    comparison["source_head"] = source_head
    native_raw = _wire(native)
    comparison_raw = _wire(comparison)
    native_path = root / f"native-{name}-v3.json"
    comparison_path = root / f"comparison-{name}-v4.json"
    native_path.write_bytes(native_raw)
    comparison_path.write_bytes(comparison_raw)
    native_pin = ProtocolPin(
        hashlib.sha256(native_raw).hexdigest(), len(native_raw), source_head
    )
    comparison_pin = ProtocolPin(
        hashlib.sha256(comparison_raw).hexdigest(),
        len(comparison_raw),
        source_head,
    )
    return (
        native_path,
        native_pin,
        comparison_path,
        comparison_pin,
        native,
        comparison,
    )


def test_b631_native_pair_requires_explicit_closed_source_profile(
    tmp_path: Path,
) -> None:
    from tools.market_replay_oracle.protocol_pair import load_declared_protocol_pair

    native_path, native_pin, comparison_path, comparison_pin, native, comparison = (
        _candidate_pair_files(tmp_path)
    )

    with pytest.raises(ValueError):
        load_declared_protocol_pair(
            native_path, native_pin, comparison_path, comparison_pin
        )

    pair = load_declared_protocol_pair(
        native_path,
        native_pin,
        comparison_path,
        comparison_pin,
        source_profile=B631_SOURCE_PROFILE,
    )

    assert pair.native_canonical == _canonical(native)
    assert pair.comparison_canonical == _canonical(comparison)
    assert pair.native_pin == native_pin
    assert pair.comparison_pin == comparison_pin
    assert pair.native_pin.source_head == B631_SOURCE_HEAD
    assert pair.comparison_pin.source_head == B631_SOURCE_HEAD


def test_b631_pair_constructor_requires_explicit_closed_source_profile(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict

    from tools.market_replay_oracle.protocol_pair import DeclaredProtocolPair

    native_path, native_pin, comparison_path, comparison_pin, native, comparison = (
        _candidate_pair_files(tmp_path)
    )
    del native_path, comparison_path
    native_canonical = _canonical(native)
    comparison_canonical = _canonical(comparison)

    with pytest.raises(ValueError, match="unsupported pinned H protocol"):
        DeclaredProtocolPair(
            native_pin,
            comparison_pin,
            native_canonical,
            comparison_canonical,
        )

    selected = DeclaredProtocolPair(
        native_pin,
        comparison_pin,
        native_canonical,
        comparison_canonical,
        source_profile=B631_SOURCE_PROFILE,
    )
    assert selected.native_canonical == native_canonical
    assert selected.comparison_canonical == comparison_canonical
    assert set(asdict(selected)) == {
        "native_pin",
        "comparison_pin",
        "native_canonical",
        "comparison_canonical",
        "scope",
    }
    assert selected == DeclaredProtocolPair(
        native_pin,
        comparison_pin,
        native_canonical,
        comparison_canonical,
        source_profile=B631_SOURCE_PROFILE,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_fa36_native_pair_matches_fresh_declarations_and_requires_opt_in(
    tmp_path: Path,
) -> None:
    from tools.market_replay_oracle import (
        DEFAULT_PROTOCOL_SOURCE_PROFILE,
        FA36_NATIVE_PROTOCOL_SOURCE_PROFILE,
    )
    from tools.market_replay_oracle.protocol_pair import load_declared_protocol_pair
    from tools.market_replay_oracle.protocol_source_profiles import (
        resolve_protocol_source_profile,
    )

    assert FA36_NATIVE_PROTOCOL_SOURCE_PROFILE == FA36_PROFILE
    profile = resolve_protocol_source_profile(FA36_NATIVE_PROTOCOL_SOURCE_PROFILE)
    assert DEFAULT_PROTOCOL_SOURCE_PROFILE == "d361-native-v3-v4"
    assert profile.profile_id == FA36_PROFILE
    assert profile.source_head == FA36_SOURCE_HEAD
    assert profile.native_v3_canonical_sha256 == FA36_NATIVE_SHA256
    assert profile.comparison_v4_canonical_sha256 == FA36_COMPARISON_SHA256

    native_path, native_pin, comparison_path, comparison_pin, native, comparison = (
        _candidate_pair_files(
            tmp_path,
            source_head=FA36_SOURCE_HEAD,
            name="fa36",
        )
    )
    assert hashlib.sha256(_canonical(native)).hexdigest() == FA36_NATIVE_SHA256
    assert hashlib.sha256(_canonical(comparison)).hexdigest() == FA36_COMPARISON_SHA256

    with pytest.raises(ValueError):
        load_declared_protocol_pair(
            native_path, native_pin, comparison_path, comparison_pin
        )

    pair = load_declared_protocol_pair(
        native_path,
        native_pin,
        comparison_path,
        comparison_pin,
        source_profile=FA36_PROFILE,
    )
    assert pair.native_pin == native_pin
    assert pair.comparison_pin == comparison_pin
    assert pair.native_pin.source_head == FA36_SOURCE_HEAD
    assert pair.comparison_pin.source_head == FA36_SOURCE_HEAD
    assert pair.native_canonical == _canonical(native)
    assert pair.comparison_canonical == _canonical(comparison)


@pytest.mark.parametrize("mutated_leg", ["native", "comparison"])
def test_fa36_profile_rejects_mixed_and_rehashed_declaration_mutations(
    tmp_path: Path,
    mutated_leg: str,
) -> None:
    from tools.market_replay_oracle.protocol_pair import load_declared_protocol_pair

    native_path, native_pin, comparison_path, comparison_pin, native, comparison = (
        _candidate_pair_files(
            tmp_path,
            source_head=FA36_SOURCE_HEAD,
            name="fa36",
        )
    )
    mixed_comparison_pin = ProtocolPin(
        comparison_pin.sha256,
        comparison_pin.raw_bytes,
        B631_SOURCE_HEAD,
    )
    with pytest.raises(ValueError):
        load_declared_protocol_pair(
            native_path,
            native_pin,
            comparison_path,
            mixed_comparison_pin,
            source_profile=FA36_PROFILE,
        )

    if mutated_leg == "native":
        native["capital"]["per_account"] = 99999.0
        raw = _wire(native)
        native_path.write_bytes(raw)
        native_pin = ProtocolPin(
            hashlib.sha256(raw).hexdigest(), len(raw), FA36_SOURCE_HEAD
        )
    else:
        comparison["economics"]["dataset_fee"] = 0.0004
        raw = _wire(comparison)
        comparison_path.write_bytes(raw)
        comparison_pin = ProtocolPin(
            hashlib.sha256(raw).hexdigest(), len(raw), FA36_SOURCE_HEAD
        )

    with pytest.raises(ValueError, match="frozen|contract|differs"):
        load_declared_protocol_pair(
            native_path,
            native_pin,
            comparison_path,
            comparison_pin,
            source_profile=FA36_PROFILE,
        )


def test_fa36_profile_rejects_unknown_source_head_and_profile_id(
    tmp_path: Path,
) -> None:
    from tools.market_replay_oracle.protocol_pair import load_declared_protocol_pair

    native_path, native_pin, comparison_path, comparison_pin, native, comparison = (
        _candidate_pair_files(
            tmp_path,
            source_head=FA36_SOURCE_HEAD,
            name="fa36",
        )
    )
    with pytest.raises(ValueError, match="unsupported.*profile"):
        load_declared_protocol_pair(
            native_path,
            native_pin,
            comparison_path,
            comparison_pin,
            source_profile="fa36-unreviewed-v3-v4",
        )

    unknown_head = "f" * 40
    native["source_head"] = unknown_head
    comparison["source_head"] = unknown_head
    native_raw = _wire(native)
    comparison_raw = _wire(comparison)
    native_path.write_bytes(native_raw)
    comparison_path.write_bytes(comparison_raw)
    with pytest.raises(ValueError):
        load_declared_protocol_pair(
            native_path,
            ProtocolPin(
                hashlib.sha256(native_raw).hexdigest(), len(native_raw), unknown_head
            ),
            comparison_path,
            ProtocolPin(
                hashlib.sha256(comparison_raw).hexdigest(),
                len(comparison_raw),
                unknown_head,
            ),
            source_profile=FA36_PROFILE,
        )


def test_b631_pair_rejects_unknown_mixed_and_mutated_profiles(
    tmp_path: Path,
) -> None:
    from tools.market_replay_oracle.protocol_pair import load_declared_protocol_pair

    native_path, native_pin, comparison_path, comparison_pin, native, comparison = (
        _candidate_pair_files(tmp_path)
    )

    with pytest.raises(ValueError, match="unsupported.*profile"):
        load_declared_protocol_pair(
            native_path,
            native_pin,
            comparison_path,
            comparison_pin,
            source_profile="unreviewed-profile",
        )

    mixed_comparison_pin = ProtocolPin(
        comparison_pin.sha256,
        comparison_pin.raw_bytes,
        "d361b6e3c3cdc83a8f5f05da9f82e1f72b66e3e5",
    )
    with pytest.raises(ValueError):
        load_declared_protocol_pair(
            native_path,
            native_pin,
            comparison_path,
            mixed_comparison_pin,
            source_profile=B631_SOURCE_PROFILE,
        )

    mutated_comparison = copy.deepcopy(comparison)
    mutated_comparison["economics"]["dataset_fee"] = 0.0004
    mutated_raw = _wire(mutated_comparison)
    comparison_path.write_bytes(mutated_raw)
    mutated_pin = ProtocolPin(
        hashlib.sha256(mutated_raw).hexdigest(), len(mutated_raw), B631_SOURCE_HEAD
    )
    with pytest.raises(ValueError, match="frozen|contract|differs"):
        load_declared_protocol_pair(
            native_path,
            native_pin,
            comparison_path,
            mutated_pin,
            source_profile=B631_SOURCE_PROFILE,
        )


def test_b631_exact_fill_loader_keeps_profile_out_of_binding_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import fields
    from pathlib import PurePosixPath

    from tests.tools.market_replay_oracle.test_exact_fill_binding import _binding
    from tests.tools.market_replay_oracle.test_protocol_v4 import _document
    from tools.market_replay_oracle import (
        DatasetContentPin,
        NativeSourcePin,
        exact_fill_binding,
    )
    from tools.market_replay_oracle.exact_fill_binding import (
        ExactFillAttemptBinding,
        load_pinned_exact_fill_attempt,
    )
    from tools.market_replay_oracle.protocol import ProtocolPin

    protocol = copy.deepcopy(_document())
    protocol["source_head"] = B631_SOURCE_HEAD
    protocol_raw = _wire(protocol)
    protocol_path = tmp_path / "comparison-b631-v4.json"
    protocol_path.write_bytes(protocol_raw)
    protocol_pin = ProtocolPin(
        hashlib.sha256(protocol_raw).hexdigest(),
        len(protocol_raw),
        B631_SOURCE_HEAD,
    )
    dataset = protocol["dataset"]
    content = _binding().content
    dataset_pin = DatasetContentPin(
        dataset["dataset_id"],
        dataset["input_sha256"]["manifest.json"],
        dataset["input_sha256"]["arrays.npz"],
        7453,
        5864784,
    )
    source_pin = NativeSourcePin(B631_SOURCE_HEAD, "2" * 64)
    source_reads: list[NativeSourcePin] = []

    def verify_source(_root: Path, pin: NativeSourcePin) -> dict[str, str]:
        source_reads.append(pin)
        return {"head_sha": pin.head_sha, "snapshot_sha256": pin.snapshot_sha256}

    monkeypatch.setattr(
        "tools.market_replay_oracle.exact_fill_binding.verify_native_source",
        verify_source,
    )
    monkeypatch.setattr(
        "tools.market_replay_oracle.exact_fill_binding.load_pinned_v4",
        lambda *_args: content,
    )
    monkeypatch.setattr(
        "tools.market_replay_oracle.exact_fill_binding._dataset_still_pinned",
        lambda *_args: None,
    )

    posix_dataset_root = PurePosixPath(dataset["root"])
    assert str(posix_dataset_root) == dataset["root"]
    assert not posix_dataset_root.is_absolute()
    with monkeypatch.context() as posix_paths:
        posix_paths.setattr(exact_fill_binding, "Path", PurePosixPath)
        with pytest.raises(ValueError, match="caller Dataset root/pins differ"):
            load_pinned_exact_fill_attempt(
                protocol_path,
                protocol_pin,
                tmp_path,
                source_pin,
                Path(dataset["root"]),
                dataset_pin,
                source_profile=B631_SOURCE_PROFILE,
            )
    assert source_reads == []

    class DeclaredAbsolutePath:
        """Adapt the frozen Windows Dataset root; Dataset I/O is stubbed below."""

        def __init__(self, value: str | Path) -> None:
            self.value = str(value)

        def is_absolute(self) -> bool:
            return True

        def __str__(self) -> str:
            return self.value

    monkeypatch.setattr(exact_fill_binding, "Path", DeclaredAbsolutePath)

    binding = load_pinned_exact_fill_attempt(
        protocol_path,
        protocol_pin,
        tmp_path,
        source_pin,
        Path(dataset["root"]),
        dataset_pin,
        source_profile=B631_SOURCE_PROFILE,
    )

    payload = binding.payload()
    with pytest.raises(ValueError, match="unsupported exact-fill static binding"):
        ExactFillAttemptBinding(_canonical(payload), content)
    explicitly_selected = ExactFillAttemptBinding(
        _canonical(payload), content, source_profile=B631_SOURCE_PROFILE
    )
    assert payload["protocol"] == {
        "sha256": protocol_pin.sha256,
        "bytes": protocol_pin.raw_bytes,
        "source_head": B631_SOURCE_HEAD,
    }
    assert explicitly_selected.payload() == payload
    assert "source_profile" not in payload
    assert "source_profile" not in {field.name for field in fields(binding)}
    assert len(source_reads) == 2 and all(pin == source_pin for pin in source_reads)


def test_b631_static_readback_requires_explicit_native_selection_and_rechecks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.tools.market_replay_oracle.test_exact_fill_binding import _binding
    from tools.market_b4_oracle import offline_replay_consumer as consumer
    from tools.market_b4_oracle.frozen_market_replay import (
        FrozenMarketBinding,
        FrozenMarketPolicy,
    )
    from tools.market_replay_oracle import DatasetContentPin, NativeSourcePin
    from tools.market_replay_oracle.profiles import effective_profile

    native_path, native_pin, comparison_path, comparison_pin, _, comparison = (
        _candidate_pair_files(tmp_path)
    )
    from tools.market_replay_oracle.protocol_pair import load_declared_protocol_pair

    pair = load_declared_protocol_pair(
        native_path,
        native_pin,
        comparison_path,
        comparison_pin,
        source_profile=B631_SOURCE_PROFILE,
    )
    content = _binding().content
    native_document = json.loads(pair.native_canonical)
    binding = FrozenMarketBinding.from_declared_protocol_pair(
        protocol_pair=pair,
        content=content,
        configuration_id="H24_RIDGE_A1",
        scenario="BASEL0",
        native_role="nonrl",
        selected_symbol_index=0,
    )
    risk = native_document["policy"]["risk"]
    policy = FrozenMarketPolicy.from_effective_profile(
        dataset_id=content.dataset_id,
        profile=effective_profile("BASEL0", binding.window.actor_advances, risk),
        risk=risk,
    )
    dataset = comparison["dataset"]
    pins = consumer.StaticInputReadback(
        protocol_v3_path=native_path,
        protocol_v3_pin=native_pin,
        protocol_v4_path=comparison_path,
        protocol_v4_pin=comparison_pin,
        source_root=tmp_path,
        source_pin=NativeSourcePin(B631_SOURCE_HEAD, "2" * 64),
        dataset_root=Path(dataset["root"]),
        dataset_pin=DatasetContentPin(
            dataset["dataset_id"],
            dataset["input_sha256"]["manifest.json"],
            dataset["input_sha256"]["arrays.npz"],
            7453,
            5864784,
        ),
        source_profile=B631_SOURCE_PROFILE,
    )
    from tools.market_replay_oracle import exact_fill_binding

    class DeclaredAbsolutePath:
        """Stand in for the Windows Dataset path; artifact I/O is stubbed below."""

        def __init__(self, value: str | Path) -> None:
            self.value = str(value)

        def is_absolute(self) -> bool:
            return True

        def __str__(self) -> str:
            return self.value

    monkeypatch.setattr(
        exact_fill_binding,
        "verify_native_source",
        lambda _root, pin: {
            "head_sha": pin.head_sha,
            "snapshot_sha256": pin.snapshot_sha256,
        },
    )
    monkeypatch.setattr(exact_fill_binding, "load_pinned_v4", lambda *_args: content)
    monkeypatch.setattr(
        exact_fill_binding, "_dataset_still_pinned", lambda *_args: None
    )
    monkeypatch.setattr(exact_fill_binding, "Path", DeclaredAbsolutePath)
    monkeypatch.setattr(consumer, "load_pinned_v4", lambda *_args: content)

    pair_profiles: list[object] = []
    real_pair_loader = consumer.load_declared_protocol_pair

    def observe_pair_loader(*args: object, **kwargs: object) -> object:
        pair_profiles.append(kwargs.get("source_profile"))
        return real_pair_loader(*args, **kwargs)

    exact_profiles: list[object] = []
    real_exact_loader = consumer.load_pinned_exact_fill_attempt

    def observe_exact_loader(*args: object, **kwargs: object) -> object:
        exact_profiles.append(kwargs.get("source_profile"))
        return real_exact_loader(*args, **kwargs)

    monkeypatch.setattr(consumer, "load_declared_protocol_pair", observe_pair_loader)
    monkeypatch.setattr(
        consumer, "load_pinned_exact_fill_attempt", observe_exact_loader
    )

    readback = consumer._readback_static_inputs(
        pins=pins,
        binding=binding,
        content=content,
        policy=policy,
    )

    assert readback[0] == pair
    assert readback[0].native_pin.source_head == B631_SOURCE_HEAD
    assert readback[1].payload()["protocol"]["source_head"] == B631_SOURCE_HEAD
    assert "source_profile" not in readback[1].payload()
    assert pair_profiles == [B631_SOURCE_PROFILE, B631_SOURCE_PROFILE]
    assert exact_profiles == [B631_SOURCE_PROFILE]


def test_candidate_profile_cannot_be_used_for_comparison_only_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.tools.market_replay_oracle.test_exact_fill_binding import _binding
    from tests.tools.market_replay_oracle.test_protocol import _document
    from tools.market_b4_oracle import offline_replay_consumer as consumer
    from tools.market_b4_oracle.frozen_market_replay import (
        FrozenMarketBinding,
        FrozenMarketPolicy,
        FrozenMarketWindow,
    )
    from tools.market_replay_oracle import DatasetContentPin, NativeSourcePin
    from tools.market_replay_oracle.profiles import effective_profile

    content = _binding().content
    comparison = copy.deepcopy(_document())
    dataset = comparison["dataset"]
    pins = consumer.StaticInputReadback(
        protocol_v3_path=tmp_path / "native.json",
        protocol_v3_pin=ProtocolPin("0" * 64, 1, B631_SOURCE_HEAD),
        protocol_v4_path=tmp_path / "comparison.json",
        protocol_v4_pin=ProtocolPin("0" * 64, 1, B631_SOURCE_HEAD),
        source_root=tmp_path,
        source_pin=NativeSourcePin(B631_SOURCE_HEAD, "2" * 64),
        dataset_root=Path(dataset["root"]),
        dataset_pin=DatasetContentPin(
            dataset["dataset_id"],
            dataset["input_sha256"]["manifest.json"],
            dataset["input_sha256"]["arrays.npz"],
            7453,
            5864784,
        ),
        source_profile=B631_SOURCE_PROFILE,
    )
    binding = FrozenMarketBinding._comparison_fixture(
        content=content,
        window=FrozenMarketWindow(2903, 9430, 9502),
        selected_symbol_index=0,
        expected_horizon_seconds=24 * 3600,
        observation_feature_indices=tuple(range(6)),
        native_role="nonrl",
        initial_capital=100000.0,
    )
    risk = _document()["policy"]["risk"]
    policy = FrozenMarketPolicy.from_effective_profile(
        dataset_id=content.dataset_id,
        profile=effective_profile("BASEL0", binding.window.actor_advances, risk),
        risk=risk,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("comparison-only binding reached native profile readback")

    monkeypatch.setattr(consumer, "load_declared_protocol_pair", forbidden)
    with pytest.raises(ValueError, match="native arm"):
        consumer._readback_static_inputs(
            pins=pins,
            binding=binding,
            content=content,
            policy=policy,
        )
