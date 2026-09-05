from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.workflows.universal_trade_rl_u1_test_support import (
    build_u1_workflow_fixture,
)
from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.workflows.universal_trade_rl_u1_contract import (
    UniversalTradeRLU1Contract,
)

_EXPECTED_FILENAMES = {"normalizer.json", "u1_contract.json"}
_EXPECTED_NORMALIZER_KEYS = {
    "version",
    "artifact_digest",
    "statistics_digest",
    "contract_digest",
    "train_symbols",
    "source_dataset_digests",
    "knowledge_cutoff_ns",
    "universe_manifest_digest",
    "provenance_digest",
    "clip_value",
    "channels",
}
_EXPECTED_CHANNEL_KEYS = {
    "timeframe",
    "feature_names",
    "mean",
    "scale",
    "per_symbol_sample_counts",
}


def _runner():
    try:
        return importlib.import_module(
            "trade_rl.workflows.universal_trade_rl_u1_runner"
        )
    except ModuleNotFoundError:
        pytest.fail("Universal Trade RL U1 artifact runner is not implemented")


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _fixture_contract_and_normalizer() -> tuple[
    UniversalTradeRLU1Contract,
    UniversalTradeSequenceNormalizer,
]:
    fixture = build_u1_workflow_fixture()
    contract = fixture.build_contract()
    normalizer = fixture.environment.sequence_normalizer
    assert isinstance(contract, UniversalTradeRLU1Contract)
    assert isinstance(normalizer, UniversalTradeSequenceNormalizer)
    return contract, normalizer


def _materialize(
    output: Path,
) -> tuple[UniversalTradeRLU1Contract, UniversalTradeSequenceNormalizer]:
    contract, normalizer = _fixture_contract_and_normalizer()
    _runner().materialize_universal_trade_rl_u1(
        contract=contract,
        normalizer=normalizer,
        output_root=output,
    )
    return contract, normalizer


def test_u1_materialization_is_exact_canonical_and_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "u1"
    contract, normalizer = _materialize(output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    _runner().materialize_universal_trade_rl_u1(
        contract=contract,
        normalizer=normalizer,
        output_root=output,
    )
    after = {path.name: path.read_bytes() for path in output.iterdir()}

    assert set(before) == _EXPECTED_FILENAMES
    assert after == before
    assert all(content.endswith(b"\n") for content in after.values())
    assert all(not content.endswith(b"\n\n") for content in after.values())

    for content in after.values():
        payload = json.loads(content)
        assert content == _canonical_json_bytes(payload)

    contract_payload = json.loads(after["u1_contract.json"])
    assert UniversalTradeRLU1Contract.from_payload(contract_payload) == contract

    normalizer_payload = json.loads(after["normalizer.json"])
    assert set(normalizer_payload) == _EXPECTED_NORMALIZER_KEYS
    assert normalizer_payload["artifact_digest"] == normalizer.digest
    assert normalizer_payload["statistics_digest"] == normalizer.statistics_digest
    assert normalizer_payload["contract_digest"] == normalizer.contract_digest
    assert normalizer_payload["train_symbols"] == list(normalizer.train_symbols)
    assert normalizer_payload["knowledge_cutoff_ns"] == normalizer.knowledge_cutoff_ns
    assert (
        normalizer_payload["universe_manifest_digest"]
        == normalizer.universe_manifest_digest
    )
    assert normalizer_payload["provenance_digest"] == normalizer.provenance_digest
    assert normalizer_payload["clip_value"] == normalizer.clip_value
    channels = normalizer_payload["channels"]
    assert isinstance(channels, list)
    assert len(channels) == len(normalizer.channels)
    assert all(set(channel) == _EXPECTED_CHANNEL_KEYS for channel in channels)


def test_u1_materialization_rejects_existing_artifact_drift_without_repair(
    tmp_path: Path,
) -> None:
    output = tmp_path / "u1"
    contract, normalizer = _materialize(output)
    target = output / "normalizer.json"
    target.write_text("{}\n", encoding="utf-8")
    tampered = target.read_bytes()

    with pytest.raises(ValueError, match="existing|drift"):
        _runner().materialize_universal_trade_rl_u1(
            contract=contract,
            normalizer=normalizer,
            output_root=output,
        )

    assert target.read_bytes() == tampered


def test_u1_materialization_rejects_extra_file_without_repair(tmp_path: Path) -> None:
    output = tmp_path / "u1"
    contract, normalizer = _materialize(output)
    extra = output / "extra.txt"
    extra.write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="existing|drift"):
        _runner().materialize_universal_trade_rl_u1(
            contract=contract,
            normalizer=normalizer,
            output_root=output,
        )

    assert extra.read_text(encoding="utf-8") == "unexpected"


def test_u1_second_staged_write_failure_never_publishes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, normalizer = _fixture_contract_and_normalizer()
    output = tmp_path / "u1"
    runner = _runner()
    original = runner._write_canonical_json
    calls = 0

    def fail_on_second_write(path: Path, payload: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-write failure")
        original(path, payload)

    monkeypatch.setattr(runner, "_write_canonical_json", fail_on_second_write)

    with pytest.raises(OSError, match="second-write"):
        runner.materialize_universal_trade_rl_u1(
            contract=contract,
            normalizer=normalizer,
            output_root=output,
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".u1.staging-*"))


def test_u1_materialization_rejects_contract_normalizer_identity_mismatch_before_write(
    tmp_path: Path,
) -> None:
    contract, normalizer = _fixture_contract_and_normalizer()
    mismatched = replace(contract, normalizer_digest="0" * 64, digest="")
    output = tmp_path / "u1"

    with pytest.raises(ValueError, match="normalizer|digest"):
        _runner().materialize_universal_trade_rl_u1(
            contract=mismatched,
            normalizer=normalizer,
            output_root=output,
        )

    assert not output.exists()


def test_u1_materialization_rejects_contract_provenance_mismatch_before_write(
    tmp_path: Path,
) -> None:
    contract, normalizer = _fixture_contract_and_normalizer()
    mismatched = replace(
        contract,
        normalizer_provenance_digest="0" * 64,
        digest="",
    )
    output = tmp_path / "u1"

    with pytest.raises(ValueError, match="provenance|normalizer"):
        _runner().materialize_universal_trade_rl_u1(
            contract=mismatched,
            normalizer=normalizer,
            output_root=output,
        )

    assert not output.exists()


def test_u1_materialization_rejects_preexisting_empty_output_directory(
    tmp_path: Path,
) -> None:
    contract, normalizer = _fixture_contract_and_normalizer()
    output = tmp_path / "u1"
    output.mkdir()

    with pytest.raises(ValueError, match="existing"):
        _runner().materialize_universal_trade_rl_u1(
            contract=contract,
            normalizer=normalizer,
            output_root=output,
        )
