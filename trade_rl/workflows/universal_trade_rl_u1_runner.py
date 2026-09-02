"""Atomic materialization of frozen Universal Trade RL U1 artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Final

from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.workflows.universal_trade_rl_u1_contract import (
    UniversalTradeRLU1Contract,
)

_NORMALIZER_FILENAME: Final = "normalizer.json"
_CONTRACT_FILENAME: Final = "u1_contract.json"
_ARTIFACT_FILENAMES: Final = tuple(sorted((_NORMALIZER_FILENAME, _CONTRACT_FILENAME)))


def _canonical_json_bytes(payload: object) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{encoded}\n".encode("utf-8")


def _write_canonical_json(path: Path, payload: object) -> None:
    data = _canonical_json_bytes(payload)
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_identity_match(
    contract: UniversalTradeRLU1Contract,
    normalizer: UniversalTradeSequenceNormalizer,
) -> None:
    if not isinstance(contract, UniversalTradeRLU1Contract):
        raise TypeError("U1 materialization contract is invalid")
    if not isinstance(normalizer, UniversalTradeSequenceNormalizer):
        raise TypeError("U1 materialization normalizer is invalid")
    if contract.normalizer_digest != normalizer.digest:
        raise ValueError("U1 contract normalizer digest mismatch")
    if contract.normalizer_provenance_digest != normalizer.provenance_digest:
        raise ValueError("U1 contract normalizer provenance mismatch")
    if contract.policy_contract_digest != normalizer.contract_digest:
        raise ValueError("U1 contract policy/normalizer identity mismatch")
    if contract.universe_manifest_digest != normalizer.universe_manifest_digest:
        raise ValueError("U1 contract universe/normalizer identity mismatch")
    if contract.normalizer_knowledge_cutoff_ns != normalizer.knowledge_cutoff_ns:
        raise ValueError("U1 contract normalizer knowledge cutoff mismatch")
    if contract.normalizer_clip_value != normalizer.clip_value:
        raise ValueError("U1 contract normalizer clip value mismatch")


def _normalizer_payload(normalizer: UniversalTradeSequenceNormalizer) -> dict[str, object]:
    return {
        "version": normalizer.version,
        "artifact_digest": normalizer.digest,
        "statistics_digest": normalizer.statistics_digest,
        "contract_digest": normalizer.contract_digest,
        "train_symbols": normalizer.train_symbols,
        "source_dataset_digests": normalizer.source_dataset_digests,
        "knowledge_cutoff_ns": normalizer.knowledge_cutoff_ns,
        "universe_manifest_digest": normalizer.universe_manifest_digest,
        "provenance_digest": normalizer.provenance_digest,
        "clip_value": normalizer.clip_value,
        "channels": tuple(channel.digest_payload() for channel in normalizer.channels),
    }


def _artifact_payloads(
    contract: UniversalTradeRLU1Contract,
    normalizer: UniversalTradeSequenceNormalizer,
) -> dict[str, object]:
    return {
        _NORMALIZER_FILENAME: _normalizer_payload(normalizer),
        _CONTRACT_FILENAME: contract.to_payload(),
    }


def _require_existing_identical(
    output_root: Path,
    *,
    payloads: dict[str, object],
) -> None:
    if not output_root.is_dir():
        raise ValueError("existing Universal Trade RL U1 output root drifted")
    observed_names = tuple(sorted(path.name for path in output_root.iterdir()))
    if observed_names != _ARTIFACT_FILENAMES:
        raise ValueError("existing Universal Trade RL U1 output artifacts drifted")
    for name in _ARTIFACT_FILENAMES:
        expected = _canonical_json_bytes(payloads[name])
        observed = (output_root / name).read_bytes()
        if observed != expected:
            raise ValueError("existing Universal Trade RL U1 output artifact drifted")


def materialize_universal_trade_rl_u1(
    *,
    contract: UniversalTradeRLU1Contract,
    normalizer: UniversalTradeSequenceNormalizer,
    output_root: str | Path,
) -> tuple[UniversalTradeRLU1Contract, UniversalTradeSequenceNormalizer]:
    """Validate and atomically publish one immutable U1 generation."""

    _require_identity_match(contract, normalizer)
    output_root = Path(output_root)
    payloads = _artifact_payloads(contract, normalizer)

    if output_root.exists():
        _require_existing_identical(output_root, payloads=payloads)
        return contract, normalizer

    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=parent,
        )
    )
    published = False
    try:
        _write_canonical_json(
            staging / _NORMALIZER_FILENAME,
            payloads[_NORMALIZER_FILENAME],
        )
        _write_canonical_json(
            staging / _CONTRACT_FILENAME,
            payloads[_CONTRACT_FILENAME],
        )
        _fsync_directory(staging)
        if output_root.exists():
            raise ValueError(
                "existing Universal Trade RL U1 output root appeared during publish"
            )
        os.replace(staging, output_root)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

    return contract, normalizer


__all__ = ["materialize_universal_trade_rl_u1"]
