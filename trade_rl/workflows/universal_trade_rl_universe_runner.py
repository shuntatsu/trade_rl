"""Atomic U0 universe materialization runner for Universal Trade RL."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from trade_rl.workflows.universal_trade_rl_run_identity import (
    UniversalTradeRLRunIdentity,
    UniversalTradeRLRunStage,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    load_universal_trade_rl_source_catalog,
    load_universal_trade_rl_universe_config,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)

_UNIVERSE_FILENAME: Final = "universe.json"
_IDENTITY_FILENAME: Final = "identity.json"
_ARTIFACT_FILENAMES: Final = (_IDENTITY_FILENAME, _UNIVERSE_FILENAME)


def _canonical_json_bytes(payload: object) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{encoded}\n".encode()


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


def _build_artifacts(
    *,
    config_path: Path,
    source_catalog_path: Path,
) -> tuple[UniversalTradeRLUniverseManifest, UniversalTradeRLRunIdentity]:
    config = load_universal_trade_rl_universe_config(config_path)
    sources = load_universal_trade_rl_source_catalog(source_catalog_path)
    universe = build_universal_trade_rl_universe_manifest(
        config=config,
        sources=sources,
    )
    identity = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION,
        universe_manifest_digest=universe.digest,
        model_config_digest=None,
        fit_provenance_digests=(),
    )
    return universe, identity


def _artifact_payloads(
    universe: UniversalTradeRLUniverseManifest,
    identity: UniversalTradeRLRunIdentity,
) -> dict[str, object]:
    return {
        _UNIVERSE_FILENAME: universe.to_payload(),
        _IDENTITY_FILENAME: identity.to_payload(),
    }


def _require_existing_identical(
    output_root: Path,
    *,
    payloads: dict[str, object],
) -> None:
    if not output_root.is_dir():
        raise ValueError("existing Universal Trade RL output root drifted")
    observed_names = tuple(sorted(path.name for path in output_root.iterdir()))
    if observed_names != _ARTIFACT_FILENAMES:
        raise ValueError("existing Universal Trade RL output artifacts drifted")
    for name in _ARTIFACT_FILENAMES:
        expected = _canonical_json_bytes(payloads[name])
        observed = (output_root / name).read_bytes()
        if observed != expected:
            raise ValueError("existing Universal Trade RL output artifact drifted")


def materialize_universal_trade_rl_universe(
    *,
    config_path: str | Path,
    source_catalog_path: str | Path,
    output_root: str | Path,
) -> tuple[UniversalTradeRLUniverseManifest, UniversalTradeRLRunIdentity]:
    """Validate and atomically publish one complete immutable U0 materialization."""

    config_path = Path(config_path)
    source_catalog_path = Path(source_catalog_path)
    output_root = Path(output_root)
    universe, identity = _build_artifacts(
        config_path=config_path,
        source_catalog_path=source_catalog_path,
    )
    payloads = _artifact_payloads(universe, identity)

    if output_root.exists():
        _require_existing_identical(output_root, payloads=payloads)
        return universe, identity

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
            staging / _UNIVERSE_FILENAME, payloads[_UNIVERSE_FILENAME]
        )
        _write_canonical_json(
            staging / _IDENTITY_FILENAME, payloads[_IDENTITY_FILENAME]
        )
        _fsync_directory(staging)
        if output_root.exists():
            raise ValueError(
                "existing Universal Trade RL output root appeared during publish"
            )
        os.replace(staging, output_root)
        published = True
        _fsync_directory(parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return universe, identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-rl-universe")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-catalog", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        universe, identity = materialize_universal_trade_rl_universe(
            config_path=args.config,
            source_catalog_path=args.source_catalog,
            output_root=args.output_root,
        )
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "error": str(error),
                    "production_status": "NO-GO",
                    "status": "rejected",
                },
                sort_keys=True,
            )
        )
        return 5
    print(
        json.dumps(
            {
                "identity_digest": identity.digest,
                "production_status": "NO-GO",
                "status": "materialized",
                "universe_manifest_digest": universe.digest,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "cli_main",
    "materialize_universal_trade_rl_universe",
]
