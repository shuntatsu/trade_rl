"""Prepare once and strictly restore artificial long-plan inputs without refits."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

import numpy as np

from tools.tier_d_nonrl.restore import closed, restore_costs, restore_matrix
from tools.tier_d_nonrl.resume import freeze_inputs
from tools.tier_d_nonrl.tier_d_native import PairDeclaration
from tools.tier_d_nonrl.tier_d_protocol import source_head, source_pins
from tools.tier_d_nonrl.tier_d_records import (
    load_gzip,
    output_manifest,
    read_json,
    write_json,
)
from tools.tier_d_nonrl.tier_d_runner import declare_matrix, prepare_inputs
from trade_rl.artifacts import content_digest
from trade_rl.data import load_market_dataset_artifact
from trade_rl.evaluation.allocation_market_inputs import PreparedAllocationMarketInputs
from trade_rl.evaluation.allocation_settled_forecast_comparison import (
    SettledNonRLComparisonPlan,
)
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.strategies.forecasts.simple_stream_io import (
    PublishedSimpleReturnStreamArtifact,
    load_simple_return_stream_artifact,
)

from .fixture import write_full_artificial_dataset
from .spec import profile_protocol, profile_source_pins, validate_profile_source


def _envelope(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_head": protocol["source_head"],
        "driver_sha256": source_pins(),
        "profile_sha256": profile_source_pins(),
        "protocol_digest": content_digest(protocol),
        "python": sys.version,
        "numpy": np.__version__,
        "native": build_candidate_run_provenance(),
    }


def prepare(root: Path) -> dict[str, Any]:
    """Fresh artificial inputs only; freeze full plans before any native reset."""
    validate_profile_source({"source_head": source_head()})
    root.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    state: dict[str, Any] = {"status": "PREPARING", "native_calls": 0}
    write_json(root / "preparation.json", state)
    try:
        dataset = write_full_artificial_dataset(root / "dataset")
        protocol = profile_protocol(dataset, root / "dataset")
        write_json(root / "protocol.json", protocol)
        envelope = _envelope(protocol)
        write_json(root / "provenance.json", envelope)
        prepared, tables = prepare_inputs(root, root / "dataset", dataset, protocol)
        pairs = declare_matrix(prepared, tables, protocol, envelope["native"])
        matrix = SettledNonRLComparisonPlan(
            tuple(pair.controls for pair in pairs),
            tuple(protocol["accounts"]),
            tuple(protocol["scenarios"]),
        )
        frozen = freeze_inputs(
            root, pairs, prepared, tables, matrix, protocol, envelope
        )
        validate_profile_source(protocol)
        state.update(
            status="PREPARED",
            matrix_digest=matrix.digest,
            freeze_digest=content_digest(frozen),
        )
    except Exception as error:
        state.update(
            status="FAILED", error={"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        state["elapsed_seconds"] = perf_counter() - started
        write_json(root / "preparation.json", state, replace=True)
        write_json(root / "output-manifest.json", output_manifest(root))
    return state


def _check_manifest(root: Path) -> None:
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("manifest root must not use symlinks")
    manifest = read_json(root / "output-manifest.json")
    if type(manifest) is not dict or not manifest:
        raise ValueError("prepared manifest is missing")
    for name in manifest:
        if (
            type(name) is not str
            or not name
            or "\\" in name
            or ":" in name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
        ):
            raise ValueError("prepared manifest path is invalid")
    canonical = root.resolve(strict=True)
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(canonical):
            raise ValueError("prepared manifest containment differs")
    if manifest != output_manifest(root):
        raise ValueError("prepared manifest bytes differ")


@dataclass(frozen=True)
class PreparedContext:
    root: Path
    protocol: dict[str, Any]
    envelope: dict[str, Any]
    inputs: PreparedAllocationMarketInputs
    pair: PairDeclaration
    matrix: SettledNonRLComparisonPlan


def load_prepared(root: Path) -> PreparedContext:
    """Verify the complete artifact set before loading or executing its inputs."""
    validate_profile_source({"source_head": source_head()})
    _check_manifest(root)
    protocol = read_json(root / "protocol.json")
    dataset = load_market_dataset_artifact(root / "dataset")
    if protocol != profile_protocol(dataset, root / "dataset"):
        raise ValueError("prepared artificial protocol differs")
    envelope = read_json(root / "provenance.json")
    if envelope != _envelope(protocol):
        raise ValueError("prepared source/runtime provenance differs")
    state = read_json(root / "preparation.json")
    if state["status"] != "PREPARED" or state["native_calls"] != 0:
        raise ValueError("requires complete preparation without native execution")
    frozen = read_json(root / "freeze.json")
    if (
        content_digest(frozen) != state["freeze_digest"]
        or frozen["protocol_digest"] != content_digest(protocol)
        or frozen["provenance_digest"] != content_digest(envelope)
        or frozen["matrix_digest"] != state["matrix_digest"]
    ):
        raise ValueError("prepared freeze binding differs")
    for key, name in (
        ("matrix_file", "native-plan.json.gz"),
        ("costs_file", "costs.json.gz"),
    ):
        if frozen[key] != {"path": name, **output_manifest(root)[name]}:
            raise ValueError("prepared frozen artifact binding differs")
    matrix = restore_matrix(
        load_gzip(root / "native-plan.json.gz"), expected_digest=state["matrix_digest"]
    )
    tables = restore_costs(
        load_gzip(root / "costs.json.gz"), expected_digest=frozen["costs_digest"]
    )
    config = protocol["configurations"][0]
    if (
        len(matrix.controls) != 1
        or list(matrix.accounts) != protocol["accounts"]
        or list(matrix.scenarios) != protocol["scenarios"]
        or set(frozen["forecasts"]) != {config["id"]}
    ):
        raise ValueError("prepared matrix roster differs")
    forecast = closed(
        frozen["forecasts"][config["id"]], {"path", "sha256", "bytes", "dataset_id"}
    )
    relative = f"forecasts/{config['id']}.json"
    if forecast != {
        "path": relative,
        "dataset_id": dataset.dataset_id,
        **output_manifest(root)[relative],
    }:
        raise ValueError("prepared forecast binding differs")
    stream = load_simple_return_stream_artifact(
        root / relative,
        expected_digest=forecast["sha256"],
        expected_dataset_id=dataset.dataset_id,
    )
    inputs = PreparedAllocationMarketInputs(
        dataset,
        stream,
        PublishedSimpleReturnStreamArtifact(
            root / relative, forecast["sha256"], dataset.dataset_id
        ),
        tuple(
            tuple(row for row in tables[72, "BASEL0"] if row.symbol == symbol)
            for symbol in dataset.symbols
        ),
    )
    pair = PairDeclaration(
        protocol["accounts"][0], "BASEL0", config["id"], 0, matrix.controls[0]
    )
    expected_pair = {
        "configuration": pair.configuration,
        "account": pair.account,
        "scenario": pair.scenario,
        "symbol_index": 0,
    }
    if frozen["pairs"] != [expected_pair] or len(stream.vintages) != 3:
        raise ValueError("prepared pair/vintage roster differs")
    _check_manifest(root)
    return PreparedContext(root, protocol, envelope, inputs, pair, matrix)
