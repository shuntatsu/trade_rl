"""Read-only reuse admission and exclusive copying of fully pinned artifacts."""

from __future__ import annotations

import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_settled_forecast_comparison import (
    SettledNonRLComparisonPlan,
)
from trade_rl.evaluation.allocation_terminal_evidence import (
    TerminalAllocationExecutionReceipt,
)
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream
from trade_rl.strategies.forecasts.simple_stream_io import (
    load_simple_return_stream_artifact,
)

from .restore import CostTables, closed, costs_payload, restore_costs, restore_matrix
from .tier_d_protocol import file_sha256
from .tier_d_records import load_gzip, output_manifest, read_json, save_gzip, write_json


def _same(a: object, b: object) -> None:
    if canonical_json_bytes(a) != canonical_json_bytes(b):
        raise ValueError(
            "resume source/protocol/provenance identity changed; start fresh without reuse"
        )


def _file(root: Path, relative: str) -> Path:
    if type(relative) is not str or not relative or "\\" in relative or ":" in relative:
        raise ValueError("invalid retained artifact path")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or any(p in {"..", "."} for p in parts):
        raise ValueError("retained artifact path escapes its attempt")
    path = root.joinpath(*parts)
    if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError("retained artifact must resolve inside its source attempt")
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError("retained artifacts must not use symlinks")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("retained artifact must be a regular file")
    return path


def _manifest(root: Path) -> dict[str, Any]:
    value = read_json(_file(root, "output-manifest.json"))
    if type(value) is not dict or not value:
        raise ValueError("source has no complete output manifest")
    for relative, pin in value.items():
        pin = closed(pin, {"sha256", "bytes"})
        path = _file(root, relative)
        if (
            type(pin["bytes"]) is not int
            or pin["bytes"] < 0
            or path.stat().st_size != pin["bytes"]
            or file_sha256(path) != pin["sha256"]
        ):
            raise ValueError("retained artifact bytes differ from the source manifest")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("source attempt contains a symlink")
    _same(value, output_manifest(root))
    return value


def freeze_inputs(
    root: Path,
    pairs: tuple[Any, ...],
    prepared: dict[str, Any],
    tables: CostTables,
    matrix: SettledNonRLComparisonPlan,
    protocol: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    costs = costs_payload(tables)
    cost_file = save_gzip(root, "costs.json.gz", costs)
    plan_file = save_gzip(root, "native-plan.json.gz", matrix.payload)
    forecasts = {}
    for configuration, inputs in prepared.items():
        path = inputs.artifact.path
        digest = inputs.artifact.digest
        if file_sha256(path) != digest or inputs.stream.digest != digest:
            raise ValueError("published forecast differs before freeze")
        forecasts[configuration] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "dataset_id": inputs.stream.dataset_id,
        }
    frozen = {
        "schema": "tier_d_frozen_inputs_v1",
        "protocol_digest": content_digest(protocol),
        "provenance_digest": content_digest(provenance),
        "matrix_digest": matrix.digest,
        "matrix_file": plan_file,
        "costs_digest": content_digest(costs),
        "costs_file": cost_file,
        "forecasts": forecasts,
        "pairs": [
            {
                "configuration": p.configuration,
                "account": p.account,
                "scenario": p.scenario,
                "symbol_index": p.symbol_index,
            }
            for p in pairs
        ],
    }
    write_json(root / "freeze.json", frozen)
    return frozen


@dataclass(frozen=True)
class ResumeInputs:
    source: Path
    manifest_digest: str
    manifest: dict[str, Any]
    frozen: dict[str, Any]
    matrix: SettledNonRLComparisonPlan
    tables: CostTables
    streams: dict[str, FrozenSimpleReturnStream]
    files: tuple[str, ...]

    def materialize(self, destination: Path) -> None:
        # All reuse was validated before destination creation; recheck each copy.
        for relative in self.files:
            source = _file(self.source, relative)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            pin = self.manifest[relative]
            if (
                target.stat().st_size != pin["bytes"]
                or file_sha256(target) != pin["sha256"]
            ):
                raise ValueError(
                    "retained bytes changed during exclusive copy; keep failed retry"
                )
        write_json(
            destination / "resume.json",
            {
                "source": str(self.source),
                "source_manifest_sha256": self.manifest_digest,
                "reused_files": list(self.files),
            },
        )


def preflight_resume(
    source: Path,
    destination: Path,
    protocol: dict[str, Any],
    provenance: dict[str, Any],
) -> ResumeInputs:
    """Validate the complete reuse set before any destination side effect."""
    try:
        if source.is_symlink() or destination.exists() or destination.is_symlink():
            raise ValueError(
                "resume requires a new destination and a regular source directory"
            )
        source, destination = source.resolve(strict=True), destination.resolve()
        if destination.is_relative_to(source):
            raise ValueError("new attempt must not be inside the preserved source")
        manifest_digest = file_sha256(_file(source, "output-manifest.json"))
        manifest = _manifest(source)
        _same(read_json(source / "protocol.json"), protocol)
        _same(read_json(source / "provenance.json"), provenance)
        state = read_json(source / "attempt.json")
        if (
            state["status"] not in {"FAILED", "COMPLETED"}
            or not state.get("matrix_digest")
            or not state.get("freeze_digest")
        ):
            raise ValueError(
                "resume source requires a terminal attempt and a complete frozen matrix"
            )
        frozen = closed(
            read_json(source / "freeze.json"),
            {
                "schema",
                "protocol_digest",
                "provenance_digest",
                "matrix_digest",
                "matrix_file",
                "costs_digest",
                "costs_file",
                "forecasts",
                "pairs",
            },
        )
        if (
            frozen["schema"] != "tier_d_frozen_inputs_v1"
            or content_digest(frozen) != state["freeze_digest"]
            or frozen["protocol_digest"] != content_digest(protocol)
            or frozen["provenance_digest"] != content_digest(provenance)
            or frozen["matrix_digest"] != state["matrix_digest"]
        ):
            raise ValueError("source freeze differs from its original pins")
        files = ["freeze.json"]
        for metadata_key in ("matrix_file", "costs_file"):
            metadata = closed(frozen[metadata_key], {"path", "sha256", "bytes"})
            _same(
                manifest[metadata["path"]],
                {k: metadata[k] for k in ("sha256", "bytes")},
            )
            files.append(metadata["path"])
        matrix = restore_matrix(
            load_gzip(_file(source, frozen["matrix_file"]["path"])),
            expected_digest=frozen["matrix_digest"],
        )
        tables = restore_costs(
            load_gzip(_file(source, frozen["costs_file"]["path"])),
            expected_digest=frozen["costs_digest"],
        )
        _same(list(matrix.accounts), protocol["accounts"])
        _same(list(matrix.scenarios), protocol["scenarios"])
        configurations = {c["id"]: c for c in protocol["configurations"]}
        if type(frozen["forecasts"]) is not dict or set(frozen["forecasts"]) != set(
            configurations
        ):
            raise ValueError("source forecast roster differs")
        streams = {}
        for configuration, metadata in frozen["forecasts"].items():
            metadata = closed(metadata, {"path", "sha256", "bytes", "dataset_id"})
            _same(
                manifest[metadata["path"]],
                {k: metadata[k] for k in ("sha256", "bytes")},
            )
            if metadata["dataset_id"] != protocol["dataset"]["dataset_id"]:
                raise ValueError("source forecast Dataset differs")
            streams[configuration] = load_simple_return_stream_artifact(
                _file(source, metadata["path"]),
                expected_digest=metadata["sha256"],
                expected_dataset_id=metadata["dataset_id"],
            )
            files.append(metadata["path"])
        declarations = {}
        if type(frozen["pairs"]) is not list:
            raise ValueError("source pair roster must be a list")
        for pair in frozen["pairs"]:
            pair = closed(
                pair, {"configuration", "account", "scenario", "symbol_index"}
            )
            pair_key = pair["account"], pair["scenario"]
            if (
                pair_key in declarations
                or pair["configuration"] not in configurations
                or type(pair["symbol_index"]) is not int
            ):
                raise ValueError("duplicate or foreign source pair")
            declarations[pair_key] = pair
        if set(declarations) != {g.key for g in matrix.controls}:
            raise ValueError("source pair roster differs from frozen matrix")
        plans = {}
        for group in matrix.ordered_groups:
            pair = declarations[group.key]
            common = group.nonrl_plan.native_plan.payload["common"]
            configuration = pair["configuration"]
            symbol = protocol["dataset"]["fit_symbols"][pair["symbol_index"]]
            scenario = "COST2L0" if pair["scenario"] == "COST2L0" else "BASEL0"
            rows = [
                row.payload()
                for row in tables[
                    configurations[configuration]["horizon_hours"], scenario
                ]
                if row.symbol == symbol
            ]
            rows.sort(key=lambda row: (row["symbol"], row["decision_time"]))
            if (
                common["symbol_index"] != pair["symbol_index"]
                or common["dataset_id"] != protocol["dataset"]["dataset_id"]
                or common["forecast_context_digest"]
                != frozen["forecasts"][configuration]["sha256"]
                or common["cost_beliefs_digest"] != content_digest(rows)
                or common["implementation_digest"]
                != provenance["native"]["implementation_digest"]
                or common["runtime_environment_digest"]
                != provenance["native"]["runtime_environment_digest"]
            ):
                raise ValueError(
                    "source matrix/forecast/cost/provenance binding differs"
                )
            for role, plan in zip(protocol["roles"], group.plans, strict=True):
                plans[*group.key, role] = plan
        expected = [
            (a, s, r)
            for a in protocol["accounts"]
            for s in protocol["scenarios"]
            for r in protocol["roles"]
        ]
        if [
            (s["account"], s["scenario"], s["role"]) for s in state["arms"]
        ] != expected:
            raise ValueError("source arm roster differs")
        for slot in state["arms"]:
            arm_key = slot["account"], slot["scenario"], slot["role"]
            relative = f"receipts/{arm_key[0]}_{arm_key[1]}_{arm_key[2]}.json.gz"
            if slot["state"] == "COMPLETED":
                metadata = closed(slot["receipt"], {"path", "sha256", "bytes"})
                if (
                    metadata["path"] != relative
                    or slot["plan_digest"] != plans[arm_key].digest
                ):
                    raise ValueError(
                        "completed source slot references a foreign receipt"
                    )
                _same(manifest[relative], {k: metadata[k] for k in ("sha256", "bytes")})
                raw = load_gzip(_file(source, relative))
                receipt = TerminalAllocationExecutionReceipt.from_payload(
                    raw, expected_plan=plans[arm_key]
                )
                if (
                    raw["status"] != "completed"
                    or raw["settlement_complete"] is not True
                    or receipt.digest != slot["receipt_digest"]
                ):
                    raise ValueError("completed source receipt is stale or partial")
                files.append(relative)
            elif (
                slot["state"]
                not in {
                    "UNATTEMPTED",
                    "FAILED",
                    "INTERRUPTED",
                    "CANCELLED_BEFORE_START",
                }
                or relative in manifest
            ):
                raise ValueError(
                    "source has an ambiguous, active, or unaccounted receipt"
                )
        if state["status"] == "COMPLETED" and any(
            s["state"] != "COMPLETED" for s in state["arms"]
        ):
            raise ValueError("source claims completion without every receipt")
        _same(manifest, _manifest(source))
        if file_sha256(source / "output-manifest.json") != manifest_digest:
            raise ValueError("source manifest changed during reuse validation")
        return ResumeInputs(
            source,
            manifest_digest,
            manifest,
            frozen,
            matrix,
            tables,
            streams,
            tuple(files),
        )
    except (OSError, KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError(
            "invalid or incomplete resume source; preserve it and start fresh"
        ) from error
