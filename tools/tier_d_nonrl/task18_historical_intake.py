"""Verify the retained Task18 R2 checkpoint without importing archived code.

Only the exact published 2026-10-15 release is admitted by the public entry
point. Hash opaque members (including forecasts and nested gzip/NPZ files)
without decoding them or extracting the ZIP. Parse only bounded manifest,
protocol, provenance and attempt-status metadata. Dataset declarations are
not proof that the original market arrays or runtime are available. The v2
inventory also verifies every pinned original R2 Python source in its Git tree
and both archived native provenance digests; the runtime is a declaration,
not an observation of a restored installation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import stat
import subprocess
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.tier_d_nonrl.task18_recipe import (
    SourceFilePin,
    Task18Protocol,
    build_task18_protocol,
)
from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import open_regular_binary, read_verified_bytes

_TAG = "research/issue810-development-checkpoint-20261015"
_RELEASE_SOURCE = "30c5ab54fdb9db7f03b947c1687a09c8b929c461"
_RELEASE_TREE = "75744f807c54539f355499fc9b204820f2d0489c"
_R2_SOURCE_FILE_COUNT = 288
_R2_IMPLEMENTATION_DIGEST = (
    "71e69fa9ad2087c0740a0c77efa3f3da9937bc5ce5a17623a0790e16601dec3d"
)
_R2_RUNTIME_DIGEST = "0c56fc67d16f583d49b5dea7a3878e3fcba0f58e6abd0e9dd66e51393c994cc9"
_R2_PROTOCOL = "ee21a2ce876f0e79a3ec518a963320fafe74430bef1944e9df918c760dfe43a1"
_R2_MATRIX = "0fdc9537019ac3feffd834de5220c7845b27ae252d449e47a0cd0a91417dc79b"
_PREFIX = "output/issue810-tier-d-nonrl-preparation/"
_R2 = _PREFIX + "market-20261014-process-tree-r2/"
_R1 = _PREFIX + "market-20261014T1305Z/"
_ARCHIVE_NAME = "issue810-development-checkpoint-20261015.zip"
_MAX_METADATA = 4 * 1024 * 1024
_MAX_MEMBER = 32 * 1024 * 1024
_MAX_UNCOMPRESSED = 192 * 1024 * 1024
_MAX_MEMBERS = 128
_DRIVER_NAMES = (
    "tier_d_decomposition.py",
    "tier_d_fixture.py",
    "tier_d_native.py",
    "tier_d_process_tree.py",
    "tier_d_protocol.py",
    "tier_d_records.py",
    "tier_d_runner.py",
    "tier_d_supervisor.py",
)


@dataclass(frozen=True, slots=True)
class _Pin:
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        require_sha256(self.sha256, field="historical member SHA-256")
        if type(self.bytes) is not int or not 0 <= self.bytes <= _MAX_MEMBER:
            raise ValueError("historical member size exceeds bound")


# Outer archive is slightly larger than a single member. Its exact size is
# externally pinned rather than admitted by the generic member-size bound.
_ASSETS = {
    _ARCHIVE_NAME: (
        "03bcbd613a516e9e0504d5c074297a896d3e680d87c29adcdbc2fad2d4dc1556",
        34_538_439,
    ),
    "MANIFEST.json": (
        "9e1c4ff1d0f30e9fcb98fc915edcf10348e8dfdce1a04a14b1b7acabe312935e",
        18_512,
    ),
    "SHA256SUMS": (
        "50a74c7fea1968ea826fdc14fd0d029af5345688439e7bf5bc6bdcff9a5cd4c6",
        112,
    ),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate historical metadata key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError("nonfinite historical metadata constant")


def _json(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > _MAX_METADATA:
        raise ValueError("historical metadata exceeds size bound")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
        if type(value) is not dict:
            raise ValueError("historical metadata must be an object")
        # Also reject overflowed finite-syntax numbers (e.g. 1e999).
        canonical_json_bytes(value)
    except (UnicodeError, RecursionError, TypeError, OverflowError) as exc:
        raise ValueError("invalid historical metadata") from exc
    return value


def _member_pins(value: object) -> dict[str, _Pin]:
    if type(value) is not dict or not 0 < len(value) <= _MAX_MEMBERS:
        raise ValueError("historical member roster exceeds bound or is empty")
    pins: dict[str, _Pin] = {}
    for path, item in value.items():
        if type(item) is not dict or item.keys() != {"sha256", "bytes"}:
            raise ValueError("historical member pin schema differs")
        pin = _Pin(**item)
        SourceFilePin(path, pin.sha256)  # Same strict repo-relative path grammar.
        pins[path] = pin
    if len({name.casefold() for name in pins}) != len(pins):
        raise ValueError("case-aliased historical member names")
    return pins


def _verify_members(
    archive: zipfile.ZipFile, manifest_raw: bytes, pins: dict[str, _Pin]
) -> int:
    if any(name.casefold() == "manifest.json" for name in pins):
        raise ValueError("historical manifest must not pin itself")
    expected = {**pins, "MANIFEST.json": _Pin(_sha(manifest_raw), len(manifest_raw))}
    infos = archive.infolist()
    if (
        len(infos) > _MAX_MEMBERS
        or len(infos) != len(expected)
        or {i.filename for i in infos} != expected.keys()
    ):
        raise ValueError("historical ZIP member roster differs")
    total = sum(pin.bytes for pin in expected.values())
    if total > _MAX_UNCOMPRESSED:
        raise ValueError("historical ZIP aggregate size exceeds bound")
    # Check all declared names, types and sizes before expanding any member.
    for info in infos:
        if (
            info.orig_filename != info.filename
            or info.is_dir()
            or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)
            or info.flag_bits & 1
            or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            or info.file_size != expected[info.filename].bytes
        ):
            raise ValueError("historical ZIP member type or size differs")
    for info in infos:
        pin = expected[info.filename]
        digest = hashlib.sha256()
        size = 0
        with archive.open(info) as stream:
            while chunk := stream.read(min(1024 * 1024, pin.bytes - size + 1)):
                size += len(chunk)
                if size > pin.bytes:
                    raise ValueError("historical ZIP member exceeds declared size")
                digest.update(chunk)
        if size != pin.bytes or digest.hexdigest() != pin.sha256:
            raise ValueError("historical ZIP member pin mismatch")
    return total


def _read_metadata(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    if archive.getinfo(name).file_size > _MAX_METADATA:
        raise ValueError("historical metadata exceeds size bound")
    return _json(archive.read(name))


def _nested_manifest(
    archive: zipfile.ZipFile, pins: dict[str, _Pin], prefix: str
) -> dict[str, _Pin]:
    # The released output-manifest is itself an opaque release-pinned member.
    actual = _member_pins(_read_metadata(archive, prefix + "output-manifest.json"))
    expected = {
        name.removeprefix(prefix): pin
        for name, pin in pins.items()
        if name.startswith(prefix) and name != prefix + "output-manifest.json"
    }
    if actual != expected:
        raise ValueError("historical attempt manifest differs from release roster")
    return actual


def _expected_roster(recipe: Task18Protocol) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (f"{arm.configuration}_{arm.symbol}", arm.role.lower(), arm.scenario)
        for arm in recipe.arms
    )


def _stopped_attempt(
    data: dict[str, Any], recipe: Task18Protocol, *, interrupted: int
) -> dict[str, Any]:
    arms = data["arms"]
    if (
        data["scope"] != "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY"
        or data["status"] != "FAILED"
        or type(arms) is not list
        or len(arms) != len(recipe.arms)
        or tuple((arm["account"], arm["role"], arm["scenario"]) for arm in arms)
        != _expected_roster(recipe)
        or any(arm["receipt"] is not None for arm in arms)
        or [arm["state"] for arm in arms]
        != ["INTERRUPTED"] * interrupted + ["UNATTEMPTED"] * (len(arms) - interrupted)
    ):
        raise ValueError("historical stopped attempt roster or status differs")
    return {
        "status": "FAILED",
        "state_counts": dict(Counter(arm["state"] for arm in arms)),
        "completion_receipt_references": 0,
    }


def _check_protocol(
    protocol: dict[str, Any], recipe: Task18Protocol, expected_digest: str
) -> None:
    ref = recipe.historical_reference
    ds = protocol["dataset"]
    budget = protocol["budget"]
    if (
        _sha(canonical_json_bytes(protocol)) != expected_digest
        or protocol["source_head"] != _RELEASE_SOURCE
        or protocol["schema"] != "issue810_tier_d_nonrl48_result_blind_v1"
        or protocol["scope"] != "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY"
        or ds["dataset_id"] != ref.dataset_id
        or ds["input_sha256"]
        != {
            "manifest.json": ref.manifest_sha256,
            "arrays.npz": ref.arrays_sha256,
        }
        or ds["schema"] != f"market_dataset_artifact_v{ref.artifact_schema}"
        or ds["identity"] != f"market_dataset_identity_v{ref.identity_schema}"
        or ds["build"] != f"market_build_v{ref.market_build_version}"
        or ds["dimensions"]
        != [
            recipe.dataset.row_count,
            len(recipe.dataset.symbols),
            recipe.dataset.features_per_symbol,
        ]
        or tuple(ds["feature_names"]) != recipe.dataset.feature_names
        or budget["advances_per_arm"] != recipe.index_plan.native_advances_per_arm
        or budget["native_arms"] != len(recipe.arms)
        or budget["matched_pairs"] != len(recipe.pairs)
        or budget["ridge_vintages"] != recipe.index_plan.ridge_vintage_count
        or budget["public_prepare_fits"]
        != recipe.index_plan.public_cost_preparation_fit_count
    ):
        raise ValueError("historical protocol or declared Dataset binding differs")


def _git_output(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(repo), *args],
        timeout=10,
    )


def _audit_reference_source(repo: Path, recipe: Task18Protocol) -> int:
    ref = recipe.historical_reference
    if _git_output(repo, "cat-file", "-t", ref.source_commit).strip() != b"commit":
        raise ValueError("historical recipe source is not a commit")
    for pin in ref.source_file_pins:
        if (
            _sha(_git_output(repo, "show", f"{ref.source_commit}:{pin.path}"))
            != pin.sha256
        ):
            raise ValueError("historical recipe source pin mismatch")
    return len(ref.source_file_pins)


def _audit_r2_native_provenance(repo: Path, native: object) -> dict[str, Any]:
    """Verify the exact original Python source roster, not a live runtime."""
    if (
        type(native) is not dict
        or native.keys()
        != {
            "schema_version",
            "implementation",
            "implementation_digest",
            "runtime_environment",
            "runtime_environment_digest",
            "research_context_digest",
        }
        or native["schema_version"] != "candidate_run_provenance_v1"
        or native["research_context_digest"] is not None
    ):
        raise ValueError("historical R2 native provenance schema differs")
    implementation = native["implementation"]
    runtime = native["runtime_environment"]
    if (
        type(implementation) is not dict
        or implementation.keys() != {"schema_version", "files"}
        or implementation["schema_version"] != "candidate_run_implementation_v1"
        or type(runtime) is not dict
        or runtime.get("schema_version") != "candidate_run_runtime_v1"
        or native["implementation_digest"] != _R2_IMPLEMENTATION_DIGEST
        or native["runtime_environment_digest"] != _R2_RUNTIME_DIGEST
        or _sha(canonical_json_bytes(implementation)) != _R2_IMPLEMENTATION_DIGEST
        or _sha(canonical_json_bytes(runtime)) != _R2_RUNTIME_DIGEST
    ):
        raise ValueError("historical R2 implementation/runtime declaration differs")

    files = implementation["files"]
    if type(files) is not list or len(files) != _R2_SOURCE_FILE_COUNT:
        raise ValueError("historical R2 Python source roster differs")
    pins: dict[str, str] = {}
    for item in files:
        if type(item) is not dict or item.keys() != {"path", "sha256"}:
            raise ValueError("historical R2 Python source pin schema differs")
        pin = SourceFilePin(**item)
        if not pin.path.endswith(".py") or pin.path in pins:
            raise ValueError("historical R2 Python source roster differs")
        pins[pin.path] = pin.sha256
    if len({path.casefold() for path in pins}) != len(pins):
        raise ValueError("case-aliased historical R2 Python source roster")

    if _git_output(
        repo, "cat-file", "-t", _RELEASE_SOURCE
    ).strip() != b"commit" or _git_output(
        repo, "rev-parse", f"{_RELEASE_SOURCE}^{{tree}}"
    ).strip() != _RELEASE_TREE.encode("ascii"):
        raise ValueError("historical release commit/tree differs")
    # NUL-separated tree records avoid Git path quoting and worktree/CRLF
    # conversion. The retained release pins match the original raw Git blobs.
    listing = _git_output(
        repo, "ls-tree", "-r", "-z", "--full-tree", _RELEASE_SOURCE, "--", "trade_rl/"
    )
    blobs: dict[str, str] = {}
    for record in listing.split(b"\0"):
        if not record:
            continue
        identity, raw_path = record.split(b"\t", 1)
        mode, kind, oid = identity.split()
        if not raw_path.startswith(b"trade_rl/"):
            raise ValueError("historical R2 Git source is outside the package")
        path = raw_path.removeprefix(b"trade_rl/").decode("utf-8")
        if not path.endswith(".py"):
            continue
        if mode not in {b"100644", b"100755"} or kind != b"blob" or path in blobs:
            raise ValueError("historical R2 source is not a unique regular Python blob")
        blobs[path] = oid.decode("ascii")
    if blobs.keys() != pins.keys():
        raise ValueError("historical R2 Git Python source roster differs")
    for path, digest in pins.items():
        if _sha(_git_output(repo, "cat-file", "blob", blobs[path])) != digest:
            raise ValueError("historical R2 Python source pin mismatch")

    return {
        "source_commit": _RELEASE_SOURCE,
        "source_tree": _RELEASE_TREE,
        "python_source_pins_verified": len(pins),
        "implementation_digest": _R2_IMPLEMENTATION_DIGEST,
        "runtime_environment_digest": _R2_RUNTIME_DIGEST,
        "runtime_environment_declaration": runtime,
    }


def _verify_opaque_file(path: Path, expected_sha256: str, *, limit: int) -> int:
    """Hash only one regular-file snapshot, with no market deserialization."""
    require_sha256(expected_sha256, field="original Dataset SHA-256")
    digest = hashlib.sha256()
    size = 0
    with open_regular_binary(path, field="original Dataset file") as stream:
        while chunk := stream.read(min(1024 * 1024, limit - size + 1)):
            size += len(chunk)
            if size > limit:
                raise ValueError("original Dataset file exceeds byte limit")
            digest.update(chunk)
    if size == 0 or digest.hexdigest() != expected_sha256:
        raise ValueError("original Dataset file digest mismatch")
    return size


def _verify_dataset_bytes(root: Path, recipe: Task18Protocol) -> dict[str, int]:
    ref = recipe.historical_reference
    return {
        "manifest.json": _verify_opaque_file(
            root / "manifest.json", ref.manifest_sha256, limit=_MAX_METADATA
        ),
        "arrays.npz": _verify_opaque_file(
            root / "arrays.npz", ref.arrays_sha256, limit=128 * 1024 * 1024
        ),
    }


def inspect_historical_checkpoint(
    root: Path, *, repo: Path, dataset_directory: Path | None = None
) -> dict[str, Any]:
    """Return a read-only HOLD inventory of the exact released R1/R2 bytes."""
    assets = {
        name: read_verified_bytes(
            root / name, expected_digest=digest, expected_size_bytes=size, field=name
        )
        for name, (digest, size) in _ASSETS.items()
    }
    if (
        assets["SHA256SUMS"].strip()
        != f"{_ASSETS[_ARCHIVE_NAME][0]}  {_ARCHIVE_NAME}".encode()
    ):
        raise ValueError("historical checksum declaration differs")
    manifest_raw = assets["MANIFEST.json"]
    manifest = _json(manifest_raw)
    if (
        manifest.keys()
        != {
            "files",
            "market_result",
            "schema",
            "scope",
            "source_head",
            "source_sha256",
            "tag",
        }
        or manifest["tag"] != _TAG
        or manifest["schema"] != "issue810_development_checkpoint_v1"
        or manifest["scope"] != "IN_PROGRESS_CONSUMED_DEVELOPMENT_SOURCE_AND_EVIDENCE"
        or manifest["source_head"] != _RELEASE_SOURCE
        or manifest["market_result"] != "NOT_ESTABLISHED"
    ):
        raise ValueError("historical release identity differs")
    pins = _member_pins(manifest["files"])
    drivers = manifest["source_sha256"]
    if (
        type(drivers) is not dict
        or set(drivers) != set(_DRIVER_NAMES)
        or any(
            pins[_PREFIX + name].sha256 != digest for name, digest in drivers.items()
        )
    ):
        raise ValueError("historical release source declaration differs")
    recipe = build_task18_protocol()
    ref = recipe.historical_reference
    with zipfile.ZipFile(io.BytesIO(assets[_ARCHIVE_NAME])) as archive:
        total = _verify_members(archive, manifest_raw, pins)
        if len(pins) != 86 or total != 164_241_586:
            raise ValueError("historical release exact inventory differs")
        # All outer/member pins have passed; decode only named metadata,
        # never forecasts, native-plan gzip, NPZ or archived Python.
        r2_pins = _nested_manifest(archive, pins, _R2)
        _nested_manifest(archive, pins, _R1)
        protocol = _read_metadata(archive, _R2 + "protocol.json")
        _check_protocol(protocol, recipe, _R2_PROTOCOL)
        provenance = _read_metadata(archive, _R2 + "provenance.json")
        if (
            provenance["source_head"] != _RELEASE_SOURCE
            or provenance["protocol_digest"] != _R2_PROTOCOL
            or provenance["driver_sha256"] != drivers
        ):
            raise ValueError("historical R2 source/protocol provenance differs")
        r2 = _read_metadata(archive, _R2 + "attempt.json")
        if r2["matrix_digest"] != _R2_MATRIX:
            raise ValueError("historical R2 matrix identity differs")
        r2_summary = _stopped_attempt(r2, recipe, interrupted=1)
        r1_summary = _stopped_attempt(
            _read_metadata(archive, _R1 + "attempt.json"), recipe, interrupted=0
        )
        forecast_paths = tuple(
            f"forecasts/{c.name}.json" for c in recipe.configurations
        )
        if (
            set(name for name in r2_pins if name.startswith("forecasts/"))
            != set(forecast_paths)
            or "native-plan.json.gz" not in r2_pins
            or any(
                name.startswith("receipts/") or name.endswith(".npz")
                for name in r2_pins
            )
        ):
            raise ValueError("historical R2 plan/forecast/receipt inventory differs")
        plan = r2_pins["native-plan.json.gz"]
        forecasts = {name: asdict(r2_pins[name]) for name in forecast_paths}
    original_r2_provenance = _audit_r2_native_provenance(repo, provenance["native"])
    matched = _audit_reference_source(repo, recipe)
    dataset_sizes = (
        None
        if dataset_directory is None
        else _verify_dataset_bytes(dataset_directory, recipe)
    )
    reference_by_name = {
        Path(pin.path).name: pin.sha256 for pin in ref.source_file_pins
    }
    driver_matches = tuple(
        sorted(name for name in drivers if drivers[name] == reference_by_name.get(name))
    )
    return {
        "schema": "task18_historical_checkpoint_intake_v2",
        "scope": "READ_ONLY_CONSUMED_DEVELOPMENT_ARCHIVE_INVENTORY",
        "release_tag": _TAG,
        "release_assets": {
            name: {"sha256": digest, "bytes": size}
            for name, (digest, size) in _ASSETS.items()
        },
        "verified_archive_members": len(pins) + 1,
        "verified_uncompressed_bytes": total,
        "release_source_commit": _RELEASE_SOURCE,
        "original_r2_provenance": original_r2_provenance,
        "r1": r1_summary,
        "r2": {
            **r2_summary,
            "matrix_digest": _R2_MATRIX,
            "protocol_digest": _R2_PROTOCOL,
            "protocol_file": asdict(r2_pins["protocol.json"]),
            "native_plan_file": asdict(plan),
            "forecast_files": forecasts,
            "verified_output_manifest_files": len(r2_pins),
        },
        "recipe_source_commit": ref.source_commit,
        "recipe_source_pins_verified": matched,
        "recipe_protocol_digest": ref.protocol_digest,
        "archive_driver_count": len(drivers),
        "archive_drivers_byte_identical_to_recipe": driver_matches,
        "declared_dataset_matches_recipe": True,
        "dataset_id": ref.dataset_id,
        "dataset_manifest_sha256": ref.manifest_sha256,
        "dataset_arrays_sha256": ref.arrays_sha256,
        "original_market_dataset_bytes_verified": dataset_sizes is not None,
        "original_market_dataset_verified_sizes": dataset_sizes,
        "original_market_dataset_decoded": False,
        "source_commit_matches_recipe": _RELEASE_SOURCE == ref.source_commit,
        "protocol_digest_matches_recipe": _R2_PROTOCOL == ref.protocol_digest,
        "native_plan_decoded": False,
        "forecast_values_decoded": False,
        "archived_code_executed": False,
        "original_runtime_verified": False,
        "recovery_ready": False,
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-directory", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        help="explicit original consumed Dataset directory; opaque SHA checks only",
    )
    args = parser.parse_args(argv)
    try:
        report = inspect_historical_checkpoint(
            args.release_directory,
            repo=args.repo,
            dataset_directory=args.dataset_directory,
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        zipfile.BadZipFile,
        subprocess.SubprocessError,
    ):
        parser.exit(
            2,
            "historical checkpoint intake refused; exact evidence unavailable or inconsistent\n",
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    # Successful verification is a useful inventory, but recovery remains HOLD.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
