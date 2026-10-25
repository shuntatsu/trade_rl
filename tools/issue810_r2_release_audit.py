"""Fail-closed, no-network verification of the original Issue #810 R2 ZIP.

This verifies only published ZIP/member bytes and checkpoint source declarations.
It must not interpret results, admit a new run, or certify G0-G4/unused/live.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

R2_ZIP_BYTES = 34_538_439
R2_ZIP_SHA256 = "03bcbd613a516e9e0504d5c074297a896d3e680d87c29adcdbc2fad2d4dc1556"
R2_MANIFEST_BYTES = 18_512
R2_MANIFEST_SHA256 = "9e1c4ff1d0f30e9fcb98fc915edcf10348e8dfdce1a04a14b1b7acabe312935e"
R2_HEAD = "30c5ab54fdb9db7f03b947c1687a09c8b929c461"
R2_SOURCE_DIR = "output/issue810-tier-d-nonrl-preparation/"
R2_ROOT = R2_SOURCE_DIR + "market-20261014-process-tree-r2/"
R2_SOURCE_FILES = frozenset(
    {
        "tier_d_decomposition.py",
        "tier_d_fixture.py",
        "tier_d_native.py",
        "tier_d_process_tree.py",
        "tier_d_protocol.py",
        "tier_d_records.py",
        "tier_d_runner.py",
        "tier_d_supervisor.py",
    }
)
R2_ROOT_FILES = frozenset(
    {
        "attempt.json",
        "forecasts/H24_RIDGE_A1.json",
        "forecasts/H24_RIDGE_A10.json",
        "forecasts/H72_RIDGE_A1.json",
        "forecasts/H72_RIDGE_A10.json",
        "native-plan.json.gz",
        "operator-stop.json",
        "output-manifest.json",
        "progress.jsonl",
        "protocol.json",
        "provenance.json",
        "supervisor.json",
        "worker.log",
    }
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SHA_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
MAX_MEMBERS = 200
MAX_UNCOMPRESSED = 512 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


class ArchiveAuditError(ValueError):
    """The release identity, ZIP archive, or a manifest entry is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchiveAuditError(message)


def _sha_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _checked_json(data: bytes, *, name: str) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ArchiveAuditError(f"duplicate JSON field: {name}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ArchiveAuditError(f"non-finite JSON value: {value}")

    try:
        obj = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError) as exc:
        raise ArchiveAuditError(f"invalid JSON: {name}: {exc}") from exc
    _require(isinstance(obj, dict), f"JSON root must be an object: {name}")
    return obj


def _safe_path(name: str) -> None:
    _require(isinstance(name, str) and bool(name), "empty ZIP member path")
    _require(
        "\\" not in name and "\x00" not in name,
        f"invalid ZIP member path: {name}",
    )
    _require(
        not name.startswith("/") and ":" not in name,
        f"absolute ZIP path: {name}",
    )
    parts = name.rstrip("/").split("/")
    _require(
        all(part not in ("", ".", "..") for part in parts),
        f"unsafe ZIP path: {name}",
    )
    _require(
        PurePosixPath(name.rstrip("/")).as_posix() == name.rstrip("/"),
        f"noncanonical ZIP path: {name}",
    )


def _manifest_files(manifest: dict[str, Any]) -> dict[str, tuple[int, str]]:
    expected_fields = {
        "files",
        "market_result",
        "schema",
        "scope",
        "source_head",
        "source_sha256",
        "tag",
    }
    _require(
        set(manifest) == expected_fields,
        "R2 checkpoint manifest root fields differ from the pinned schema",
    )
    _require(
        manifest["schema"] == "issue810_development_checkpoint_v1",
        "manifest schema changed",
    )
    _require(
        manifest["market_result"] == "NOT_ESTABLISHED",
        "market result was reclassified",
    )
    _require(
        manifest["scope"] == "IN_PROGRESS_CONSUMED_DEVELOPMENT_SOURCE_AND_EVIDENCE",
        "checkpoint market/data scope changed",
    )
    _require(manifest["source_head"] == R2_HEAD, "R2 historical source HEAD differs")
    _require(
        _SHA_COMMIT.fullmatch(manifest["source_head"]) is not None,
        "invalid commit SHA",
    )
    _require(
        manifest["tag"] == "research/issue810-development-checkpoint-20261015",
        "release tag changed",
    )
    records = manifest["files"]
    _require(
        isinstance(records, dict) and len(records) == 86,
        "R2 must declare 86 files",
    )
    verified: dict[str, tuple[int, str]] = {}
    for path, raw in records.items():
        _safe_path(path)
        _require(not path.endswith("/"), "manifest lists a directory")
        _require(path != "MANIFEST.json", "manifest must not attest its own digest")
        _require(isinstance(raw, dict), f"manifest file has invalid metadata: {path}")
        _require(
            set(raw) == {"bytes", "sha256"},
            f"unexpected manifest entry fields: {path}",
        )
        size, digest = raw["bytes"], raw["sha256"]
        _require(
            type(size) is int and 0 <= size <= MAX_UNCOMPRESSED,
            f"invalid member size: {path}",
        )
        _require(
            isinstance(digest, str) and _DIGEST.fullmatch(digest) is not None,
            f"invalid member digest: {path}",
        )
        verified[path] = (size, digest)
    declared_sources = manifest["source_sha256"]
    _require(isinstance(declared_sources, dict), "missing source-pins map")
    _require(set(declared_sources) == R2_SOURCE_FILES, "R2 source file roster changed")
    for name, digest in declared_sources.items():
        path = R2_SOURCE_DIR + name
        _require(path in verified, f"missing archived source: {name}")
        _require(
            verified[path][1] == digest,
            f"source-pins disagree with file manifest: {name}",
        )
    observed_root = {p.removeprefix(R2_ROOT) for p in verified if p.startswith(R2_ROOT)}
    _require(observed_root == R2_ROOT_FILES, "original R2 root file roster changed")
    _require("CHECKPOINT.md" in verified, "missing original checkpoint explanation")
    return verified


def audit_archive(
    archive_path: Path,
    manifest_path: Path,
    *,
    archive_sha256: str = R2_ZIP_SHA256,
    archive_size: int = R2_ZIP_BYTES,
    manifest_sha256: str = R2_MANIFEST_SHA256,
    manifest_size: int = R2_MANIFEST_BYTES,
) -> dict[str, object]:
    """Verify every exact member with streaming SHA256; do not parse outcome files.

    Digest/size overrides exist for synthetic test fixtures only. The CLI pins
    the real public release constants unconditionally.
    """
    _require(
        _DIGEST.fullmatch(archive_sha256) is not None,
        "bad expected archive SHA",
    )
    _require(
        _DIGEST.fullmatch(manifest_sha256) is not None,
        "bad expected manifest SHA",
    )
    _require(archive_path.is_file(), "R2 ZIP file is absent")
    _require(manifest_path.is_file(), "R2 release MANIFEST.json is absent")
    observed_zip_sha, observed_zip_bytes = _sha_file(archive_path)
    _require(
        observed_zip_bytes == archive_size,
        "R2 ZIP byte length differs from release",
    )
    _require(observed_zip_sha == archive_sha256, "R2 ZIP SHA256 differs from release")
    observed_manifest_sha, observed_manifest_bytes = _sha_file(manifest_path)
    _require(
        observed_manifest_bytes == manifest_size,
        "external MANIFEST byte length differs",
    )
    _require(
        observed_manifest_sha == manifest_sha256,
        "external MANIFEST SHA256 differs",
    )
    manifest_bytes = manifest_path.read_bytes()
    manifest = _checked_json(manifest_bytes, name="external manifest")
    listed_files = _manifest_files(manifest)

    try:
        with zipfile.ZipFile(archive_path, "r", allowZip64=True) as archive:
            infos = archive.infolist()
            _require(len(infos) <= MAX_MEMBERS, "ZIP has too many members")
            names: set[str] = set()
            files: dict[str, zipfile.ZipInfo] = {}
            total = 0
            for info in infos:
                _safe_path(info.filename)
                _require(info.filename not in names, "duplicate ZIP member path")
                names.add(info.filename)
                _require(
                    not info.flag_bits & 1,
                    f"encrypted ZIP member: {info.filename}",
                )
                mode = (info.external_attr >> 16) & 0xFFFF
                _require(
                    stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR),
                    f"unsupported ZIP file type: {info.filename}",
                )
                _require(
                    info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                    f"unsupported ZIP compression: {info.filename}",
                )
                if info.is_dir():
                    _require(info.file_size == 0, "directory entry has content")
                    continue
                total += info.file_size
                _require(
                    total <= MAX_UNCOMPRESSED,
                    "archive uncompressed data exceeds cap",
                )
                files[info.filename] = info
            expected_names = set(listed_files) | {"MANIFEST.json"}
            _require(
                set(files) == expected_names,
                "ZIP file roster differs from R2 manifest",
            )
            for name in names - set(files):
                _require(
                    any(p.startswith(name) for p in expected_names),
                    f"orphan ZIP directory: {name}",
                )
            _require(len(files) == 87, "R2 ZIP must have 86 files plus MANIFEST")
            for filename, info in sorted(files.items()):
                expected_size, expected_sha = (
                    (manifest_size, manifest_sha256)
                    if filename == "MANIFEST.json"
                    else listed_files[filename]
                )
                _require(
                    info.file_size == expected_size,
                    f"ZIP member length mismatch: {filename}",
                )
                digest = hashlib.sha256()
                count = 0
                with archive.open(info, "r") as source:
                    for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                        digest.update(chunk)
                        count += len(chunk)
                        _require(
                            count <= expected_size,
                            f"member decompressed size overflow: {filename}",
                        )
                _require(count == expected_size, f"read size mismatch: {filename}")
                _require(
                    digest.hexdigest() == expected_sha,
                    f"member SHA256 mismatch: {filename}",
                )
            with archive.open("MANIFEST.json") as source:
                _require(
                    source.read() == manifest_bytes,
                    "embedded and external manifest bytes differ",
                )
    except (zipfile.BadZipFile, RuntimeError, EOFError, OSError) as exc:
        raise ArchiveAuditError(f"ZIP stream failed strict readback: {exc}") from exc

    return {
        "schema": "issue810_r2_archived_bytes_audit_v1",
        "state": "ZIP_AND_ALL87_MEMBERS_BYTE_VERIFIED",
        "release_tag": "research/issue810-development-checkpoint-20261015",
        "zip_bytes": observed_zip_bytes,
        "zip_sha256": observed_zip_sha,
        "manifest_bytes": observed_manifest_bytes,
        "manifest_sha256": observed_manifest_sha,
        "member_count": len(files),
        "verified_members": {
            name: {
                "bytes": (
                    manifest_size if name == "MANIFEST.json" else listed_files[name][0]
                ),
                "sha256": (
                    manifest_sha256
                    if name == "MANIFEST.json"
                    else listed_files[name][1]
                ),
            }
            for name in sorted(files)
        },
        "declared_source_count": len(R2_SOURCE_FILES),
        "r2_root_member_count": len(R2_ROOT_FILES),
        "market_result": manifest["market_result"],
        "source_head_declared": manifest["source_head"],
        "source_head_reproducibility": "NOT_ESTABLISHED",
        "original_48_role_fit_receipt_binding": "NOT_ESTABLISHED",
        "economic_authority": "NOT_GRANTED",
        "unused_or_live_authority": "NOT_GRANTED",
        "note": (
            "This is byte-level archival verification only. It does not "
            "authenticate old execution semantics, completed per-arm receipts, "
            "an independent reference account, or economic-run authorization."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = audit_archive(args.archive, args.manifest)
        with args.report.open("x", encoding="utf-8") as output:
            output.write(
                json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            )
    except (ArchiveAuditError, OSError) as exc:
        parser.exit(2, f"R2 ARCHIVE NOT VERIFIED: {exc}\n")
    print("R2 ARCHIVE BYTE VERIFIED ONLY: 87 members, no economic authority")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
