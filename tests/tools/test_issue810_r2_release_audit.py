"""TDD counterexamples for byte-verifying a source-only R2 checkpoint ZIP."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import warnings
import zipfile
from pathlib import Path

import pytest

from tools import issue810_r2_release_audit as release_audit
from tools.issue810_r2_release_audit import (
    R2_HEAD,
    R2_ROOT,
    R2_ROOT_FILES,
    R2_SOURCE_DIR,
    R2_SOURCE_FILES,
    ArchiveAuditError,
    _checked_json,
    audit_archive,
    main,
)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _members() -> dict[str, bytes]:
    members: dict[str, bytes] = {"CHECKPOINT.md": b"synthetic checkpoint, no returns"}
    for name in sorted(R2_SOURCE_FILES):
        members[R2_SOURCE_DIR + name] = ("fake source " + name).encode()
    for name in sorted(R2_ROOT_FILES):
        members[R2_ROOT + name] = ("synthetic frozen metadata " + name).encode()
    for i in range(86 - len(members)):
        members[f"synthetic-control/evidence-{i:03d}.txt"] = str(i).encode()
    assert len(members) == 86
    return members


def _manifest(members: dict[str, bytes]) -> dict[str, object]:
    pins = {name: _sha(members[R2_SOURCE_DIR + name]) for name in R2_SOURCE_FILES}
    return {
        "schema": "issue810_development_checkpoint_v1",
        "scope": "IN_PROGRESS_CONSUMED_DEVELOPMENT_SOURCE_AND_EVIDENCE",
        "tag": "research/issue810-development-checkpoint-20261015",
        "market_result": "NOT_ESTABLISHED",
        "source_head": R2_HEAD,
        "source_sha256": pins,
        "files": {
            name: {"bytes": len(content), "sha256": _sha(content)}
            for name, content in members.items()
        },
    }


def _create(
    tmp_path: Path,
    *,
    members: dict[str, bytes] | None = None,
    manifest: dict[str, object] | None = None,
    extra: tuple[tuple[str, bytes], ...] = (),
    embedded_manifest: bytes | None = None,
    fake_symlink: bool = False,
) -> tuple[Path, Path, dict[str, object]]:
    raw_members = _members() if members is None else members
    body = _manifest(raw_members) if manifest is None else manifest
    manifest_bytes = json.dumps(body, sort_keys=True).encode() + b"\n"
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_bytes(manifest_bytes)
    zip_path = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in raw_members.items():
            archive.writestr(name, content)
        selected_manifest = (
            manifest_bytes if embedded_manifest is None else embedded_manifest
        )
        archive.writestr("MANIFEST.json", selected_manifest)
        for name, content in extra:
            if fake_symlink:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, content)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr(name, content)
    args: dict[str, object] = {
        "archive_sha256": _sha(zip_path.read_bytes()),
        "archive_size": zip_path.stat().st_size,
        "manifest_sha256": _sha(manifest_bytes),
        "manifest_size": len(manifest_bytes),
    }
    return zip_path, manifest_path, args


def _audit(bundle: tuple[Path, Path, dict[str, object]]) -> dict[str, object]:
    archive, manifest, args = bundle
    return audit_archive(archive, manifest, **args)  # type: ignore[arg-type]


def test_zip_path_replacement_cannot_separate_reported_and_verified_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zip_path, manifest_path, args = _create(tmp_path)
    original_bytes = zip_path.read_bytes()
    replacement_path = tmp_path / "repacked.zip"
    with zipfile.ZipFile(zip_path, "r") as original:
        members = [(info.filename, original.read(info)) for info in original.infolist()]
    with zipfile.ZipFile(
        replacement_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as replacement:
        for name, content in members:
            replacement.writestr(name, content)
        replacement.comment = b"different ZIP container bytes"
    replacement_bytes = replacement_path.read_bytes()
    assert replacement_bytes != original_bytes
    with zipfile.ZipFile(replacement_path, "r") as replacement:
        assert {
            info.filename: replacement.read(info) for info in replacement.infolist()
        } == dict(members)

    observed_digests: list[str] = []
    original_zipfile = release_audit.zipfile.ZipFile

    def replace_before_open(
        source: Path | io.BytesIO,
        *open_args: object,
        **open_kwargs: object,
    ) -> zipfile.ZipFile:
        zip_path.write_bytes(replacement_bytes)
        consumed_bytes = (
            source.getvalue() if isinstance(source, io.BytesIO) else source.read_bytes()
        )
        observed_digests.append(_sha(consumed_bytes))
        return original_zipfile(source, *open_args, **open_kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(release_audit.zipfile, "ZipFile", replace_before_open)
    report = audit_archive(zip_path, manifest_path, **args)  # type: ignore[arg-type]

    assert zip_path.read_bytes() == replacement_bytes
    assert observed_digests == [report["zip_sha256"]]
    assert report["zip_sha256"] == _sha(original_bytes)


def test_complete_synthetic_checkpoint_accepts_exact_87_member_bytes(
    tmp_path: Path,
) -> None:
    report = _audit(_create(tmp_path))
    assert report["state"] == "ZIP_AND_ALL87_MEMBERS_BYTE_VERIFIED"
    assert report["member_count"] == 87
    assert len(report["verified_members"]) == 87
    assert report["verified_members"]["MANIFEST.json"]["bytes"] > 100
    assert report["market_result"] == "NOT_ESTABLISHED"
    assert report["source_head_reproducibility"] == "NOT_ESTABLISHED"
    assert report["original_48_role_fit_receipt_binding"] == "NOT_ESTABLISHED"
    assert report["economic_authority"] == "NOT_GRANTED"


def test_archive_release_hash_and_size_fence(tmp_path: Path) -> None:
    zip_path, manifest_path, args = _create(tmp_path)
    with pytest.raises(ArchiveAuditError, match="ZIP SHA256"):
        audit_archive(zip_path, manifest_path, **(args | {"archive_sha256": "0" * 64}))
    with pytest.raises(ArchiveAuditError, match="ZIP byte length"):
        audit_archive(zip_path, manifest_path, **(args | {"archive_size": 3}))
    with pytest.raises(ArchiveAuditError, match="external MANIFEST SHA256"):
        audit_archive(zip_path, manifest_path, **(args | {"manifest_sha256": "0" * 64}))
    with pytest.raises(ArchiveAuditError, match="external MANIFEST byte length"):
        audit_archive(zip_path, manifest_path, **(args | {"manifest_size": 3}))


def test_missing_and_extra_members_are_not_silent(tmp_path: Path) -> None:
    omitted = _members()
    del omitted[R2_ROOT + "attempt.json"]
    bundle = _create(tmp_path, members=omitted)
    with pytest.raises(ArchiveAuditError, match="86 files"):
        _audit(bundle)

    normal = _members()
    # Keep the 86-entry manifest fixed while adding one unknown ZIP member.
    extra = _create(tmp_path, members=normal, extra=(("unexpected.txt", b"evil"),))
    with pytest.raises(ArchiveAuditError, match="ZIP file roster"):
        _audit(extra)


def test_byte_tamper_fails_even_when_zip_hash_is_newly_declared(tmp_path: Path) -> None:
    members = _members()
    expected = _manifest(members)
    original = members[R2_SOURCE_DIR + "tier_d_runner.py"]
    members[R2_SOURCE_DIR + "tier_d_runner.py"] = b"x" * len(original)
    with pytest.raises(ArchiveAuditError, match="member SHA256 mismatch"):
        _audit(_create(tmp_path, members=members, manifest=expected))


def test_source_sha_declaration_mismatch_fails_before_zip_read(tmp_path: Path) -> None:
    members = _members()
    body = _manifest(members)
    body["source_sha256"]["tier_d_runner.py"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ArchiveAuditError, match="source-pins disagree"):
        _audit(_create(tmp_path, members=members, manifest=body))


@pytest.mark.parametrize("bad_path", ["../elsewhere", "C:/evil", "\\\\evil", "a//b"])
def test_unsafe_paths_refused(tmp_path: Path, bad_path: str) -> None:
    with pytest.raises(ArchiveAuditError, match="ZIP .*path"):
        _audit(_create(tmp_path, extra=((bad_path, b"extra"),)))


def test_duplicate_file_names_refused_before_member_read(tmp_path: Path) -> None:
    name = R2_SOURCE_DIR + "tier_d_fixture.py"
    with pytest.raises(ArchiveAuditError, match="duplicate ZIP member path"):
        _audit(_create(tmp_path, extra=((name, b"fake source"),)))


def test_untrusted_symlink_and_manifest_mismatch_refused(tmp_path: Path) -> None:
    with pytest.raises(ArchiveAuditError, match="unsupported ZIP file type"):
        bad_link = _create(
            tmp_path,
            extra=(("synthetic-control/link", b"target"),),
            fake_symlink=True,
        )
        _audit(bad_link)
    with pytest.raises(ArchiveAuditError, match="member (length|SHA256) mismatch"):
        _audit(_create(tmp_path, embedded_manifest=b"{}\n"))


def test_no_implicit_economic_approval_via_result_tag(tmp_path: Path) -> None:
    members = _members()
    metadata = _manifest(members)
    metadata["market_result"] = "WINNER"
    with pytest.raises(ArchiveAuditError, match="reclassified"):
        _audit(_create(tmp_path, members=members, manifest=metadata))


def test_malformed_json_fails_closed() -> None:
    with pytest.raises(ArchiveAuditError, match="duplicate JSON field"):
        _checked_json(b'{"schema":"A","schema":"B"}', name="test")
    with pytest.raises(ArchiveAuditError, match="non-finite JSON"):
        _checked_json(b'{"value":NaN}', name="test")


def test_cli_cannot_override_pinned_release_sha(tmp_path: Path) -> None:
    zip_path, manifest_path, _ = _create(tmp_path)
    report = tmp_path / "fake-approved.json"
    with pytest.raises(SystemExit) as failure:
        main(
            [
                "--archive",
                str(zip_path),
                "--manifest",
                str(manifest_path),
                "--report",
                str(report),
            ]
        )
    assert failure.value.code == 2
    assert not report.exists()


def test_report_output_is_write_once_after_valid_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, manifest, _ = _create(tmp_path)
    monkeypatch.setattr(
        "tools.issue810_r2_release_audit.audit_archive",
        lambda *args: {"state": "synthetic byte verification"},
    )
    report = tmp_path / "owned-report.json"
    report.write_text("DO NOT REPLACE EXISTING EVIDENCE")
    with pytest.raises(SystemExit) as failure:
        main(
            [
                "--archive",
                str(archive),
                "--manifest",
                str(manifest),
                "--report",
                str(report),
            ]
        )
    assert failure.value.code == 2
    assert report.read_text() == "DO NOT REPLACE EXISTING EVIDENCE"
