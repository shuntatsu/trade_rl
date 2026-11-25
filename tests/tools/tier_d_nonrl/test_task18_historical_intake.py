"""Result-blind historical archive admission falsifiers (no market values)."""

from __future__ import annotations

import io
import stat
import subprocess
import warnings
import zipfile
from dataclasses import replace

import pytest

from tools.tier_d_nonrl import task18_historical_intake as intake
from tools.tier_d_nonrl.task18_recipe import SourceFilePin, build_task18_protocol
from trade_rl.artifacts.canonical import canonical_json_bytes


def _pin(raw):
    return intake._Pin(intake._sha(raw), len(raw))


def _archive(members):
    data = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members:
                archive.writestr(name, raw)
    return zipfile.ZipFile(io.BytesIO(data.getvalue()))


def test_member_verification_treats_forecast_and_nested_compression_as_opaque(
    monkeypatch,
):
    # Deliberately invalid payloads: archive verification must hash them, not
    # parse, decompress nested formats, import or run historical Python.
    files = {
        "attempt/forecasts/x.json": b"not JSON",
        "attempt/native-plan.json.gz": b"not gzip",
        "fixture/arrays.npz": b"not NPZ",
        "source.py": b"raise AssertionError('must never execute')",
    }
    raw = b"pinned manifest"
    pins = {name: _pin(body) for name, body in files.items()}
    with _archive([*files.items(), ("MANIFEST.json", raw)]) as archive:
        monkeypatch.setattr(archive, "extract", lambda *a, **k: pytest.fail("extract"))
        monkeypatch.setattr(
            archive, "extractall", lambda *a, **k: pytest.fail("extractall")
        )
        monkeypatch.setattr(intake, "_json", lambda *a: pytest.fail("opaque decode"))
        assert intake._verify_members(archive, raw, pins) == sum(
            len(v) for v in files.values()
        ) + len(raw)


@pytest.mark.parametrize(
    "change",
    (
        "missing",
        "foreign",
        "duplicate",
        "wrong_size",
        "wrong_hash",
        "symlink",
        "nested_manifest",
    ),
)
def test_member_verifier_refuses_changed_roster_and_bytes(change):
    raw = b"pinned manifest"
    items = [("a", b"abc"), ("MANIFEST.json", raw)]
    pins = {"a": _pin(b"abc")}
    if change == "missing":
        items.pop(0)
    elif change == "foreign":
        items.append(("foreign", b"x"))
    elif change == "duplicate":
        items.insert(0, ("a", b"abc"))
    elif change == "wrong_size":
        items[0] = ("a", b"ab")
    elif change == "wrong_hash":
        items[0] = ("a", b"abd")
    elif change == "symlink":
        entry = zipfile.ZipInfo("a")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        items[0] = (entry, b"abc")
    else:
        items[1] = ("MANIFEST.json", b"x" * len(raw))
    with _archive(items) as archive:
        with pytest.raises(ValueError):
            intake._verify_members(archive, raw, pins)


@pytest.mark.parametrize(
    "name", ("../x", "/x", "a/../b", "a//b", "a\\b", "C:/x", "a/./b")
)
def test_pin_paths_reject_unsafe_names(name):
    with pytest.raises(ValueError):
        intake._member_pins({name: {"bytes": 1, "sha256": "a" * 64}})


def test_manifest_alias_and_aggregate_size_are_refused_before_inflation(monkeypatch):
    with pytest.raises(ValueError, match="case-aliased"):
        intake._member_pins(
            {
                "a": {"bytes": 1, "sha256": "a" * 64},
                "A": {"bytes": 1, "sha256": "b" * 64},
            }
        )
    raw = b"manifest"
    pins = {"manifest.json": _pin(b"x")}
    with _archive([("manifest.json", b"x"), ("MANIFEST.json", raw)]) as archive:
        with pytest.raises(ValueError, match="pin itself"):
            intake._verify_members(archive, raw, pins)
    with _archive([("a", b"abc"), ("MANIFEST.json", raw)]) as archive:
        monkeypatch.setattr(intake, "_MAX_UNCOMPRESSED", len(raw) + 2)
        monkeypatch.setattr(
            archive, "open", lambda *a, **k: pytest.fail("expanded oversized member")
        )
        with pytest.raises(ValueError, match="aggregate"):
            intake._verify_members(archive, raw, {"a": _pin(b"abc")})


@pytest.mark.parametrize(
    "raw", (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b"[]", b"\xff")
)
def test_metadata_json_refuses_ambiguity(raw):
    with pytest.raises(ValueError):
        intake._json(raw)


def test_metadata_bound_applies_before_read(monkeypatch):
    with _archive([("attempt.json", b"123456789")]) as archive:
        monkeypatch.setattr(intake, "_MAX_METADATA", 8)
        monkeypatch.setattr(
            archive, "read", lambda *a, **k: pytest.fail("read oversized metadata")
        )
        with pytest.raises(ValueError, match="metadata exceeds"):
            intake._read_metadata(archive, "attempt.json")


def test_nested_manifest_requires_full_subtree_not_just_referenced_files():
    prefix = "attempt/"
    body = b"opaque"
    payload = canonical_json_bytes(
        {"native-plan.json.gz": {"bytes": len(body), "sha256": intake._sha(body)}}
    )
    pins = {
        prefix + "native-plan.json.gz": _pin(body),
        prefix + "output-manifest.json": _pin(payload),
    }
    with _archive([(prefix + "output-manifest.json", payload)]) as archive:
        assert intake._nested_manifest(archive, pins, prefix) == {
            "native-plan.json.gz": _pin(body)
        }
        with pytest.raises(ValueError, match="roster"):
            intake._nested_manifest(
                archive, {**pins, prefix + "foreign": _pin(b"x")}, prefix
            )


def _attempt(recipe):
    return {
        "scope": "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY",
        "status": "FAILED",
        "arms": [
            {
                "account": account,
                "role": role,
                "scenario": scenario,
                "state": "INTERRUPTED" if i == 0 else "UNATTEMPTED",
                "receipt": None,
            }
            for i, (account, role, scenario) in enumerate(
                intake._expected_roster(recipe)
            )
        ],
    }


@pytest.mark.parametrize(
    "change",
    (
        "reorder",
        "duplicate",
        "completed",
        "receipt",
        "missing",
        "wrong_status",
        "wrong_scope",
    ),
)
def test_stopped_attempt_does_not_admit_cross_scope_or_manufactured_completion(change):
    recipe = build_task18_protocol()
    data = _attempt(recipe)
    assert intake._stopped_attempt(data, recipe, interrupted=1)["state_counts"] == {
        "INTERRUPTED": 1,
        "UNATTEMPTED": 47,
    }
    if change == "reorder":
        data["arms"][0], data["arms"][1] = data["arms"][1], data["arms"][0]
    elif change == "duplicate":
        data["arms"][1] = data["arms"][0].copy()
    elif change == "completed":
        data["arms"][0]["state"] = "COMPLETED"
    elif change == "receipt":
        data["arms"][0]["receipt"] = "foreign"
    elif change == "missing":
        data["arms"].pop()
    elif change == "wrong_status":
        data["status"] = "COMPLETED"
    else:
        data["scope"] = "SYNTHETIC_ONLY"
    with pytest.raises(ValueError):
        intake._stopped_attempt(data, recipe, interrupted=1)


def test_public_intake_refuses_before_decoding_wrong_asset(tmp_path, monkeypatch):
    (tmp_path / intake._ARCHIVE_NAME).write_bytes(b"not the pinned release")
    monkeypatch.setattr(
        intake, "_json", lambda *a: pytest.fail("decode before asset pins")
    )
    with pytest.raises(ValueError, match="size mismatch"):
        intake.inspect_historical_checkpoint(tmp_path, repo=tmp_path)


def test_reference_source_audit_ignores_local_git_replace_refs(tmp_path):
    def git(*args):
        return (
            subprocess.check_output(["git", "-C", str(tmp_path), *args])
            .decode()
            .strip()
        )

    git("init", "-q")
    path = tmp_path / "runner.py"
    path.write_bytes(b"original = True\\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "original",
    )
    first = git("rev-parse", "HEAD")
    original = path.read_bytes()
    path.write_bytes(b"original = False\\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "replacement",
    )
    second = git("rev-parse", "HEAD")
    git("replace", first, second)
    recipe = build_task18_protocol()
    ref = replace(
        recipe.historical_reference,
        source_commit=first,
        source_file_pins=(SourceFilePin("runner.py", intake._sha(original)),),
    )
    assert (
        intake._audit_reference_source(
            tmp_path, replace(recipe, historical_reference=ref)
        )
        == 1
    )


def test_explicit_dataset_verification_is_opaque_and_requires_both_exact_pins(tmp_path):
    recipe = build_task18_protocol()
    manifest = b"not JSON: intentionally opaque"
    arrays = b"not NPZ: intentionally opaque"
    (tmp_path / "manifest.json").write_bytes(manifest)
    (tmp_path / "arrays.npz").write_bytes(arrays)
    ref = replace(
        recipe.historical_reference,
        manifest_sha256=intake._sha(manifest),
        arrays_sha256=intake._sha(arrays),
    )
    artificial = replace(recipe, historical_reference=ref)
    assert intake._verify_dataset_bytes(tmp_path, artificial) == {
        "manifest.json": len(manifest),
        "arrays.npz": len(arrays),
    }
    (tmp_path / "arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        intake._verify_dataset_bytes(tmp_path, artificial)


def test_opaque_dataset_read_is_bounded(tmp_path):
    target = tmp_path / "arrays.npz"
    target.write_bytes(b"abc")
    with pytest.raises(ValueError, match="byte limit"):
        intake._verify_opaque_file(target, intake._sha(b"abc"), limit=2)


@pytest.mark.parametrize(
    "change",
    (
        "wrong_dataset",
        "wrong_arrays",
        "wrong_dimensions",
        "wrong_budget",
        "wrong_source",
        "wrong_schema",
    ),
)
def test_protocol_resealed_metadata_must_match_frozen_dataset_and_recipe(change):
    recipe = build_task18_protocol()
    ref = recipe.historical_reference
    protocol = {
        "source_head": intake._RELEASE_SOURCE,
        "schema": "issue810_tier_d_nonrl48_result_blind_v1",
        "scope": "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY",
        "dataset": {
            "dataset_id": ref.dataset_id,
            "input_sha256": {
                "manifest.json": ref.manifest_sha256,
                "arrays.npz": ref.arrays_sha256,
            },
            "schema": "market_dataset_artifact_v4",
            "identity": "market_dataset_identity_v7",
            "build": "market_build_v3",
            "dimensions": [9504, 2, 6],
            "feature_names": list(recipe.dataset.feature_names),
        },
        "budget": {
            "advances_per_arm": 6599,
            "native_arms": 48,
            "matched_pairs": 24,
            "ridge_vintages": 12,
            "public_prepare_fits": 4,
        },
    }
    intake._check_protocol(
        protocol, recipe, intake._sha(canonical_json_bytes(protocol))
    )
    if change == "wrong_dataset":
        protocol["dataset"]["dataset_id"] = "0" * 64
    elif change == "wrong_arrays":
        protocol["dataset"]["input_sha256"]["arrays.npz"] = "0" * 64
    elif change == "wrong_dimensions":
        protocol["dataset"]["dimensions"] = [9503, 2, 6]
    elif change == "wrong_budget":
        protocol["budget"]["native_arms"] = 47
    elif change == "wrong_source":
        protocol["source_head"] = "0" * 40
    else:
        protocol["dataset"]["schema"] = "market_dataset_artifact_v5"
    with pytest.raises(ValueError, match="binding differs"):
        intake._check_protocol(
            protocol, recipe, intake._sha(canonical_json_bytes(protocol))
        )
