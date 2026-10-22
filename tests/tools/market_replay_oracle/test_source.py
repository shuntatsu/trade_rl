from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tools.market_replay_oracle import source
from tools.market_replay_oracle.source import NativeSourcePin, verify_native_source

REPOSITORY = Path(__file__).resolve().parents[3]
PROFILE_SOURCE_HASHES: dict[str, frozenset[str]] = {
    "tools/tier_d_nonrl/tier_d_protocol.py": frozenset(
        {"9d679316cc4ffe614bdd8237271610c3168915f4b66993a16010b2fa812c6641"}
    ),
    "tools/tier_d_nonrl/tier_d_native.py": frozenset(
        {"dcf6dacf2d82ecec6ef957d5df13006fa4df972cb96e3d2d2ae03042afa1658e"}
    ),
    "trade_rl/simulation/execution.py": frozenset(
        {"6ce371ba1ab4d7221c214a0d722062d3d1e7791caf5da71704fecdd06f648d63"}
    ),
    "trade_rl/strategies/rl/allocation_policy.py": frozenset(
        {"627d146cffb240947b5d0322af613c7d7ddcc9f13dc01a322a9c88b328856576"}
    ),
    "trade_rl/strategies/rl/allocation_recipe_v2.py": frozenset(
        {"fee8318929cc9fd9c16f0eaae82caff96d3658a0f75d131ac1792b637311f2da"}
    ),
    "trade_rl/simulation/orders/model.py": frozenset(
        {
            "cb1c2f9e71a8ee556ffb9058f8564109cb7699e102a729b5a73afaa80c2c3a79",
            "07a01341e51d2ac54aa0ea316d59fd6ed58046309b40f73db51390503abe741d",
            "13ccc2feaaffa6225c9a8a1eda6902e130a77cfec5ef0f87383bdd0352ddecf4",
            "7031ddb5de0e0a33495bd97c0c91fb54d86a3ba49706dbe45544f7eea20d841b",
        }
    ),
    "trade_rl/artifacts/hashing.py": frozenset(
        {"cb1593dda363fea3276184a699ce730380646ad764969fec47196161d388b4fa"}
    ),
    "trade_rl/artifacts/canonical.py": frozenset(
        {"c827358950f642f6258b0e00f1c9c16ce430cb493e53f3591ba7eed570952e3f"}
    ),
    "trade_rl/risk/pretrade.py": frozenset(
        {"2124d298b18fe5332fbe0ff07b03d67ece1251029cb18c32a1f67203e0f9cc49"}
    ),
    "trade_rl/strategies/allocation.py": frozenset(
        {"7294e78ba9c3d95501fd5ef9e1c87ac1e2ca95b971c70654e2c0913c4ea9f014"}
    ),
    "trade_rl/strategies/allocation_action.py": frozenset(
        {"07792791cb4bf6e6e0b2a9c375b420f6f547c766b7cbe03315e697da596f3d43"}
    ),
    "trade_rl/evaluation/rl_allocation/env.py": frozenset(
        {"e566ffb69c5268fa4b2e8ea379f48dc4fe0d99308961f02d5a878cc5f24583a1"}
    ),
    "trade_rl/evaluation/rl_allocation/binding.py": frozenset(
        {"9750be33cda4a6622ee60fb8c3f01b67978e69f333186bc30017ecec9a0ba791"}
    ),
    "trade_rl/evaluation/objectives/clock.py": frozenset(
        {"d4f8ff01593779c04a477b628048113bc70b44c0a3819b748b8749766a846a3e"}
    ),
    "trade_rl/strategies/rl/allocation_observation_v2.py": frozenset(
        {"27e66b0b59ca15606a5ca4a081ea0254a01475afacc7fad9e578948c7a1c79a2"}
    ),
    "trade_rl/data/market_order_rules.py": frozenset(
        {"a13570349844f7a1b838273e340171c04d97b7313a0818236551f49e49023e7c"}
    ),
}
PROFILE_PATHS = tuple(sorted(PROFILE_SOURCE_HASHES))
SOURCE_ROOTS = ("trade_rl", "tools/tier_d_nonrl")


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def _make_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "source-repository"
    root.mkdir()
    subprocess.run(
        ["git", "init", "-q", str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    _git(root, "config", "user.name", "Source verifier test")
    _git(root, "config", "user.email", "source-verifier@example.invalid")
    _git(root, "config", "core.autocrlf", "false")
    for relative in PROFILE_PATHS:
        destination = root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_git(REPOSITORY, "show", f"HEAD:{relative}"))
    (root / "LICENSE").write_bytes((REPOSITORY / "LICENSE").read_bytes())
    _git(root, "add", "--all")
    _git(root, "commit", "-q", "-m", "source verifier fixture")
    return root


def _snapshot_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for relative in PROFILE_PATHS:
        raw = (root / Path(relative)).read_bytes()
        rows.append(
            {
                "path": relative,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return rows


def _pin(root: Path) -> NativeSourcePin:
    rows = _snapshot_rows(root)
    canonical = json.dumps(
        rows,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    return NativeSourcePin(head, hashlib.sha256(canonical).hexdigest())


def _commit(root: Path, relative: str) -> None:
    _git(root, "add", "--", relative)
    _git(root, "commit", "-q", "-m", "change source fixture")


def test_supported_profile_fingerprints_are_the_closed_public_contract() -> None:
    assert source.SUPPORTED_PROFILE_SOURCES == PROFILE_SOURCE_HASHES


def test_validates_clean_pinned_source_tree_and_returns_raw_and_git_pins(
    tmp_path: Path,
) -> None:
    root = _make_fixture(tmp_path)
    pin = _pin(root)

    verified = verify_native_source(root, pin)

    assert verified["head_sha"] == pin.head_sha
    assert verified["snapshot_sha256"] == pin.snapshot_sha256
    assert verified["scope"] == "STATIC_SOURCE_BYTES_NOT_RUNTIME_ATTESTATION"
    assert (
        verified["profile_source_sha256"]["trade_rl/simulation/orders/model.py"]
        == "7031ddb5de0e0a33495bd97c0c91fb54d86a3ba49706dbe45544f7eea20d841b"
    )
    returned = verified["files"]
    expected = _snapshot_rows(root)
    assert [
        {key: row[key] for key in ("path", "bytes", "sha256")} for row in returned
    ] == expected
    for row in returned:
        blob = _git(root, "show", f"HEAD:{row['path']}")
        assert row["git_blob_sha256"] == hashlib.sha256(blob).hexdigest()
        assert row["normalized_sha256"] == row["git_blob_sha256"]


def test_pin_requires_lowercase_fixed_width_hashes() -> None:
    with pytest.raises(ValueError, match="HEAD|SHA|lowercase"):
        NativeSourcePin("A" * 40, "0" * 64)
    with pytest.raises(ValueError, match="SHA|lowercase"):
        NativeSourcePin("0" * 40, "A" * 64)
    with pytest.raises(ValueError, match="HEAD|SHA"):
        NativeSourcePin("0" * 39, "0" * 64)


def test_rejects_wrong_head_and_snapshot_digest(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    pin = _pin(root)

    with pytest.raises(ValueError, match="HEAD|head"):
        verify_native_source(root, replace(pin, head_sha="0" * 40))
    with pytest.raises(ValueError, match="snapshot|digest|SHA"):
        verify_native_source(root, replace(pin, snapshot_sha256="0" * 64))


def test_crlf_worktree_is_accepted_but_raw_and_git_hashes_stay_distinct(
    tmp_path: Path,
) -> None:
    root = _make_fixture(tmp_path)
    relative = "tools/tier_d_nonrl/tier_d_protocol.py"
    path = root / relative
    git_blob = _git(root, "show", f"HEAD:{relative}")
    crlf = git_blob.replace(b"\n", b"\r\n")
    assert crlf != git_blob
    path.write_bytes(crlf)

    verified = verify_native_source(root, _pin(root))
    row = next(item for item in verified["files"] if item["path"] == relative)

    assert row["sha256"] == hashlib.sha256(crlf).hexdigest()
    assert row["git_blob_sha256"] == hashlib.sha256(git_blob).hexdigest()
    assert row["sha256"] != row["git_blob_sha256"]
    assert row["normalized_sha256"] == row["git_blob_sha256"]
    assert row["crlf_count"] == crlf.count(b"\r\n")
    assert row["lf_count"] == 0


def test_raw_mutation_cannot_be_reauthorized_by_repinning_snapshot(
    tmp_path: Path,
) -> None:
    root = _make_fixture(tmp_path)
    relative = "tools/tier_d_nonrl/tier_d_protocol.py"
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\n# changed after commit\n")

    with pytest.raises(ValueError, match="Git|blob|source|changed"):
        verify_native_source(root, _pin(root))


def test_rejects_oversized_tracked_source_blob(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    relative = "trade_rl/artifacts/canonical.py"
    (root / relative).write_bytes(b"x" * (source.MAX_SOURCE_FILE_BYTES + 1))
    _commit(root, relative)

    with pytest.raises(ValueError, match="exceeds bounded size"):
        verify_native_source(root, _pin(root))


def test_rejects_an_unreviewed_model_source_fingerprint(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    relative = "trade_rl/simulation/orders/model.py"
    path = root / relative
    path.write_bytes(path.read_bytes() + b"\n# unsupported source variant\n")
    _commit(root, relative)

    with pytest.raises(ValueError, match="profile|supported|fingerprint|source"):
        verify_native_source(root, _pin(root))


def test_rejects_missing_tracked_source(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    pin = _pin(root)
    (root / "trade_rl/artifacts/canonical.py").unlink()

    with pytest.raises(ValueError, match="missing|regular|source|file"):
        verify_native_source(root, pin)


def test_rejects_extra_untracked_source_file(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    pin = _pin(root)
    extra = root / "tools/tier_d_nonrl/extra.py"
    extra.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="untracked|extra|source|clean"):
        verify_native_source(root, pin)


def test_rejects_ignored_untracked_source_but_allows_pycache(
    tmp_path: Path,
) -> None:
    root = _make_fixture(tmp_path)
    pin = _pin(root)
    exclude = root / ".git" / "info" / "exclude"
    exclude.write_text("tools/tier_d_nonrl/ignored.py\n", encoding="utf-8")
    (root / "tools/tier_d_nonrl/ignored.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ignored|untracked|source"):
        verify_native_source(root, pin)

    (root / "tools/tier_d_nonrl/ignored.py").unlink()
    exclude.write_text("tools/tier_d_nonrl/__pycache__/\n", encoding="utf-8")
    cache_source = root / "tools/tier_d_nonrl/__pycache__" / "generated.py"
    cache_source.parent.mkdir()
    cache_source.write_text("VALUE = 3\n", encoding="utf-8")
    assert verify_native_source(root, pin)["head_sha"] == pin.head_sha


def test_unrelated_untracked_file_outside_native_namespaces_is_not_part_of_scope(
    tmp_path: Path,
) -> None:
    root = _make_fixture(tmp_path)
    (root / "notes.txt").write_text("outside source namespaces\n", encoding="utf-8")

    assert verify_native_source(root, _pin(root))["scope"] == (
        "STATIC_SOURCE_BYTES_NOT_RUNTIME_ATTESTATION"
    )


def test_rejects_symlinked_source_parent(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    pin = _pin(root)
    source_dir = root / "tools" / "tier_d_nonrl"
    moved_dir = root / "tools" / "tier_d_nonrl-moved"
    source_dir.rename(moved_dir)
    try:
        source_dir.symlink_to(moved_dir, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory symlink creation unavailable: {error}")

    with pytest.raises(ValueError, match="symlink|reparse|link|source"):
        verify_native_source(root, pin)


def test_rejects_symlinked_common_source_parent(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    pin = _pin(root)
    tools_dir = root / "tools"
    moved_dir = root / "tools-moved"
    tools_dir.rename(moved_dir)
    try:
        tools_dir.symlink_to(moved_dir, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory symlink creation unavailable: {error}")

    with pytest.raises(ValueError, match="symlink|reparse|link|source"):
        verify_native_source(root, pin)
