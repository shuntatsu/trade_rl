from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.runs import provenance as provenance_module
from trade_rl.evaluation.runs.provenance import (
    PROVENANCE_SCHEMA,
    build_candidate_run_provenance,
)


def _fake_package(root: Path, *, payload: bytes = b"VALUE = 1\n") -> Path:
    package = root / "trade_rl"
    (package / "sub").mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"\n")
    (package / "module.py").write_bytes(payload)
    (package / "sub" / "child.py").write_bytes(b"VALUE = 2\n")
    (package / "ignored.txt").write_text("not-python\n", encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "ignored.py").write_text(
        "SHOULD_NOT_BE_INCLUDED = True\n",
        encoding="utf-8",
    )
    return package


def test_candidate_run_provenance_digest_binds_implementation_and_runtime() -> None:
    payload = build_candidate_run_provenance()

    assert payload["schema_version"] == PROVENANCE_SCHEMA
    assert payload["implementation_digest"] == content_digest(payload["implementation"])
    assert payload["runtime_environment_digest"] == content_digest(
        payload["runtime_environment"]
    )
    assert len(str(payload["implementation_digest"])) == 64
    assert len(str(payload["runtime_environment_digest"])) == 64


def test_implementation_manifest_is_path_independent_and_source_byte_sensitive(
    tmp_path: Path,
) -> None:
    first = _fake_package(tmp_path / "one")
    second = _fake_package(tmp_path / "different")

    first_manifest = provenance_module._implementation_manifest(first)
    second_manifest = provenance_module._implementation_manifest(second)

    assert first_manifest == second_manifest
    assert first_manifest["files"] == [
        {
            "path": "__init__.py",
            "sha256": sha256(b"\n").hexdigest(),
        },
        {
            "path": "module.py",
            "sha256": sha256(b"VALUE = 1\n").hexdigest(),
        },
        {
            "path": "sub/child.py",
            "sha256": sha256(b"VALUE = 2\n").hexdigest(),
        },
    ]

    (second / "module.py").write_bytes(b"VALUE = 3\n")
    changed = provenance_module._implementation_manifest(second)
    assert changed != first_manifest
    assert content_digest(changed) != content_digest(first_manifest)


def test_runtime_environment_manifest_has_fixed_dependency_roster(monkeypatch) -> None:
    real_version = provenance_module.metadata.version

    def fake_version(name: str) -> str:
        if name == "lightgbm":
            raise provenance_module.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(provenance_module.metadata, "version", fake_version)
    manifest = provenance_module.runtime_environment_manifest()

    assert set(manifest) == {
        "schema_version",
        "python",
        "os",
        "machine",
        "packages",
    }
    assert set(manifest["packages"]) == {
        "trade-rl",
        "numpy",
        "gymnasium",
        "lightgbm",
        "stable-baselines3",
        "torch",
    }
    assert manifest["packages"]["lightgbm"] is None
    assert set(manifest["python"]) == {"implementation", "version"}
    assert set(manifest["os"]) == {"family", "release"}
