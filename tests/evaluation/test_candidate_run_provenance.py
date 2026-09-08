from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.evaluation.runs import provenance as provenance_module
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance


def _fake_package(root: Path, *, payload: str = "VALUE = 1\n") -> Path:
    package = root / "trade_rl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "module.py").write_text(payload, encoding="utf-8")
    (package / "ignored.txt").write_text("not-python\n", encoding="utf-8")
    return package


def test_candidate_run_provenance_is_context_bound_and_digest_complete() -> None:
    payload = build_candidate_run_provenance(research_context_digest="a" * 64)

    assert payload["schema_version"] == "candidate_run_provenance_v1"
    assert payload["research_context_digest"] == "a" * 64
    assert isinstance(payload["implementation"], dict)
    assert isinstance(payload["runtime_environment"], dict)
    assert len(str(payload["implementation_digest"])) == 64
    assert len(str(payload["runtime_environment_digest"])) == 64


def test_candidate_run_provenance_allows_standalone_unbound_run() -> None:
    payload = build_candidate_run_provenance()
    assert payload["research_context_digest"] is None


def test_candidate_run_provenance_rejects_invalid_context_digest() -> None:
    with pytest.raises(ValueError, match="research_context_digest"):
        build_candidate_run_provenance(research_context_digest="not-a-digest")


def test_implementation_manifest_changes_with_source_bytes_not_absolute_root(
    tmp_path: Path,
) -> None:
    first = _fake_package(tmp_path / "one")
    second = _fake_package(tmp_path / "two")

    first_manifest = provenance_module._implementation_manifest(first)
    second_manifest = provenance_module._implementation_manifest(second)
    assert first_manifest == second_manifest
    first_digest = provenance_module._implementation_digest(first_manifest)
    assert first_digest == provenance_module._implementation_digest(second_manifest)

    (second / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed_manifest = provenance_module._implementation_manifest(second)
    assert changed_manifest != first_manifest
    assert provenance_module._implementation_digest(changed_manifest) != first_digest


def test_implementation_manifest_hashes_only_relative_python_sources(tmp_path: Path) -> None:
    package = _fake_package(tmp_path)
    manifest = provenance_module._implementation_manifest(package)

    assert tuple(manifest) == ("__init__.py", "module.py")
    assert all(len(digest) == 64 for digest in manifest.values())
    assert not any(str(tmp_path) in key for key in manifest)


def test_runtime_manifest_has_fixed_dependency_roster(monkeypatch) -> None:
    real_version = provenance_module.metadata.version

    def fake_version(name: str) -> str:
        if name == "lightgbm":
            raise provenance_module.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(provenance_module.metadata, "version", fake_version)
    manifest = provenance_module.runtime_environment_manifest()

    assert set(manifest["distributions"]) == {
        "trade-rl",
        "numpy",
        "gymnasium",
        "lightgbm",
        "stable-baselines3",
        "torch",
    }
    assert manifest["distributions"]["lightgbm"] is None
    assert isinstance(manifest["python"], dict)
    assert isinstance(manifest["platform"], dict)
