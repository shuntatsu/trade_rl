from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _require_tooling() -> None:
    required = (
        ROOT / "scripts" / "docs" / "metadata.py",
        ROOT / "scripts" / "docs" / "manifest.py",
        ROOT / "scripts" / "docs" / "generate.py",
        ROOT / "scripts" / "docs" / "validate.py",
        ROOT / "scripts" / "docs" / "context.py",
    )
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    assert missing == []


def _build_minimal_repo(root: Path) -> None:
    _write(
        root / "docs" / ".meta.yml",
        "lifecycle: current\nauthority: supporting\n",
    )
    _write(
        root / "docs" / "reference" / ".meta.yml",
        "doc_type: reference\n",
    )
    _write(
        root / "docs" / "history" / ".meta.yml",
        "doc_type: history\nlifecycle: historical\nauthority: none\n",
    )
    _write(
        root / "docs" / "index.md",
        "---\ntitle: Docs\ndoc_type: index\ntopics: [documentation]\n---\n"
        "# Docs\n\n[Reference](reference/index.md)\n",
    )
    _write(
        root / "docs" / "reference" / "index.md",
        "---\ntitle: Reference\ndoc_type: index\ntopics: [reference]\n---\n"
        "# Reference\n\n[Architecture](architecture.md)\n",
    )
    _write(
        root / "docs" / "reference" / "architecture.md",
        "---\n"
        "title: Architecture\n"
        "authority: canonical\n"
        "topics: [architecture, identity]\n"
        "source_of_truth_for: [runtime-architecture]\n"
        "related_code: [trade_rl/rl/]\n"
        "agent_read_when: [changing runtime architecture]\n"
        "---\n"
        "# Architecture\n",
    )
    _write(
        root / "docs" / "history" / "old-design.md",
        "# Old design\n\n[stale](../../removed/path.md)\n",
    )
    (root / "trade_rl" / "rl").mkdir(parents=True)


def test_effective_metadata_merges_folder_defaults_and_front_matter(
    tmp_path: Path,
) -> None:
    _require_tooling()
    _build_minimal_repo(tmp_path)

    from scripts.docs.metadata import load_effective_metadata

    document = load_effective_metadata(
        tmp_path / "docs" / "reference" / "architecture.md",
        tmp_path / "docs",
    )

    assert document.title == "Architecture"
    assert document.doc_type == "reference"
    assert document.lifecycle == "current"
    assert document.authority == "canonical"
    assert document.topics == ("architecture", "identity")
    assert document.source_of_truth_for == ("runtime-architecture",)
    assert document.related_code == ("trade_rl/rl/",)


def test_validation_rejects_duplicate_canonical_ownership(tmp_path: Path) -> None:
    _require_tooling()
    _build_minimal_repo(tmp_path)
    _write(
        tmp_path / "docs" / "reference" / "duplicate.md",
        "---\n"
        "title: Duplicate\n"
        "authority: canonical\n"
        "topics: [architecture]\n"
        "source_of_truth_for: [runtime-architecture]\n"
        "---\n"
        "# Duplicate\n",
    )
    index = tmp_path / "docs" / "reference" / "index.md"
    index.write_text(
        index.read_text(encoding="utf-8") + "[Duplicate](duplicate.md)\n",
        encoding="utf-8",
    )

    from scripts.docs.validate import validate_documents

    errors = validate_documents(tmp_path)
    assert any("duplicate canonical ownership" in error for error in errors)
    assert any("runtime-architecture" in error for error in errors)


def test_validation_rejects_folder_type_mismatch(tmp_path: Path) -> None:
    _require_tooling()
    _build_minimal_repo(tmp_path)
    path = tmp_path / "docs" / "reference" / "architecture.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "title: Architecture\n",
            "title: Architecture\ndoc_type: runbook\n",
        ),
        encoding="utf-8",
    )

    from scripts.docs.validate import validate_documents

    errors = validate_documents(tmp_path)
    assert any("folder/type mismatch" in error for error in errors)


def test_validation_rejects_orphan_current_document(tmp_path: Path) -> None:
    _require_tooling()
    _build_minimal_repo(tmp_path)
    _write(
        tmp_path / "docs" / "reference" / "orphan.md",
        "---\ntitle: Orphan\ntopics: [orphan]\n---\n# Orphan\n",
    )

    from scripts.docs.validate import validate_documents

    errors = validate_documents(tmp_path)
    assert any("orphan current document" in error for error in errors)
    assert any("reference/orphan.md" in error for error in errors)


def test_validation_checks_current_links_but_not_historical_stale_links(
    tmp_path: Path,
) -> None:
    _require_tooling()
    _build_minimal_repo(tmp_path)

    from scripts.docs.validate import validate_documents

    assert validate_documents(tmp_path) == ()

    current = tmp_path / "docs" / "reference" / "architecture.md"
    current.write_text(
        current.read_text(encoding="utf-8") + "\n[broken](missing.md)\n",
        encoding="utf-8",
    )
    errors = validate_documents(tmp_path)
    assert any("broken current link" in error for error in errors)


def test_context_routing_prefers_current_canonical_and_excludes_history(
    tmp_path: Path,
) -> None:
    _require_tooling()
    _build_minimal_repo(tmp_path)
    _write(
        tmp_path / "docs" / "history" / "legacy-routing.md",
        "---\ntitle: Legacy routing\ntopics: [architecture]\n"
        "related_code: [trade_rl/rl/]\n---\n# Legacy routing\n",
    )

    from scripts.docs.context import documents_for_path, documents_for_topic

    by_path = documents_for_path(tmp_path, "trade_rl/rl/reward.py")
    assert [document.path.as_posix() for document in by_path] == [
        "reference/architecture.md"
    ]

    by_topic = documents_for_topic(tmp_path, "architecture")
    assert by_topic[0].path.as_posix() == "reference/architecture.md"
    assert all(document.lifecycle == "current" for document in by_topic)

    with_history = documents_for_topic(tmp_path, "architecture", include_history=True)
    assert any(document.lifecycle == "historical" for document in with_history)


def test_manifest_is_deterministic_and_has_unique_authority_map(tmp_path: Path) -> None:
    _require_tooling()
    _build_minimal_repo(tmp_path)

    from scripts.docs.manifest import build_manifest

    first = build_manifest(tmp_path)
    second = build_manifest(tmp_path)
    assert first == second
    assert "generated_at" not in first
    assert first["canonical_owners"] == {
        "runtime-architecture": "reference/architecture.md"
    }
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second,
        ensure_ascii=False,
        sort_keys=True,
    )


def test_site_staging_contains_current_docs_and_history_catalog_not_raw_history(
    tmp_path: Path,
) -> None:
    _require_tooling()
    _build_minimal_repo(tmp_path)

    from scripts.docs.generate import write_build_outputs

    output_root = tmp_path / ".docs-build"
    write_build_outputs(tmp_path, output_root)

    site_src = output_root / "site-src"
    assert (site_src / "reference" / "architecture.md").is_file()
    assert not (site_src / "history" / "old-design.md").exists()
    history_index = (site_src / "history" / "index.md").read_text(encoding="utf-8")
    assert "old-design.md" in history_index
    assert "historical" in history_index.lower()
    assert (output_root / "manifest.json").is_file()


def test_metadata_validation_reports_missing_required_current_fields(
    tmp_path: Path,
) -> None:
    _require_tooling()
    _build_minimal_repo(tmp_path)
    architecture = tmp_path / "docs" / "reference" / "architecture.md"
    architecture.write_text("# no front matter\n", encoding="utf-8")

    from scripts.docs.validate import validate_documents

    errors = validate_documents(tmp_path)
    assert any("missing required metadata" in error for error in errors)


@pytest.mark.parametrize("invalid", ["unknown", "", "CURRENT"])
def test_metadata_validation_rejects_unknown_lifecycle(
    tmp_path: Path,
    invalid: str,
) -> None:
    _require_tooling()
    _build_minimal_repo(tmp_path)
    architecture = tmp_path / "docs" / "reference" / "architecture.md"
    architecture.write_text(
        architecture.read_text(encoding="utf-8").replace(
            "title: Architecture\n",
            f"title: Architecture\nlifecycle: {invalid!r}\n",
        ),
        encoding="utf-8",
    )

    from scripts.docs.validate import validate_documents

    errors = validate_documents(tmp_path)
    assert any("invalid lifecycle" in error for error in errors)
