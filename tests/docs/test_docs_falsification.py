from __future__ import annotations

from pathlib import Path

from scripts.docs._core import validate_repository


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _minimal_repo(tmp_path: Path) -> Path:
    _write(tmp_path / "trade_rl" / "rl" / "rewards.py", "VALUE = 1\n")
    _write(
        tmp_path / "docs" / ".meta.yml",
        """schema_version: trade_rl_docs_metadata_v1
pages:
  A.md:
    title: A
    doc_type: reference
    lifecycle: current
    authority: canonical
    topics: [a]
    source_of_truth_for: [a-contract]
    related_code: [trade_rl/rl/rewards.py]
    site: {section: Reference, order: 10}
""",
    )
    _write(tmp_path / "docs" / "A.md", "# A\n")
    _write(
        tmp_path / "docs" / "history" / ".meta.yml",
        """schema_version: trade_rl_docs_metadata_v1
defaults:
  doc_type: history
  lifecycle: historical
  authority: none
""",
    )
    _write(tmp_path / "docs" / "history" / "old.md", "# Old\n")
    return tmp_path


def test_duplicate_canonical_contract_key_fails_closed(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    meta = root / "docs" / ".meta.yml"
    meta.write_text(
        meta.read_text(encoding="utf-8")
        + """  B.md:
    title: B
    doc_type: reference
    lifecycle: current
    authority: canonical
    topics: [b]
    source_of_truth_for: [a-contract]
    related_code: [trade_rl/rl/rewards.py]
    site: {section: Reference, order: 20}
""",
        encoding="utf-8",
    )
    _write(root / "docs" / "B.md", "# B\n")

    errors = validate_repository(root)
    assert any("duplicate canonical source_of_truth_for" in error for error in errors)


def test_unregistered_current_markdown_is_rejected(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    _write(root / "docs" / "UNREGISTERED.md", "# Unregistered\n")

    errors = validate_repository(root)
    assert any("missing effective metadata" in error for error in errors)


def test_history_cannot_escalate_to_current_authority(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    _write(
        root / "docs" / "history" / "old.md",
        """---
lifecycle: current
authority: canonical
source_of_truth_for: [stolen-contract]
---
# Old
""",
    )

    errors = validate_repository(root)
    assert any("history must remain historical and non-authoritative" in error for error in errors)


def test_related_code_pattern_must_resolve(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    meta = root / "docs" / ".meta.yml"
    meta.write_text(
        meta.read_text(encoding="utf-8").replace(
            "trade_rl/rl/rewards.py", "trade_rl/does-not-exist/**/*.py"
        ),
        encoding="utf-8",
    )

    errors = validate_repository(root)
    assert any("related_code pattern matched nothing" in error for error in errors)


def test_operations_directory_rejects_reference_page(tmp_path: Path) -> None:
    root = _minimal_repo(tmp_path)
    _write(
        root / "docs" / "operations" / ".meta.yml",
        """schema_version: trade_rl_docs_metadata_v1
pages:
  bad.md:
    title: Bad
    doc_type: reference
    lifecycle: current
    authority: supporting
    topics: [ops]
    site: {section: Operations, order: 10}
""",
    )
    _write(root / "docs" / "operations" / "bad.md", "# Bad\n")

    errors = validate_repository(root)
    assert any("docs/operations/ only accepts runbook/index" in error for error in errors)
