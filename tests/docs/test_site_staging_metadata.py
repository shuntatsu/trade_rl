from __future__ import annotations

from pathlib import Path

import yaml


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _front_matter(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---"
    end = lines.index("---", 1)
    payload = yaml.safe_load("\n".join(lines[1:end]))
    assert isinstance(payload, dict)
    return payload


def test_site_staging_derives_tags_from_governance_topics(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / ".meta.yml",
        "lifecycle: current\nauthority: supporting\ntopics: [documentation]\n",
    )
    _write(
        tmp_path / "docs" / "index.md",
        "---\ntitle: Docs\ndoc_type: index\ntopics: [documentation, navigation]\n---\n"
        "# Docs\n\n[Reference](reference/index.md)\n",
    )
    _write(
        tmp_path / "docs" / "reference" / ".meta.yml",
        "doc_type: reference\n",
    )
    _write(
        tmp_path / "docs" / "reference" / "index.md",
        "---\ntitle: Reference\ndoc_type: index\ntopics: [reference]\n---\n"
        "# Reference\n\n[Page](page.md)\n",
    )
    _write(
        tmp_path / "docs" / "reference" / "page.md",
        "---\ntitle: Page\ntopics: [architecture, identity]\n---\n# Page\n",
    )

    from scripts.docs.generate import write_build_outputs

    output = tmp_path / ".docs-build"
    write_build_outputs(tmp_path, output)
    staged = _front_matter(output / "site-src" / "reference" / "page.md")
    assert staged["tags"] == ["architecture", "identity"]
    assert "doc_type" not in staged
    assert "authority" not in staged
    assert "source_of_truth_for" not in staged
