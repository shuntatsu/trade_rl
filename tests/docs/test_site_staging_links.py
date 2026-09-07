from __future__ import annotations

from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_site_staging_rewrites_repository_links_outside_docs_to_github(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "AGENTS.md", "# Agent instructions\n")
    _write(tmp_path / "examples" / "demo.json", "{}\n")
    _write(
        tmp_path / "docs" / ".meta.yml",
        "lifecycle: current\nauthority: supporting\ntopics: [documentation]\n",
    )
    _write(
        tmp_path / "docs" / "index.md",
        "---\ntitle: Docs\ndoc_type: index\ntopics: [documentation]\n---\n"
        "# Docs\n\n[Reference](reference/index.md)\n[Agents](../AGENTS.md)\n",
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
        "---\ntitle: Page\ntopics: [reference]\n---\n"
        "# Page\n\n[Demo](../../examples/demo.json)\n",
    )

    from scripts.docs.generate import write_build_outputs

    output = tmp_path / ".docs-build"
    write_build_outputs(tmp_path, output)

    staged_index = (output / "site-src" / "index.md").read_text(encoding="utf-8")
    staged_page = (output / "site-src" / "reference" / "page.md").read_text(
        encoding="utf-8"
    )
    assert (
        "https://github.com/shuntatsu/trade_rl/blob/main/AGENTS.md" in staged_index
    )
    assert (
        "https://github.com/shuntatsu/trade_rl/blob/main/examples/demo.json"
        in staged_page
    )
