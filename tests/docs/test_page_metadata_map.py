from __future__ import annotations

from pathlib import Path


def test_page_metadata_map_overrides_folder_defaults_before_front_matter(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    reference = docs / "reference"
    reference.mkdir(parents=True)
    (docs / ".meta.yml").write_text(
        "lifecycle: current\nauthority: supporting\ntopics: [documentation]\n",
        encoding="utf-8",
    )
    (reference / ".meta.yml").write_text(
        "doc_type: reference\n"
        "pages:\n"
        "  architecture.md:\n"
        "    authority: canonical\n"
        "    topics: [architecture, identity]\n"
        "    source_of_truth_for: [runtime-architecture]\n"
        "    related_code: [trade_rl/rl/]\n",
        encoding="utf-8",
    )
    page = reference / "architecture.md"
    page.write_text(
        "---\ntopics: [architecture, runtime]\n---\n# Architecture\n",
        encoding="utf-8",
    )

    from scripts.docs.metadata import load_effective_metadata

    document = load_effective_metadata(page, docs)
    assert document.authority == "canonical"
    assert document.source_of_truth_for == ("runtime-architecture",)
    assert document.related_code == ("trade_rl/rl/",)
    assert document.topics == ("architecture", "runtime")
