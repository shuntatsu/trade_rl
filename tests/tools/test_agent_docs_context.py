"""Network-free oracles for section routing; no Claude host or market data."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.agent_docs_context import SECTION_GROUPS, render, split_sections

ROOT = Path(__file__).resolve().parents[2]


def _fixture() -> str:
    headings = dict.fromkeys(
        name for group in SECTION_GROUPS.values() for name in group
    )
    return "# Canonical instructions\n\n" + "".join(
        f"## {name}\nmarker for {name}\n\n" for name in headings
    )


def test_single_topic_keeps_original_section_order_and_text() -> None:
    original = _fixture()
    preamble, sections = split_sections(original)
    selected = set(SECTION_GROUPS["core"] + SECTION_GROUPS["guide"])
    expected = preamble + "".join(
        chunk for heading, chunk in sections.items() if heading in selected
    )
    assert render(original, ("guide", "core")) == expected
    assert render(original, ("all",)) == original
    assert len(render(original, ("core",))) < len(original)


def test_unknown_or_missing_section_fails_closed() -> None:
    original = _fixture()
    with pytest.raises(ValueError, match="unmapped sections"):
        split_sections(original + "## New permanent rule\nDo not drop me.\n")
    with pytest.raises(ValueError, match="missing sections"):
        split_sections(
            original.replace("## Agent Eval protocol\n", "### Agent Eval protocol\n")
        )
    with pytest.raises(ValueError, match="duplicate section"):
        split_sections(original + "## 目的\nDuplicate.\n")


def test_fenced_code_heading_is_not_a_routing_boundary() -> None:
    original = _fixture().replace(
        "marker for 目的",
        "marker for 目的\n```bash\n## Not a heading\n```",
    )
    assert split_sections(original)
    assert "## Not a heading" in render(original, ("core",))


def test_rejects_accidental_all_plus_topic() -> None:
    with pytest.raises(ValueError, match="alone"):
        render(_fixture(), ("all", "core"))


def test_actual_contract_is_covered_and_core_is_reduced() -> None:
    source = (ROOT / "docs" / "AGENTS.md").read_text(encoding="utf-8")
    assert render(source, ("all",)) == source
    core = render(source, ("core",))
    assert len(core) < len(source)
    assert "tested PR head contains current `main`" in core
    assert "source-packet" in render(source, ("tooling",))
    assert "G0" in core


def test_rejects_unclosed_fence_masquerading_as_sections() -> None:
    original = _fixture()
    with pytest.raises(ValueError, match="unclosed"):
        split_sections(original.replace("## 目的\n", "## 目的\n```\n"))


def test_specialist_topic_always_includes_common_safety_rules() -> None:
    original = _fixture()
    specialist = render(original, ("guide",))
    assert specialist == render(original, ("core", "guide"))
    assert "## Git / PR integration safety" in specialist
    assert "## Research Assurance Gate (#687)" in specialist


def test_indented_open_fence_hides_inside_atx_heading() -> None:
    source = _fixture().replace(
        "marker for 目的",
        "marker for 目的\n   ```markdown\n## Not a real section\n   ```",
    )
    assert "## Not a real section" in render(source, ("core",))


def test_indented_close_fence_reveals_following_sections() -> None:
    source = _fixture().replace(
        "marker for 目的",
        "marker for 目的\n```markdown\ntext\n   ```",
    )
    assert render(source, ("all",)) == source


def test_indented_and_tabbed_atx_heading_are_valid() -> None:
    source = (
        _fixture()
        .replace("## 目的\n", "  ##\t目的\n")
        .replace("## 読む順序\n", "   ## 読む順序\n")
    )
    assert render(source, ("all",)) == source


def test_terminal_unclosed_fence_cannot_hide_new_section() -> None:
    source = _fixture() + "\n```markdown\n## New mandatory rule\n"
    with pytest.raises(ValueError, match="unclosed"):
        split_sections(source)


def test_terminal_closed_fence_is_still_allowed() -> None:
    source = _fixture() + "\n```markdown\n## Not a section\n```\n"
    assert render(source, ("all",)) == source


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (_fixture() + "## UNMAPPED mandatory rule\nnew policy\n", "unmapped sections"),
        (_fixture() + "\n```python\n## UNMAPPED\n", "unclosed"),
        (
            _fixture().replace(
                "## Agent Eval protocol\n",
                "### Agent Eval protocol\n",
            ),
            "missing sections",
        ),
    ],
)
def test_all_route_fails_closed_for_bad_sections(source: str, error: str) -> None:
    """The full-text mode must not bypass canonical section validation."""
    with pytest.raises(ValueError, match=error):
        render(source, ("all",))
