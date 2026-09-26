from __future__ import annotations

import pytest

from tools import ppo_4h_gemini_review as review


@pytest.mark.parametrize("fence", ("```", "~~~"))
def test_markdown_section_ignores_headings_inside_fences(fence: str) -> None:
    text = (
        "## Target\n"
        "before\n"
        f"{fence}text\n"
        "## fake boundary\n"
        f"{fence}\n"
        "after\n"
        "## Next\n"
        "outside\n"
    )

    section = review._markdown_section(text, "## Target")

    assert "## fake boundary" in section
    assert "after" in section
    assert "## Next" not in section


def test_markdown_section_rejects_duplicate_real_heading() -> None:
    text = "## Target\nfirst\n## Next\n## Target\nsecond\n"

    with pytest.raises(ValueError, match="missing or ambiguous"):
        review._markdown_section(text, "## Target")


def test_markdown_section_handles_crlf_without_false_fence_boundary() -> None:
    text = "## Target\r\n~~~text\r\n## fake boundary\r\n~~~\r\nafter\r\n## Next\r\n"

    section = review._markdown_section(text, "## Target")

    assert "## fake boundary" in section
    assert "after" in section
    assert "## Next" not in section
