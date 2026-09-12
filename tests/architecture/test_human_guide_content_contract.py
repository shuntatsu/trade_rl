from pathlib import Path

import pytest

from guide.tools.content_contract import (
    GuideContractError,
    extract_markdown_section,
    section_sha256,
    validate_source_sections,
)


def test_extract_markdown_section_ignores_fenced_heading_like_lines() -> None:
    markdown = """# Root

Intro.

```text
## Target
not a heading
```

## Target
Body.

### Child
Child body.

## Next
After.
"""

    assert extract_markdown_section(markdown, "Target") == (
        "## Target\nBody.\n\n### Child\nChild body.\n"
    )


def test_extract_markdown_section_rejects_missing_or_duplicate_heading() -> None:
    with pytest.raises(GuideContractError, match="missing heading"):
        extract_markdown_section("# Root\n", "Missing")

    duplicate = "# Root\n\n## Same\nOne.\n\n## Same\nTwo.\n"
    with pytest.raises(GuideContractError, match="duplicate heading"):
        extract_markdown_section(duplicate, "Same")


def test_validate_source_sections_accepts_reviewed_digest(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "architecture" / "demo.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Demo\n\n## Contract\nStable.\n", encoding="utf-8")
    section = extract_markdown_section(source.read_text(encoding="utf-8"), "Contract")

    validate_source_sections(
        tmp_path,
        [
            {
                "path": "docs/architecture/demo.md",
                "heading": "Contract",
                "sha256": section_sha256(section),
            }
        ],
    )


def test_validate_source_sections_rejects_stale_digest(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "research" / "current-status.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Status\n\n## M2\nChanged.\n", encoding="utf-8")

    with pytest.raises(GuideContractError, match="stale source fingerprint"):
        validate_source_sections(
            tmp_path,
            [
                {
                    "path": "docs/research/current-status.md",
                    "heading": "M2",
                    "sha256": "0" * 64,
                }
            ],
        )


def test_validate_source_sections_rejects_non_authoritative_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(GuideContractError, match="non-authoritative source path"):
        validate_source_sections(
            tmp_path,
            [
                {
                    "path": "README.md",
                    "heading": "Overview",
                    "sha256": "0" * 64,
                }
            ],
        )
