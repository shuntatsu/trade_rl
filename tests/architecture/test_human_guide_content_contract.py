from pathlib import Path

import guide.tools.content_contract as contract


def _assert_contract_error(callable_object, message: str) -> None:
    try:
        callable_object()
    except contract.GuideContractError as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"expected GuideContractError containing {message!r}")


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

    assert contract.extract_markdown_section(markdown, "Target") == (
        "## Target\nBody.\n\n### Child\nChild body.\n"
    )


def test_extract_markdown_section_rejects_missing_or_duplicate_heading() -> None:
    _assert_contract_error(
        lambda: contract.extract_markdown_section("# Root\n", "Missing"),
        "missing heading",
    )

    duplicate = "# Root\n\n## Same\nOne.\n\n## Same\nTwo.\n"
    _assert_contract_error(
        lambda: contract.extract_markdown_section(duplicate, "Same"),
        "duplicate heading",
    )


def test_validate_source_sections_accepts_reviewed_digest(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "architecture" / "demo.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Demo\n\n## Contract\nStable.\n", encoding="utf-8")
    section = contract.extract_markdown_section(
        source.read_text(encoding="utf-8"), "Contract"
    )

    contract.validate_source_sections(
        tmp_path,
        [
            {
                "path": "docs/architecture/demo.md",
                "heading": "Contract",
                "sha256": contract.section_sha256(section),
            }
        ],
    )


def test_validate_source_sections_rejects_stale_digest(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "research" / "current-status.md"
    source.parent.mkdir(parents=True)
    source.write_text("# Status\n\n## M2\nChanged.\n", encoding="utf-8")

    _assert_contract_error(
        lambda: contract.validate_source_sections(
            tmp_path,
            [
                {
                    "path": "docs/research/current-status.md",
                    "heading": "M2",
                    "sha256": "0" * 64,
                }
            ],
        ),
        "stale source fingerprint",
    )


def test_validate_source_sections_rejects_non_authoritative_path(
    tmp_path: Path,
) -> None:
    _assert_contract_error(
        lambda: contract.validate_source_sections(
            tmp_path,
            [
                {
                    "path": "README.md",
                    "heading": "Overview",
                    "sha256": "0" * 64,
                }
            ],
        ),
        "non-authoritative source path",
    )
