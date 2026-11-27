"""Contract for review-only Dependabot update proposals.

The manifest is checked for YAML syntax separately; these tests pin the
repository's substantive update scope without adding a YAML runtime dependency.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / ".github" / "dependabot.yml"
ECOSYSTEM_PATTERN = re.compile(r'^  - package-ecosystem: "([a-z-]+)"$', re.M)


def _ecosystem_sections(source: str | None = None) -> dict[str, str]:
    if source is None:
        source = MANIFEST.read_text(encoding="utf-8")
    assert source.startswith("version: 2\nupdates:\n")
    # Every non-comment line outside an update mapping must be one of the
    # canonical list entries. A regex search alone misses extra flow entries,
    # replacement root keys and appended YAML documents.
    for line in source.splitlines()[2:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith("    ") and ECOSYSTEM_PATTERN.fullmatch(line) is None:
            raise AssertionError(f"noncanonical update layout: {line}")
    matches = list(ECOSYSTEM_PATTERN.finditer(source))
    names = [match.group(1) for match in matches]
    assert names == ["github-actions", "uv", "npm", "docker-compose"]
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(source)
        sections[match.group(1)] = source[start:end]
    return sections


def _direct_mapping_fields(section: str, indent: int) -> dict[str, str | None]:
    """Read canonical direct keys in the manifest's fixed block layout."""
    prefix = " " * indent
    fields: dict[str, str | None] = {}
    for line in section.splitlines():
        if (
            not line.startswith(prefix)
            or line.startswith(prefix + " ")
            or not line.strip()
            or line.lstrip().startswith("#")
        ):
            continue
        match = re.fullmatch(r"([a-z][a-z0-9-]*):(?: (.*))?", line[indent:])
        assert match is not None, f"noncanonical direct update field: {line}"
        key, value = match.groups()
        assert key not in fields, f"duplicate direct update field: {key}"
        fields[key] = value
    return fields


def _require_version_update_limit(section: str, expected: int) -> None:
    """Require one canonical direct scalar, not a comment or nested value."""
    fields = _direct_mapping_fields(section, 4)
    assert fields.get("open-pull-requests-limit") == str(expected)


def _require_update_scope(section: str, directory: str, limit: int) -> None:
    fields = _direct_mapping_fields(section, 4)
    assert fields.get("directory") == f'"{directory}"'
    assert fields.get("open-pull-requests-limit") == str(limit)
    assert "schedule" in fields and fields["schedule"] is None

    # Only the schedule's own direct children count; a comment or interval
    # elsewhere in the ecosystem mapping cannot establish the cadence.
    _, separator, suffix = section.partition("\n    schedule:\n")
    assert separator
    schedule_lines = []
    for line in suffix.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith("      "):
            break
        schedule_lines.append(line)
    schedule = _direct_mapping_fields("\n".join(schedule_lines), 6)
    assert schedule.get("interval") == '"weekly"'


def test_update_roster_and_weekly_review_limits() -> None:
    sections = _ecosystem_sections()
    expected = {
        "github-actions": ("/", 5),
        "uv": ("/", 0),
        "npm": ("/guide", 2),
        "docker-compose": ("/deploy/market-store", 1),
    }
    for ecosystem, (directory, limit) in expected.items():
        section = sections[ecosystem]
        _require_update_scope(section, directory, limit)

    assert (ROOT / "uv.lock").is_file()
    assert (ROOT / "guide" / "package-lock.json").is_file()
    assert (ROOT / "deploy" / "market-store" / "compose.yml").is_file()


def test_uv_version_updates_hold_for_pinned_incompatible_updater() -> None:
    """A new uv binary alone does not establish successful hosted Dependabot."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["uv"]["required-version"] == "==0.13.10"
    uv = _ecosystem_sections()["uv"]
    _require_version_update_limit(uv, 0)


@pytest.mark.parametrize(
    "replacement",
    [
        "    # open-pull-requests-limit: 0\n    open-pull-requests-limit: 1",
        "    open-pull-requests-limit: 01",
        "    open-pull-requests-limit: 0x1",
        "    open-pull-requests-limit: 0.5",
        "    open-pull-requests-limit: 0\n    open-pull-requests-limit: 1",
        '    open-pull-requests-limit: 0\n    "open-pull-requests-limit": 1',
        "    open-pull-requests-limit: 0\n    'open-pull-requests-limit': 1",
        "    open-pull-requests-limit: 1\n    # open-pull-requests-limit: 0",
        "    open-pull-requests-limit: 1\n    unused:\n"
        "      open-pull-requests-limit: 0",
        "    # open-pull-requests-limit: 0",
        "    unused:\n      open-pull-requests-limit: 0",
        "    open-pull-requests-limit: 0\n    <<: {open-pull-requests-limit: 1}",
    ],
)
def test_uv_hold_rejects_noncanonical_or_enabled_limit(replacement: str) -> None:
    section = _ecosystem_sections()["uv"]
    original = "    open-pull-requests-limit: 0\n"
    assert section.count(original) == 1
    changed = section.replace(original, replacement + "\n", 1)
    with pytest.raises(AssertionError):
        _require_version_update_limit(changed, 0)


@pytest.mark.parametrize(
    "suffix",
    [
        '  - {package-ecosystem: uv, directory: "/other", '
        "schedule: {interval: weekly}, open-pull-requests-limit: 1}\n",
        'updates: [{package-ecosystem: uv, directory: "/", '
        "schedule: {interval: weekly}, open-pull-requests-limit: 1}]\n",
        '"updates": [{package-ecosystem: uv, directory: "/", '
        "schedule: {interval: weekly}, open-pull-requests-limit: 1}]\n",
        '---\nversion: 2\nupdates: [{package-ecosystem: uv, directory: "/", '
        "schedule: {interval: weekly}, open-pull-requests-limit: 1}]\n",
    ],
)
def test_update_roster_refuses_hidden_entries(suffix: str) -> None:
    source = MANIFEST.read_text(encoding="utf-8") + suffix
    with pytest.raises(AssertionError, match="noncanonical update layout"):
        _ecosystem_sections(source)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        (
            '    directory: "/guide"',
            '    # directory: "/guide"\n    directory: "/other"',
        ),
        (
            '    directory: "/guide"',
            '    directory: "/other"\n    ignored:\n      directory: "/guide"',
        ),
        (
            '    directory: "/guide"',
            '    directory: "/guide"\n    "directory": "/other"',
        ),
        (
            '      interval: "weekly"',
            '      # interval: "weekly"\n      interval: "daily"',
        ),
        (
            '      interval: "weekly"',
            '      interval: "daily"\n      unused:\n        interval: "weekly"',
        ),
        (
            '    schedule:\n      interval: "weekly"',
            '    schedule:\n      interval: "daily"\n'
            '    other:\n      interval: "weekly"',
        ),
        (
            '      interval: "weekly"',
            '      interval: "weekly"\n      interval: "daily"',
        ),
        (
            '      interval: "weekly"',
            '      interval: "weekly"\n      "interval": "daily"',
        ),
        (
            '    schedule:\n      interval: "weekly"',
            '    schedule: {interval: daily}\n    # interval: "weekly"',
        ),
    ],
)
def test_update_scope_refuses_directory_and_schedule_decoys(
    original: str, replacement: str
) -> None:
    section = _ecosystem_sections()["npm"]
    assert section.count(original) == 1
    changed = section.replace(original, replacement, 1)
    with pytest.raises(AssertionError):
        _require_update_scope(changed, "/guide", 2)


def test_python_lockfile_only_recipe_is_retained_for_reenable() -> None:
    uv = _ecosystem_sections()["uv"]
    assert 'versioning-strategy: "lockfile-only"' in uv
    assert 'applies-to: "version-updates"' in uv
    assert "compatible-lockfile:" in uv
    assert '          - "minor"\n          - "patch"' in uv
    assert 'dependency-name: "*"' in uv
    assert "version-update:semver-major" in uv


def test_guide_runtime_and_build_updates_remain_separate() -> None:
    npm = _ecosystem_sections()["npm"]
    assert 'versioning-strategy: "increase-if-necessary"' in npm
    assert "guide-runtime-minor-patch:" in npm
    assert "guide-dev-minor-patch:" in npm
    assert npm.count('applies-to: "version-updates"') == 2
    assert npm.count('dependency-type: "production"') == 1
    assert npm.count('dependency-type: "development"') == 1
    assert npm.count('          - "minor"\n          - "patch"') == 2
    assert "version-update:semver-major" in npm


def test_market_store_updates_reject_major_without_automatic_deploy() -> None:
    compose = _ecosystem_sections()["docker-compose"]
    assert 'dependency-name: "*"' in compose
    assert "version-update:semver-major" in compose
    source = MANIFEST.read_text(encoding="utf-8")
    assert "auto-merge" not in source
    assert "insecure-external-code-execution" not in source


def test_locked_advisory_monitor_is_read_only_and_separate_from_updater() -> None:
    workflow = (ROOT / ".github/workflows/ci-uv-advisory-audit.yml").read_text(
        encoding="utf-8"
    )
    assert 'cron: "30 19 * * 1"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" in workflow and "branches:\n      - main" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "enable-cache: false" in workflow
    assert 'version: "0.13.10"' in workflow
    assert (
        "ref: " + chr(36) + "{{ github.event.pull_request.head.sha || github.sha }}"
    ) in workflow
    assert "ref: }" not in workflow
    assert "set -euo pipefail" in workflow
    assert "UV_PYTHON_DOWNLOADS: never" in workflow
    assert '[[ "$GITHUB_EVENT_NAME" == "pull_request" ]]' in workflow
    assert "python -m tools.uv_advisory_audit --report-only" in workflow
    assert "python -m tools.uv_advisory_audit | tee" in workflow
    assert "git diff --exit-code HEAD -- pyproject.toml uv.lock" in workflow
    for forbidden in (
        "contents: write",
        "pull-requests: write",
        "id-token: write",
        "pull_request_target:",
        "secrets.",
        "uv sync",
        "uv lock --upgrade",
        "continue-on-error",
        "git push",
        "gh pr merge",
    ):
        assert forbidden not in workflow
    _require_version_update_limit(_ecosystem_sections()["uv"], 0)
