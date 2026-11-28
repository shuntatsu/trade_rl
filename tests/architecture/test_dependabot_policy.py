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
    schedule = _direct_mapping_fields("\n".join(_direct_block(section, "schedule")), 6)
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


def _direct_block(section: str, key: str) -> tuple[str, ...]:
    """Read one canonical direct block, excluding only blank lines and comments."""
    fields = _direct_mapping_fields(section, 4)
    assert key in fields and fields[key] is None
    lines = section.splitlines()
    start = lines.index(f"    {key}:") + 1
    block = []
    for line in lines[start:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith("      "):
            break
        block.append(line)
    return tuple(block)


def _require_update_recipe(
    section: str,
    *,
    strategy: str | None,
    groups: tuple[tuple[str, str | None], ...],
) -> None:
    """Pin the complete canonical recipe, rather than matching decoy strings.

    Unknown direct options and extra nested rule entries require review too.
    This is a deliberately fixed-layout policy guard, not a YAML parser.
    """
    fields = _direct_mapping_fields(section, 4)
    expected_fields = {"directory", "schedule", "open-pull-requests-limit", "ignore"}
    if strategy is not None:
        expected_fields.add("versioning-strategy")
        assert fields.get("versioning-strategy") == f'"{strategy}"'
    if groups:
        expected_fields.add("groups")
    assert set(fields) == expected_fields

    if groups:
        expected_group_lines = []
        for name, dependency_type in groups:
            expected_group_lines.extend(
                (f"      {name}:", '        applies-to: "version-updates"')
            )
            if dependency_type is not None:
                expected_group_lines.append(
                    f'        dependency-type: "{dependency_type}"'
                )
            expected_group_lines.extend(
                (
                    "        patterns:",
                    '          - "*"',
                    "        update-types:",
                    '          - "minor"',
                    '          - "patch"',
                )
            )
        assert _direct_block(section, "groups") == tuple(expected_group_lines)
    assert _direct_block(section, "ignore") == (
        '      - dependency-name: "*"',
        "        update-types:",
        '          - "version-update:semver-major"',
    )


def test_python_lockfile_only_recipe_is_retained_for_reenable() -> None:
    _require_update_recipe(
        _ecosystem_sections()["uv"],
        strategy="lockfile-only",
        groups=(("compatible-lockfile", None),),
    )


def test_guide_runtime_and_build_updates_remain_separate() -> None:
    _require_update_recipe(
        _ecosystem_sections()["npm"],
        strategy="increase-if-necessary",
        groups=(
            ("guide-runtime-minor-patch", "production"),
            ("guide-dev-minor-patch", "development"),
        ),
    )


def test_market_store_updates_reject_major_without_automatic_deploy() -> None:
    _require_update_recipe(
        _ecosystem_sections()["docker-compose"], strategy=None, groups=()
    )
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


@pytest.mark.parametrize(
    ("ecosystem", "original", "replacement"),
    [
        (
            "uv",
            '    versioning-strategy: "lockfile-only"',
            '    # versioning-strategy: "lockfile-only"\n    versioning-strategy: "increase"',
        ),
        (
            "npm",
            '    versioning-strategy: "increase-if-necessary"',
            '    versioning-strategy: "increase"\n    # versioning-strategy: "increase-if-necessary"',
        ),
        (
            "uv",
            '    versioning-strategy: "lockfile-only"',
            '    versioning-strategy: "increase"\n    decoy:\n      versioning-strategy: "lockfile-only"',
        ),
        (
            "npm",
            '    versioning-strategy: "increase-if-necessary"',
            '    versioning-strategy: "increase-if-necessary"\n    "versioning-strategy": "increase"',
        ),
        (
            "uv",
            '        applies-to: "version-updates"',
            '        # applies-to: "version-updates"\n        applies-to: "security-updates"',
        ),
        (
            "npm",
            "      guide-runtime-minor-patch:",
            "      # guide-runtime-minor-patch:\n      other-group:",
        ),
        (
            "npm",
            '        dependency-type: "production"',
            '        # dependency-type: "production"\n        dependency-type: "development-other"',
        ),
        (
            "uv",
            '          - "minor"\n          - "patch"',
            '          - "minor"\n          - "patch"\n          - "major"',
        ),
        (
            "npm",
            '          - "minor"\n          - "patch"',
            '          - "minor"\n          - "patch"\n          - "major"',
        ),
        (
            "uv",
            '          - "version-update:semver-major"',
            '          # - "version-update:semver-major"\n          - "version-update:semver-patch"',
        ),
        (
            "docker-compose",
            '          - "version-update:semver-major"',
            '          # - "version-update:semver-major"\n          - "version-update:semver-patch"',
        ),
        ("uv", '          - "*"', '          - "one-dependency"'),
    ],
)
def test_update_recipe_refuses_decoy_or_broadened_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ecosystem: str,
    original: str,
    replacement: str,
) -> None:
    source = MANIFEST.read_text(encoding="utf-8")
    section = _ecosystem_sections(source)[ecosystem]
    assert original in section
    changed = source.replace(section, section.replace(original, replacement, 1), 1)
    manifest = tmp_path / "dependabot.yml"
    manifest.write_text(changed, encoding="utf-8")
    monkeypatch.setitem(globals(), "MANIFEST", manifest)
    check = {
        "uv": test_python_lockfile_only_recipe_is_retained_for_reenable,
        "npm": test_guide_runtime_and_build_updates_remain_separate,
        "docker-compose": test_market_store_updates_reject_major_without_automatic_deploy,
    }[ecosystem]
    with pytest.raises(AssertionError):
        check()


def test_update_recipe_allows_comment_only_changes() -> None:
    section = _ecosystem_sections()["uv"]
    section = section.replace(
        '        applies-to: "version-updates"',
        "        # The actual scalar is unchanged.\n"
        '        applies-to: "version-updates"',
        1,
    )
    _require_update_recipe(
        section, strategy="lockfile-only", groups=(("compatible-lockfile", None),)
    )


def test_repo_owned_uv_proposals_are_main_only_and_read_only() -> None:
    workflow = (ROOT / ".github/workflows/uv-lock-proposals.yml").read_text(
        encoding="utf-8"
    )
    for required in (
        "on:\n"
        "  push:\n"
        "    branches:\n"
        "      - main\n"
        "    paths:\n"
        "      - .github/workflows/uv-lock-proposals.yml\n"
        "      - tools/uv_lock_proposal.py\n"
        "      - tools/uv_advisory_audit.py\n"
        "      - pyproject.toml\n"
        "  schedule:",
        'cron: "45 19 * * 1"',
        "workflow_dispatch:",
        "permissions:\n  contents: read",
        "github.repository == 'shuntatsu/trade_rl' && github.ref == 'refs/heads/main'",
        "persist-credentials: false",
        'version: "0.13.10"',
        "enable-cache: false",
        "ref: " + chr(36) + "{{ github.sha }}",
        "REQUESTED_PACKAGES: " + chr(36) + "{{ inputs.packages }}",
        'python -m tools.uv_lock_proposal --head "$SOURCE_HEAD"',
        '--packages "$REQUESTED_PACKAGES"',
        '--output "$RUNNER_TEMP/uv-lock-proposal"',
        "git diff --exit-code HEAD -- pyproject.toml uv.lock",
        "path: " + chr(36) + "{{ runner.temp }}/uv-lock-proposal",
        "if-no-files-found: error",
    ):
        assert required in workflow
    for forbidden in (
        "contents: write",
        "pull-requests: write",
        "pull_request_target:",
        "secrets.",
        "continue-on-error",
        "uv sync",
        "git push",
        "gh pr",
    ):
        assert forbidden not in workflow
    assert workflow.index("Verify original checkout") < workflow.index(
        "Upload review-only proposal"
    )
    upload = workflow.split("Upload review-only proposal", 1)[1]
    assert "if: always()" not in upload
    _require_version_update_limit(_ecosystem_sections()["uv"], 0)
