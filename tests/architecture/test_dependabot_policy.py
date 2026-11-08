"""Contract for review-only Dependabot update proposals.

The manifest is checked for YAML syntax separately; these tests pin the
repository's substantive update scope without adding a YAML runtime dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / ".github" / "dependabot.yml"
ECOSYSTEM_PATTERN = re.compile(r'^  - package-ecosystem: "([a-z-]+)"$', re.M)


def _ecosystem_sections() -> dict[str, str]:
    source = MANIFEST.read_text(encoding="utf-8")
    assert source.startswith("version: 2\nupdates:\n")
    matches = list(ECOSYSTEM_PATTERN.finditer(source))
    names = [match.group(1) for match in matches]
    assert names == ["github-actions", "uv", "npm", "docker-compose"]
    return {
        match.group(1): source[match.end() : matches[i + 1].start()]
        if i + 1 < len(matches)
        else source[match.end() :]
        for i, match in enumerate(matches)
    }


def test_update_roster_and_weekly_review_limits() -> None:
    sections = _ecosystem_sections()
    expected = {
        "github-actions": ("/", 5),
        "uv": ("/", 2),
        "npm": ("/guide", 2),
        "docker-compose": ("/deploy/market-store", 1),
    }
    for ecosystem, (directory, limit) in expected.items():
        section = sections[ecosystem]
        assert f'directory: "{directory}"' in section
        assert 'interval: "weekly"' in section
        assert f"open-pull-requests-limit: {limit}" in section

    assert (ROOT / "uv.lock").is_file()
    assert (ROOT / "guide" / "package-lock.json").is_file()
    assert (ROOT / "deploy" / "market-store" / "compose.yml").is_file()


def test_python_updates_are_compatible_lockfile_only() -> None:
    uv = _ecosystem_sections()["uv"]
    assert 'versioning-strategy: "lockfile-only"' in uv
    assert 'applies-to: "version-updates"' in uv
    assert "compatible-lockfile:" in uv
    assert '          - "minor"\n          - "patch"' in uv
    assert 'dependency-name: "*"' in uv
    assert 'version-update:semver-major' in uv


def test_guide_runtime_and_build_updates_remain_separate() -> None:
    npm = _ecosystem_sections()["npm"]
    assert 'versioning-strategy: "increase-if-necessary"' in npm
    assert "guide-runtime-minor-patch:" in npm
    assert "guide-dev-minor-patch:" in npm
    assert npm.count('applies-to: "version-updates"') == 2
    assert npm.count('dependency-type: "production"') == 1
    assert npm.count('dependency-type: "development"') == 1
    assert npm.count('          - "minor"\n          - "patch"') == 2
    assert 'version-update:semver-major' in npm


def test_market_store_updates_reject_major_without_automatic_deploy() -> None:
    compose = _ecosystem_sections()["docker-compose"]
    assert 'dependency-name: "*"' in compose
    assert 'version-update:semver-major' in compose
    source = MANIFEST.read_text(encoding="utf-8")
    assert "auto-merge" not in source
    assert "insecure-external-code-execution" not in source
