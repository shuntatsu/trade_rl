from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

CHECKOUT_SHA = "34e114876b0b11c390a56381ad16ebd13914f8d5"
SETUP_UV_SHA = "d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86"
CONFIGURE_PAGES_SHA = "983d7736d9b0ae728b81ab479565c72886d7745b"
UPLOAD_PAGES_SHA = "7b1f4a764d45c48632c6b24a0339c27f5614fb0b"
DEPLOY_PAGES_SHA = "d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e"


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_zensical_configuration_uses_generated_current_site_source() -> None:
    config = _load_yaml(ROOT / "mkdocs.yml")
    assert config["docs_dir"] == ".docs-build/site-src"
    assert config["site_dir"] == "site"
    assert config["repo_url"] == "https://github.com/shuntatsu/trade_rl"
    assert config["edit_uri"] == "edit/main/docs/"

    theme = config["theme"]
    assert isinstance(theme, dict)
    assert theme["variant"] == "modern"
    assert theme["language"] == "ja"
    features = set(theme["features"])
    for required in (
        "navigation.indexes",
        "navigation.path",
        "navigation.sections",
        "navigation.top",
        "search.highlight",
        "toc.follow",
    ):
        assert required in features

    assert config["plugins"] == ["search", "meta", "tags"]
    assert "stylesheets/extra.css" in config["extra_css"]


def test_zensical_navigation_exposes_current_sections_and_generated_indexes() -> None:
    config = _load_yaml(ROOT / "mkdocs.yml")
    nav_text = str(config["nav"])
    for required in (
        "getting-started/index.md",
        "guides/index.md",
        "reference/index.md",
        "research/index.md",
        "operations/index.md",
        "performance/index.md",
        "legal/index.md",
        "generated/topics.md",
        "generated/authority.md",
        "history/index.md",
    ):
        assert required in nav_text


def test_documentation_workflow_builds_every_pr_and_deploys_only_main() -> None:
    workflow_path = ROOT / ".github" / "workflows" / "docs.yml"
    text = workflow_path.read_text(encoding="utf-8")
    workflow = _load_yaml(workflow_path)
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert "build" in jobs
    assert "deploy" in jobs

    build = jobs["build"]
    deploy = jobs["deploy"]
    assert isinstance(build, dict)
    assert isinstance(deploy, dict)
    assert build["permissions"] == {"contents": "read"}
    assert deploy["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    assert deploy["needs"] == "build"
    assert "github.event_name == 'push'" in str(deploy["if"])
    assert "refs/heads/main" in str(deploy["if"])
    assert deploy["environment"]["name"] == "github-pages"

    for action_sha in (
        CHECKOUT_SHA,
        SETUP_UV_SHA,
        CONFIGURE_PAGES_SHA,
        UPLOAD_PAGES_SHA,
        DEPLOY_PAGES_SHA,
    ):
        assert action_sha in text

    for required_command in (
        "uv run pytest -q tests/docs",
        "uv run python scripts/docs/validate.py",
        "uv run python scripts/docs/generate.py",
        "uvx --from zensical==0.0.59 zensical build --strict",
    ):
        assert required_command in text


def test_pages_workflow_does_not_deploy_pull_requests() -> None:
    workflow = _load_yaml(ROOT / ".github" / "workflows" / "docs.yml")
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    deploy = jobs["deploy"]
    assert isinstance(deploy, dict)
    condition = str(deploy["if"])
    assert "pull_request" not in condition
    assert "push" in condition
