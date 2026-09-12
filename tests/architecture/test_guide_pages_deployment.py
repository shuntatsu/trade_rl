from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy-guide.yml"


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_pages_deployment_is_bound_to_successful_same_repo_main_push_ci() -> None:
    text = _workflow()
    for required in (
        "workflow_run:",
        'workflows: ["CI"]',
        "types: [completed]",
        "github.event.workflow_run.conclusion == 'success'",
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_branch == 'main'",
        "github.event.workflow_run.head_repository.full_name == github.repository",
        "ref: ${{ github.event.workflow_run.head_sha }}",
    ):
        assert required in text


def test_pages_deployment_has_minimal_permissions_and_exact_pins() -> None:
    text = _workflow()
    for required in (
        "permissions: {}",
        "contents: read",
        "pages: write",
        "id-token: write",
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020",
        "actions/configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d",
        "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9",
        "actions/deploy-pages@368f82528645a54fb793d4d04e342629a3f51346",
    ):
        assert required in text


def test_pages_artifact_and_public_smoke_are_guide_only() -> None:
    text = _workflow()
    assert "path: guide/dist" in text
    assert "npm run source-check" in text
    assert "npm run build" in text
    assert "GUIDE_PUBLIC_URL:" in text
    assert "npm run e2e:public" in text
    assert (ROOT / "guide" / "playwright.public.config.ts").is_file()
    assert (ROOT / "guide" / "e2e" / "public-smoke.spec.ts").is_file()
