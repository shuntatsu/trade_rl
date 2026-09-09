from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

REQUIRED_DOC_FILES = {
    "AGENTS.md",
    "README.md",
    "architecture/lean-core.md",
    "architecture/package-boundaries.md",
    "architecture/controlled-experiment-loop.md",
    "research/current-status.md",
}
EPHEMERAL_DOC_ROOTS = {"plans", "specs"}

CURRENT_MARKDOWN = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    DOCS / "README.md",
    DOCS / "AGENTS.md",
    DOCS / "architecture" / "lean-core.md",
    DOCS / "architecture" / "package-boundaries.md",
    DOCS / "architecture" / "controlled-experiment-loop.md",
    DOCS / "research" / "current-status.md",
    ROOT / "LICENSES" / "LICENSING.md",
    ROOT / "LICENSES" / "PROVENANCE.md",
    ROOT / "LICENSES" / "THIRD_PARTY_NOTICES.md",
)

CURRENT_AUTHORITY_DOCS = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    DOCS / "README.md",
    DOCS / "AGENTS.md",
    DOCS / "architecture" / "lean-core.md",
    DOCS / "architecture" / "package-boundaries.md",
    DOCS / "architecture" / "controlled-experiment-loop.md",
    DOCS / "research" / "current-status.md",
)

COMPLIANCE_GIT_BLOBS = {
    "LICENSE": "0a041280bd00a9d068f503b8ee7ce35214bd24a1",
    "LICENSES/GPL-3.0-or-later.txt": "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7",
    "LICENSES/LGPL-3.0-or-later.txt": "0a041280bd00a9d068f503b8ee7ce35214bd24a1",
    "LICENSES/LICENSING.md": "a3066b942bd59ddae06e4c3ac53f907f84ee2a79",
    "LICENSES/MIT.txt": "9d78c611c71ee12f697b773e518a91b895e0f999",
    "LICENSES/PROVENANCE.md": "13af87f7f87d4602bc4c3f8d5123c5c6476de09b",
    "LICENSES/THIRD_PARTY_NOTICES.md": "f07145046b624c54344e5d925dc51efe66825ac5",
}

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FULL_SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")
TRANSIENT_PR_RE = re.compile(r"\b(?:PR|pull request)\s*#\d+\b", re.IGNORECASE)


def _doc_files() -> set[str]:
    result: set[str] = set()
    for path in DOCS.rglob("*"):
        if path.is_file():
            result.add(path.relative_to(DOCS).as_posix())
    return result


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def test_docs_tree_contains_current_authorities_and_only_active_ephemeral_docs() -> (
    None
):
    files = _doc_files()
    assert REQUIRED_DOC_FILES <= files

    for relative in sorted(files - REQUIRED_DOC_FILES):
        path = Path(relative)
        assert path.parts[0] in EPHEMERAL_DOC_ROOTS, relative
        assert path.suffix == ".md", relative
        text = (DOCS / path).read_text(encoding="utf-8")
        assert "Status: Active" in text, relative

    assert not (DOCS / "history").exists()
    assert not (DOCS / "archive").exists()


def test_phase4b_preview_workflow_is_absent() -> None:
    path = ROOT / ".github" / "workflows" / "phase4b-format-preview.yml"
    assert not path.exists()


def test_root_agent_entry_routes_to_docs_contract() -> None:
    root_agents = ROOT / "AGENTS.md"
    assert root_agents.is_file()
    text = root_agents.read_text(encoding="utf-8")
    assert "docs/README.md" in text
    assert "docs/AGENTS.md" in text


def test_docs_index_routes_to_every_current_doc() -> None:
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    for target in (
        "AGENTS.md",
        "architecture/lean-core.md",
        "architecture/package-boundaries.md",
        "architecture/controlled-experiment-loop.md",
        "research/current-status.md",
    ):
        assert target in index


def test_agent_contract_defines_update_and_retention_policy() -> None:
    contract = (DOCS / "AGENTS.md").read_text(encoding="utf-8")
    for required in (
        "Git history",
        "LICENSES/",
        "architecture/",
        "research/",
        "specs/",
        "plans/",
        "Status: Active",
        "docs/history",
        "docs/archive",
    ):
        assert required in contract


def test_current_docs_preserve_core_and_research_contracts() -> None:
    lean_core = (DOCS / "architecture" / "lean-core.md").read_text(encoding="utf-8")
    package_boundaries = (DOCS / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    research = (DOCS / "research" / "current-status.md").read_text(encoding="utf-8")

    for required in (
        "quantity-preserving hold",
        "MarketExecutor + BookState",
        "feature_available_time <= decision_time",
        "SHORT",
        "FLAT",
        "LONG",
    ):
        assert required in lean_core

    for required in (
        "artifacts / data / integrations / risk / simulation / strategies / evaluation",
        "_validation -> standard library only",
        "evaluation/experiments/",
        "private module path",
    ):
        assert required in package_boundaries

    for required in (
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
        "ppo",
        "cash",
        "constant_long",
        "constant_short",
        "fit_symbol_names",
        "no winner",
        "real-data development comparison not run",
        "python -m trade_rl.evaluation.runs.candidate",
        "fee adverse",
        "spread adverse",
        "+1 decision latency",
        "Production/live order routing",
    ):
        assert required in research


def test_controlled_experiment_loop_is_durable_current_architecture() -> None:
    contract = (DOCS / "architecture" / "controlled-experiment-loop.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "Study-owned EvidenceSet",
        "ACCEPT_CANDIDATE",
        "FAILED",
        "INVALID",
        "WINNER",
        "NO_WINNER",
        "sealed unused-future",
        "freeze_study",
    ):
        assert required in contract
    assert not (DOCS / "specs").exists()
    assert not (DOCS / "plans").exists()


def test_current_relative_markdown_links_resolve() -> None:
    for path in CURRENT_MARKDOWN:
        text = path.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_without_fragment = target.split("#", 1)[0]
            if not target_without_fragment:
                continue
            resolved = (path.parent / target_without_fragment).resolve()
            assert resolved.exists(), (path.relative_to(ROOT), target)


def test_current_authority_docs_do_not_embed_transient_pr_or_full_sha_refs() -> None:
    for path in CURRENT_AUTHORITY_DOCS:
        text = path.read_text(encoding="utf-8")
        assert FULL_SHA_RE.search(text) is None, path.relative_to(ROOT)
        assert TRANSIENT_PR_RE.search(text) is None, path.relative_to(ROOT)


def test_permanent_compliance_material_matches_verified_base() -> None:
    for relative_path, expected_blob_sha in COMPLIANCE_GIT_BLOBS.items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert _git_blob_sha(path) == expected_blob_sha, relative_path


def test_removed_monolithic_doc_is_not_referenced() -> None:
    old_path = "docs/trade_rl_lean_redesign_20260908.md"
    offenders: list[str] = []
    for path in ROOT.rglob("*.md"):
        if old_path in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_root_readme_uses_current_docs_entry_point() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/README.md" in readme
    assert "docs/trade_rl_lean_redesign_20260908.md" not in readme
