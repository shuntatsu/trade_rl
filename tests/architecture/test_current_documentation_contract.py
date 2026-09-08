from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

EXPECTED_DOC_FILES = {
    "README.md",
    "AGENTS.md",
    "architecture/lean-core.md",
    "architecture/package-boundaries.md",
    "research/current-status.md",
}

EXPECTED_DOC_DIRECTORIES = {"architecture", "research"}

COMPLIANCE_GIT_BLOBS = {
    "LICENSE": "0a041280bd00a9d068f503b8ee7ce35214bd24a1",
    "LICENSES/GPL-3.0-or-later.txt": "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7",
    "LICENSES/LGPL-3.0-or-later.txt": "0a041280bd00a9d068f503b8ee7ce35214bd24a1",
    "LICENSES/LICENSING.md": "a3066b942bd59ddae06e4c3ac53f907f84ee2a79",
    "LICENSES/MIT.txt": "9d78c611c71ee12f697b773e518a91b895e0f999",
    "LICENSES/PROVENANCE.md": "13af87f7f87d4602bc4c3f8d5123c5c6476de09b",
    "LICENSES/THIRD_PARTY_NOTICES.md": "f07145046b624c54344e5d925dc51efe66825ac5",
}

CURRENT_MARKDOWN = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    DOCS / "README.md",
    DOCS / "AGENTS.md",
    DOCS / "architecture" / "lean-core.md",
    DOCS / "architecture" / "package-boundaries.md",
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
    DOCS / "research" / "current-status.md",
)

PACKAGE_PATHS = {
    "_validation.py": ROOT / "trade_rl" / "_validation.py",
    "artifacts/": ROOT / "trade_rl" / "artifacts",
    "data/artifacts/": ROOT / "trade_rl" / "data" / "artifacts",
    "data/build/": ROOT / "trade_rl" / "data" / "build",
    "data/features/": ROOT / "trade_rl" / "data" / "features",
    "integrations/binance/": ROOT / "trade_rl" / "integrations" / "binance",
    "strategies/rules/": ROOT / "trade_rl" / "strategies" / "rules",
    "strategies/forecasts/": ROOT / "trade_rl" / "strategies" / "forecasts",
    "strategies/rl/": ROOT / "trade_rl" / "strategies" / "rl",
    "simulation/orders/": ROOT / "trade_rl" / "simulation" / "orders",
    "simulation/stateful/": ROOT / "trade_rl" / "simulation" / "stateful",
    "simulation/targets/": ROOT / "trade_rl" / "simulation" / "targets",
    "simulation/diagnostics/": ROOT / "trade_rl" / "simulation" / "diagnostics",
    "evaluation/gates/": ROOT / "trade_rl" / "evaluation" / "gates",
    "evaluation/comparison/": ROOT / "trade_rl" / "evaluation" / "comparison",
    "evaluation/robustness/": ROOT / "trade_rl" / "evaluation" / "robustness",
    "evaluation/runs/": ROOT / "trade_rl" / "evaluation" / "runs",
}

OLD_CURRENT_PATHS = {
    "docs/trade_rl_lean_redesign_20260908.md",
    "trade_rl.evaluation.candidate_run",
    "trade_rl.domain",
    "trade_rl.integrations.binance.py",
    "trade_rl.strategies.ridge",
    "trade_rl.simulation.orders.py",
}

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FULL_SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")
TRANSIENT_PR_RE = re.compile(r"\b(?:PR|pull request)\s*#\d+\b", re.IGNORECASE)


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _read(path: Path) -> str:
    assert path.is_file(), path.relative_to(ROOT)
    return path.read_text(encoding="utf-8")


def test_current_documentation_tree_is_minimal_and_has_no_working_tree_archive() -> None:
    observed_files = {
        str(path.relative_to(DOCS)) for path in DOCS.rglob("*.md") if path.is_file()
    }
    assert observed_files == EXPECTED_DOC_FILES

    observed_directories = {
        path.name for path in DOCS.iterdir() if path.is_dir() and path.name != "__pycache__"
    }
    assert observed_directories == EXPECTED_DOC_DIRECTORIES

    assert not (DOCS / "history").exists()
    assert not (DOCS / "archive").exists()
    assert not (DOCS / "plans").exists()
    assert not (DOCS / "specs").exists()
    assert not (DOCS / "trade_rl_lean_redesign_20260908.md").exists()


def test_root_and_docs_portal_route_humans_and_agents_to_current_authorities() -> None:
    readme = _read(ROOT / "README.md")
    root_agents = _read(ROOT / "AGENTS.md")
    docs_readme = _read(DOCS / "README.md")

    assert "docs/README.md" in readme
    assert "docs/research/current-status.md" in readme
    assert "python -m trade_rl.evaluation.runs.candidate" in readme

    assert "docs/README.md" in root_agents
    assert "docs/AGENTS.md" in root_agents
    assert len(root_agents) < 1800

    for target in (
        "architecture/lean-core.md",
        "architecture/package-boundaries.md",
        "research/current-status.md",
        "AGENTS.md",
        "../LICENSES/LICENSING.md",
        "../LICENSES/PROVENANCE.md",
    ):
        assert target in docs_readme


def test_agent_contract_defines_routing_update_and_retention_rules() -> None:
    agents = _read(DOCS / "AGENTS.md")
    for heading in (
        "## Before working",
        "## Evidence and authority",
        "## Documentation routing",
        "## Update triggers",
        "## Retention",
        "## Before finishing",
    ):
        assert heading in agents

    for token in (
        "docs/architecture/",
        "docs/research/",
        "docs/specs/",
        "docs/plans/",
        "Git history",
        "lowercase-kebab-case.md",
        "LICENSES/",
    ):
        assert token in agents

    assert "docs/history/" in agents
    assert "docs/archive/" in agents


def test_package_boundary_document_matches_paths_that_exist_in_source_tree() -> None:
    package_doc = _read(DOCS / "architecture" / "package-boundaries.md")
    for documented_path, source_path in PACKAGE_PATHS.items():
        assert source_path.exists(), source_path.relative_to(ROOT)
        assert documented_path in package_doc

    assert "evaluation/experiments/" in package_doc
    assert "not part of the current cleanup/runtime path" in package_doc


def test_lean_core_preserves_causality_accounting_and_independent_replay_contracts() -> None:
    lean_core = _read(DOCS / "architecture" / "lean-core.md")
    for contract in (
        "feature_available_time <= decision_time",
        "label_end_time < fit_cutoff",
        "MarketExecutor",
        "BookState",
        "quantity hold",
        "SHORT",
        "FLAT",
        "LONG",
        "independently",
        "fit_symbol_names",
    ):
        assert contract in lean_core


def test_research_status_is_explicit_about_what_has_and_has_not_been_validated() -> None:
    status = _read(DOCS / "research" / "current-status.md")
    for statement in (
        "M1 lean core: **complete**",
        "M2 comparison infrastructure: **complete**",
        "M2 real-data development comparison: **not run yet**",
        "M3 frozen final evaluation / stress / deletion: **not started**",
        "Profitability claim: **none**",
        "Production/live order routing: **not authorized**",
    ):
        assert statement in status

    for candidate in (
        "trend",
        "mean_reversion",
        "ridge24",
        "lightgbm24",
        "ppo",
        "cash",
        "constant_long",
        "constant_short",
    ):
        assert f"`{candidate}`" in status

    assert "fit_symbol_names" in status
    assert "evaluation" in status
    assert "no winner" in status.lower()


def test_current_authority_docs_do_not_use_retired_paths_or_transient_pr_sha_refs() -> None:
    for path in CURRENT_AUTHORITY_DOCS:
        text = _read(path)
        for retired in OLD_CURRENT_PATHS:
            assert retired not in text, (path.relative_to(ROOT), retired)
        assert FULL_SHA_RE.search(text) is None, path.relative_to(ROOT)
        assert TRANSIENT_PR_RE.search(text) is None, path.relative_to(ROOT)


def test_all_current_relative_markdown_links_resolve() -> None:
    for path in CURRENT_MARKDOWN:
        text = _read(path)
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_without_fragment = target.split("#", 1)[0]
            if not target_without_fragment:
                continue
            resolved = (path.parent / target_without_fragment).resolve()
            assert resolved.exists(), (path.relative_to(ROOT), target)


def test_permanent_license_and_provenance_material_is_byte_identical_to_phase4a_base() -> None:
    for relative_path, expected_blob_sha in COMPLIANCE_GIT_BLOBS.items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert _git_blob_sha(path) == expected_blob_sha, relative_path
