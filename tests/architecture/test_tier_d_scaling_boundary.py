"""Artificial prefix profiling stays repository-only and result-blind."""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _imports(source: str) -> set[str]:
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    return imported


def test_scaling_tooling_has_only_artificial_repository_dependencies():
    owner = ROOT / "tools/tier_d_scaling"
    sources = tuple(owner.glob("*.py"))
    assert sources
    allowed_prefixes = (
        "trade_rl.data",
        "trade_rl.artifacts",
        "trade_rl.evaluation.allocation_market_inputs",
        "trade_rl.evaluation.allocation_settled_forecast_comparison",
        "trade_rl.evaluation.allocation_terminal_execution",
        "trade_rl.evaluation.runs",
        "trade_rl.evaluation.allocation_terminal_evidence",
        "trade_rl.strategies.forecasts.simple_stream_io",
        "tools.tier_d_nonrl",
    )
    forbidden_literals = (
        "BTCUSDT",
        "ETHUSDT",
        "Task18",
        "C:\\dev\\trade_rl",
        "C:/dev/trade_rl",
        "output/issue810-tier-d-nonrl-preparation",
    )
    for path in sources:
        source = path.read_text(encoding="utf-8")
        for module in _imports(source):
            top_level = module.split(".", maxsplit=1)[0]
            assert (
                top_level in sys.stdlib_module_names
                or top_level == "numpy"
                or any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in allowed_prefixes
                )
            ), (path.name, module)
        assert not any(value in source for value in forbidden_literals), path.name

    for path in (ROOT / "trade_rl").rglob("*.py"):
        assert not any(
            module == "tools.tier_d_scaling"
            or module.startswith("tools.tier_d_scaling.")
            for module in _imports(path.read_text(encoding="utf-8"))
        ), path.relative_to(ROOT)


def test_scaling_source_has_lf_and_both_research_typecheck_routes():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "tools/tier_d_scaling/*.py text eol=lf" in attributes
    for path in (ROOT / "tools/tier_d_scaling").glob("*.py"):
        assert b"\r\n" not in path.read_bytes(), path.name

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert workflow.count("tools/tier_d_scaling") == 2
    assert workflow.count("name: Research transport types") == 2


def test_docs_bound_the_profile_to_artificial_prefixes():
    status = (ROOT / "docs/research/current-status.md").read_text(encoding="utf-8")
    section = status.split("## Issue #888 artificial scaling prefix profile", 1)[1]
    section = section.split("\n## ", 1)[0]
    for required in (
        "9,504 hourly bars × 2 artificial symbols × 6 features",
        "100, 400, and",
        "1,600 advances",
        "H72",
        "alpha 10",
        "ARTIFICIAL_A",
        "BASEL0",
        "one representative NONRL arm",
        "CASH",
        "UNATTEMPTED",
        "first-vintage actor advances",
        "No real input",
        "economic admission",
        "48-arm",
        "release byte-equivalence",
    ):
        assert required in section

    boundaries = (ROOT / "docs/architecture/package-boundaries.md").read_text(
        encoding="utf-8"
    )
    assert "`tools/tier_d_scaling/`" in boundaries
    assert "repository-only" in boundaries
