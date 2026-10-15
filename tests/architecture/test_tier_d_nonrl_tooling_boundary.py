"""The maintained Tier D driver composes native APIs without a second ledger."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_tier_d_source_bytes_have_portable_lf_checkout():
    # Market admission compares working bytes with committed blobs. A Windows
    # autocrlf checkout must preserve the reviewed source bytes for that check.
    attributes = (ROOT / ".gitattributes").read_text("utf-8")
    assert "tools/tier_d_nonrl/*.py text eol=lf" in attributes
    for path in (ROOT / "tools/tier_d_nonrl").glob("*.py"):
        assert b"\r\n" not in path.read_bytes(), path.name


def test_tier_d_source_has_a_maintained_cli_and_normal_verification_routing():
    owner = ROOT / "tools/tier_d_nonrl"
    assert (owner / "__init__.py").is_file()
    assert (owner / "__main__.py").is_file()
    assert (ROOT / "tests/tools/tier_d_nonrl/test_runner.py").is_file()
    workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")
    assert workflow.count("tools/tier_d_nonrl") == 2
    assert not any(
        (ROOT / "output/issue810-tier-d-nonrl-preparation" / name).exists()
        for name in ("tier_d_runner.py", "tier_d_supervisor.py", "test_runner.py")
    )


def test_tier_d_tooling_does_not_implement_a_ledger_or_import_live_final_or_ppo():
    owner = ROOT / "tools/tier_d_nonrl"
    paths = tuple(owner.glob("*.py"))
    assert paths
    forbidden = (
        "torch",
        "stable_baselines3",
        "ccxt",
        "requests",
        "trade_rl.evaluation.final_test",
        "trade_rl.evaluation.rl_allocation.training",
        "trade_rl.simulation.accounting",
    )
    for path in paths:
        tree = ast.parse(path.read_text("utf-8"))
        imports = {
            name
            for node in ast.walk(tree)
            for name in (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
                if isinstance(node, ast.ImportFrom) and node.module
                else []
            )
        }
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imports
            for prefix in forbidden
        ), path.name
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        assert not calls & {"BookState", "env.step", "env.reset"}, path.name
