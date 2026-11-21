"""Architecture and import-boundary tests for the agy anchor port."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import trade_rl.strategies as strategies
from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
ANCHOR = PACKAGE / "strategies" / "agy_anchor"
MODULES = ("primitives.py", "inputs.py", "signals.py", "targets.py", "policy.py")
BASE = "trade_rl.strategies.agy_anchor"
FORBIDDEN_NAMES = frozenset(
    {
        "BookState",
        "MarketExecutor",
        "SharedPortfolioSession",
        "execute_native_target_transition",
        "execute_stateful_orders",
        "run_shared_portfolio_replay",
    }
)
FORBIDDEN_THIRD_PARTY = frozenset(
    {"torch", "stable_baselines3", "gymnasium", "requests", "httpx", "urllib3"}
)


def _trade_rl_imports(path: Path) -> set[str]:
    # ``from pkg import module`` also records ``pkg`` itself; ignore the package node
    return {
        name
        for name in ImportCollector(PACKAGE).collect(path)
        if within_module(name, "trade_rl") and name != BASE
    }


def _third_party(path: Path) -> set[str]:
    return {
        name.split(".", 1)[0]
        for name in ImportCollector(PACKAGE).collect(path)
        if name.split(".", 1)[0] not in sys.stdlib_module_names
        and not within_module(name, "trade_rl")
    }


def test_layout_and_no_facade_reexport() -> None:
    assert {path.name for path in ANCHOR.glob("*.py")} == {"__init__.py", *MODULES}
    assert "agy_anchor" not in strategies.__all__
    init_tree = ast.parse((ANCHOR / "__init__.py").read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom)) for node in init_tree.body
    )


def test_layers_keep_their_dependency_direction() -> None:
    assert _trade_rl_imports(ANCHOR / "primitives.py") == set()
    assert _trade_rl_imports(ANCHOR / "inputs.py") == {"trade_rl.data.market"}
    assert _trade_rl_imports(ANCHOR / "signals.py") == {
        f"{BASE}.primitives",
        f"{BASE}.inputs",
    }
    assert _trade_rl_imports(ANCHOR / "targets.py") == {
        f"{BASE}.primitives",
        f"{BASE}.signals",
    }
    assert _trade_rl_imports(ANCHOR / "policy.py") == {
        f"{BASE}.signals",
        f"{BASE}.targets",
        "trade_rl.strategies.portfolio",
    }


def test_only_numpy_and_no_io_network_ml_or_second_backtester() -> None:
    forbidden_prefixes = (
        "trade_rl.evaluation",
        "trade_rl.risk",
        "trade_rl.simulation",
        "trade_rl.integrations",
        "trade_rl.strategies.rl",
        "trade_rl.strategies.forecasts",
    )
    for module in MODULES:
        path = ANCHOR / module
        assert _third_party(path) <= {"numpy"}, module
        assert not _third_party(path) & FORBIDDEN_THIRD_PARTY, module
        for imported in _trade_rl_imports(path):
            assert not any(within_module(imported, p) for p in forbidden_prefixes), (
                module,
                imported,
            )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert not FORBIDDEN_NAMES & (names | attributes), module
        assert "random" not in names | attributes, module
        assert "open" not in names, module


def test_importing_the_package_does_not_load_torch_or_sb3() -> None:
    code = (
        "import sys; import trade_rl.strategies.agy_anchor.policy; "
        "bad = [m for m in ('torch', 'stable_baselines3') if m in sys.modules]; "
        "assert not bad, bad"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
