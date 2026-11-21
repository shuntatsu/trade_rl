"""Architecture and import-boundary tests for the out-of-sample trend test."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from trade_rl import evaluation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "trade_rl" / "evaluation" / "trend_oos"
RUNNER_PATH = PACKAGE_DIR / "runner.py"
TOOL_PATH = ROOT / "tools" / "trend_oos.py"

_NETWORK_MODULES = frozenset(
    {
        "aiohttp",
        "ftplib",
        "http",
        "httpx",
        "requests",
        "smtplib",
        "socket",
        "ssl",
        "subprocess",
        "urllib",
    }
)
_FORBIDDEN_ML_MODULES = frozenset({"gymnasium", "stable_baselines3", "torch"})
_FORBIDDEN_SECOND_BACKTESTER_NAMES = frozenset(
    {
        "BookState",
        "MarketExecutor",
        "SharedPortfolioSession",
        "execute_native_target_transition",
        "execute_stateful_orders",
    }
)


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _referenced_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def test_trend_oos_has_no_network_no_store_no_ml_and_no_second_backtester() -> None:
    for py_path in sorted(PACKAGE_DIR.glob("*.py")):
        for module in _imported_roots(py_path):
            root = module.split(".", 1)[0]
            assert root not in _NETWORK_MODULES, f"{py_path.name} imports {module}"
            assert root not in _FORBIDDEN_ML_MODULES, f"{py_path.name} imports {module}"
            assert not module.startswith("trade_rl.integrations"), py_path.name
    for py_path in sorted(PACKAGE_DIR.glob("*.py")):
        overlap = _FORBIDDEN_SECOND_BACKTESTER_NAMES & _referenced_names(py_path)
        assert not overlap, f"{py_path.name} references {sorted(overlap)}"


def test_the_runner_calls_the_shared_replay_once_with_settlement_and_ledger() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_shared_portfolio_replay"
    ]
    assert len(calls) == 1
    keywords = {k.arg: k.value for k in calls[0].keywords if k.arg is not None}
    for name in ("settle_terminal_position", "capture_ledger_evidence"):
        assert isinstance(keywords.get(name), ast.Constant)
        assert keywords[name].value is True


def test_the_cell_summary_is_always_called_with_the_execution_cost() -> None:
    """Without ``execution_cost`` the core7 summary cannot value terminal dust."""
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_summarize_replay_cell"
    ]
    assert len(calls) == 1
    assert "execution_cost" in {k.arg for k in calls[0].keywords}


def test_the_policy_is_lowturns_and_nothing_here_redefines_it() -> None:
    names: set[str] = set()
    for py_path in PACKAGE_DIR.glob("*.py"):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        names.update(
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        )
    assert "LowturnPolicy" not in names and "PortfolioPolicy" not in names
    assert not (PACKAGE_DIR / "policies.py").exists()
    runner_imports = _imported_roots(RUNNER_PATH)
    assert "trade_rl.evaluation.lowturn_screen.policies" in runner_imports


def test_layers_keep_their_dependency_direction() -> None:
    decision = _imported_roots(PACKAGE_DIR / "decision.py")
    assert not any(
        module.startswith(
            (
                "trade_rl.data",
                "trade_rl.simulation",
                "trade_rl.risk",
                "trade_rl.strategies",
                "trade_rl.evaluation.replay",
                "trade_rl.evaluation.trend_oos.runner",
            )
        )
        for module in decision
    )
    signals = _imported_roots(PACKAGE_DIR / "signals.py")
    assert not any(
        module.startswith(
            (
                "trade_rl.strategies",
                "trade_rl.simulation",
                "trade_rl.risk",
                "trade_rl.evaluation.replay",
            )
        )
        for module in signals
    )
    protocol = _imported_roots(PACKAGE_DIR / "protocol.py")
    assert not any(
        module.startswith(("trade_rl.data", "trade_rl.evaluation.replay"))
        for module in protocol
    )


def test_only_the_tool_wires_the_market_store_and_the_forward_embargo_policy() -> None:
    tool = _imported_roots(TOOL_PATH)
    assert "trade_rl.integrations.market_store.embargo" in tool
    package_text = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGE_DIR.glob("*.py")
    )
    assert "EmbargoPolicy" not in package_text
    assert "ForwardEmbargoPolicy" in TOOL_PATH.read_text(encoding="utf-8")


def test_importing_the_package_does_not_load_torch_or_sb3_and_is_not_a_facade() -> None:
    code = (
        "import sys; import trade_rl.evaluation.trend_oos; "
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
    assert not {"run_stage", "OosCell", "StageWindow"} & set(evaluation.__all__)
