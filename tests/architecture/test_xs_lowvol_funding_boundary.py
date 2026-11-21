"""Architecture and import-boundary tests for the ``xs_lowvol_funding_v1`` test."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from trade_rl import evaluation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "trade_rl" / "evaluation" / "xs_lowvol_funding"
RUNNER_PATH = PACKAGE_DIR / "runner.py"
TOOL_PATH = ROOT / "tools" / "xs_lowvol_funding.py"
MODULES = {
    "__init__.py",
    "protocol.py",
    "universe.py",
    "policy.py",
    "data.py",
    "decision.py",
    "runner.py",
}

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
#: Policies other protocols own; this package must not redefine them.
_FOREIGN_POLICIES = frozenset(
    {"AgyAnchorPolicy", "LowturnPolicy", "CashAccountPolicy", "AnchorCashAccountPolicy"}
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


def _calls(path: Path, name: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def test_the_package_layout_is_exactly_the_documented_modules() -> None:
    assert {path.name for path in PACKAGE_DIR.glob("*.py")} == MODULES


def test_no_network_no_store_no_ml_and_no_second_backtester() -> None:
    for py_path in sorted(PACKAGE_DIR.glob("*.py")):
        for module in _imported_roots(py_path):
            root = module.split(".", 1)[0]
            assert root not in _NETWORK_MODULES, f"{py_path.name} imports {module}"
            assert root not in _FORBIDDEN_ML_MODULES, f"{py_path.name} imports {module}"
            assert not module.startswith("trade_rl.integrations"), py_path.name
        overlap = _FORBIDDEN_SECOND_BACKTESTER_NAMES & _referenced_names(py_path)
        assert not overlap, f"{py_path.name} references {sorted(overlap)}"


def test_the_runner_calls_the_shared_replay_once_with_settlement_and_ledger() -> None:
    calls = _calls(RUNNER_PATH, "run_shared_portfolio_replay")
    assert len(calls) == 1
    keywords = {k.arg: k.value for k in calls[0].keywords if k.arg is not None}
    for name in ("settle_terminal_position", "capture_ledger_evidence"):
        assert isinstance(keywords.get(name), ast.Constant)
        assert keywords[name].value is True
    assert "execution_cost" in keywords and "risk" in keywords
    # nothing else in the package replays
    for py_path in PACKAGE_DIR.glob("*.py"):
        if py_path != RUNNER_PATH:
            assert "run_shared_portfolio_replay" not in _referenced_names(py_path)


def test_the_cell_summary_is_always_called_with_the_execution_cost() -> None:
    """Without ``execution_cost`` the core7 summary cannot value terminal dust."""
    calls = _calls(RUNNER_PATH, "_summarize_replay_cell")
    assert len(calls) == 1
    assert "execution_cost" in {k.arg for k in calls[0].keywords}


def test_one_policy_class_and_the_shared_runner_helpers_are_reused() -> None:
    names: set[str] = set()
    for py_path in PACKAGE_DIR.glob("*.py"):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        names.update(
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        )
    assert not _FOREIGN_POLICIES & names
    assert "XsPolicy" in names
    runner_imports = _imported_roots(RUNNER_PATH)
    assert "trade_rl.evaluation.trend_oos" in runner_imports  # shared runner helpers
    assert "trade_rl.evaluation.core7_mtf_study" in runner_imports
    decision_imports = _imported_roots(PACKAGE_DIR / "decision.py")
    assert "trade_rl.evaluation.trend_oos.decision" in decision_imports  # OosCell
    assert "trade_rl.evaluation.lowturn_screen.decision" in decision_imports  # DSR


def test_the_dataset_is_assembled_once_with_the_shared_market_dataset_contract() -> (
    None
):
    data_path = PACKAGE_DIR / "data.py"
    assert len(_calls(data_path, "MarketDataset")) == 1
    assert "trade_rl.evaluation.core7_mtf_study" in _imported_roots(data_path)


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
                "trade_rl.evaluation.xs_lowvol_funding.runner",
                "trade_rl.evaluation.xs_lowvol_funding.data",
                "trade_rl.evaluation.xs_lowvol_funding.policy",
            )
        )
        for module in decision
    )
    protocol = _imported_roots(PACKAGE_DIR / "protocol.py")
    assert not any(
        module.startswith(
            (
                "trade_rl.data",
                "trade_rl.evaluation.replay",
                "trade_rl.evaluation.xs_lowvol_funding",
            )
        )
        for module in protocol
    )
    universe = _imported_roots(PACKAGE_DIR / "universe.py")
    assert not any(
        module.startswith(
            (
                "trade_rl.data",
                "trade_rl.simulation",
                "trade_rl.risk",
                "trade_rl.strategies",
                "trade_rl.evaluation.replay",
            )
        )
        for module in universe
    )
    policy = _imported_roots(PACKAGE_DIR / "policy.py")
    assert not any(
        module.startswith(
            ("trade_rl.simulation", "trade_rl.evaluation.replay", "trade_rl.risk")
        )
        for module in policy
    )
    data = _imported_roots(PACKAGE_DIR / "data.py")
    assert not any(
        module.startswith(
            (
                "trade_rl.evaluation.replay",
                "trade_rl.evaluation.xs_lowvol_funding.runner",
                "trade_rl.evaluation.xs_lowvol_funding.policy",
            )
        )
        for module in data
    )


def test_only_the_tool_wires_the_market_store_and_the_forward_embargo_policy() -> None:
    tool = _imported_roots(TOOL_PATH)
    assert "trade_rl.integrations.market_store.embargo" in tool
    package_text = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGE_DIR.glob("*.py")
    )
    assert "EmbargoPolicy" not in package_text
    assert "ForwardEmbargoPolicy" in TOOL_PATH.read_text(encoding="utf-8")
    # no spot market is read anywhere
    assert "SPOT" not in TOOL_PATH.read_text(encoding="utf-8")


def test_the_reused_packages_do_not_depend_on_this_package() -> None:
    for name in ("agy_anchor_oos", "trend_oos", "lowturn_screen", "core7_mtf_study"):
        for path in (ROOT / "trade_rl" / "evaluation" / name).glob("*.py"):
            assert "xs_lowvol_funding" not in "\n".join(_imported_roots(path)), path


def test_importing_the_package_does_not_load_torch_or_sb3_and_is_not_a_facade() -> None:
    code = (
        "import sys; import trade_rl.evaluation.xs_lowvol_funding; "
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
    assert not {"run_stage", "OosCell", "StageWindow", "XsPolicy"} & set(
        evaluation.__all__
    )
