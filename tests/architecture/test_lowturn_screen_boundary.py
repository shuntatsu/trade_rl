"""Architecture and import-boundary tests for the low-turnover family screen."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from trade_rl import evaluation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "trade_rl" / "evaluation" / "lowturn_screen"
RUNNER_PATH = PACKAGE_DIR / "runner.py"

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


def test_lowturn_screen_has_no_network_no_ml_and_no_second_backtester() -> None:
    for py_path in sorted(PACKAGE_DIR.glob("*.py")):
        for module in _imported_roots(py_path):
            root = module.split(".", 1)[0]
            assert root not in _NETWORK_MODULES, f"{py_path.name} imports {module}"
            assert root not in _FORBIDDEN_ML_MODULES, f"{py_path.name} imports {module}"
            assert not module.startswith("trade_rl.integrations"), py_path.name
    for py_path in sorted(PACKAGE_DIR.glob("*.py")):
        overlap = _FORBIDDEN_SECOND_BACKTESTER_NAMES & _referenced_names(py_path)
        assert not overlap, f"{py_path.name} references {sorted(overlap)}"


def test_runner_calls_the_shared_replay_once_with_settlement_and_ledger_evidence() -> (
    None
):
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


def test_layers_keep_their_dependency_direction() -> None:
    signals = _imported_roots(PACKAGE_DIR / "signals.py")
    forbidden = (
        "trade_rl.strategies",
        "trade_rl.simulation",
        "trade_rl.risk",
        "trade_rl.evaluation.replay",
        "trade_rl.evaluation.core7_mtf_study",
    )
    assert not any(module.startswith(forbidden) for module in signals)
    decision = _imported_roots(PACKAGE_DIR / "decision.py")
    assert not any(
        module.startswith(
            (
                "trade_rl.data",
                "trade_rl.simulation",
                "trade_rl.risk",
                "trade_rl.strategies",
            )
        )
        or module in ("trade_rl.evaluation.lowturn_screen.runner",)
        or module.startswith("trade_rl.evaluation.replay")
        for module in decision
    )
    policies = _imported_roots(PACKAGE_DIR / "policies.py")
    assert not any(
        module.startswith(
            ("trade_rl.evaluation.replay", "trade_rl.simulation", "trade_rl.risk")
        )
        for module in policies
    )


def test_importing_the_package_does_not_load_torch_or_sb3_and_is_not_a_facade() -> None:
    code = (
        "import sys; import trade_rl.evaluation.lowturn_screen; "
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
    assert not {"run_screen", "LowturnPolicy", "ScreenCell"} & set(evaluation.__all__)
