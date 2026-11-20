"""Architecture and import-boundary tests for the Core7 MTF Shadow-Anchor PPO stack."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from trade_rl import evaluation, strategies
from trade_rl.data.features import mtf6_signals
from trade_rl.evaluation.core7_mtf_study import protocol
from trade_rl.strategies import rl as strategies_rl

ROOT = Path(__file__).resolve().parents[2]
MTF6_SIGNALS_PATH = ROOT / "trade_rl" / "data" / "features" / "mtf6_signals.py"
SHADOW_ANCHOR_PATH = ROOT / "trade_rl" / "strategies" / "rl" / "shadow_anchor_mtf.py"
STUDY_DIR = ROOT / "trade_rl" / "evaluation" / "core7_mtf_study"
RUNNER_PATH = STUDY_DIR / "runner.py"

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
            for alias in node.names:
                modules.add(alias.name)
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


def test_mtf6_signals_stays_within_data_layer() -> None:
    imports = _imported_roots(MTF6_SIGNALS_PATH)
    forbidden_prefixes = (
        "trade_rl.strategies",
        "trade_rl.evaluation",
        "trade_rl.simulation",
        "trade_rl.integrations",
        "trade_rl.risk",
    )
    for mod in imports:
        assert not mod.startswith(forbidden_prefixes), f"forbidden data import: {mod}"
        assert mod.split(".", 1)[0] not in _FORBIDDEN_ML_MODULES
        assert mod.split(".", 1)[0] not in _NETWORK_MODULES


def test_shadow_anchor_mtf_stays_within_strategy_layer_and_pure_numpy() -> None:
    imports = _imported_roots(SHADOW_ANCHOR_PATH)
    forbidden_prefixes = (
        "trade_rl.strategies.forecasts",
        "trade_rl.evaluation",
        "trade_rl.simulation",
        "trade_rl.integrations",
        "trade_rl.risk",
    )
    for mod in imports:
        assert not mod.startswith(forbidden_prefixes), (
            f"forbidden strategy import: {mod}"
        )
        assert mod.split(".", 1)[0] not in _FORBIDDEN_ML_MODULES
        assert mod.split(".", 1)[0] not in _NETWORK_MODULES


def test_core7_mtf_study_no_network_and_no_second_backtester() -> None:
    for py_path in sorted(STUDY_DIR.glob("*.py")):
        imports = _imported_roots(py_path)
        for mod in imports:
            assert mod.split(".", 1)[0] not in _NETWORK_MODULES, (
                f"{py_path.name} imports network module {mod}"
            )
            assert mod.split(".", 1)[0] not in _FORBIDDEN_ML_MODULES

    runner_names = _referenced_names(RUNNER_PATH)
    overlap = _FORBIDDEN_SECOND_BACKTESTER_NAMES & runner_names
    assert not overlap, f"runner references direct simulation types: {sorted(overlap)}"

    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    replay_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_shared_portfolio_replay"
    ]
    assert len(replay_calls) == 1
    kw = {k.arg: k.value for k in replay_calls[0].keywords if k.arg is not None}
    assert isinstance(kw.get("settle_terminal_position"), ast.Constant)
    assert kw["settle_terminal_position"].value is True
    assert isinstance(kw.get("capture_ledger_evidence"), ast.Constant)
    assert kw["capture_ledger_evidence"].value is True


def test_importing_core7_modules_does_not_load_torch_or_sb3() -> None:
    data_code = (
        "import sys; "
        "import trade_rl.data.features.mtf6_signals; "
        "bad = [m for m in ('torch', 'stable_baselines3', 'gymnasium') if m in sys.modules]; "
        "assert not bad, bad"
    )
    proc_data = subprocess.run(
        [sys.executable, "-c", data_code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_data.returncode == 0, proc_data.stderr or proc_data.stdout

    code = (
        "import sys; "
        "import trade_rl.strategies.rl.shadow_anchor_mtf; "
        "import trade_rl.evaluation.core7_mtf_study; "
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


def test_data_end_exclusive_and_public_facades_unchanged() -> None:
    assert mtf6_signals.DATA_END_EXCLUSIVE_ISO == protocol.DATA_END_EXCLUSIVE
    assert mtf6_signals.DATA_END_EXCLUSIVE_ISO == "2026-08-01T00:00:00"

    new_names = {
        "Mtf6SignalPack",
        "ShadowAnchorMlpWeights",
        "ShadowAnchorMtfConfig",
        "ShadowAnchorMtfPolicy",
        "CashHoldPortfolioPolicy",
        "build_mtf6_signal_pack",
        "run_study",
    }
    assert not (new_names & set(strategies.__all__))
    assert not (new_names & set(strategies_rl.__all__))
    assert not (new_names & set(evaluation.__all__))
