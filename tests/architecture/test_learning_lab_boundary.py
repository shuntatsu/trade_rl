"""Architecture and import-boundary tests for the learning-lab harness."""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

from trade_rl import evaluation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "trade_rl" / "evaluation" / "learning_lab"
TOOL_PATH = ROOT / "tools" / "learning_lab.py"
MODULES = {
    "__init__.py",
    "splits.py",
    "metrics.py",
    "registry.py",
    "synthetic.py",
    "overlay_env.py",
    "baselines.py",
    "agents.py",
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
_ML_MODULES = frozenset({"torch", "stable_baselines3", "sb3_contrib", "lightgbm"})
#: A second backtester would need these; the engine is reached only through the session
#: seam and the shared replay.
_FORBIDDEN_SECOND_BACKTESTER_NAMES = frozenset(
    {
        "BookState",
        "MarketExecutor",
        "execute_native_target_transition",
        "execute_stateful_orders",
    }
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _names(path: Path) -> set[str]:
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


def _calls(path: Path, attribute: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    )


def test_the_package_layout_is_exactly_the_documented_modules() -> None:
    assert {path.name for path in PACKAGE_DIR.glob("*.py")} == MODULES


def test_no_network_no_store_and_no_second_backtester() -> None:
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        for module in _imports(path):
            root = module.split(".", 1)[0]
            assert root not in _NETWORK_MODULES, f"{path.name} imports {module}"
            assert not module.startswith("trade_rl.integrations"), path.name
            assert not module.startswith("tools"), path.name
        overlap = _FORBIDDEN_SECOND_BACKTESTER_NAMES & _names(path)
        assert not overlap, f"{path.name} references {sorted(overlap)}"


def test_ml_runtimes_are_loaded_lazily_and_only_by_the_agent_module() -> None:
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        imported = {module.split(".", 1)[0] for module in _imports(path)}
        assert not imported & _ML_MODULES, f"{path.name} imports an ML runtime"
    agent_text = (PACKAGE_DIR / "agents.py").read_text(encoding="utf-8")
    assert "importlib.import_module" in agent_text
    code = (
        "import sys, importlib, pkgutil; "
        "import trade_rl.evaluation.learning_lab as p; "
        "[importlib.import_module(f'{p.__name__}.{m.name}') "
        "for m in pkgutil.iter_modules(p.__path__)]; "
        "bad = [m for m in ('torch', 'stable_baselines3', 'sb3_contrib', 'lightgbm') "
        "if m in sys.modules]; assert not bad, bad"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_the_environment_steps_the_engine_session_and_owns_no_pnl() -> None:
    env = PACKAGE_DIR / "overlay_env.py"
    imported = _imports(env)
    assert "trade_rl.simulation.stateful.portfolio_session" in imported
    assert "trade_rl.simulation.objectives.contract" in imported
    assert "trade_rl.strategies.portfolio_observation" in imported
    assert _calls(env, "advance") == 1
    assert _calls(env, "settle_next_interval") == 1
    names = _names(env)
    assert "net_equity_increment" in names and "SharedPortfolioSession" in names
    # the reward is never computed from prices, quantities or fills here
    for forbidden in ("mark_prices", "exact_quantities", "interval_cost"):
        assert forbidden not in names


def test_baselines_replay_through_the_shared_engine_replay_only() -> None:
    baselines = PACKAGE_DIR / "baselines.py"
    assert "trade_rl.evaluation.replay" in _imports(baselines)
    tree = ast.parse(baselines.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_shared_portfolio_replay"
    ]
    assert len(calls) == 1
    # the anchor and its cash-account scale are reused, never redefined
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert not {"AgyAnchorPolicy", "AnchorCashAccountPolicy"} & classes
    assert "trade_rl.evaluation.agy_anchor_oos.cash" in _imports(baselines)
    assert "trade_rl.strategies.agy_anchor.policy" in _imports(baselines)


def test_pure_modules_stay_pure() -> None:
    forbidden = (
        "trade_rl.data",
        "trade_rl.simulation",
        "trade_rl.risk",
        "trade_rl.strategies",
        "trade_rl.evaluation",
        "gymnasium",
    )
    for name in ("splits.py", "registry.py", "metrics.py"):
        for module in _imports(PACKAGE_DIR / name):
            assert not module.startswith(forbidden), f"{name} imports {module}"
    registry_roots = {m.split(".", 1)[0] for m in _imports(PACKAGE_DIR / "registry.py")}
    assert registry_roots <= set(sys.stdlib_module_names) | {"__future__"}


def test_the_registry_is_only_ever_opened_for_append() -> None:
    text = (PACKAGE_DIR / "registry.py").read_text(encoding="utf-8")
    assert "O_APPEND" in text
    assert "write_text" not in text and "write_bytes" not in text
    assert "open(" not in text.replace("os.open(", "")


def test_the_real_data_gate_lives_only_in_the_tool_and_uses_the_g4_verifier() -> None:
    package_text = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGE_DIR.glob("*.py")
    )
    assert "fetch_comment" not in package_text and "G4AuthError" not in package_text
    assert "core7_g4_comment" in _names(TOOL_PATH)
    tool_names = _names(TOOL_PATH)
    assert {
        "fetch_comment",
        "require_protocol_digest",
        "require_protocol_id",
    } <= tool_names
    assert "confirm_unedited" in tool_names
    assert "--authorization-url" in TOOL_PATH.read_text(encoding="utf-8")


def test_nothing_outside_the_package_depends_on_it() -> None:
    for path in (ROOT / "trade_rl").rglob("*.py"):
        if PACKAGE_DIR in path.parents:
            continue
        assert not any(
            module.startswith("trade_rl.evaluation.learning_lab")
            for module in _imports(path)
        ), path.relative_to(ROOT)
    assert "learning_lab" not in set(evaluation.__all__)
    assert not any(name.startswith("Overlay") for name in evaluation.__all__)


def test_the_learning_lab_extra_pins_exact_versions_of_the_training_stack() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    extra = project["optional-dependencies"]["learning-lab"]
    names = {requirement.split("==")[0].split("[")[0] for requirement in extra}
    assert names == {"stable-baselines3", "sb3-contrib", "torch", "lightgbm"}
    assert all("==" in requirement for requirement in extra)
    train = project["optional-dependencies"]["train-sb3"]
    pinned = {r.split("==")[0]: r.split("==")[1] for r in train}
    for requirement in extra:
        name, _, version = requirement.partition("==")
        if name in pinned:
            assert pinned[name] == version  # one training stack, not two
    assert not any(r.startswith("sb3-contrib") for r in project["dependencies"])
