from __future__ import annotations

import sys
from pathlib import Path

import trade_rl.evaluation as evaluation
from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
RESIDUAL = PACKAGE / "evaluation" / "residual_rl"
MODULES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "env.py",
    "evaluate.py",
    "market.py",
    "protocol.py",
    "study.py",
    "train.py",
}
ALLOWED_TRADE_RL = (
    "trade_rl.artifacts",
    "trade_rl.evaluation.residual_rl",
    "trade_rl.strategies.allocation.cost_aware",
    "trade_rl.strategies.portfolio",
)
FORBIDDEN_TRADE_RL = (
    "trade_rl.data",
    "trade_rl.integrations",
    "trade_rl.simulation",
    "trade_rl.risk",
    "trade_rl.strategies.rl",
    "trade_rl.evaluation.paper",
    "trade_rl.evaluation.forward_study",
    "trade_rl.evaluation.forward_acquisition",
    "trade_rl.evaluation.final_test",
)
NETWORK_OR_LIVE = {"socket", "ssl", "http", "urllib", "requests", "ftplib", "smtplib"}


def _imports(path: Path) -> set[str]:
    return ImportCollector(PACKAGE).collect(path)


def test_residual_rl_package_layout_is_closed() -> None:
    assert {path.name for path in RESIDUAL.glob("*.py")} == MODULES
    assert not [
        path
        for path in RESIDUAL.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    ]


def test_residual_rl_is_not_part_of_the_evaluation_public_api() -> None:
    assert "residual_rl" not in " ".join(evaluation.__all__)
    init_text = (PACKAGE / "evaluation" / "__init__.py").read_text(encoding="utf-8")
    assert "residual_rl" not in init_text


def test_residual_rl_reads_no_dataset_network_or_live_source() -> None:
    for path in sorted(RESIDUAL.glob("*.py")):
        imported = _imports(path)
        trade_rl_imports = {n for n in imported if within_module(n, "trade_rl")}
        for name in trade_rl_imports:
            assert any(within_module(name, ok) for ok in ALLOWED_TRADE_RL), (
                path.name,
                name,
            )
            assert not any(within_module(name, bad) for bad in FORBIDDEN_TRADE_RL), (
                path.name,
                name,
            )
        top_level = {name.split(".", 1)[0] for name in imported}
        assert not top_level & NETWORK_OR_LIVE, path.name


def test_only_numpy_and_gymnasium_are_imported_statically() -> None:
    for path in sorted(RESIDUAL.glob("*.py")):
        third_party = {
            name.split(".", 1)[0]
            for name in _imports(path)
            if name.split(".", 1)[0] not in sys.stdlib_module_names
            and not within_module(name, "trade_rl")
        }
        assert third_party <= {"numpy", "gymnasium"}, (path.name, third_party)


def test_sb3_and_torch_are_loaded_lazily_and_only_by_train() -> None:
    for path in sorted(RESIDUAL.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        mentions = "stable_baselines3" in text or '"torch"' in text
        assert mentions == (path.name == "train.py"), path.name
    train = (RESIDUAL / "train.py").read_text(encoding="utf-8")
    assert "import_module" in train
    assert "from stable_baselines3" not in train and "import torch" not in train


def test_study_artifacts_are_lf_only() -> None:
    for path in sorted(
        (ROOT / "report").glob("issue810-residual-rl-synthetic-20261030*.json")
    ):
        assert b"\r" not in path.read_bytes(), path.name


def test_residual_rl_is_documented_as_synthetic_only() -> None:
    boundaries = (ROOT / "docs" / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    assert "evaluation/residual_rl/" in boundaries
    assert "never a profitability claim" in boundaries
    status = (ROOT / "docs" / "research" / "current-status.md").read_text(
        encoding="utf-8"
    )
    assert "synthetic residual-PPO over the cost-aware allocator" in status
    assert (
        "real-data RL remains blocked behind the fee-only learnability check" in status
    )
