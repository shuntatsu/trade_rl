"""Architecture contract for the SD1 signal-measurement tools (#1146)."""

from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.evaluation as evaluation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
MEASURE = PACKAGE / "evaluation" / "signal_measurement"
DOCS = ROOT / "docs"
MODULES = (
    "__init__.py",
    "ic.py",
    "leakage.py",
    "multitest.py",
    "payoff.py",
    "profile.py",
    "significance.py",
    "targets.py",
)

_IC = "trade_rl.evaluation.signal_measurement.ic"
ALLOWED_TRADE_RL_IMPORTS = {
    "__init__.py": {
        _IC,
        "trade_rl.evaluation.signal_measurement.leakage",
        "trade_rl.evaluation.signal_measurement.multitest",
        "trade_rl.evaluation.signal_measurement.payoff",
        "trade_rl.evaluation.signal_measurement.profile",
        "trade_rl.evaluation.signal_measurement.significance",
        "trade_rl.evaluation.signal_measurement.targets",
    },
    "ic.py": set(),
    "leakage.py": {_IC},
    "payoff.py": {_IC},
    "multitest.py": {"trade_rl.evaluation.comparison.bootstrap", _IC},
    "profile.py": {_IC},
    "significance.py": {_IC},
    "targets.py": {_IC},
}
FORBIDDEN_TOP_LEVEL = {
    "pandas",
    "scipy",
    "sklearn",
    "lightgbm",
    "torch",
    "stable_baselines3",
    "socket",
    "ssl",
    "urllib",
    "http",
    "requests",
    "subprocess",
    "sqlite3",
    "psycopg",
    "os",
    "shutil",
    "pathlib",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_package_layout_is_the_eight_pure_modules_only() -> None:
    assert {path.name for path in MEASURE.glob("*.py")} == set(MODULES)
    assert not [
        path
        for path in MEASURE.rglob("*")
        if path.is_dir() and path.name != "__pycache__"
    ]
    assert "signal_measurement" not in evaluation.__all__


def test_modules_are_pure_numpy_with_no_dataset_network_or_filesystem_access() -> None:
    for name in MODULES:
        path = MEASURE / name
        imports = _imports(path)
        top = {item.split(".", 1)[0] for item in imports}
        assert not top & FORBIDDEN_TOP_LEVEL, (name, top & FORBIDDEN_TOP_LEVEL)
        trade_rl = {item for item in imports if item.split(".", 1)[0] == "trade_rl"}
        assert trade_rl == ALLOWED_TRADE_RL_IMPORTS[name], (name, trade_rl)
        text = path.read_text(encoding="utf-8")
        for forbidden in (
            "market_store",
            "final_test",
            "forward_study",
            "integrations",
            "signal_diagnosis",
            "open(",
            "read_bytes",
            "read_text",
        ):
            assert forbidden not in text, (name, forbidden)


def test_no_runner_cli_or_main_entrypoint_exists_in_sd1() -> None:
    assert not (MEASURE / "__main__.py").exists()
    assert not (MEASURE / "cli.py").exists()
    for path in MEASURE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "argparse" not in text and "__main__" not in text, path.name
        assert "def main" not in text, path.name


def test_no_other_package_depends_on_the_measurement_tools_yet() -> None:
    for path in PACKAGE.rglob("*.py"):
        if MEASURE in path.parents:
            continue
        assert "signal_measurement" not in path.read_text(encoding="utf-8"), (
            path.relative_to(ROOT)
        )


def test_current_docs_record_the_package_and_that_it_is_synthetic_only() -> None:
    boundaries = (DOCS / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )

    assert (
        "signal_measurement/{__init__.py,ic.py,leakage.py,multitest.py,payoff.py,profile.py,significance.py,targets.py}"
        in boundaries
    )
    section = boundaries[boundaries.index("## Signal measurement boundary") :]
    for required in (
        "synthetic",
        "demean",
        "not beta removal",
        "SD2",
        "reads no dataset",
    ):
        assert required in section, required
