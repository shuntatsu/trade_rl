"""Architecture contract for the SD2 signal-diagnosis runner (#1146)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
RUN = PACKAGE / "evaluation" / "signal_diagnosis_run"
DOCS = ROOT / "docs"
MODULES = (
    "__init__.py",
    "__main__.py",
    "cli.py",
    "fitting.py",
    "leakage.py",
    "measure.py",
    "panel.py",
    "runner.py",
)
FORBIDDEN_TOP_LEVEL = {
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "stable_baselines3",
    "socket",
    "ssl",
    "urllib",
    "http",
    "requests",
    "sqlite3",
    "psycopg",
    "shutil",
}
FORBIDDEN_TEXT = (
    "market_store",
    "final_test",
    "forward_acquisition",
    "forward_study",
    "integrations.binance.transport",
    "urlopen",
)
# Only these modules touch the filesystem or the Git process boundary.
IO_MODULES = {"cli.py", "runner.py"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_package_layout_is_the_eight_modules_only() -> None:
    assert {path.name for path in RUN.glob("*.py")} == set(MODULES)
    assert not [
        path for path in RUN.rglob("*") if path.is_dir() and path.name != "__pycache__"
    ]


def test_no_network_store_final_or_forward_study_dependency() -> None:
    for name in MODULES:
        path = RUN / name
        top = {item.split(".", 1)[0] for item in _imports(path)}
        assert not top & FORBIDDEN_TOP_LEVEL, (name, top & FORBIDDEN_TOP_LEVEL)
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_TEXT:
            assert forbidden not in text, (name, forbidden)
        if name not in IO_MODULES:
            assert "subprocess" not in top, name
            for io_call in ("open(", "read_bytes", "read_text", "write_text"):
                assert io_call not in text, (name, io_call)


def test_only_the_leakage_module_reaches_the_binance_feature_contract() -> None:
    for name in MODULES:
        trade_rl = {i for i in _imports(RUN / name) if i.startswith("trade_rl.integr")}
        if name == "leakage.py":
            assert trade_rl == {"trade_rl.integrations.binance.dataset"}
        else:
            assert not trade_rl, name


def test_no_other_package_depends_on_the_runner() -> None:
    for path in PACKAGE.rglob("*.py"):
        if RUN in path.parents:
            continue
        assert "signal_diagnosis_run" not in path.read_text(encoding="utf-8"), (
            path.relative_to(ROOT)
        )


def test_the_runner_reads_every_threshold_from_the_protocol() -> None:
    text = "\n".join((RUN / name).read_text(encoding="utf-8") for name in MODULES)
    for literal in ("0.0014", "14 /", "2000", "block_days=20", "q = 0.1", "seed=0"):
        assert literal not in text, literal


def test_current_docs_record_the_runner_and_its_authorization_boundary() -> None:
    boundaries = (DOCS / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    assert (
        "signal_diagnosis_run/{__init__.py,__main__.py,cli.py,fitting.py,"
        "leakage.py,measure.py,panel.py,runner.py}" in boundaries
    )
    section = boundaries[boundaries.index("## Signal diagnosis run boundary") :]
    for required in (
        "G4 authorization",
        "dirty",
        "single run",
        "reads no market value",
        "no other package imports it",
        "before the loader parses any array",
    ):
        assert required in section, required
