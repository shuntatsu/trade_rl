"""Architecture contract for the SD0 signal-diagnosis protocol (#1146)."""

from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.evaluation as evaluation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
SIGNAL = PACKAGE / "evaluation" / "signal_diagnosis"
DOCS = ROOT / "docs"
MODULES = ("__init__.py", "protocol.py", "verdict.py")

ALLOWED_TRADE_RL_IMPORTS = {
    "protocol.py": {
        "trade_rl.artifacts.canonical",
        "trade_rl.evaluation.forward_study.registry",
    },
    "verdict.py": {"trade_rl.evaluation.signal_diagnosis.protocol"},
}
FORBIDDEN_TOP_LEVEL = {
    "numpy",
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


def test_package_layout_is_protocol_and_pure_verdict_only() -> None:
    assert {path.name for path in SIGNAL.glob("*.py")} == set(MODULES)
    assert not [
        path
        for path in SIGNAL.rglob("*")
        if path.is_dir() and path.name != "__pycache__"
    ]
    assert "signal_diagnosis" not in evaluation.__all__


def test_modules_are_pure_no_dataset_network_filesystem_or_integration_imports() -> (
    None
):
    for name in MODULES:
        path = SIGNAL / name
        imports = _imports(path)
        top = {item.split(".", 1)[0] for item in imports}
        assert not top & FORBIDDEN_TOP_LEVEL, (name, top & FORBIDDEN_TOP_LEVEL)
        trade_rl = {item for item in imports if item.split(".", 1)[0] == "trade_rl"}
        assert trade_rl == ALLOWED_TRADE_RL_IMPORTS.get(name, set()), (name, trade_rl)
        text = path.read_text(encoding="utf-8")
        # The store, FS1 scoring and the final-test chain are boundaries, not inputs.
        for forbidden in (
            "market_store",
            "final_test",
            "integrations",
            "open(",
            "read_bytes",
            "read_text",
        ):
            assert forbidden not in text, (name, forbidden)


def test_no_runner_cli_or_main_entrypoint_exists_in_sd0() -> None:
    assert not (SIGNAL / "__main__.py").exists()
    assert not (SIGNAL / "cli.py").exists()
    for path in SIGNAL.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "argparse" not in text and "__main__" not in text, path.name
        assert "def main" not in text, path.name


def test_other_packages_do_not_depend_on_the_signal_diagnosis_protocol_yet() -> None:
    for root in (
        PACKAGE / "data",
        PACKAGE / "simulation",
        PACKAGE / "strategies",
        PACKAGE / "integrations",
        PACKAGE / "risk",
        PACKAGE / "evaluation" / "forward_study",
        PACKAGE / "evaluation" / "forward_acquisition",
        PACKAGE / "evaluation" / "dev_realdata_study",
        PACKAGE / "evaluation" / "final_test",
    ):
        for path in root.rglob("*.py"):
            assert "signal_diagnosis" not in path.read_text(encoding="utf-8"), (
                path.relative_to(ROOT)
            )


def test_current_docs_record_the_protocol_and_the_planned_not_executed_status() -> None:
    boundaries = (DOCS / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    assurance = (DOCS / "architecture" / "research-assurance.md").read_text(
        encoding="utf-8"
    )
    status = (DOCS / "research" / "current-status.md").read_text(encoding="utf-8")
    from trade_rl.evaluation.signal_diagnosis.protocol import (
        default_signal_diagnosis_protocol,
        protocol_digest,
    )

    digest = protocol_digest(default_signal_diagnosis_protocol())

    assert "evaluation/signal_diagnosis/" in boundaries
    assert (
        "report/issue1146-sd0-signal-diagnosis-protocol-v2-20261110.json" in boundaries
    )
    assert "signal_diagnosis/{__init__.py,protocol.py,verdict.py}" in boundaries
    assert "## Research-specific contract: signal diagnosis SD0" in assurance
    for heading in (
        "### G0 — SD0 question, hypotheses, and falsifiers",
        "### G1 — SD0 fixed mechanism and data scope",
        "### G2 — SD0 independent oracles and counterexamples",
    ):
        assert heading in assurance, heading
    assert digest in assurance
    section = status[status.index("## Issue #1146") :]
    for required in (
        "planned, not executed",
        "known-used",
        "post hoc",
        "no result",
        "SD2",
        "G4 lease",
        digest,
    ):
        assert required in section, required
