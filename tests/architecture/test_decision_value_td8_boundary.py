"""Architecture contract for the corrected decision-value (TD8) design (#1025)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "trade_rl" / "evaluation"
STRATEGIES = ROOT / "trade_rl" / "strategies"
DOCS = ROOT / "docs"

NEW_EVALUATION_MODULES = (
    "decision_value_ledger_oracle.py",
    "decision_value_feature_transform.py",
    "decision_value_reachable_labels.py",
    "decision_value_margin.py",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_value_policies_stay_free_of_evaluation_and_replay_imports() -> None:
    for name in ("decision_value.py", "decision_value_shallow.py"):
        imports = _imported_modules(STRATEGIES / name)
        offenders = {
            module
            for module in imports
            if module.startswith(("trade_rl.evaluation", "trade_rl.simulation"))
        }
        assert not offenders, (name, offenders)


def test_new_study_modules_exist_and_are_documented_in_package_boundaries() -> None:
    boundaries = (DOCS / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    for name in NEW_EVALUATION_MODULES:
        assert (EVALUATION / name).is_file(), name
        assert name in boundaries, name


def test_study_runner_uses_the_new_modules_and_the_reference_stays_in_the_study() -> (
    None
):
    imports = _imported_modules(EVALUATION / "decision_value_dev_study.py")
    for name in NEW_EVALUATION_MODULES:
        assert f"trade_rl.evaluation.{name[:-3]}" in imports, name
    # The label and margin modules receive policies from the caller; they must
    # not import the study runner (no cycle, no hidden B0 dependency).
    for name in ("decision_value_reachable_labels.py", "decision_value_margin.py"):
        assert "trade_rl.evaluation.decision_value_dev_study" not in _imported_modules(
            EVALUATION / name
        ), name


def test_current_docs_record_the_corrected_design_and_the_withdrawn_calibration() -> (
    None
):
    lean_core = (DOCS / "architecture" / "lean-core.md").read_text(encoding="utf-8")
    for term in (
        "TD8補正設計",
        "到達可能状態ラベル",
        "HOLD相対action rule",
        "事前固定のmargin選択",
        "入力特徴の一様変換",
        "事後prior撤回",
        "切り詰めprefix近似",
    ):
        assert term in lean_core, term
    status = (DOCS / "research" / "current-status.md").read_text(encoding="utf-8")
    for term in (
        "TD8: Corrected decision-value design",
        "That calibration is withdrawn",
        "exploratory re-run on consumed Tier D development data",
        "NOT ESTABLISHED",
        "truncated-prefix approximation",
        "clip rule was chosen after the diagnosis",
    ):
        assert term in status, term
