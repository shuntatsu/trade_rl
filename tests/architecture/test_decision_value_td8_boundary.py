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


def test_failure_receipt_contract_is_documented_and_exported() -> None:
    lean_core = (DOCS / "architecture" / "lean-core.md").read_text(encoding="utf-8")
    for term in (
        "decision_value_incomplete_pair_attempt_v3",
        "decision_value_incomplete_pair_attempt_v2",
        "load_incomplete_pair_attempt",
        "write_once_no_restart",
        "receipt_digest",
        "fit_admitted=false",
        "causality_probe",
        "observed_view_digest",
    ):
        assert term in lean_core, term
    status = (DOCS / "research" / "current-status.md").read_text(encoding="utf-8")
    assert "Failure-receipt hardening" in status
    assert "decision_value_incomplete_pair_attempt_v3" in status
    assert "decision_value_incomplete_pair_attempt_v2" in status
    source = (EVALUATION / "decision_value_dev_study.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    exported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            assert isinstance(node.value, ast.List)
            exported = {
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant)
            }
    assert "load_incomplete_pair_attempt" in exported
    # The receipt reuses the shared canonical/digest/verified-file helpers rather
    # than a parallel artifact system.
    imports = _imported_modules(EVALUATION / "decision_value_dev_study.py")
    assert {
        "trade_rl.artifacts.canonical",
        "trade_rl.artifacts.hashing",
        "trade_rl.artifacts.verified_file",
    } <= imports


def _function_calls(path: Path, function: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return {
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            }
    raise AssertionError(f"{function} not found")


def test_failure_receipt_writer_and_reader_share_one_validator() -> None:
    """The writer must not duplicate the reader's rules (#1025 receipt Low 1)."""
    source = EVALUATION / "decision_value_dev_study.py"
    # Reader and writer both go through the same decode/validate functions.
    assert "_decode_incomplete_receipt" in _function_calls(
        source, "load_incomplete_pair_attempt"
    )
    assert "_decode_incomplete_receipt" in _function_calls(
        source, "_encode_incomplete_receipt"
    )
    assert "_validate_incomplete_pair_record" in _function_calls(
        source, "_decode_incomplete_receipt"
    )
    assert "_validate_incomplete_pair_record" in _function_calls(
        source, "_receipt_pair"
    )
    # The claim happens only after the receipt is encoded and self-checked.
    persist = ast.parse(source.read_text(encoding="utf-8"))
    body = next(
        node
        for node in ast.walk(persist)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_persist_incomplete_pair_attempt"
    )
    order = [
        child.func.id if isinstance(child.func, ast.Name) else child.func.attr
        for child in ast.walk(body)
        if isinstance(child, ast.Call)
        and isinstance(child.func, (ast.Name, ast.Attribute))
    ]
    assert order.index("_encode_incomplete_receipt") < order.index("mkdir")


def test_failure_receipt_numeric_bounds_and_versions_are_documented() -> None:
    lean_core = (DOCS / "architecture" / "lean-core.md").read_text(encoding="utf-8")
    status = (DOCS / "research" / "current-status.md").read_text(encoding="utf-8")
    for term in (
        "nonfinite:",
        "unvalidated_source_record",
        "16,384",
        "4,000",
        "4,096",
        "v2",
        "timestamps",
        "tradable",
        "feature_available",
        "global_features",
    ):
        assert term in lean_core, term
    assert "nonfinite:" in status
    for text in (lean_core, status):
        # Digests are content digests of package source / runtime manifest, not
        # an unqualified "checkout-independent" guarantee.
        assert "checkout-independent" not in text.split("TD8の比較不能pair")[-1][:6000]
        assert "checkout非依存" not in text
        assert "genuinely insolvent" not in text
