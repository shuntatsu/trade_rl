from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from trade_rl.release.selection_authorization import SelectionProposal
from trade_rl.simulation.runtime_promotion import (
    ExecutionPromotionEvidence,
    RuntimeMode,
    build_execution_promotion_report,
    load_execution_promotion_report,
    write_execution_promotion_report,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"


def _state_module():
    sys.path.insert(0, str(EXAMPLE_ROOT))
    return importlib.import_module("run_full_research_state")


def _report(*, allowed: bool = True):
    return build_execution_promotion_report(
        requested=RuntimeMode.DUAL_SHADOW,
        evidence=ExecutionPromotionEvidence(
            capability_passed=allowed,
            causal_bridge_passed=allowed,
            funding_passed=allowed,
            terminal_flat_passed=allowed,
            exact_parity_passed=False,
            determinism_passed=False,
            performance_approved=False,
        ),
    )


def _proposal(*, runtime_digest: str | None) -> SelectionProposal:
    return SelectionProposal.create(
        walk_forward_run_digest="1" * 64,
        gate_evidence_digest="2" * 64,
        execution_sensitivity_digest="3" * 64,
        dataset_id="4" * 64,
        selected_configuration="candidate-a",
        candidate_config_digest="5" * 64,
        seeds=(7, 11),
        git_commit="a" * 40,
        dependency_digest="6" * 64,
        resume_checkpoint_digests=(),
        runtime_promotion_report_digest=runtime_digest,
    )


def test_retain_runtime_promotion_report_copies_allowed_evidence(tmp_path: Path) -> None:
    module = _state_module()
    report = _report()
    source = write_execution_promotion_report(tmp_path / "source.json", report)
    work_root = tmp_path / "generation"
    work_root.mkdir()

    retained = module._retain_runtime_promotion_report(
        str(source),
        work_root=work_root,
    )

    assert retained == report
    retained_path = work_root / "runtime-promotion-report.json"
    assert retained_path.is_file()
    assert load_execution_promotion_report(retained_path) == report


def test_retain_runtime_promotion_report_rejects_denied_evidence(tmp_path: Path) -> None:
    module = _state_module()
    report = _report(allowed=False)
    source = write_execution_promotion_report(tmp_path / "source.json", report)
    work_root = tmp_path / "generation"
    work_root.mkdir()

    with pytest.raises(ValueError, match="runtime promotion report is not allowed"):
        module._retain_runtime_promotion_report(
            str(source),
            work_root=work_root,
        )

    assert not (work_root / "runtime-promotion-report.json").exists()


def test_retained_runtime_promotion_must_match_signed_selection(tmp_path: Path) -> None:
    module = _state_module()
    report = _report()
    source = write_execution_promotion_report(tmp_path / "source.json", report)
    work_root = tmp_path / "generation"
    work_root.mkdir()
    module._retain_runtime_promotion_report(str(source), work_root=work_root)

    assert (
        module._require_retained_runtime_promotion(
            _proposal(runtime_digest=report.digest),
            work_root=work_root,
        )
        == report
    )

    with pytest.raises(
        ValueError,
        match="selection proposal runtime promotion report digest mismatch",
    ):
        module._require_retained_runtime_promotion(
            _proposal(runtime_digest="f" * 64),
            work_root=work_root,
        )


def test_unsigned_retained_runtime_promotion_is_rejected(tmp_path: Path) -> None:
    module = _state_module()
    report = _report()
    source = write_execution_promotion_report(tmp_path / "source.json", report)
    work_root = tmp_path / "generation"
    work_root.mkdir()
    module._retain_runtime_promotion_report(str(source), work_root=work_root)

    with pytest.raises(
        ValueError,
        match="selection proposal does not authorize runtime promotion evidence",
    ):
        module._require_retained_runtime_promotion(
            _proposal(runtime_digest=None),
            work_root=work_root,
        )


def test_full_research_source_binds_and_rechecks_runtime_promotion() -> None:
    state = (EXAMPLE_ROOT / "run_full_research_state.py").read_text(encoding="utf-8")

    assert "runtime_promotion_report_digest=" in state
    assert "_require_retained_runtime_promotion(proposal" in state
    assert 'parser.add_argument("--runtime-promotion-report")' in state
