from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from research.issue519_portable_exp002 import preregister, verify_prereg
from trade_rl.evaluation.experiments import inspect_study


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".mutation.lock"
    }


def test_preregister_only_adds_result_blind_experiment_0002_definition(
    tmp_path: Path,
) -> None:
    root = Path(os.environ["ISSUE519_RECOVERED_ROOT"])
    exp1_root = root / "study/experiments/0001"
    before = _tree_manifest(exp1_root)

    index = preregister(root, issue_number=519)

    assert _tree_manifest(exp1_root) == before
    snapshot = inspect_study(root / "study")
    assert snapshot.experiment_sequences == (1, 2)
    assert snapshot.terminal_sequences == (1,)
    assert not snapshot.frozen

    exp2_root = root / "study/experiments/0002"
    assert (exp2_root / "definition.json").is_file()
    assert not (exp2_root / "candidate").exists()
    assert not (exp2_root / "verification.json").exists()
    assert not (exp2_root / "comparison.json").exists()
    assert not (exp2_root / "decision.json").exists()

    assert index["schema_version"] == "canonical_m2_portable_exp002_prereg_index_v1"
    assert index["issue_number"] == 519
    assert index["experiment_sequence"] == 2
    assert index["factor"] == "RULE_SIGNAL"
    assert index["resolved_changed_paths"] == ["signal_index", "signal_name"]
    assert index["baseline_signal_name"] == "1h__log_return_24bar"
    assert index["baseline_signal_index"] == 2
    assert index["candidate_signal_name"] == "1d__log_return_1bar"
    assert index["candidate_signal_index"] == 114
    assert index["candidate_executed"] is False
    assert index["result_inspected_before_preregistration"] is False
    assert index["daily_refresh_counts"] == {
        "ADAUSDT": 731,
        "BNBUSDT": 731,
        "BTCUSDT": 731,
        "ETHUSDT": 731,
        "XRPUSDT": 731,
    }
    assert index["baseline_intent_transition_median"] == 1601.0
    assert index["candidate_intent_transition_median"] == 423.0

    stored = json.loads(
        (root / "portable-exp002-prereg-index.json").read_text(encoding="utf-8")
    )
    assert stored == index

    report_root = tmp_path / "verify"
    report = verify_prereg(root, report_root)
    assert report["verification_passed"] is True
    assert report["definition_is_result_blind"] is True
    assert report["candidate_absent"] is True
    assert (report_root / "exp002-prereg-verification.json").is_file()
