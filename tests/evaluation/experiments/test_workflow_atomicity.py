from __future__ import annotations

from pathlib import Path

import pytest

from tests.evaluation.experiments.test_workflow import _created_study
from trade_rl.evaluation.experiments import inspect_study, run_baseline


def test_baseline_analysis_failure_publishes_no_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import workflow as workflow_module

    root, dataset_root, _ = _created_study(tmp_path, monkeypatch)

    def fail_analysis(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic analysis failure")

    monkeypatch.setattr(workflow_module, "analyze_evidence_set", fail_analysis)

    with pytest.raises(RuntimeError, match="synthetic analysis failure"):
        run_baseline(root, dataset_root=dataset_root)

    assert not (root / "baseline").exists()
    assert inspect_study(root).baseline is None
