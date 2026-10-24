from __future__ import annotations

from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def test_attempt_api_is_owned_by_the_runs_facade() -> None:
    import trade_rl.evaluation as evaluation
    import trade_rl.evaluation.runs as runs
    from trade_rl.evaluation.runs import ridge_comparison_attempt as owner

    assert runs.run_ridge_comparison_attempt is owner.run_ridge_comparison_attempt
    assert runs.PublishedRidgeComparisonAttempt is owner.PublishedRidgeComparisonAttempt
    assert not hasattr(evaluation, "run_ridge_comparison_attempt")


def test_attempt_owner_does_not_cross_into_experiment_or_artifact_codecs() -> None:
    owner_path = PACKAGE / "evaluation" / "runs" / "ridge_comparison_attempt.py"
    imports = ImportCollector(PACKAGE).collect_direct(owner_path)

    assert not any(
        within_module(imported, "trade_rl.evaluation.experiments")
        for imported in imports
    )
    assert "trade_rl.evaluation.runs.artifact" not in imports
    assert "trade_rl.evaluation.runs.ridge_comparison_artifact" in imports
