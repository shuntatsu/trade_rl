"""Source-authorization regression for the corrected TD8 study runner (#1025)."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from tests.evaluation.test_decision_value_dev_study import _compact_market
from trade_rl.evaluation import decision_value_dev_study as study


def test_historical_real_td8_authorization_does_not_authorize_revised_code() -> None:
    """The original SHA-approved run is immutable; revised software is NOT G4."""
    protocol = study.default_2024q1_decision_value_dev_protocol()
    assert (
        protocol.to_mapping()["schema_version"]
        == "decision_value_dev_study_protocol_v2"
    )
    assert (
        protocol.protocol_digest
        == "e7ec28ae162a14849002df769164519edd430f32d40014ee8d673929543c8cd7"
    )
    fake = replace(_compact_market(100), dataset_id=protocol.dataset_id)
    with patch.object(
        study, "fit_ridge_forecast", side_effect=AssertionError("FITTED")
    ):
        with pytest.raises(ValueError, match="independent|authorization|G0|research"):
            study.run_decision_value_dev_study(
                fake,
                protocol,
                actual_artifact_digest=protocol.expected_artifact_digest,
            )


def test_corrected_td8_requires_synthetic_plan_and_verified_artifact_before_fit() -> (
    None
):
    """Synthetic provenance and published artifact are admission conditions."""
    original = study.default_2024q1_decision_value_dev_protocol()
    synthetic = _compact_market(90).with_content_identity(
        {"generator": "unit_test_fixed_seed", "synthetic_only": True}
    )
    protocol = replace(
        original,
        dataset_id=synthetic.dataset_id,
        expected_artifact_digest="e" * 64,
        study_plan=replace(
            original.study_plan,
            data_scope="synthetic_only",
            real_scope_authorization_digest=None,
        ),
    )
    assert (
        protocol.to_mapping()["schema_version"]
        == "decision_value_dev_study_protocol_v3_software"
    )
    with patch.object(
        study, "fit_ridge_forecast", side_effect=AssertionError("FITTED")
    ):
        with pytest.raises(ValueError, match="artifact_root|published|verified"):
            study.run_decision_value_dev_study(
                synthetic,
                protocol,
                actual_artifact_digest=protocol.expected_artifact_digest,
            )


def test_corrected_td8_rejects_canonically_published_non_synthetic_dataset(
    tmp_path,
) -> None:
    from trade_rl.data.artifacts.publication import publish_market_dataset_artifact

    original = study.default_2024q1_decision_value_dev_protocol()
    declared_real = _compact_market(90).with_content_identity(
        {"source": "unreviewed historical dataset"}
    )
    artifact = publish_market_dataset_artifact(tmp_path / "dataset", declared_real)
    protocol = replace(
        original,
        dataset_id=declared_real.dataset_id,
        expected_artifact_digest=artifact.artifact_digest,
        study_plan=replace(
            original.study_plan,
            data_scope="synthetic_only",
            real_scope_authorization_digest=None,
        ),
    )
    with patch.object(
        study, "fit_ridge_forecast", side_effect=AssertionError("FITTED")
    ):
        with pytest.raises(ValueError, match="synthetic_only|provenance"):
            study.run_decision_value_dev_study(
                declared_real,
                protocol,
                artifact_root=artifact.root,
            )


def test_synthetic_td8_declared_arms_and_scenarios_must_match_execution_roster() -> (
    None
):
    """A smaller study declaration must never silently execute six arms anyway."""
    historical = study.default_2024q1_decision_value_dev_protocol()
    software = replace(
        historical,
        study_plan=replace(
            historical.study_plan,
            data_scope="synthetic_only",
            real_scope_authorization_digest=None,
        ),
    )
    with pytest.raises(ValueError, match="learner|roster|arm"):
        replace(
            software,
            study_plan=replace(
                software.study_plan,
                learner_roster=("linear",),
            ),
        )
    with pytest.raises(ValueError, match="scenario"):
        replace(software, scenarios=(("base", 1.0, 0),))
    with pytest.raises(ValueError, match="scenario"):
        replace(
            software,
            scenarios=(
                ("base", 1.0, 0),
                ("base", 2.0, 0),
                ("latency_1bar", 1.0, 1),
                ("latency_2bar", 1.0, 2),
            ),
        )


def test_synthetic_td8_existing_output_dir_is_rejected_before_model_fit(
    tmp_path,
) -> None:
    from trade_rl.data.artifacts.publication import publish_market_dataset_artifact

    historical = study.default_2024q1_decision_value_dev_protocol()
    dataset = _compact_market(90).with_content_identity(
        {"synthetic_only": True, "generator": "test_prior_output"}
    )
    artifact = publish_market_dataset_artifact(tmp_path / "dataset", dataset)
    protocol = replace(
        historical,
        dataset_id=dataset.dataset_id,
        expected_artifact_digest=artifact.artifact_digest,
        study_plan=replace(
            historical.study_plan,
            data_scope="synthetic_only",
            real_scope_authorization_digest=None,
        ),
    )
    dest = tmp_path / "already_written"
    dest.mkdir()
    (dest / "result.json").write_text("untouched", encoding="utf-8")
    with patch.object(
        study, "fit_ridge_forecast", side_effect=AssertionError("FITTED")
    ):
        with pytest.raises(FileExistsError, match="output_dir"):
            study.run_decision_value_dev_study(
                dataset,
                protocol,
                artifact_root=artifact.root,
                output_dir=dest,
            )
    assert (dest / "result.json").read_text(encoding="utf-8") == "untouched"


def test_synthetic_td8_never_trusts_supplied_artifact_digest_without_loading(
    tmp_path,
) -> None:
    from trade_rl.data.artifacts.publication import publish_market_dataset_artifact

    historical = study.default_2024q1_decision_value_dev_protocol()
    declared = _compact_market(90).with_content_identity(
        {"synthetic_only": True, "generator": "A"}
    )
    another = _compact_market(90).with_content_identity(
        {"synthetic_only": True, "generator": "B"}
    )
    good_artifact = publish_market_dataset_artifact(tmp_path / "good", declared)
    wrong_artifact = publish_market_dataset_artifact(tmp_path / "wrong", another)
    protocol = replace(
        historical,
        dataset_id=declared.dataset_id,
        expected_artifact_digest=good_artifact.artifact_digest,
        study_plan=replace(
            historical.study_plan,
            data_scope="synthetic_only",
            real_scope_authorization_digest=None,
        ),
    )
    with patch.object(
        study, "fit_ridge_forecast", side_effect=AssertionError("FITTED")
    ):
        with pytest.raises(ValueError, match="published artifact digest"):
            study.run_decision_value_dev_study(
                declared,
                protocol,
                artifact_root=wrong_artifact.root,
                actual_artifact_digest=good_artifact.artifact_digest,
            )


def test_incomplete_pair_attempt_is_write_once_and_preserves_both_outcomes(
    tmp_path,
) -> None:
    import json

    from trade_rl.evaluation.decision_value_reachable_labels import (
        IneligiblePairOutcome,
    )

    arm = {
        "terminal_marked_equity": "893.0",
        "terminal_cash": "10.0",
        "termination_reason": "risk_termination",
        "terminal_exact_quantities": ["4", "0"],
        "pending_order_remainders": [["a" * 64, 4.0]],
    }
    row = IneligiblePairOutcome(
        intervention_index=1,
        prefix_start_index=0,
        terminal_index=7,
        candidate_action=(1, 0),
        reason_codes=("baseline_terminated", "candidate_terminated"),
        baseline=arm,
        candidate=arm,
    )
    failure = study.IncompletePairedLabelsError("b1", (row,))
    output = tmp_path / "attempt"
    study._persist_incomplete_pair_attempt(
        output,
        failure,
        dataset_id="1" * 64,
        protocol_digest="f" * 64,
    )
    recorded = json.loads((output / "incomplete-pairs.json").read_text())
    assert recorded["verdict"] == "INVALID / BLOCKED"
    assert recorded["ineligible_pairs"] == 1
    assert recorded["pairs"][0]["baseline"]["terminal_marked_equity"] == "893.0"
    assert recorded["pairs"][0]["candidate"]["termination_reason"] == "risk_termination"
    with pytest.raises(FileExistsError):
        study._persist_incomplete_pair_attempt(
            output,
            failure,
            dataset_id="1" * 64,
            protocol_digest="f" * 64,
        )
    assert len(tuple(output.iterdir())) == 1


def test_software_runner_persists_incomplete_receipt_before_any_fit(
    tmp_path, monkeypatch
) -> None:
    import json

    from trade_rl.data.artifacts.publication import publish_market_dataset_artifact
    from trade_rl.evaluation.decision_value_reachable_labels import (
        IneligiblePairOutcome,
    )

    dataset = _compact_market(90).with_content_identity(
        {"synthetic_only": True, "generator": "failed-label-attempt"}
    )
    artifact = publish_market_dataset_artifact(tmp_path / "dataset", dataset)
    archived = study.default_2024q1_decision_value_dev_protocol()
    block = study.DecisionValueBlockSpec(
        block_id="blocked",
        train_start_index=0,
        train_stop_index=35,
        train_step=24,
        fit_cutoff_index=60,
        eval_start_index=61,
        eval_stop_index=80,
    )
    protocol = replace(
        archived,
        dataset_id=dataset.dataset_id,
        expected_artifact_digest=artifact.artifact_digest,
        blocks=(block,),
        study_plan=replace(
            archived.study_plan,
            data_scope="synthetic_only",
            real_scope_authorization_digest=None,
        ),
    )
    arm = {
        "terminal_marked_equity": "90000.0",
        "terminal_cash": "90000.0",
        "termination_reason": "insolvency",
        "terminal_exact_quantities": ["0", "0"],
        "pending_order_remainders": [],
        "insolvent": True,
    }
    row = IneligiblePairOutcome(
        intervention_index=50,
        prefix_start_index=20,
        terminal_index=60,
        candidate_action=(1, 0),
        reason_codes=("baseline_insolvent", "candidate_insolvent"),
        baseline=arm,
        candidate=arm,
    )
    failure = study.IncompletePairedLabelsError(block.block_id, (row,))

    def stop_before_fit(*args, **kwargs):
        raise failure

    monkeypatch.setattr(study, "_train_block", stop_before_fit)
    dest = tmp_path / "attempt"
    with pytest.raises(study.IncompletePairedLabelsError):
        study.run_decision_value_dev_study(
            dataset, protocol, artifact_root=artifact.root, output_dir=dest
        )
    assert (dest / "incomplete-pairs.json").is_file()
    assert not (dest / "result.json").exists()
    receipt = json.loads((dest / "incomplete-pairs.json").read_text())
    assert receipt["block_id"] == "blocked"
    assert receipt["pairs"][0]["baseline"]["termination_reason"] == "insolvency"


def test_incomplete_attempt_does_not_replace_concurrently_created_empty_dir(
    tmp_path, monkeypatch
) -> None:
    """Write-once must be atomic even when another writer claims the path."""
    from pathlib import Path

    from trade_rl.evaluation.decision_value_reachable_labels import (
        IneligiblePairOutcome,
    )

    row = IneligiblePairOutcome(
        intervention_index=1,
        prefix_start_index=0,
        terminal_index=7,
        candidate_action=(1, 0),
        reason_codes=("candidate_terminated",),
        baseline={"terminal_marked_equity": "1000"},
        candidate={"terminal_marked_equity": "900"},
    )
    error = study.IncompletePairedLabelsError("b1", (row,))
    destination = tmp_path / "race"
    original_mkdir = Path.mkdir
    raced = False

    def concurrent_writer(path: Path, *args, **kwargs):
        nonlocal raced
        if path == destination and not raced:
            raced = True
            original_mkdir(path, exist_ok=False)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", concurrent_writer)
    with pytest.raises(FileExistsError):
        study._persist_incomplete_pair_attempt(
            destination,
            error,
            dataset_id="1" * 64,
            protocol_digest="f" * 64,
        )
    assert destination.is_dir()
    assert tuple(destination.iterdir()) == ()
