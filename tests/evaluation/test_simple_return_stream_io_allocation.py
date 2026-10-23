"""Published predictions reuse existing allocation without another ledger."""

from dataclasses import replace

import pytest

from tests.evaluation.test_forecast_allocation import setup
from tests.strategies.test_simple_return_stream_io import io_api
from trade_rl.evaluation.forecast_allocation import propose_forecast_target


def test_reloaded_file_drives_the_literal_existing_allocation_proposal(tmp_path):
    api = io_api()
    executor, book, orders, arguments = setup()
    original = arguments["stream"]
    api.publish_simple_return_stream_artifact(tmp_path / "stream.json", original)
    arguments["stream"] = api.load_simple_return_stream_artifact(
        tmp_path / "stream.json",
        expected_digest=original.digest,
        expected_dataset_id=executor.dataset.dataset_id,
    )
    proposal = propose_forecast_target(executor, book, orders, **arguments)
    # Mature simple returns 1,-0.5,1,-0.5 have mean .25, variance .5625.
    # Maximize (.25-.025)w-.5625w²: w=.2; no realized market result is used.
    assert proposal.inputs.expected_simple_return == pytest.approx(0.25)
    assert proposal.inputs.return_variance == pytest.approx(0.5625)
    assert proposal.target_weight == pytest.approx(0.2)
    assert proposal.objective_value == pytest.approx(0.0225)
    original_proposal = propose_forecast_target(
        executor, book, orders, **(arguments | {"stream": original})
    )
    assert proposal == original_proposal


def test_valid_file_does_not_override_current_allocation_snapshot_checks(tmp_path):
    api = io_api()
    executor, book, orders, arguments = setup()
    stream = arguments["stream"]
    api.publish_simple_return_stream_artifact(tmp_path / "stream.json", stream)
    arguments["stream"] = api.load_simple_return_stream_artifact(
        tmp_path / "stream.json",
        expected_digest=stream.digest,
        expected_dataset_id=executor.dataset.dataset_id,
    )
    features = executor.dataset.features.copy()
    features[6, 0, 0] = 42
    executor.dataset = replace(executor.dataset, features=features)
    with pytest.raises(ValueError, match="decision snapshot"):
        propose_forecast_target(executor, book, orders, **arguments)
