"""All TD8 action heads must use the identical pre-intervention observation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.evaluation.test_decision_value_fit import _CANDIDATES, _fit_batch
from tests.evaluation.test_decision_value_vintage_evidence import _context, _labels
from trade_rl.evaluation.decision_value_fit import (
    fit_conservative_linear_decision_value,
    fit_synthetic_linear_decision_value,
    fit_synthetic_shallow_decision_value,
)
from trade_rl.evaluation.paired_portfolio_label import (
    build_synthetic_paired_label_batch,
)


def _different_feature_snapshot(sample):
    original = sample.observation
    features = list(original.features)
    features[0] = (float(features[0][0]) + 0.25, *features[0][1:])
    return replace(sample, observation=replace(original, features=tuple(features)))


def test_v4_batch_refuses_one_head_with_different_features_at_same_state():
    _, dataset, provider = _context()
    samples = list(_labels(dataset, provider, require_v4_evidence=True).samples)
    assert samples[0].decision_time_ns == samples[1].decision_time_ns
    samples[1] = _different_feature_snapshot(samples[1])
    with pytest.raises(ValueError, match="state|observation|population|feature"):
        build_synthetic_paired_label_batch(
            tuple(samples),
            fit_cutoff=dataset.timestamps[330],
            evaluation_start=dataset.timestamps[331],
            expected_reference_policy_digest="d" * 64,
            require_v4_evidence=True,
        )


@pytest.mark.parametrize(
    "fitter",
    (
        fit_synthetic_linear_decision_value,
        fit_conservative_linear_decision_value,
        fit_synthetic_shallow_decision_value,
    ),
)
def test_td8_model_fit_refuses_mixed_observations_for_one_decision(fitter):
    source = _fit_batch()
    samples = list(source.samples)
    assert samples[0].decision_time_ns == samples[1].decision_time_ns
    samples[1] = _different_feature_snapshot(samples[1])
    batch = build_synthetic_paired_label_batch(
        tuple(samples),
        fit_cutoff=np.datetime64(source.fit_cutoff_ns, "ns"),
        evaluation_start=np.datetime64(source.evaluation_start_ns, "ns"),
        expected_reference_policy_digest=samples[0].reference_policy_digest,
    )
    with pytest.raises(ValueError, match="state|observation|population|feature"):
        fitter(batch, candidate_actions=_CANDIDATES, solver_version="td8_v2")
