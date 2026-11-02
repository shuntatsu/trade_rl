"""TD8 receipts bind the recorded teacher to the actual native prefix identity.

A forged SHA over every changed field is not a signed source attestation.
This test only checks internal canonical consistency, not external authenticity.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.evaluation.test_decision_value_vintage_evidence import _context, _labels
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.paired_portfolio_label import (
    build_synthetic_paired_label_batch,
)


def test_v4_rejects_a_collectively_rehashed_teacher_native_prefix_swap() -> None:
    """An entire decision state must not change its claimed teacher prefix."""
    _, market, provider = _context()
    originals = _labels(market, provider, require_v4_evidence=True).samples
    first_time = originals[0].decision_time_ns
    forged = []
    for sample in originals:
        if sample.decision_time_ns != first_time:
            forged.append(sample)
            continue
        vintage = sample.reference_vintage
        evidence = sample.pair_evidence
        assert vintage is not None and evidence is not None
        false_vintage = replace(
            vintage, prefix_start_index=vintage.prefix_start_index + 1
        )
        forged_evidence = {
            **evidence,
            "prefix_start_index": false_vintage.prefix_start_index,
            "reference_vintage": false_vintage.to_mapping(),
            "prefix_identity_digest": "f" * 64,
        }
        forged_evidence["evidence_digest"] = content_digest(
            {k: v for k, v in forged_evidence.items() if k != "evidence_digest"}
        )
        forged.append(
            replace(
                sample,
                reference_vintage=false_vintage,
                pair_evidence=forged_evidence,
                pair_evidence_digest=forged_evidence["evidence_digest"],
            )
        )
    with pytest.raises(ValueError, match="prefix|vintage|identity"):
        build_synthetic_paired_label_batch(
            tuple(forged),
            fit_cutoff=market.timestamps[330],
            evaluation_start=market.timestamps[331],
            expected_reference_policy_digest="d" * 64,
            require_v4_evidence=True,
        )
