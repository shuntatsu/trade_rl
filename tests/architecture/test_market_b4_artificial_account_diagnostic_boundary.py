from __future__ import annotations

from dataclasses import fields
from inspect import signature

import tools.market_b4_oracle as market_b4_oracle
from tools.market_b4_oracle import offline_replay_consumer


def test_artificial_account_diagnostic_stays_internal_and_non_authorizing() -> None:
    assert callable(
        offline_replay_consumer.consume_artificial_reference_account_diagnostic
    )
    assert not hasattr(
        market_b4_oracle, "consume_artificial_reference_account_diagnostic"
    )
    assert (
        "reference_account_diagnostic"
        not in signature(offline_replay_consumer.consume_indexed_market_arm).parameters
    )

    result_fields = {
        field.name: field
        for field in fields(
            offline_replay_consumer.ArtificialReferenceAccountDiagnostic
        )
    }
    for authority_field in (
        "outer_attempt_authenticated",
        "root_attempt_observed",
        "full_b4_authorized",
        "research_authorized",
        "economic_evidence",
        "live_authorized",
    ):
        assert result_fields[authority_field].default is False
    assert result_fields["independent_arithmetic_checked"].default is True
