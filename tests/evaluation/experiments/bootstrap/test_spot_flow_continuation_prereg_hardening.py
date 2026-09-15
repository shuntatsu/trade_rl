from __future__ import annotations

import pytest

from trade_rl.evaluation.experiments.bootstrap.spot_flow_continuation_prereg import (
    SpotFlowContinuationProtocol,
    canonical_spot_flow_continuation_protocol,
)


def test_protocol_binds_complete_spot_source_authority_and_structural_result() -> None:
    protocol = canonical_spot_flow_continuation_protocol()

    assert protocol.spot_source_protocol_seal_run_id == 34928427543
    assert protocol.spot_source_protocol_seal_artifact_id == 10381040623
    assert protocol.spot_source_protocol_seal_artifact_api_digest == (
        "84b3637ab61dcc6ba9942a2b41907db326bc5fd48643e437cc4505ff17f9f62f"
    )
    assert protocol.spot_source_protocol_fresh_artifact_id == 10381305016
    assert protocol.spot_source_protocol_fresh_artifact_api_digest == (
        "d15223aa90a1feaeb8cd85baad9d374422f55d4cfb7d262e52ecc9e7978440a0"
    )
    assert protocol.spot_source_validator_head == (
        "a8234444cb4306c37d80c47832200f8303f8928c"
    )
    assert protocol.spot_source_validator_verification_run_id == 34930082702
    assert protocol.spot_source_execution_run_id == 34930477567
    assert protocol.spot_source_publisher_artifact_id == 10381815598
    assert protocol.spot_source_publisher_artifact_api_digest == (
        "faad983c85f496f6c6f88fa6adacdd0f53af785d35e3c15778ace17378d7a0a7"
    )
    assert protocol.spot_source_fresh_artifact_id == 10381247817
    assert protocol.spot_source_fresh_artifact_api_digest == (
        "0dc8faba79881ce36489bbf83f2526b7aae6fd6441539e46e0bec92f660b020f"
    )
    assert protocol.spot_source_usable_rows == 13_598_332
    assert protocol.spot_source_malformed_rows == 0
    assert protocol.spot_source_invalid_sentinel_rows == 0


def test_protocol_freezes_event_time_and_feature_transform_boundaries() -> None:
    protocol = canonical_spot_flow_continuation_protocol()

    assert protocol.provider_order_required is True
    assert protocol.event_time_strictly_before_decision is True
    assert protocol.archive_publication_time_used_as_market_event_time is False
    assert protocol.clipping_allowed is False
    assert protocol.winsorization_allowed is False
    assert protocol.zscore_allowed is False
    assert protocol.volume_scaling_allowed is False
    assert protocol.volatility_scaling_allowed is False
    assert protocol.funding_combination_allowed is False
    assert protocol.basis_combination_allowed is False
    assert protocol.price_return_input_allowed is False
    assert protocol.cross_sectional_normalization_allowed is False
    assert protocol.rank_transform_allowed is False
    assert protocol.learned_coefficient_allowed is False
    assert protocol.feature_threshold_allowed is False


def test_protocol_freezes_target_validity_and_calibration_numeric_gates() -> None:
    protocol = canonical_spot_flow_continuation_protocol()

    assert protocol.require_target_rows_present is True
    assert protocol.require_target_rows_contiguous is True
    assert protocol.require_target_rows_information_available is True
    assert protocol.require_target_rows_active is True
    assert protocol.require_target_rows_tradable is True
    assert protocol.require_target_open_finite_positive is True
    assert protocol.target_2023_or_later_allowed is False

    assert protocol.weighted_regression_allowed is False
    assert protocol.robust_regression_allowed is False
    assert protocol.require_numerator_finite is True
    assert protocol.require_denominator_finite_positive is True
    assert protocol.require_beta_finite is True
    assert protocol.no_magnitude_threshold_after_results is True
    assert protocol.training_relation_executed is False


def test_hardened_boolean_fields_reject_bool_int_spoofing() -> None:
    payload = canonical_spot_flow_continuation_protocol().to_payload()
    payload["provider_order_required"] = 1
    with pytest.raises(ValueError):
        SpotFlowContinuationProtocol.from_payload(payload)
