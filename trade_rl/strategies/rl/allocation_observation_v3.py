"""Pure frozen-feature v3 projection; runtime and fitting admission stay above."""

from __future__ import annotations

import numpy as np

from trade_rl.strategies.allocation_action import AllocationDecision
from trade_rl.strategies.allocation_snapshot import AllocationAccountSnapshot
from trade_rl.strategies.rl.allocation_observation_encoder_v2 import (
    encode_allocation_observation_v2,
)
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_preprocessing import (
    AllocationFeaturePreprocessing,
    _native_json,
)


def allocation_observation_payload_v3(
    schema: AllocationObservationSchema,
    preprocessing: AllocationFeaturePreprocessing,
) -> dict[str, object]:
    """Detached v2 economic layout plus explicitly frozen first-F feature meaning."""
    if (
        type(schema) is not AllocationObservationSchema
        or type(preprocessing) is not AllocationFeaturePreprocessing
        or type(schema.feature_names) is not tuple
        or schema.feature_names != preprocessing.normalizer.feature_names
    ):
        raise ValueError("v3 requires matching frozen feature and schema declarations")
    raw = schema.payload()
    _native_json(raw)
    return raw | {
        "schema": "allocation_account_observation_v3",
        "raw_feature_projection": "frozen_prefix_standardization_float64_then_float32_v1",
        "feature_preprocessing": preprocessing.payload(),
        "feature_preprocessing_digest": preprocessing.digest,
    }


def encode_allocation_observation_v3(
    snapshot: AllocationAccountSnapshot,
    decision: AllocationDecision,
    *,
    schema: AllocationObservationSchema,
    preprocessing: AllocationFeaturePreprocessing,
    feature_config_digest: str,
    source_normalization_digest: str,
    episode_steps: int,
) -> np.ndarray:
    """Transform raw first-F values once; preserve v2 economic suffix bytes.

    This live-input API retains v2 raw admissibility, not terminal-vector handling.
    Caller-supplied build/clock facts are checked, not authenticated. Full fit
    provenance and actual runtime policy-window checks remain upper responsibilities.
    """
    allocation_observation_payload_v3(schema, preprocessing)
    values = encode_allocation_observation_v2(
        snapshot, decision, schema=schema, episode_steps=episode_steps
    )
    values[: len(schema.feature_names)] = preprocessing.transform(
        decision.feature_values,
        feature_names=decision.feature_names,
        feature_config_digest=feature_config_digest,
        source_normalization_digest=source_normalization_digest,
        decision_time_ns=int(
            decision.baseline.context.decision_time.astype("datetime64[ns]").astype(
                np.int64
            )
        ),
    )
    return values


__all__ = ["allocation_observation_payload_v3", "encode_allocation_observation_v3"]
