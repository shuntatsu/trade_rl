"""Frozen-prefix receipt linkage; no actual Dataset or fitting authentication."""

from __future__ import annotations

from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.strategies.rl.allocation_preprocessing import (
    AllocationFeaturePreprocessing,
    _native_json,
)
from trade_rl.strategies.rl.allocation_protocol_receipt import (
    _validate_protocol_optimization,
)
from trade_rl.strategies.rl.allocation_receipt_time import parse_source_timestamp
from trade_rl.strategies.rl.allocation_training_receipt import (
    validate_allocation_training,
)


def preprocessing_fit_payload(
    declaration: AllocationFeaturePreprocessing,
) -> dict[str, object]:
    return {
        "schema": "allocation_ppo_preprocessing_fit_receipt_v1",
        "declaration_digest": declaration.digest,
        "fit_source_dataset_id": declaration.normalizer.source_dataset_id,
        "feature_config_digest": declaration.feature_config_digest,
        "source_normalization_digest": declaration.source_normalization_digest,
        "fit_consumption_digest": declaration.fit_consumption_digest,
        "policy_start_index": declaration.policy_start_index,
        "policy_start_time_ns": declaration.policy_start_time_ns,
    }


def validate_allocation_training_v4(
    training: object, recipe: dict[str, Any], recipe_digest: str
) -> None:
    if (
        type(recipe) is not dict
        or type(recipe.get("schema")) is not str
        or recipe["schema"] != "allocation_ppo_recipe_v3"
    ):
        raise ValueError("v4 training requires recipe v3")
    try:
        _native_json(training)
    except RecursionError as error:
        raise ValueError("v4 training requires finite native JSON") from error
    extras = {
        "schema",
        "protocol",
        "protocol_digest",
        "optimization",
        "preprocessing_fit",
    }
    if (
        not isinstance(training, dict)
        or not extras <= training.keys()
        or training["schema"] != "allocation_ppo_training_receipt_v4"
    ):
        raise ValueError("v4 training requires its explicit preprocessing fields")
    common = {key: value for key, value in training.items() if key not in extras}
    validate_allocation_training(common, recipe, recipe_digest)
    _validate_protocol_optimization(training)
    declaration = AllocationFeaturePreprocessing.from_payload(
        recipe["observation"]["feature_preprocessing"]
    )
    if canonical_json_bytes(training["preprocessing_fit"]) != canonical_json_bytes(
        preprocessing_fit_payload(declaration)
    ):
        raise ValueError("preprocessing fit receipt differs from its declaration")
    source = training["source"]
    if (
        source["dataset_id"] != declaration.normalizer.source_dataset_id
        or source["start_index"] != declaration.policy_start_index
        or int(
            parse_source_timestamp(source["decision_start"])
            .astype("datetime64[ns]")
            .astype("int64")
        )
        != declaration.policy_start_time_ns
    ):
        raise ValueError("preprocessing fit provenance differs from training source")
