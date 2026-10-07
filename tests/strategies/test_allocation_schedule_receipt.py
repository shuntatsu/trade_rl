"""Independent declared schedule receipts; no fitted-model authenticity claim."""

from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from tests.strategies.test_allocation_protocol_receipt import manifest_v3
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest
from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
    AllocationTrainingWindow,
)


def manifest_v5():
    raw = manifest_v3()
    old = raw["training"]
    sources, windows = [], []
    for day in (1, 2):
        source, objective = deepcopy(old["source"]), deepcopy(old["objective"])
        source["decision_counts"] = [2] * 16
        source["decision_start"] = f"2026-01-0{day}T06:00:00.000000000"
        source["terminal_time"] = f"2026-01-0{day}T22:00:00.000000000"
        objective["evaluation_start"] = f"2026-01-0{day}T06:00:00Z"
        objective["evaluation_stop_exclusive"] = f"2026-01-0{day}T22:00:00Z"
        envelope = source | {"decision_counts": [1] * 16}
        window = AllocationTrainingWindow(
            "train",
            source["dataset_id"],
            "S0",
            6,
            22,
            int(datetime(2026, 1, day, 6, tzinfo=UTC).timestamp()) * 10**9,
            int(datetime(2026, 1, day, 22, tzinfo=UTC).timestamp()) * 10**9,
            content_digest({k: v for k, v in envelope.items() if k != "dataset_id"}),
        )
        windows.append(window)
        sources.append(
            {"window_id": window.window_id, "source": source, "objective": objective}
        )
    schedule = AllocationTrainingSchedule(
        tuple(windows), tuple(w.window_id for w in windows)
    )
    raw["schema"] = "allocation_ppo_inference_bundle_v5"
    raw["training"] = {
        k: v
        for k, v in old.items()
        if k not in {"source", "objective", "observation_consumption"}
    } | {
        "schema": "allocation_ppo_schedule_training_receipt_v1",
        "recipe_digest": raw["recipe_digest"],
        "schedule": schedule.payload(),
        "schedule_digest": schedule.digest,
        "sources": sources,
        "usage": {
            "schema": "allocation_training_schedule_usage_v1",
            "schedule_digest": schedule.digest,
            "sampler": "cyclic_declared_order_v1",
            "reset_semantics": "independent_account_v1",
            "rollout_boundary_semantics": "continue_account_v1",
            "windows": [
                {
                    "window_id": w.window_id,
                    "reset_count": 3 if i == 0 else 2,
                    "decision_count": 32,
                }
                for i, w in enumerate(windows)
            ],
        },
        "consumption": {
            "schema": "allocation_training_schedule_consumption_v1",
            "windows": [
                {
                    "window_id": w.window_id,
                    "decision_indices": list(range(6, 22)),
                    "decision_counts": [2] * 16,
                    "observation_indices": list(range(6, 22)),
                }
                for w in windows
            ],
        },
    }
    return raw


def test_scheduled_receipt_has_explicit_version_and_preserves_legacy_bytes():
    before = canonical_json_bytes(manifest_v3())
    raw = manifest_v5()
    try:
        actual = validate_allocation_manifest(raw)
    except ValueError as error:
        pytest.fail(f"scheduled inference manifest is not admitted: {error}")
    assert actual == raw
    actual["training"]["usage"]["windows"][0]["reset_count"] = 99
    assert raw["training"]["usage"]["windows"][0]["reset_count"] == 3
    assert canonical_json_bytes(manifest_v3()) == before
    for schema in (
        "allocation_ppo_inference_bundle_v2",
        "allocation_ppo_inference_bundle_v3",
        "allocation_ppo_inference_bundle_v4",
    ):
        with pytest.raises(ValueError):
            validate_allocation_manifest(raw | {"schema": schema})


def repin_first_source(raw):
    training = raw["training"]
    source = training["sources"][0]["source"]
    envelope = source | {"decision_counts": [1] * 16}
    window = training["schedule"]["windows"][0]
    window["source_digest"] = content_digest(
        {k: v for k, v in envelope.items() if k != "dataset_id"}
    )
    identity = AllocationTrainingWindow.from_payload(window).window_id
    training["schedule"]["train_window_ids"][0] = identity
    training["schedule_digest"] = content_digest(training["schedule"])
    training["usage"]["schedule_digest"] = training["schedule_digest"]
    for field in ("usage", "consumption"):
        training[field]["windows"][0]["window_id"] = identity
    training["sources"][0]["window_id"] = identity


@pytest.mark.parametrize(
    "key,value", [("feature_consumption_digest", "bad"), ("start_index", 6.0)]
)
def test_rehashed_source_tokens_and_indices_are_not_coerced(key, value):
    raw = manifest_v5()
    raw["training"]["sources"][0]["source"][key] = value
    repin_first_source(raw)
    with pytest.raises(ValueError):
        validate_allocation_manifest(raw)


@pytest.mark.parametrize(
    "path,value",
    [
        (("schedule_digest",), "f" * 64),
        (("recipe_digest",), "f" * 64),
        (("seed",), True),
        (("requested_timesteps",), 63),
        (("actual_timesteps",), 32),
        (("clock", "economic_horizon_seconds"), 7200),
        (("clock", "gamma"), 0.99),
        (("protocol_digest",), "f" * 64),
        (("protocol", "ppo", "n_steps"), 16),
        (("optimization", "successful_optimizer_step_calls"), 19),
        (("usage", "windows", 0, "reset_count"), 2),
        (("usage", "windows", 0, "decision_count"), 31),
        (("consumption", "windows", 0, "decision_indices"), list(range(7, 23))),
        (("consumption", "windows", 0, "decision_counts"), [1] * 16),
        (("consumption", "windows", 0, "observation_indices"), list(range(6, 23))),
        (("sources", 0, "source", "feature_consumption_digest"), "f" * 64),
        (("sources", 0, "objective", "capital", "initial_equities"), [999.0]),
        (("sources", 0, "objective", "evaluation_start"), "2026-01-02T06:00:00Z"),
        (("sources",), []),
        (("usage", "extra"), 0),
    ],
)
def test_scheduled_manifest_tamper_fails_before_optional_loader(
    tmp_path, monkeypatch, path, value
):
    from trade_rl.strategies.rl import allocation_artifact as module

    raw = manifest_v5()
    target = raw["training"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path[0] == "protocol":
        raw["training"]["protocol_digest"] = content_digest(raw["training"]["protocol"])
    data = b"never deserialize this invalid receipt"
    raw["policy_sha256"] = sha256(data).hexdigest()
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "policy.zip").write_bytes(data)
    (root / "manifest.json").write_bytes(canonical_json_bytes(raw))
    monkeypatch.setattr(module, "_load_policy", lambda _: pytest.fail("loader reached"))
    with pytest.raises(ValueError):
        module.load_allocation_policy(
            root,
            expected_digest=content_digest(raw),
            expected_recipe_digest=raw["recipe_digest"],
        )


@pytest.mark.parametrize("change", ["order", "role", "range", "excluded", "alias"])
def test_schedule_declaration_and_native_tokens_are_closed(change):
    raw = manifest_v5()
    training = raw["training"]
    schedule = training["schedule"]
    if change == "order":
        schedule["train_window_ids"].reverse()
    elif change == "role":
        schedule["windows"][0]["role"] = "held_out"
    elif change == "range":
        schedule["windows"][0]["stop_index"] = 23
    elif change == "excluded":
        schedule["windows"][1]["role"] = "validation"
    else:

        class Alias(str):
            pass

        training["schedule_digest"] = Alias(training["schedule_digest"])
    if change != "alias":
        training["schedule_digest"] = content_digest(schedule)
    with pytest.raises(ValueError):
        validate_allocation_manifest(raw)


def direct_reader_case(*, frozen=False):
    raw = manifest_v5()
    if frozen:
        from tests.strategies.test_allocation_preprocessing_receipt import manifest_v4

        previous = manifest_v4()
        raw["recipe"] = previous["recipe"]
        raw["recipe_digest"] = previous["recipe_digest"]
        raw["training"]["recipe_digest"] = raw["recipe_digest"]
        raw["training"]["preprocessing_fit"] = previous["training"]["preprocessing_fit"]
        for record in raw["training"]["sources"]:
            record["objective"]["deployment_recipe_digest"] = raw["recipe_digest"]
    return raw


@pytest.mark.parametrize("frozen", [False, True])
def test_public_schedule_reader_accepts_valid_v2_v3_without_mutating_inputs(frozen):
    from trade_rl.strategies.rl.allocation_schedule_receipt import (
        validate_allocation_schedule_training,
    )

    raw = direct_reader_case(frozen=frozen)
    before = canonical_json_bytes(raw)
    validate_allocation_schedule_training(
        raw["training"], raw["recipe"], raw["recipe_digest"]
    )
    assert canonical_json_bytes(raw) == before


@pytest.mark.parametrize(
    "change",
    [
        "v1",
        "unknown",
        "tag_alias",
        "root_alias",
        "deep_alias",
        "malformed_field",
        "changed_payload",
        "wrong_pin",
        "pin_alias",
        "malformed_pin",
        "nonfinite",
    ],
)
def test_public_schedule_reader_closes_recipe_lane_and_payload_pin(change):
    from trade_rl.strategies.rl.allocation_schedule_receipt import (
        validate_allocation_schedule_training,
    )

    class StringAlias(str):
        pass

    class MappingAlias(dict):
        pass

    raw = direct_reader_case()
    recipe, pin = raw["recipe"], raw["recipe_digest"]
    if change in ("v1", "unknown"):
        recipe["schema"] = "allocation_ppo_recipe_v1" if change == "v1" else "unknown"
    elif change == "tag_alias":
        recipe["schema"] = StringAlias(recipe["schema"])
    elif change == "root_alias":
        recipe = MappingAlias(recipe)
    elif change == "deep_alias":
        recipe["action"]["mode"] = StringAlias(recipe["action"]["mode"])
    elif change == "malformed_field":
        recipe["action"]["scale"] = "invalid"
    elif change == "changed_payload":
        recipe["action"]["scale"] = 0.25
    elif change in ("wrong_pin", "malformed_pin"):
        pin = "f" * 64 if change == "wrong_pin" else "bad"
        raw["training"]["recipe_digest"] = pin
        for record in raw["training"]["sources"]:
            record["objective"]["deployment_recipe_digest"] = pin
    elif change == "pin_alias":
        pin = StringAlias(pin)
    else:
        recipe["allocator"]["risk_aversion"] = float("nan")
    with pytest.raises(ValueError):
        validate_allocation_schedule_training(raw["training"], recipe, pin)
