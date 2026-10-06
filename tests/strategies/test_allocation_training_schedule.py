from dataclasses import FrozenInstanceError, replace

import pytest

from trade_rl.artifacts import canonical_json_bytes, content_digest


def capability():
    from trade_rl.strategies.rl.allocation_training_schedule import (
        AllocationTrainingSchedule,
        AllocationTrainingWindow,
    )

    return AllocationTrainingSchedule, AllocationTrainingWindow


def window(
    *,
    role="train",
    dataset="a",
    symbol="S0",
    start=6,
    stop=10,
    start_ns=6_000,
    stop_ns=10_000,
    source="b",
):
    _, AllocationTrainingWindow = capability()
    return AllocationTrainingWindow(
        role=role,
        dataset_id=dataset * 64,
        symbol=symbol,
        start_index=start,
        stop_index=stop,
        decision_start_ns=start_ns,
        terminal_time_ns=stop_ns,
        source_digest=source * 64,
    )


def schedule(*windows):
    AllocationTrainingSchedule, _ = capability()
    train = tuple(
        value.window_id
        for value in sorted(
            (value for value in windows if value.role == "train"),
            key=lambda value: (value.decision_start_ns, value.window_id),
        )
    )
    return AllocationTrainingSchedule(tuple(windows), train_window_ids=train)


def test_schedule_is_closed_immutable_and_roundtrips_exact_bytes():
    AllocationTrainingSchedule, _ = capability()
    first = window()
    second = window(
        start=12,
        stop=16,
        start_ns=12_000,
        stop_ns=16_000,
        source="c",
    )
    validation = window(
        role="validation",
        start=20,
        stop=24,
        start_ns=20_000,
        stop_ns=24_000,
        source="d",
    )
    held = window(
        role="held_out",
        start=30,
        stop=34,
        start_ns=30_000,
        stop_ns=34_000,
        source="e",
    )
    declaration = schedule(first, second, validation, held)
    payload = declaration.payload()
    assert payload["schema"] == "allocation_training_schedule_v1"
    assert payload["sampler"] == "cyclic_declared_order_v1"
    assert payload["reset_semantics"] == "independent_account_v1"
    assert payload["rollout_boundary_semantics"] == "continue_account_v1"
    assert payload["train_window_ids"] == [first.window_id, second.window_id]
    assert declaration.digest == content_digest(payload)
    restored = AllocationTrainingSchedule.from_payload(payload)
    assert canonical_json_bytes(restored.payload()) == canonical_json_bytes(payload)
    assert restored.digest == declaration.digest
    with pytest.raises(FrozenInstanceError):
        declaration.windows = ()  # type: ignore[misc]


def test_window_identity_excludes_whole_dataset_lineage_but_payload_retains_it():
    first = window()
    relined = replace(first, dataset_id="f" * 64)
    assert first.window_id == relined.window_id
    assert first.identity_payload() == relined.identity_payload()
    assert first.payload() != relined.payload()
    assert schedule(first).digest != schedule(relined).digest


def test_window_allows_pre_epoch_clocks_when_order_is_valid():
    _, AllocationTrainingWindow = capability()
    value = AllocationTrainingWindow(
        role="train",
        dataset_id="a" * 64,
        symbol="S0",
        start_index=0,
        stop_index=2,
        decision_start_ns=-7_200_000_000_000,
        terminal_time_ns=-3_600_000_000_000,
        source_digest="b" * 64,
    )
    assert value.decision_start_ns < value.terminal_time_ns < 0


def test_window_identity_does_not_depend_on_later_windows():
    first = window()
    baseline = schedule(first)
    future = window(
        role="held_out",
        start=30,
        stop=34,
        start_ns=30_000,
        stop_ns=34_000,
        source="e",
    )
    extended = schedule(first, future)
    assert baseline.windows[0].window_id == extended.windows[0].window_id
    assert baseline.windows[0].payload() == extended.windows[0].payload()
    assert baseline.digest != extended.digest


@pytest.mark.parametrize(
    "mutator",
    [
        lambda values: values + (values[0],),
        lambda values: (
            values[0],
            replace(
                values[0],
                source_digest="f" * 64,
                start_index=8,
                stop_index=12,
                decision_start_ns=8_000,
                terminal_time_ns=12_000,
            ),
        ),
        lambda values: (
            values[0],
            window(
                role="held_out",
                start=2,
                stop=5,
                start_ns=2_000,
                stop_ns=5_000,
                source="e",
            ),
        ),
    ],
)
def test_schedule_rejects_duplicate_overlap_or_future_role_before_training(mutator):
    AllocationTrainingSchedule, _ = capability()
    values = (window(),)
    bad = mutator(values)
    train_ids = tuple(value.window_id for value in bad if value.role == "train")
    with pytest.raises(ValueError):
        AllocationTrainingSchedule(bad, train_window_ids=train_ids)


def test_schedule_rejects_same_symbol_clock_overlap_across_dataset_lineages():
    AllocationTrainingSchedule, _ = capability()
    first = window(dataset="a", source="b")
    relined_overlap = window(
        dataset="f",
        start=8,
        stop=12,
        start_ns=8_000,
        stop_ns=12_000,
        source="c",
    )
    with pytest.raises(ValueError, match="overlap"):
        AllocationTrainingSchedule(
            (first, relined_overlap),
            train_window_ids=(first.window_id, relined_overlap.window_id),
        )


def test_schedule_rejects_validation_or_heldout_in_training_order():
    AllocationTrainingSchedule, _ = capability()
    first = window()
    validation = window(
        role="validation",
        start=20,
        stop=24,
        start_ns=20_000,
        stop_ns=24_000,
        source="d",
    )
    with pytest.raises(ValueError, match="training order"):
        AllocationTrainingSchedule(
            (first, validation),
            train_window_ids=(first.window_id, validation.window_id),
        )


def test_schedule_requires_causal_train_order_not_randomized_order():
    AllocationTrainingSchedule, _ = capability()
    first = window()
    second = window(
        start=12,
        stop=16,
        start_ns=12_000,
        stop_ns=16_000,
        source="c",
    )
    with pytest.raises(ValueError, match="chronological"):
        AllocationTrainingSchedule(
            (first, second),
            train_window_ids=(second.window_id, first.window_id),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset_id": "not-a-digest"},
        {"source_digest": "not-a-digest"},
        {"symbol": ""},
        {"start_index": True},
        {"stop_index": 6},
        {"decision_start_ns": 10_000, "terminal_time_ns": 10_000},
        {"role": "test"},
    ],
)
def test_window_rejects_malformed_identity_clock_and_role(changes):
    _, AllocationTrainingWindow = capability()
    values = dict(
        role="train",
        dataset_id="a" * 64,
        symbol="S0",
        start_index=6,
        stop_index=10,
        decision_start_ns=6_000,
        terminal_time_ns=10_000,
        source_digest="b" * 64,
    )
    with pytest.raises(ValueError):
        AllocationTrainingWindow(**(values | changes))
