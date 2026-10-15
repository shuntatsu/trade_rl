"""Pure prefix observations: manufactured callbacks, no native execution or data."""

from dataclasses import FrozenInstanceError
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.tier_d_nonrl.progress import ArmProgressSpec


def api():
    try:
        return import_module("tools.tier_d_scaling.probe")
    except ModuleNotFoundError as error:
        if error.name == "tools.tier_d_scaling.probe":
            pytest.fail("bounded prefix observation helper is missing")
        raise


def spec():
    return ArmProgressSpec(
        "H72_A10", "ARTIFICIAL_A", "BASEL0", "nonrl", "a" * 64, 10, 13, 15
    )


EVENTS = (
    ("actor", 10, 11),
    ("actor", 11, 12),
    ("actor", 12, 13),
    ("reserve", 13, 14),
    ("reserve", 14, 15),
)


def fixed_clock():
    values = iter((20.0, 22.5))
    return lambda: next(values)


def test_expected_stop_keeps_exact_exception_and_actual_frontier():
    module = api()
    actual, forwarded, caught = [], [], []

    def invoke(callback):
        try:
            for event in EVENTS:
                actual.append(event)
                callback(*event)
        except Exception as error:
            error.terminal_execution_receipt = {
                "scope": "MANUFACTURED",
                "rows": actual.copy(),
            }
            caught.append(error)
            raise
        pytest.fail("technical stop was not raised")

    result = module.observe_prefix(
        invoke,
        spec(),
        target=4,
        on_advance=lambda *event: forwarded.append(event),
        clock=fixed_clock(),
    )
    assert result.expected_stop and result.error is caught[0]
    assert str(result.error) == "predeclared artificial native prefix reached"
    assert actual == forwarded == list(EVENTS[:4])
    assert (
        result.count,
        result.actor_advances,
        result.reserve_advances,
        result.processing_index,
    ) == (4, 3, 1, 14)
    assert result.elapsed_seconds == 2.5 and result.returned is None
    assert result.error.terminal_execution_receipt["rows"] == actual
    with pytest.raises(FrozenInstanceError):
        result.count = 99


def test_foreign_same_type_is_never_expected_stop():
    module = api()
    first = module.observe_prefix(
        lambda callback: callback(*EVENTS[0]), spec(), target=1
    )
    foreign = type(first.error)(str(first.error))

    def invoke(callback):
        callback(*EVENTS[0])
        raise foreign

    result = module.observe_prefix(invoke, spec(), target=4)
    assert not result.expected_stop and result.error is foreign
    assert result.count == 1 and result.processing_index == 11


@pytest.mark.parametrize("count", [0, 2])
def test_early_normal_return_retains_object_without_fabricating_target(count):
    module, returned = api(), object()

    def invoke(callback):
        for event in EVENTS[:count]:
            callback(*event)
        return returned

    result = module.observe_prefix(invoke, spec(), target=4)
    assert not result.expected_stop and result.error is None
    assert result.returned is returned and result.count == count
    assert result.processing_index == 10 + count


def test_other_native_error_keeps_identity_and_known_prefix():
    module, error = api(), LookupError("manufactured native failure")

    def invoke(callback):
        callback(*EVENTS[0])
        raise error

    result = module.observe_prefix(invoke, spec(), target=5)
    assert result.error is error and not result.expected_stop
    assert result.count == result.actor_advances == 1 and result.reserve_advances == 0


@pytest.mark.parametrize("target", [True, False, 0, -1, 6, 1.0, "1"])
def test_invalid_target_rejects_before_invocation_or_profile_io(tmp_path, target):
    with pytest.raises(ValueError, match="target"):
        api().observe_prefix(
            lambda _: pytest.fail("invoked invalid target"),
            spec(),
            target=target,
            profile_path=tmp_path / "absent" / "profile.prof",
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "event",
    [
        ("reserve", 10, 11),
        ("actor", 11, 12),
        ("actor", 10, 12),
        ("actor", True, 11),
        ("actor", 10, 11.0),
        ("ACTOR", 10, 11),
    ],
)
def test_malformed_first_event_rejects_without_counting_or_forwarding(event):
    forwarded = []
    result = api().observe_prefix(
        lambda callback: callback(*event),
        spec(),
        target=5,
        on_advance=lambda *value: forwarded.append(value),
    )
    assert isinstance(result.error, ValueError) and not result.expected_stop
    assert result.count == 0 and result.processing_index == 10 and forwarded == []


def test_duplicate_after_valid_event_retains_only_valid_frontier():
    def invoke(callback):
        callback(*EVENTS[0])
        callback(*EVENTS[0])

    result = api().observe_prefix(invoke, spec(), target=5)
    assert isinstance(result.error, ValueError)
    assert result.count == 1 and result.processing_index == 11


def test_journal_failure_still_counts_known_actual_advance_after_forward_attempt():
    error, forwarded = OSError("manufactured journal publication failure"), []

    def journal(*event):
        forwarded.append(event)
        raise error

    result = api().observe_prefix(
        lambda callback: callback(*EVENTS[0]), spec(), target=1, on_advance=journal
    )
    assert result.error is error and not result.expected_stop
    assert forwarded == [EVENTS[0]]
    assert result.count == result.actor_advances == 1 and result.processing_index == 11


def test_no_profile_path_never_constructs_profiler_or_opens_file(monkeypatch):
    module, returned = api(), object()
    monkeypatch.setattr(
        module, "_make_profiler", lambda: pytest.fail("constructed profiler")
    )
    monkeypatch.setattr(
        Path, "open", lambda *args, **kwargs: pytest.fail("unexpected IO")
    )
    result = module.observe_prefix(lambda _: returned, spec(), target=1)
    assert result.returned is returned and result.profile_publication_failure is None


def test_actual_cprofile_preserves_fake_facts_and_expected_stop(tmp_path):
    import pstats

    module, outcomes = api(), []
    for path in (None, tmp_path / "prefix.prof"):
        actual = []

        def invoke(callback):
            for event in EVENTS:
                actual.append(event)
                callback(*event)

        observed = module.observe_prefix(
            invoke, spec(), target=4, profile_path=path, clock=fixed_clock()
        )
        outcomes.append(
            (
                actual,
                observed.count,
                observed.actor_advances,
                observed.reserve_advances,
                observed.processing_index,
                observed.elapsed_seconds,
                observed.expected_stop,
                str(observed.error),
            )
        )
        assert observed.profile_publication_failure is None
    assert outcomes[0] == outcomes[1]
    assert any(
        key[2] == "invoke" for key in pstats.Stats(str(tmp_path / "prefix.prof")).stats
    )


@pytest.mark.parametrize("native_failure", [False, True])
def test_profile_publication_failure_never_masks_primary_or_return(
    tmp_path, monkeypatch, native_failure
):
    module, error, returned = api(), LookupError("primary"), object()
    order = []

    def dump(path):
        order.append("dump")
        raise PermissionError("manufactured profile publication failure")

    profiler = SimpleNamespace(
        enable=lambda: order.append("enable"),
        disable=lambda: order.append("disable"),
        dump_stats=dump,
    )
    monkeypatch.setattr(module, "_make_profiler", lambda: profiler)

    def invoke(callback):
        order.append("invoke")
        callback(*EVENTS[0])
        if native_failure:
            raise error
        return returned

    result = module.observe_prefix(
        invoke,
        spec(),
        target=4,
        profile_path=tmp_path / "profile.prof",
        clock=fixed_clock(),
    )
    assert order == ["enable", "invoke", "disable", "dump"]
    assert result.error is (error if native_failure else None)
    assert result.returned is (None if native_failure else returned)
    assert result.count == 1 and not result.expected_stop
    assert result.elapsed_seconds == 2.5
    assert (
        "manufactured profile publication failure" in result.profile_publication_failure
    )


def test_profile_destination_collision_preserves_prior_bytes_and_return(tmp_path):
    path, returned = tmp_path / "old.prof", object()
    path.write_bytes(b"retained prior diagnostic")
    result = api().observe_prefix(
        lambda _: returned, spec(), target=1, profile_path=path
    )
    assert result.returned is returned and result.error is None
    assert result.profile_publication_failure is not None
    assert path.read_bytes() == b"retained prior diagnostic"
