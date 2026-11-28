"""Direct shared-validator boundary tests, independent of adapter normalization."""

from types import MappingProxyType

import pytest

from tools.github_workflow_attempts import (
    validate_attempt_job_states,
    validate_workflow_state,
)


class _ForeignDict(dict):
    def get(self, *args, **kwargs):
        raise AssertionError("foreign lookup invoked")


class _ForeignList(list):
    def __iter__(self):
        raise AssertionError("foreign iteration invoked")


class _ForeignTuple(tuple):
    def __iter__(self):
        raise AssertionError("foreign iteration invoked")


class _ForeignText(str):
    def __eq__(self, other):
        raise AssertionError("foreign equality invoked")

    __hash__ = str.__hash__


def _row():
    return {"name": "job", "status": "completed", "conclusion": "success"}


@pytest.mark.parametrize("key", ["status", "conclusion", "extension"])
@pytest.mark.parametrize("boundary", ["state", "job"])
def test_foreign_keys_rejected_before_lookup(key, boundary):
    row = _row()
    row["extension"] = None
    row[_ForeignText(key)] = row.pop(key)
    with pytest.raises(ValueError, match="native JSON object"):
        if boundary == "state":
            validate_workflow_state(row, name="run")
        else:
            validate_attempt_job_states("completed", "success", [row])


@pytest.mark.parametrize("factory", [_ForeignDict, MappingProxyType, list])
@pytest.mark.parametrize("boundary", ["state", "job"])
def test_foreign_mapping_rejected_before_lookup(factory, boundary):
    row = factory(_row())
    with pytest.raises(ValueError, match="native JSON object"):
        if boundary == "state":
            validate_workflow_state(row, name="run")
        else:
            validate_attempt_job_states("completed", "success", [row])


@pytest.mark.parametrize("factory", [_ForeignList, _ForeignTuple, iter])
def test_foreign_roster_rejected_before_iteration(factory):
    with pytest.raises(ValueError, match="native list or tuple"):
        validate_attempt_job_states("completed", "success", factory([_row()]))


def test_lazy_roster_is_not_executed():
    def rows():
        raise AssertionError("lazy roster executed")
        yield _row()

    with pytest.raises(ValueError, match="native list or tuple"):
        validate_attempt_job_states("completed", "success", rows())


@pytest.mark.parametrize(
    ("status", "conclusion"),
    [
        ("unknown", None),
        ("completed", None),
        ("queued", "success"),
        ("completed", "unknown"),
        (_ForeignText("completed"), "success"),
        ("completed", _ForeignText("success")),
    ],
)
@pytest.mark.parametrize("jobs", [[], [_row()]])
def test_run_state_is_validated_even_if_no_jobs(status, conclusion, jobs):
    with pytest.raises(ValueError):
        validate_attempt_job_states(status, conclusion, jobs)


@pytest.mark.parametrize("factory", [list, tuple])
@pytest.mark.parametrize(
    ("status", "conclusion"),
    [
        ("requested", None),
        ("waiting", None),
        ("pending", None),
        ("queued", None),
        ("in_progress", None),
        ("completed", "success"),
        ("completed", "failure"),
        ("completed", "cancelled"),
        ("completed", "skipped"),
        ("completed", "neutral"),
        ("completed", "timed_out"),
        ("completed", "action_required"),
        ("completed", "stale"),
    ],
)
def test_native_state_and_roster_remain_valid(factory, status, conclusion):
    assert validate_workflow_state(
        {"status": status, "conclusion": conclusion}, name="run"
    ) == (status, conclusion)
    assert validate_attempt_job_states(status, conclusion, factory([_row()])) is None


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_successful_run_still_rejects_failed_job(conclusion):
    row = _row()
    row["conclusion"] = conclusion
    with pytest.raises(ValueError, match="successful run contains"):
        validate_attempt_job_states("completed", "success", [row])
