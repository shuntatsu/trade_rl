"""Reject non-JSON object implementations at direct payload decoder boundaries."""

import json

import pytest

from tools.agent_repo.coordination.serde import (
    status_mapping,
    task_status_from_payload,
)


class _StatusDict(dict):
    def __getitem__(self, key):
        if key == "phase":
            return "complete"
        return super().__getitem__(key)


class _TextKey(str):
    pass


class _SpoofedDict:
    @property
    def __class__(self):
        return dict

    def __iter__(self):
        return iter(("phase", "condition", "reason"))

    def __getitem__(self, key):
        return {"phase": "complete", "condition": "healthy", "reason": None}[key]


@pytest.mark.parametrize(
    "payload",
    [
        _StatusDict(phase="ready", condition="healthy", reason=None),
        _SpoofedDict(),
        {_TextKey("phase"): "ready", "condition": "healthy", "reason": None},
    ],
)
def test_status_decoder_rejects_non_native_json_objects(payload):
    with pytest.raises(ValueError, match="task status must be a JSON object"):
        task_status_from_payload(payload)


def test_nested_status_decoder_rejects_non_native_json_objects():
    with pytest.raises(ValueError, match="task status must be a JSON object"):
        status_mapping(
            {"T1": _StatusDict(phase="ready", condition="healthy", reason=None)}
        )


def test_native_json_status_roster_remains_valid():
    payload = json.loads('{"T1":{"phase":"ready","condition":"healthy","reason":null}}')
    assert status_mapping(payload)["T1"].phase.value == "ready"
