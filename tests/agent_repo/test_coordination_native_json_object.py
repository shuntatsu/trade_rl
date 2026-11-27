"""Reject non-JSON object implementations at direct payload decoder boundaries."""

import json

import pytest

from tools.agent_repo.coordination.serde import (
    dashboard_snapshot_from_payload,
    packet_sequence,
    status_mapping,
    string_set,
    task_packet_from_payload,
    task_status_from_payload,
    task_status_record_from_payload,
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


class _TrapDict(dict):
    def __iter__(self):
        pytest.fail("must not enumerate foreign JSON object")

    def __getitem__(self, key):
        pytest.fail("must not look up foreign JSON object")

    def items(self):
        pytest.fail("must not enumerate foreign JSON items")

    def keys(self):
        pytest.fail("must not enumerate foreign JSON keys")


class _TrapList(list):
    def __iter__(self):
        pytest.fail("must not enumerate foreign JSON array")


class _DropList(list):
    def __iter__(self):
        return iter(())


@pytest.mark.parametrize(
    "decoder",
    [
        task_packet_from_payload,
        task_status_from_payload,
        task_status_record_from_payload,
        dashboard_snapshot_from_payload,
        status_mapping,
    ],
)
@pytest.mark.parametrize(
    "payload", [_TrapDict(phase="ready"), _SpoofedDict(), {_TextKey("phase"): "ready"}]
)
def test_all_object_decoders_reject_foreign_objects_before_access(decoder, payload):
    with pytest.raises(ValueError, match="must be a JSON object"):
        decoder(payload)


def _packet_payload():
    from tests.agent_repo.test_coordination_model import _packet

    packet = _packet()
    return dict(packet.canonical_payload(), title=packet.title)


@pytest.mark.parametrize("path", ["write_scope", "dependency", "statuses"])
def test_nested_object_decoders_reject_before_access(path):
    packet = _packet_payload()
    if path == "write_scope":
        packet["write_scope"] = _TrapDict(packet["write_scope"])
        with pytest.raises(ValueError, match="write_scope must be a JSON object"):
            task_packet_from_payload(packet)
    elif path == "dependency":
        packet["dependencies"] = [_TrapDict(packet["dependencies"][0])]
        with pytest.raises(ValueError, match="dependency must be a JSON object"):
            task_packet_from_payload(packet)
    else:
        with pytest.raises(ValueError, match="task status must be a JSON object"):
            status_mapping({"T1": _TrapDict(phase="ready")})


@pytest.mark.parametrize(
    "field",
    [
        "dependencies",
        "non_goals",
        "acceptance_criteria",
        "test_oracle",
        "write_scope.allow",
    ],
)
@pytest.mark.parametrize("wrapper", [_DropList, _TrapList])
def test_task_decoder_rejects_foreign_json_arrays(field, wrapper):
    packet = _packet_payload()
    target = packet
    if field == "write_scope.allow":
        target, field = packet["write_scope"], "allow"
    assert target[field]
    target[field] = wrapper(target[field])
    with pytest.raises(ValueError, match="must be a JSON array"):
        task_packet_from_payload(packet)


@pytest.mark.parametrize("wrapper", [_DropList, _TrapList])
def test_packet_sequence_and_string_set_reject_foreign_json_arrays(wrapper):
    with pytest.raises(ValueError, match="must be a JSON array"):
        packet_sequence(wrapper([_packet_payload()]))
    with pytest.raises(ValueError, match="must be a JSON array"):
        string_set(wrapper(["T1"]), field_name="IDs")


def test_native_packet_json_roundtrip_preserves_contract():
    payload = json.loads(json.dumps(_packet_payload()))
    packet = task_packet_from_payload(payload)
    assert packet.canonical_payload() == {
        key: value for key, value in payload.items() if key != "title"
    }
    assert packet_sequence([payload]) == (packet,)
    assert string_set(["T1"], field_name="IDs") == frozenset({"T1"})


class _ForeignValue(str):
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


def test_task_schema_must_be_a_native_string():
    packet = _packet_payload()
    packet["schema"] = _ForeignValue("not-agent-task")
    with pytest.raises(ValueError, match="schema"):
        task_packet_from_payload(packet)


def test_status_text_must_be_a_native_string():
    with pytest.raises(ValueError, match="reason must be a string"):
        task_status_from_payload(
            {
                "phase": "ready",
                "condition": "healthy",
                "reason": _ForeignValue("reason"),
            }
        )
