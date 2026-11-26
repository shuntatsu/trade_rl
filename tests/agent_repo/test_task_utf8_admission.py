from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools.agent_repo.__main__ import main
from tools.agent_repo.coordination.model import ExecutionMode, TaskPacket, WriteScope


def _packet() -> TaskPacket:
    return TaskPacket(
        task_id="T1",
        task_revision=1,
        parent_issue=1067,
        title="Unicode " + chr(0x1F600),
        objective="caf" + chr(233),
        execution_mode=ExecutionMode.WRITE,
        write_scope=WriteScope(allow=("tools/**",), deny=("docs/old",)),
        resource_keys=("schema:caf" + chr(233),),
        base_sha="a" * 40,
    )


@pytest.mark.parametrize(
    "field",
    [
        "title",
        "objective",
        "deliverable",
        "non_goals",
        "resource_keys",
        "acceptance_criteria",
        "test_oracle",
    ],
)
@pytest.mark.parametrize("bad", [chr(0xD800), chr(0xDFFF)])
def test_task_refuses_unencodable_text(field: str, bad: str) -> None:
    value = "schema:" + bad if field == "resource_keys" else "text" + bad
    replacement = value if field in {"title", "objective", "deliverable"} else (value,)
    with pytest.raises(ValueError, match="valid Unicode text"):
        replace(_packet(), **{field: replacement})


@pytest.mark.parametrize("field", ["allow", "deny"])
def test_write_scope_refuses_unencodable_text(field: str) -> None:
    with pytest.raises(ValueError, match="valid Unicode text"):
        replace(_packet().write_scope, **{field: ("path" + chr(0xD800),)})


@pytest.mark.parametrize("value", [None, 3, [], {}])
def test_deliverable_requires_text(value: object) -> None:
    with pytest.raises(ValueError, match="deliverable.*string"):
        replace(_packet(), deliverable=value)


@pytest.mark.parametrize("command", ["digest", "ready"])
@pytest.mark.parametrize("field", ["resource_keys", "objective", "write_scope.allow"])
def test_task_cli_refuses_escaped_surrogate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, command: str
) -> None:
    packet = _packet()
    payload = {"title": packet.title, **packet.canonical_payload()}
    bad = ("schema:" if field == "resource_keys" else "text") + chr(0xD800)
    if field == "write_scope.allow":
        payload["write_scope"]["allow"] = [bad]
    else:
        payload[field] = bad if field == "objective" else [bad]
    data = (
        payload
        if command == "digest"
        else {
            "packets": [payload],
            "statuses": {
                "T1": {"phase": "ready", "condition": "healthy", "reason": None}
            },
            "leased_task_ids": [],
        }
    )
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data, ensure_ascii=True), encoding="ascii")
    assert main(["task", command, str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "valid Unicode text" in captured.err


def test_valid_unicode_task_preserves_digest() -> None:
    from tools.agent_repo.coordination.model import TaskPhase, TaskStatus
    from tools.agent_repo.coordination.scheduler import ready_tasks

    packet = _packet()
    assert (
        packet.contract_digest()
        == "3283f33eb9faaffc33c35c357ce77fab1375242edb423f7489a25be2356a3fe9"
    )
    assert ready_tasks(
        (packet,), {"T1": TaskStatus(TaskPhase.READY)}, leased_task_ids=()
    ) == ("T1",)


@pytest.mark.parametrize("command", ["digest", "ready"])
@pytest.mark.parametrize(
    ("key", "diagnostic"),
    [
        ("mystery:" + chr(27) + "[2Jspoofed", "unknown resource namespace"),
        ("schema:line" + chr(10) + "spoofed", "invalid resource key"),
        ("file:../" + chr(0x202E) + "spoofed", "invalid file resource key"),
        ("file:part?" + chr(27) + "spoofed", "invalid file resource key"),
    ],
)
def test_resource_admission_diagnostic_does_not_render_untrusted_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    key: str,
    diagnostic: str,
    command: str,
) -> None:
    packet = _packet()
    payload = {"title": packet.title, **packet.canonical_payload()}
    payload["resource_keys"] = [key]
    data = (
        payload
        if command == "digest"
        else {
            "packets": [payload],
            "statuses": {
                "T1": {"phase": "ready", "condition": "healthy", "reason": None}
            },
            "leased_task_ids": [],
        }
    )
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data, ensure_ascii=True), encoding="ascii")
    assert main(["task", command, str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == diagnostic + "\n"


@pytest.mark.parametrize(
    ("field", "diagnostic"),
    [
        ("statuses", "status references unknown task"),
        ("leased_task_ids", "lease references unknown task"),
    ],
)
@pytest.mark.parametrize("control", [chr(27), chr(10), chr(0x202E)])
def test_ready_unknown_id_diagnostic_omits_untrusted_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    diagnostic: str,
    control: str,
) -> None:
    packet = _packet()
    payload = {"title": packet.title, **packet.canonical_payload()}
    status = {"phase": "ready", "condition": "healthy", "reason": None}
    data = {"packets": [payload], "statuses": {"T1": status}, "leased_task_ids": []}
    key = "unknown" + control + "spoofed"
    if field == "statuses":
        data["statuses"][key] = status
    else:
        data["leased_task_ids"] = [key]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data, ensure_ascii=True), encoding="ascii")
    assert main(["task", "ready", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == diagnostic + "\n"
