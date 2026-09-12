from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_SHA = "a" * 40
DIGEST = "f" * 64


def _run(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not current else f"{ROOT}{os.pathsep}{current}"
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_repo", *args],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _packet(
    task_id: str,
    *,
    risk: str = "medium",
    resource_keys: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema": "agent_task_v1",
        "task_id": task_id,
        "task_revision": 1,
        "parent_issue": 500,
        "title": task_id,
        "objective": f"Implement {task_id}",
        "execution_mode": "write",
        "non_goals": [],
        "dependencies": [],
        "write_scope": {"allow": [f"tools/{task_id}/**"], "deny": ["trade_rl/**"]},
        "resource_keys": resource_keys or [],
        "capabilities": ["repository_read", "lease_branch_write"],
        "acceptance_criteria": ["works"],
        "test_oracle": ["deterministic"],
        "base_sha": BASE_SHA,
        "risk": risk,
        "deliverable": "pull_request",
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_task_cli_digest_ready_status_and_dashboard_are_network_free(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "packet.json"
    _write_json(packet_path, _packet("T1"))

    snapshot_path = tmp_path / "snapshot.json"
    _write_json(
        snapshot_path,
        {
            "packets": [
                _packet("T1", risk="low", resource_keys=["identity:shared"]),
                _packet("T2", risk="high", resource_keys=["identity:shared"]),
                _packet("T3", resource_keys=["authority:other"]),
            ],
            "statuses": {
                task_id: {"phase": "ready", "condition": "healthy", "reason": None}
                for task_id in ("T1", "T2", "T3")
            },
            "leased_task_ids": [],
            "evidence_task_ids": [],
        },
    )

    status_path = tmp_path / "status.json"
    _write_json(
        status_path,
        {
            "task_id": "T1",
            "task_revision": 1,
            "task_contract_digest": DIGEST,
            "phase": "review",
            "condition": "stale",
            "base_sha": BASE_SHA,
            "owner": None,
            "lease_epoch": None,
            "lease_branch": None,
            "head_sha": None,
            "pr_id": "507",
            "checkpoint": "reviewable HEAD",
            "blocking_reason": None,
            "stale_reason": "head_changed",
        },
    )

    dashboard_path = tmp_path / "dashboard.json"
    _write_json(
        dashboard_path,
        {
            "parent_title": "#500 Agent Coordination Plane",
            "main_sha": "b" * 40,
            "tasks": [
                {
                    "task_id": "T2",
                    "title": "Second",
                    "phase": "complete",
                    "condition": "healthy",
                },
                {
                    "task_id": "T1",
                    "title": "First",
                    "phase": "review",
                    "condition": "stale",
                    "owner": "agent-a",
                    "pr_id": "507",
                    "dependency_summary": "waits T0",
                    "stale_reason": "head_changed",
                },
            ],
        },
    )

    before = sorted(path.relative_to(tmp_path) for path in tmp_path.iterdir())

    digest = _run(tmp_path, "task", "digest", str(packet_path))
    ready = _run(tmp_path, "task", "ready", str(snapshot_path))
    rendered = _run(tmp_path, "task", "status-render", str(status_path))
    assert rendered.returncode == 0, rendered.stderr
    rendered_payload = json.loads(rendered.stdout)
    status_text = tmp_path / "status.txt"
    status_text.write_text(rendered_payload["comment"], encoding="utf-8")
    parsed = _run(tmp_path, "task", "status-parse", str(status_text))
    dashboard = _run(tmp_path, "task", "dashboard", str(dashboard_path))

    assert digest.returncode == 0, digest.stderr
    digest_payload = json.loads(digest.stdout)
    assert digest_payload["task_id"] == "T1"
    assert len(digest_payload["contract_digest"]) == 64

    assert ready.returncode == 0, ready.stderr
    assert json.loads(ready.stdout) == {"ready_task_ids": ["T1", "T3"]}

    assert parsed.returncode == 0, parsed.stderr
    parsed_payload = json.loads(parsed.stdout)
    assert parsed_payload["task_id"] == "T1"
    assert parsed_payload["phase"] == "review"
    assert parsed_payload["condition"] == "stale"

    assert dashboard.returncode == 0, dashboard.stderr
    dashboard_text = json.loads(dashboard.stdout)["dashboard"]
    assert dashboard_text.index("T1") < dashboard_text.index("T2")
    assert "1 / 2 complete" in dashboard_text
    assert "review / stale" in dashboard_text

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.iterdir())
    assert after == before + [Path("status.txt")]


def test_task_cli_invalid_input_fails_closed_without_stdout(tmp_path: Path) -> None:
    invalid_packet = tmp_path / "invalid.json"
    _write_json(invalid_packet, {"schema": "agent_task_v1", "task_id": "T1"})

    result = _run(tmp_path, "task", "digest", str(invalid_packet))

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr
