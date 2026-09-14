from __future__ import annotations

import json
from pathlib import Path

from tests.research.test_issue541_stage_a import _write_fixture, _write_json
from tools.issue541_stage_a import verify_stage_a_bridge


def test_bridge_allows_current_implementation_plan_and_context_only(tmp_path: Path) -> None:
    original = tmp_path / "original"
    replay = tmp_path / "replay"
    _write_fixture(original, implementation="1" * 64, fingerprint="2" * 64)
    _write_fixture(replay, implementation="3" * 64, fingerprint="4" * 64)

    for root, implementation in ((original, "1" * 64), (replay, "3" * 64)):
        plan_path = root / "study" / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["implementation_digest"] = implementation
        plan["runtime_environment_digest"] = "e" * 64
        _write_json(plan_path, plan)

    replay_manifest_path = replay / "study" / "baseline" / "evidence" / "manifest.json"
    replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    replay_manifest["research_context_digest"] = "7" * 64
    _write_json(replay_manifest_path, replay_manifest)

    for seed in (0, 1):
        provenance_path = (
            replay
            / "study"
            / "baseline"
            / "evidence"
            / "runs"
            / f"seed-{seed}"
            / "provenance.json"
        )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["research_context_digest"] = "7" * 64
        _write_json(provenance_path, provenance)

    report = verify_stage_a_bridge(original, replay)

    assert report["status"] == "PASS"
    assert report["study_plan_implementation_changed"] is True
    assert report["research_context_changed"] is True
    assert report["raw_returns_exact"] is True
    assert report["economic_summaries_exact"] is True
