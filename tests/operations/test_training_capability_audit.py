from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from trade_rl.artifacts.hashing import content_digest
from trade_rl.operations import _training_capability_audit_impl as impl
from trade_rl.operations.training_capability_audit import run_training_capability_audit


class _TrainingResult:
    pass


def test_run_training_capability_audit_preserves_report_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "audit"
    root.mkdir()
    (root / "stale.txt").write_text("stale", encoding="utf-8")

    results = {name: _TrainingResult() for name in ("ppo", "sac", "td3", "tqc")}

    def train_algorithm(output_root: Path, algorithm: str):
        assert output_root == root
        return {"algorithm": algorithm, "status": "pass"}, results[algorithm]

    monkeypatch.setattr(impl, "_train_algorithm", train_algorithm)
    monkeypatch.setattr(
        impl, "_behavior_cloning_training", lambda _: {"status": "pass"}
    )
    monkeypatch.setattr(impl, "_export_ppo", lambda _: {"status": "pass"})
    monkeypatch.setattr(
        impl,
        "_resume_replay",
        lambda output_root, source: {
            "source_matches_sac": source is results["sac"],
            "status": "pass",
        },
    )
    monkeypatch.setattr(
        impl, "_residual_feature_training", lambda _: {"status": "pass"}
    )
    monkeypatch.setattr(impl, "_sequence_training", lambda _: {"status": "pass"})
    monkeypatch.setattr(impl, "_resume_ppo", lambda _: {"status": "pass"})

    report = run_training_capability_audit(root)

    assert not (root / "stale.txt").exists()
    assert report["schema_version"] == "full_training_capability_audit_v1"
    algorithms = cast(dict[str, object], report["algorithms"])
    replay_resume = cast(dict[str, object], report["replay_resume"])
    assert set(algorithms) == {"ppo", "sac", "td3", "tqc"}
    assert replay_resume["source_matches_sac"] is True

    unsigned = dict(report)
    digest = unsigned.pop("digest")
    assert digest == content_digest(unsigned)

    expected_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    assert (root / "audit-report.json").read_bytes() == expected_bytes
    assert json.loads(expected_bytes) == report
