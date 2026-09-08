from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import (
    _config,
    _dataset,
    _fake_execute,
    _plan,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import publish_market_dataset_artifact
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.evidence import (
    execute_evidence_set,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.store import StudyStore


def test_load_evidence_set_rejects_tampered_member_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    artifact = publish_market_dataset_artifact(dataset_root, dataset)
    plan = _plan(dataset, artifact)
    monkeypatch.setattr(evidence_module, "execute_candidate_run", _fake_execute())
    store = StudyStore(tmp_path / "study")
    context = content_digest({"kind": "baseline", "study": plan.digest})

    execute_evidence_set(
        store=store,
        target=Path("evidence/baseline"),
        dataset_root=dataset_root,
        plan=plan,
        config=_config(),
        research_context_digest=context,
    )

    summary_path = (
        store.root / "evidence" / "baseline" / "runs" / "seed-2" / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["dataset_id"] = "f" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="digest|Run"):
        load_evidence_set(store.root / "evidence" / "baseline")
