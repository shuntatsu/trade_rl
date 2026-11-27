"""Evidence fingerprints must not erase ambiguous input JSON members."""

import json
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence_identity import _published_evidence
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.evidence import load_evidence_set


@pytest.mark.parametrize("case", ["root-conflict", "nested-equal", "escaped-alias"])
def test_public_evidence_loader_rejects_duplicate_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    store, plan, published = _published_evidence(tmp_path, monkeypatch)
    root = store.root / "evidence" / "baseline"
    manifest = root / "manifest.json"
    text = manifest.read_text(encoding="utf-8")
    if case == "root-conflict":
        needle = f'"fingerprint":"{published.fingerprint}"'
        replacement = '"fingerprint":"' + "0" * 64 + '",' + needle
    else:
        needle = f'"ppo_seed":{plan.ppo_seeds[0]}'
        duplicate = (
            needle
            if case == "nested-equal"
            else needle.replace("ppo_seed", r"\u0070po_seed")
        )
        replacement = duplicate + "," + needle
    assert text.count(needle) == 1
    manifest.write_text(text.replace(needle, replacement), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="duplicate JSON member"):
        load_evidence_set(root, expected_plan=plan)


def test_evidence_loader_preserves_valid_noncompact_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, plan, published = _published_evidence(tmp_path, monkeypatch)
    root = store.root / "evidence" / "baseline"
    manifest = root / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert load_evidence_set(root, expected_plan=plan).evidence == published
