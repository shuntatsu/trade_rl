from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.evaluation.experiments.bootstrap.test_workflow import (
    _config_path,
    _install_fakes,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    bootstrap_canonical_m2_study,
    inspect_canonical_m2_bootstrap,
)
from trade_rl.integrations.binance import vision_cache_path


def _published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _install_fakes(monkeypatch)
    output = tmp_path / "canonical-m2"
    bootstrap_canonical_m2_study(_config_path(tmp_path), output)
    return output


def _rewrite_manifest(output: Path, mutate) -> None:
    path = output / "bootstrap-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    body = dict(payload)
    body.pop("bootstrap_digest", None)
    payload["bootstrap_digest"] = content_digest(body)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_self_consistent_raw_source_tamper_is_rejected_by_frozen_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _published(tmp_path, monkeypatch)
    manifest = json.loads(
        (output / "bootstrap-manifest.json").read_text(encoding="utf-8")
    )
    url = manifest["raw_source_roster"][0]["url"]
    cache = vision_cache_path(output / "source" / "vision-cache", url)
    payload = cache.read_bytes() + b"tamper"
    cache.write_bytes(payload)
    sidecar = cache.with_suffix(".json")
    evidence = json.loads(sidecar.read_text(encoding="utf-8"))
    evidence["sha256"] = hashlib.sha256(payload).hexdigest()
    evidence["size_bytes"] = len(payload)
    sidecar.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="raw source roster|source evidence"):
        inspect_canonical_m2_bootstrap(output)


def test_self_consistent_manifest_dataset_digest_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _published(tmp_path, monkeypatch)
    _rewrite_manifest(
        output,
        lambda payload: payload.__setitem__("dataset_artifact_digest", "f" * 64),
    )

    with pytest.raises(ValueError, match="dataset.*digest|dataset artifact"):
        inspect_canonical_m2_bootstrap(output)


def test_study_plan_tamper_is_rejected_even_when_json_remains_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _published(tmp_path, monkeypatch)
    plan_path = output / "study" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["research_question"] = "tampered question"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="Study|study|digest"):
        inspect_canonical_m2_bootstrap(output)


def test_study_and_manifest_cannot_agree_on_a_different_dataset_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _published(tmp_path, monkeypatch)
    plan_path = output / "study" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["dataset_artifact_digest"] = "f" * 64
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    tampered_study_digest = content_digest(plan)

    def mutate(manifest: dict[str, object]) -> None:
        manifest["dataset_artifact_digest"] = "f" * 64
        manifest["study_digest"] = tampered_study_digest

    _rewrite_manifest(output, mutate)

    with pytest.raises(ValueError, match="dataset.*digest|dataset artifact|Study"):
        inspect_canonical_m2_bootstrap(output)


def test_bootstrap_config_and_manifest_cannot_be_rewritten_after_study_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _published(tmp_path, monkeypatch)
    bootstrap_path = output / "bootstrap.json"
    config = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    config["research_question"] = "post-hoc research question"
    bootstrap_path.write_text(json.dumps(config), encoding="utf-8")
    config_digest = content_digest(config)
    _rewrite_manifest(
        output,
        lambda payload: payload.__setitem__("bootstrap_config_digest", config_digest),
    )

    with pytest.raises(ValueError, match="research question|Study|study"):
        inspect_canonical_m2_bootstrap(output)
