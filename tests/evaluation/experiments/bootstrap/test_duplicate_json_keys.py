"""Ambiguous JSON must not acquire the same parsed identity as clean evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.evaluation.experiments.bootstrap.test_config import _valid_payload
from tests.evaluation.experiments.bootstrap.test_workflow import (
    _config_path,
    _install_fakes,
)
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    bootstrap_canonical_m2_study,
    inspect_canonical_m2_bootstrap,
)


@pytest.mark.parametrize(
    ("needle", "insertion"),
    (
        ('"bootstrap_seed": 17', '"bootstrap_seed": 99, "bootstrap_seed": 17'),
        ('"bootstrap_seed": 17', '"bootstrap_seed": 17, "bootstrap_seed": 17'),
        ('"gross_budget": 0.5', '"gross_budget": 0.9, "gross_budget": 0.5'),
        ('"gross_budget": 0.5', r'"gross_budg\u0065t": 0.9, "gross_budget": 0.5'),
    ),
)
def test_config_rejects_duplicate_decoded_member_names(
    tmp_path: Path, needle: str, insertion: str
) -> None:
    path = tmp_path / "bootstrap.json"
    clean = json.dumps(_valid_payload())
    ambiguous = clean.replace(needle, insertion, 1)
    assert ambiguous != clean and json.loads(ambiguous) == json.loads(clean)
    path.write_text(clean, encoding="utf-8")
    load_canonical_m2_bootstrap_config(path)
    path.write_text(ambiguous, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON member"):
        load_canonical_m2_bootstrap_config(path)


@pytest.mark.parametrize(
    "relative",
    (
        "bootstrap-manifest.json",
        "source/vision-plan.json",
        "source/vision-resolution.json",
    ),
)
def test_inspection_rejects_ambiguous_published_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    _install_fakes(monkeypatch)
    root = tmp_path / "canonical-m2"
    bootstrap_canonical_m2_study(_config_path(tmp_path), root)
    assert inspect_canonical_m2_bootstrap(root).root == root
    path = root / relative
    clean = path.read_text(encoding="utf-8")
    payload = json.loads(clean)
    key = next(iter(payload))
    ambiguous = clean.replace("{", "{" + json.dumps(key) + ":null,", 1)
    assert json.loads(ambiguous) == payload
    path.write_text(ambiguous, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON member"):
        inspect_canonical_m2_bootstrap(root)
    assert path.read_text(encoding="utf-8") == ambiguous
