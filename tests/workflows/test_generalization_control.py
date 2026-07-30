from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.workflows.generalization_control import (
    build_generalization_control_manifest,
    load_generalization_control_manifest,
    write_generalization_control_manifest,
)

DIGESTS = tuple(f"{index:064x}" for index in range(1, 7))
BASE_COMMIT = "d60c091e21d06894d074144a4518ed9dd32e5c6b"


def _build_control():
    return build_generalization_control_manifest(
        control_name="stage-a-control",
        base_commit=BASE_COMMIT,
        action_schema="portfolio_action_v3",
        policy_identity=DIGESTS[0],
        dataset_identity=DIGESTS[1],
        feature_identity=DIGESTS[2],
        execution_identity=DIGESTS[3],
        evaluation_identity=DIGESTS[4],
        seeds=(2, 0, 1),
        folds=(1, 0),
        stage_scope="stage_a_b",
    )


def test_control_manifest_binds_comparison_identities_and_round_trips(
    tmp_path: Path,
) -> None:
    manifest = _build_control()

    assert manifest.base_commit == BASE_COMMIT
    assert manifest.seeds == (0, 1, 2)
    assert manifest.folds == (0, 1)
    assert manifest.comparison_identities == {
        "dataset": DIGESTS[1],
        "evaluation": DIGESTS[4],
        "execution": DIGESTS[3],
        "feature": DIGESTS[2],
        "policy": DIGESTS[0],
    }
    path = write_generalization_control_manifest(tmp_path / "control.json", manifest)
    assert load_generalization_control_manifest(path) == manifest


def test_control_manifest_rejects_tampered_identity(tmp_path: Path) -> None:
    manifest = _build_control()
    path = write_generalization_control_manifest(tmp_path / "control.json", manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["execution_identity"] = DIGESTS[5]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        load_generalization_control_manifest(path)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"base_commit": "abc"}, "Git SHA"),
        ({"policy_identity": "ABC"}, "SHA-256"),
        ({"seeds": (0, 0)}, "unique"),
        ({"folds": ()}, "not be empty"),
        ({"stage_scope": "stage_c"}, "stage scope"),
    ],
)
def test_control_manifest_rejects_invalid_control_contract(
    overrides: dict[str, object], message: str
) -> None:
    kwargs: dict[str, object] = {
        "control_name": "stage-a-control",
        "base_commit": BASE_COMMIT,
        "action_schema": "portfolio_action_v3",
        "policy_identity": DIGESTS[0],
        "dataset_identity": DIGESTS[1],
        "feature_identity": DIGESTS[2],
        "execution_identity": DIGESTS[3],
        "evaluation_identity": DIGESTS[4],
        "seeds": (0, 1, 2),
        "folds": (0, 1),
        "stage_scope": "stage_a_b",
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        build_generalization_control_manifest(**kwargs)
