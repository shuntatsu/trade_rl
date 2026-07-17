from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.workflows.selection_authorization import (
    SelectionAuthorization,
    load_selection_authorization,
    write_selection_authorization,
)


def _authorization() -> SelectionAuthorization:
    return SelectionAuthorization.create(
        walk_forward_run_digest="a" * 64,
        gate_evidence_digest="b" * 64,
        dataset_id="c" * 64,
        selected_configuration="ppo-15m-target",
        candidate_config_digest="d" * 64,
        seeds=(0, 1, 2),
    )


def test_selection_authorization_round_trips_and_verifies(tmp_path: Path) -> None:
    authorization = _authorization()
    path = write_selection_authorization(tmp_path / "selection-authorization.json", authorization)
    loaded = load_selection_authorization(path)

    assert loaded == authorization
    loaded.verify(
        dataset_id="c" * 64,
        candidate_config_digest="d" * 64,
        seeds=(0, 1, 2),
    )


@pytest.mark.parametrize(
    "dataset_id,candidate_digest,seeds,match",
    [
        ("e" * 64, "d" * 64, (0, 1, 2), "dataset"),
        ("c" * 64, "e" * 64, (0, 1, 2), "candidate"),
        ("c" * 64, "d" * 64, (0, 1), "seed"),
    ],
)
def test_selection_authorization_rejects_identity_mismatch(
    dataset_id: str,
    candidate_digest: str,
    seeds: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _authorization().verify(
            dataset_id=dataset_id,
            candidate_config_digest=candidate_digest,
            seeds=seeds,
        )
