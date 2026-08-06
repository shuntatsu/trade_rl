from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trade_rl.studio.contracts import OverviewEvidenceSummary, StudioAlert
from trade_rl.studio.overview_evidence import summarize_overview_evidence


def report(
    *,
    status: str = "VALID",
    node_statuses: tuple[str, ...] = ("VERIFIED",),
    files: str = "VERIFIED",
) -> SimpleNamespace:
    nodes = tuple(
        SimpleNamespace(
            key="run_manifest" if index == 1 else f"node-{index}",
            required=True,
            status=value,
        )
        for index, value in enumerate(node_statuses)
    )
    return SimpleNamespace(
        status=status,
        nodes=nodes,
        files=SimpleNamespace(status=files),
    )


def test_contract_rejects_more_verified_than_required() -> None:
    with pytest.raises(ValueError):
        OverviewEvidenceSummary(
            run_resource_id="run-x",
            status="VERIFIED",
            required_count=1,
            verified_count=2,
            blocker_count=0,
        )


def test_alert_requires_stable_identity_and_accepts_missing_timestamp() -> None:
    alert = StudioAlert(
        id="dataset:no-valid",
        level="warning",
        message="missing",
        age="now",
    )
    assert alert.occurred_at is None


def test_no_run_is_unavailable() -> None:
    summary = summarize_overview_evidence(None, run_resource_id=None)
    assert summary.status == "UNAVAILABLE"
    assert summary.required_count == 0
    assert summary.verified_count == 0
    assert summary.blocker_count == 0


def test_valid_required_nodes_are_verified(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "trade_rl.studio.overview_evidence.inspect_run_evidence",
        lambda *_args, **_kwargs: report(
            node_statuses=("VERIFIED", "VERIFIED")
        ),
    )
    summary = summarize_overview_evidence(tmp_path, run_resource_id="run-x")

    assert summary.status == "VERIFIED"
    assert summary.required_count == 2
    assert summary.verified_count == 2
    assert summary.blocker_count == 0


def test_invalid_required_node_is_a_blocker(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "trade_rl.studio.overview_evidence.inspect_run_evidence",
        lambda *_args, **_kwargs: report(
            status="INVALID",
            node_statuses=("VERIFIED", "INVALID"),
            files="INVALID",
        ),
    )
    summary = summarize_overview_evidence(tmp_path, run_resource_id="run-x")

    assert summary.status == "INVALID"
    assert summary.required_count == 2
    assert summary.verified_count == 1
    assert summary.blocker_count == 1
