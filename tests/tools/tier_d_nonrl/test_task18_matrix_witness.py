"""Admission and stale-state falsifiers for the explicit matrix witness."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.tools.tier_d_nonrl import task18_matrix_witness as witness
from tests.tools.tier_d_nonrl.test_task18_recovery import _bundle
from trade_rl.artifacts.canonical import canonical_json_bytes


@pytest.fixture
def untouched(tmp_path):
    run, _, _ = _bundle(tmp_path / "receipt-fixture")
    return replace(
        run,
        arms=tuple(
            replace(
                a,
                status="NOT_ATTEMPTED",
                reason_code="NOT_STARTED",
                initial_equity_usdt=None,
                terminal_equity_usdt=None,
                max_drawdown=None,
                terminal_flat=None,
                pending_order_count=None,
                native_ledger_sha256=None,
                arm_result_sha256=None,
                external_report=None,
                error_code=None,
                progress=None,
            )
            for a in run.arms
        ),
    )


def test_witness_only_dispatches_next_untouched_recipe_slots(untouched):
    ids = witness._IDS
    assert witness._batch_ids(untouched, 4, 2) == ids[:4]
    first = replace(untouched.arms[0], status="COMPLETED", reason_code=None)
    progressed = replace(untouched, arms=(first, *untouched.arms[1:]))
    assert witness._batch_ids(progressed, 12, 4) == ids[1:13]


@pytest.mark.parametrize(
    ("count", "workers"), ((0, 1), (13, 1), (True, 1), (4, 0), (4, 5), (4, True))
)
def test_witness_bounds_concurrency(untouched, count, workers):
    with pytest.raises(ValueError):
        witness._batch_ids(untouched, count, workers)


@pytest.mark.parametrize("status", ("RUNNING", "INTERRUPTED", "FAILED"))
def test_witness_never_implicitly_retries_stopped_slots(untouched, status):
    run = replace(
        untouched, arms=(replace(untouched.arms[0], status=status), *untouched.arms[1:])
    )
    with pytest.raises(ValueError, match="explicit recovery"):
        witness._batch_ids(run, 4, 2)


def test_witness_refuses_duplicate_or_closed_roster(untouched):
    for run in (
        replace(untouched, runtime_completed=True),
        replace(untouched, arms=(untouched.arms[1], *untouched.arms[1:])),
        replace(untouched, arms=tuple(reversed(untouched.arms))),
    ):
        with pytest.raises(ValueError, match="roster"):
            witness._batch_ids(run, 4, 2)


def test_witness_stale_metadata_pin_is_rejected_before_decode(tmp_path):
    root = tmp_path.resolve()
    raw = canonical_json_bytes({"scope": "artificial"})
    expected = witness._sha(raw)
    path = root / "state.json"
    path.write_bytes(raw)
    assert witness._read_pinned(root, path.name, expected) == raw
    path.write_bytes(b"untrusted different bytes")
    with pytest.raises(ValueError, match="external pin"):
        witness._read_pinned(root, path.name, expected)
