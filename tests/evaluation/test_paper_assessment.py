from dataclasses import replace
from datetime import timedelta

import pytest

from tests.integrations.test_binance_forward import NOW
from trade_rl.evaluation.paper import assessment
from trade_rl.evaluation.paper.operations import seal_paper_study


def good_measurements():
    return assessment.Measurements(
        net_profit=120.0,
        recorded_fees=20.0,
        block_returns=(0.004, 0.004, 0.004),
        block_funding_counts=(180, 180, 180),
        maximum_drawdown=0.02,
        first_delay_seconds=1,
        final_grace_seconds=121,
        maximum_gap_seconds=61,
        terminal_flat=True,
        pending=False,
        stop_reason="terminal_close",
        quality_failures=(),
        unpaid_funding=(),
        filled_symbols=(0, 1, 2, 3),
    )


def test_paper_screen_passes_only_complete_profitable_observation_metrics():
    assert assessment.screen_reasons(good_measurements()) == []


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"net_profit": 0}, "nonpositive_net_profit"),
        ({"net_profit": 20}, "insufficient_fee_headroom"),
        ({"block_returns": (0.01, -0.001, 0.01)}, "nonpositive_block_return"),
        ({"block_returns": (0.01, float("nan"), 0.01)}, "nonfinite_metrics"),
        ({"block_funding_counts": (10, 0, 10)}, "missing_block_funding"),
        ({"maximum_drawdown": 0.1}, "drawdown_limit"),
        ({"first_delay_seconds": 181}, "initial_coverage_gap"),
        ({"maximum_gap_seconds": 181}, "observation_coverage_gap"),
        ({"final_grace_seconds": 119}, "terminal_coverage_gap"),
        ({"terminal_flat": False}, "not_flat"),
        ({"pending": True}, "pending_intent"),
        ({"stop_reason": "maximum_drawdown"}, "nonterminal_stop"),
        ({"quality_failures": ("late_funding",)}, "quality_failure"),
        ({"unpaid_funding": ({"symbol": "BTCUSDT"},)}, "unpaid_funding"),
        ({"filled_symbols": (0, 1)}, "missing_instrument_fills"),
    ],
)
def test_paper_screen_does_not_allow_profit_to_hide_failure(change, reason):
    assert reason in assessment.screen_reasons(replace(good_measurements(), **change))


def test_early_economic_evaluation_cannot_qualify_even_with_a_pinned_empty_tip(
    tmp_path,
):
    start = NOW + timedelta(minutes=5)
    root = tmp_path / "paper"
    digest = seal_paper_study(root, start_at=start, clock=lambda: NOW)
    with pytest.raises(ValueError, match="terminal deadline"):
        assessment.evaluate_paper_study(
            root,
            expected_protocol_sha256=digest,
            expected_tip=digest,
            clock=lambda: start + timedelta(days=89),
        )


def test_short_software_collection_cannot_be_relabelled_as_a_qualified_study(tmp_path):
    from trade_rl.evaluation.paper.engine import PaperSettings
    from trade_rl.evaluation.paper.operations import screen_plan
    from trade_rl.evaluation.paper.supervisor import seal_collection

    root = tmp_path / "paper"
    digest = seal_collection(
        root,
        settings=PaperSettings(
            start_at=NOW + timedelta(minutes=5), close_at=NOW + timedelta(minutes=6)
        ),
        research_plan=screen_plan(),
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="fixed prospective settings"):
        assessment.evaluate_paper_study(
            root,
            expected_protocol_sha256=digest,
            expected_tip=digest,
            clock=lambda: NOW + timedelta(days=91),
        )


def test_block_marks_are_causal_and_final_block_includes_exit_costs():
    def event(days, equity, fee=0, funding=()):
        at = NOW + timedelta(days=days)
        return {
            "event": {
                "kind": "decision",
                "payload": {
                    "request": {"at": at.isoformat()},
                    "result": {
                        "account": {"equity": equity},
                        "funding": list(funding),
                        "fills": [],
                    },
                },
            }
        }

    records = [
        event(0.001, 9990),
        event(29.999, 10100),
        event(30.001, 20000),
        event(59.999, 10200),
        event(60.001, 40000),
        event(90.001, 10250),
    ]
    status = {
        "account": {"equity": 10250, "total_cost": 20, "maximum_drawdown": 0.01},
        "terminal_flat": True,
        "pending": False,
        "stop_reason": "terminal_close",
        "quality_failures": [],
    }
    metrics = assessment.measure_observations(
        records,
        status,
        start_at=NOW,
        close_at=NOW + timedelta(days=90),
        unpaid_funding=[],
    )
    assert metrics.block_returns == pytest.approx(
        (0.01, 10200 / 10100 - 1, 10250 / 10200 - 1)
    )
    assert metrics.maximum_gap_seconds > 180


def completed_software_collection(tmp_path, monkeypatch):
    from tests.evaluation.test_paper_supervisor import collector, setup
    from trade_rl.evaluation.paper.store import PaperJournal

    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    with collector(root, digest, clock, feed) as active:
        for seconds in (2, 61, 121, 181):
            clock.value = NOW + timedelta(seconds=seconds)
            active.cycle()
    records = PaperJournal(root, expected_protocol_sha256=digest).events()
    return root, digest, records


def test_cycle_roster_reconciles_actual_commands_and_captures(tmp_path, monkeypatch):
    root, digest, records = completed_software_collection(tmp_path, monkeypatch)
    assert (
        assessment.collection_evidence_reasons(
            root,
            digest,
            records,
            start=NOW + timedelta(seconds=1),
            end=NOW + timedelta(seconds=240),
        )
        == []
    )


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("incomplete", "incomplete_collection_cycle"),
        ("empty_cycle", "cycle_command_mismatch"),
        ("unused_source", "unconsumed_or_missing_sources"),
        ("failure", "collection_failure"),
    ],
)
def test_positive_prefix_cannot_hide_incomplete_or_unused_collection(
    tmp_path, monkeypatch, defect, reason
):
    import sqlite3

    root, digest, records = completed_software_collection(tmp_path, monkeypatch)
    if defect == "incomplete":
        with sqlite3.connect(root / "collection-control.sqlite") as connection:
            connection.execute("UPDATE cycles SET complete=0 WHERE id=4")
    elif defect == "empty_cycle":
        with sqlite3.connect(root / "collection-control.sqlite") as connection:
            connection.execute(
                "INSERT INTO cycles VALUES(5,?,1)",
                ((NOW + timedelta(seconds=220)).isoformat(),),
            )
    elif defect == "unused_source":
        (root / "sources/2026-09-18/unconsumed").mkdir(parents=True)
    else:
        (root / "collection-failure.json").write_text("{}")
    assert reason in assessment.collection_evidence_reasons(
        root,
        digest,
        records,
        start=NOW + timedelta(seconds=1),
        end=NOW + timedelta(seconds=240),
    )


def test_assessment_excludes_active_collector_then_rejects_empty_complete_period(
    tmp_path,
):
    from trade_rl.evaluation.paper.supervisor import PaperCollector

    root = tmp_path / "study"
    start = NOW + timedelta(minutes=5)
    digest = seal_paper_study(root, start_at=start, clock=lambda: NOW)
    arguments = dict(
        expected_protocol_sha256=digest,
        expected_tip=digest,
        clock=lambda: start + timedelta(days=90, seconds=180),
    )
    with PaperCollector(root, expected_protocol_sha256=digest):
        with pytest.raises(RuntimeError, match="active collector"):
            assessment.evaluate_paper_study(root, **arguments)
    report = assessment.evaluate_paper_study(root, **arguments)
    assert report["decision"] == "PAPER_SCREEN_REJECTED"
    assert report["production_eligible"] is False
    assert "initial_coverage_gap" in report["reasons"]
    assert "missing_instrument_fills" in report["reasons"]
    with pytest.raises(ValueError, match="external anchor"):
        assessment.evaluate_paper_study(root, **{**arguments, "expected_tip": "a" * 64})


def test_final_assessment_requires_the_original_source_runtime(tmp_path, monkeypatch):
    root = tmp_path / "study"
    start = NOW + timedelta(minutes=5)
    digest = seal_paper_study(root, start_at=start, clock=lambda: NOW)
    monkeypatch.setattr(
        assessment, "build_candidate_run_provenance", lambda: {"changed": True}
    )
    with pytest.raises(ValueError, match="provenance"):
        assessment.evaluate_paper_study(
            root,
            expected_protocol_sha256=digest,
            expected_tip=digest,
            clock=lambda: start + timedelta(days=91),
        )


@pytest.mark.parametrize("drift", ["source", "protocol"])
def test_final_assessment_rechecks_identity_after_full_replay(
    tmp_path, monkeypatch, drift
):
    import json

    from trade_rl.evaluation.paper.supervisor import PaperCollector

    root = tmp_path / "study"
    start = NOW + timedelta(minutes=5)
    digest = seal_paper_study(root, start_at=start, clock=lambda: NOW)
    with PaperCollector(root, expected_protocol_sha256=digest):
        pass
    original = assessment.PaperEngine.unsettled_funding
    frozen = json.loads((root / "protocol.json").read_bytes())["protocol"]["study"][
        "provenance"
    ]
    changed = False
    monkeypatch.setattr(
        assessment,
        "build_candidate_run_provenance",
        lambda: {"changed": True} if changed else frozen,
    )

    def drift_during_replay(engine):
        nonlocal changed
        changed = drift == "source"
        if drift == "protocol":
            with (root / "protocol.json").open("ab") as stream:
                stream.write(b" ")
        return original(engine)

    monkeypatch.setattr(
        assessment.PaperEngine, "unsettled_funding", drift_during_replay
    )
    with pytest.raises(ValueError, match="provenance|digest"):
        assessment.evaluate_paper_study(
            root,
            expected_protocol_sha256=digest,
            expected_tip=digest,
            clock=lambda: start + timedelta(days=91),
        )
