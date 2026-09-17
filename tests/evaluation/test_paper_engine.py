import hashlib
import json
from datetime import timedelta

import numpy as np
import pytest

from tests.integrations.test_binance_forward import MS, NOW, PublicFeed
from tests.integrations.test_binance_forward_rules import RuleFeed
from trade_rl.evaluation.paper.engine import EvidenceRef, PaperEngine, PaperSettings
from trade_rl.integrations.binance.forward import capture_forward_snapshot
from trade_rl.integrations.binance.forward_rules import capture_forward_rules


class Feed(PublicFeed):
    def __init__(
        self,
        at,
        *,
        funding=None,
        thin=False,
        mark=100.5,
        next_funding=28800,
        events=None,
    ):
        super().__init__()
        self.at, self.funding, self.thin, self.mark = at, funding, thin, mark
        self.next_funding = next_funding
        self.events = events

    def _request_bytes(self, url):
        payload = json.loads(super()._request_bytes(url))
        milliseconds = int(self.at.timestamp() * 1000)
        if isinstance(payload, dict):
            for key in ("serverTime", "E", "T", "time"):
                if key in payload:
                    payload[key] = milliseconds
            if "bids" in payload:
                size = "0.01" if self.thin and "fapi" in url else "1000"
                for side in ("bids", "asks"):
                    for level in payload[side]:
                        level[1] = size
            if "markPrice" in payload:
                payload["markPrice"] = str(self.mark)
                payload["nextFundingTime"] = MS + int(self.next_funding * 1000)
        elif self.funding is not None or self.events is not None:
            events = (
                self.events[payload[0]["symbol"]]
                if self.events is not None
                else [self.funding]
            )
            for timestamp, rate in events:
                payload.append(
                    dict(
                        symbol=payload[0]["symbol"],
                        fundingTime=int(timestamp.timestamp() * 1000),
                        fundingRate=str(rate),
                        markPrice="100",
                    )
                )
        return json.dumps(payload).encode()


def market(root, name, seconds, **kwargs):
    at = NOW + timedelta(seconds=seconds)
    capture_forward_snapshot(
        root / name,
        transport=Feed(at, **kwargs),
        clock=lambda: at,
        monotonic=lambda: 0,
    )
    raw = (root / name / "snapshot.json").read_bytes()
    return EvidenceRef(name, hashlib.sha256(raw).hexdigest())


def rules(root, name="rules"):
    capture_forward_rules(
        root / name, transport=RuleFeed(), clock=lambda: NOW, monotonic=lambda: 0
    )
    return EvidenceRef(
        name, hashlib.sha256((root / name / "rules.json").read_bytes()).hexdigest()
    )


def setup(root, *, close_seconds=3600):
    settings = PaperSettings(
        start_at=NOW - timedelta(seconds=1),
        close_at=NOW + timedelta(seconds=close_seconds),
    )
    digest = PaperEngine.initialize(root, settings=settings, study={"purpose": "test"})
    return PaperEngine(root, expected_protocol_sha256=digest), rules(root), digest


def entry(root, *, thin=False, close_seconds=3600):
    engine, rule, digest = setup(root, close_seconds=close_seconds)
    decision = engine.decide(
        "d1", market=market(root, "m1", 0), rules=rule, at=NOW + timedelta(seconds=0.1)
    )
    fill = engine.execute(
        "e1", market=market(root, "m2", 1, thin=thin), at=NOW + timedelta(seconds=1.1)
    )
    return engine, rule, digest, decision, fill


def result(record):
    return record["event"]["payload"]["result"]


def test_decision_precedes_four_fills_and_reopening_reproduces_account(tmp_path):
    root = tmp_path / "paper"
    engine, _, digest, decision, fill = entry(root)
    assert result(decision)["account"]["fill_count"] == 0
    outcome = result(fill)
    assert len(outcome["fills"]) == 4
    assert all(row["unfilled_lots"] == 0 for row in outcome["fills"])
    assert outcome["account"]["fill_count"] == 4
    assert outcome["account"]["equity"] < 10000
    assert outcome["account"]["funding_pnl"] == 0
    assert outcome["stop_reason"] is None
    reopened = PaperEngine(root, expected_protocol_sha256=digest)
    assert reopened.status() == engine.status()
    assert reopened.status()["production_eligible"] is False


def test_exact_retry_does_not_duplicate_fill_and_conflicting_retry_fails(tmp_path):
    root = tmp_path / "paper"
    engine, _, digest, _, fill = entry(root)
    ref = EvidenceRef(
        "m2", hashlib.sha256((root / "m2/snapshot.json").read_bytes()).hexdigest()
    )
    reopened = PaperEngine(root, expected_protocol_sha256=digest)
    assert reopened.execute("e1", market=ref, at=NOW + timedelta(seconds=1.1)) == fill
    assert reopened.status() == engine.status()
    with pytest.raises(ValueError, match="idempotency"):
        reopened.execute("e1", market=ref, at=NOW + timedelta(seconds=1.2))


def test_funding_uses_held_quantity_at_settlement_not_current_quantity(tmp_path):
    root = tmp_path / "paper"
    engine, rule, _, _, _ = entry(root)
    engine.decide(
        "d2",
        market=market(root, "m3", 2, funding=(NOW + timedelta(seconds=0.5), 0.01)),
        rules=rule,
        at=NOW + timedelta(seconds=2.1),
    )
    assert engine.status()["account"]["funding_pnl"] == 0
    ref = market(root, "m4", 3, funding=(NOW + timedelta(seconds=2.5), 0.01))
    paid = result(
        engine.decide("d3", market=ref, rules=rule, at=NOW + timedelta(seconds=3.1))
    )
    quantity = engine.status()["account"]["quantities"][1]
    expected = -2 * quantity * 100 * 0.01
    assert paid["account"]["funding_pnl"] == pytest.approx(expected)
    assert len(paid["funding"]) == 2
    engine.decide(
        "d4",
        market=market(root, "m5", 4, funding=(NOW + timedelta(seconds=2.5), 0.01)),
        rules=rule,
        at=NOW + timedelta(seconds=4.1),
    )
    assert engine.status()["account"]["funding_pnl"] == pytest.approx(expected)


def test_asymmetric_fills_stop_and_later_quotes_actually_flatten(tmp_path):
    root = tmp_path / "paper"
    engine, rule, _, _, fill = entry(root, thin=True)
    assert result(fill)["stop_reason"] == "unmatched_hedge"
    assert not engine.status()["terminal_flat"]
    engine.decide(
        "exit-d",
        market=market(root, "m3", 2),
        rules=rule,
        at=NOW + timedelta(seconds=2.1),
    )
    closed = result(
        engine.execute(
            "exit-e", market=market(root, "m4", 3), at=NOW + timedelta(seconds=3.1)
        )
    )
    # Small perpetual residuals violate the frozen minimum notional: preserve them.
    assert closed["account"]["quantities"][0] == 0
    assert closed["account"]["quantities"][1] != 0
    assert not engine.status()["terminal_flat"]


def test_terminal_close_requires_later_actual_fills(tmp_path):
    root = tmp_path / "paper"
    engine, rule, digest, _, _ = entry(root, close_seconds=3)
    engine.decide(
        "close-d",
        market=market(root, "m3", 3),
        rules=rule,
        at=NOW + timedelta(seconds=3.1),
    )
    assert not engine.status()["terminal_flat"]
    engine.execute(
        "close-e", market=market(root, "m4", 4), at=NOW + timedelta(seconds=4.1)
    )
    assert engine.status()["terminal_flat"]
    assert engine.status()["account"]["fill_count"] == 8
    assert (
        engine.status() == PaperEngine(root, expected_protocol_sha256=digest).status()
    )


def test_stale_capture_failed_append_and_reused_capacity_do_not_change_book(tmp_path):
    root = tmp_path / "paper"
    engine, rule, digest = setup(root)
    ref = market(root, "m1", 0)
    other = PaperEngine(root, expected_protocol_sha256=digest)
    engine.decide("d1", market=ref, rules=rule, at=NOW + timedelta(seconds=0.1))
    unchanged = other.status()
    with pytest.raises(ValueError, match="stale"):
        other.decide("other", market=ref, rules=rule, at=NOW + timedelta(seconds=0.1))
    assert other.status() == unchanged
    unchanged = engine.status()
    with pytest.raises(ValueError, match="after|stale"):
        engine.execute("bad", market=ref, at=NOW + timedelta(seconds=1))
    assert engine.status() == unchanged
    with pytest.raises(ValueError, match="stale"):
        engine.execute(
            "late", market=market(root, "m2", 1), at=NOW + timedelta(seconds=20)
        )
    assert engine.status() == unchanged


def test_gap_cancels_pending_intent_and_preserves_positions(tmp_path):
    root = tmp_path / "paper"
    engine, _, digest, _, _ = entry(root)
    before = engine.status()["account"]["quantities"]
    observed_at = engine.status()["observed_at"]
    engine.gap("gap1", at=NOW + timedelta(seconds=10), reason="source unavailable")
    status = engine.status()
    assert status["account"]["quantities"] == before
    assert status["stop_reason"] == "data_gap"
    assert status["quality_failures"] == ["source unavailable"]
    assert status["observed_at"] == observed_at
    assert status["last_command_at"] == (NOW + timedelta(seconds=10)).isoformat()
    assert status == PaperEngine(root, expected_protocol_sha256=digest).status()


def test_changed_evidence_or_path_escape_is_rejected(tmp_path):
    root = tmp_path / "paper"
    engine, rule, digest = setup(root)
    ref = market(root, "m1", 0)
    with pytest.raises(ValueError, match="path"):
        engine.decide(
            "bad", market=EvidenceRef("../m1", ref.sha256), rules=rule, at=NOW
        )
    engine.decide("d1", market=ref, rules=rule, at=NOW + timedelta(seconds=0.1))
    (root / "m1/BTCUSDT_mark.raw").write_text("{}")
    with pytest.raises(ValueError, match="hash|digest|size"):
        PaperEngine(root, expected_protocol_sha256=digest)


def test_revised_funding_never_rewrites_cash(tmp_path):
    root = tmp_path / "paper"
    engine, rule, _, _, _ = entry(root)
    event_at = NOW + timedelta(seconds=2)
    engine.decide(
        "d2",
        market=market(root, "m3", 3, funding=(event_at, 0.01)),
        rules=rule,
        at=NOW + timedelta(seconds=3.1),
    )
    funding = engine.status()["account"]["funding_pnl"]
    engine.decide(
        "d3",
        market=market(root, "m4", 4, funding=(event_at, 0.02)),
        rules=rule,
        at=NOW + timedelta(seconds=4.1),
    )
    assert engine.status()["account"]["funding_pnl"] == funding
    assert engine.status()["stop_reason"] == "funding_revision"
    assert engine.status()["quality_failures"]


def test_exact_quantity_accessor_does_not_round_accumulated_fills():
    from fractions import Fraction

    from trade_rl.simulation import BookState

    book = BookState.zero(1, 100)
    for _ in range(3):
        book.execute_fill(
            symbol_index=0,
            quantity=0.1,
            fill_prices=np.ones(1),
            cost_amount=0,
            turnover=0,
            lot_size=0.1,
            lot_count=1,
        )
    assert book.exact_quantities == (Fraction(3, 10),)


def test_no_new_entry_when_close_time_arrives_after_saved_decision(tmp_path):
    root = tmp_path / "paper"
    engine, rule, _ = setup(root, close_seconds=0.5)
    engine.decide(
        "d", market=market(root, "m1", 0), rules=rule, at=NOW + timedelta(seconds=0.1)
    )
    execution = result(
        engine.execute(
            "e", market=market(root, "m2", 1), at=NOW + timedelta(seconds=1.1)
        )
    )
    assert execution["cancelled"] and not execution["fills"]
    assert execution["terminal_flat"] and execution["account"]["fill_count"] == 0


def test_funding_credit_cannot_erase_previous_drawdown_stop(tmp_path):
    root = tmp_path / "paper"
    engine, rule, _, _, _ = entry(root)
    execution = result(
        engine.decide(
            "d2",
            market=market(
                root, "m3", 2, mark=200, funding=(NOW + timedelta(seconds=1.5), 1)
            ),
            rules=rule,
            at=NOW + timedelta(seconds=2.1),
        )
    )
    assert execution["account"]["equity"] > 9900
    assert execution["account"]["maximum_drawdown"] > 0.2
    assert execution["stop_reason"] == "maximum_drawdown"
    assert all(
        row["lot_count"] < 0 if row["symbol_index"] % 2 == 0 else row["lot_count"] > 0
        for row in execution["orders"]
    )


def test_settlement_received_after_exit_still_uses_the_old_held_quantity(tmp_path):
    root = tmp_path / "paper"
    engine, rule, _, _, _ = entry(root, close_seconds=3)
    quantity = engine.status()["account"]["quantities"][1]
    engine.decide(
        "d2", market=market(root, "m3", 3), rules=rule, at=NOW + timedelta(seconds=3.1)
    )
    engine.execute("e2", market=market(root, "m4", 4), at=NOW + timedelta(seconds=4.1))
    paid = result(
        engine.decide(
            "d3",
            market=market(root, "m5", 5, funding=(NOW + timedelta(seconds=2), -0.01)),
            rules=rule,
            at=NOW + timedelta(seconds=5.1),
        )
    )
    assert paid["terminal_flat"]
    assert paid["account"]["funding_pnl"] == pytest.approx(2 * quantity)
    assert paid["account"]["funding_pnl"] < 0


def test_same_timestamp_fill_has_not_held_the_previous_settlement(tmp_path):
    root = tmp_path / "paper"
    engine, rule, _, _, _ = entry(root)
    engine.decide(
        "d2",
        market=market(root, "m3", 2, funding=(NOW + timedelta(seconds=1.1), 0.01)),
        rules=rule,
        at=NOW + timedelta(seconds=2.1),
    )
    assert engine.status()["account"]["funding_pnl"] == 0


def test_missing_and_late_funding_remain_visible_quality_failures(tmp_path):
    root = tmp_path / "paper"
    engine, rule, _ = setup(root)
    engine.decide(
        "d1",
        market=market(root, "m1", 0, next_funding=2),
        rules=rule,
        at=NOW + timedelta(seconds=0.1),
    )
    engine.execute(
        "e1",
        market=market(root, "m2", 1, next_funding=2),
        at=NOW + timedelta(seconds=1.1),
    )
    engine.decide(
        "d2",
        market=market(root, "m3", 100),
        rules=rule,
        at=NOW + timedelta(seconds=100.1),
    )
    engine.decide(
        "d3",
        market=market(root, "m4", 190),
        rules=rule,
        at=NOW + timedelta(seconds=190.1),
    )
    assert engine.status()["stop_reason"] == "missing_funding"
    execution = result(
        engine.execute(
            "e2",
            market=market(root, "m5", 191, funding=(NOW + timedelta(seconds=2), 0.01)),
            at=NOW + timedelta(seconds=191.1),
        )
    )
    assert execution["terminal_flat"]
    assert any(
        reason.startswith("late_funding:") for reason in execution["quality_failures"]
    )
    assert any(
        reason.startswith("missing_funding:")
        for reason in execution["quality_failures"]
    )


def test_unacknowledged_commit_recovers_without_executing_twice(tmp_path, monkeypatch):
    root = tmp_path / "paper"
    engine, rule, digest = setup(root)
    engine.decide(
        "d", market=market(root, "m1", 0), rules=rule, at=NOW + timedelta(seconds=0.1)
    )
    ref = market(root, "m2", 1)
    append = engine.journal.append

    def commit_then_crash(*args, **kwargs):
        append(*args, **kwargs)
        raise RuntimeError("lost acknowledgement")

    monkeypatch.setattr(engine.journal, "append", commit_then_crash)
    with pytest.raises(RuntimeError, match="acknowledgement"):
        engine.execute("e", market=ref, at=NOW + timedelta(seconds=1.1))
    assert engine.status()["account"]["fill_count"] == 0
    recovered = PaperEngine(root, expected_protocol_sha256=digest)
    assert recovered.status()["account"]["fill_count"] == 4
    recovered.execute("e", market=ref, at=NOW + timedelta(seconds=1.1))
    assert recovered.status()["account"]["fill_count"] == 4


def test_rehashed_fictional_account_result_fails_replay(tmp_path):
    from trade_rl.artifacts import canonical_json_bytes
    from trade_rl.evaluation.paper.store import PaperJournal

    root = tmp_path / "original"
    _, _, _, decision, _ = entry(root)
    destination = tmp_path / "altered"
    protocol = json.loads((root / "protocol.json").read_bytes())["protocol"]
    digest = PaperJournal.initialize(destination, protocol)
    import shutil

    for name in ("m1", "rules"):
        shutil.copytree(root / name, destination / name)
    payload = json.loads(canonical_json_bytes(decision["event"]["payload"]))
    payload["result"]["account"]["cash"] += 1
    PaperJournal(destination, expected_protocol_sha256=digest).append(
        "d1", "decision", payload, expected_parent=digest
    )
    with pytest.raises(ValueError, match="replay result"):
        PaperEngine(destination, expected_protocol_sha256=digest)


def test_unmatched_terminal_exit_stays_disqualified_even_after_eventual_flatness(
    tmp_path,
):
    root = tmp_path / "paper"
    engine, rule, _, _, _ = entry(root, close_seconds=3)
    engine.decide(
        "d2", market=market(root, "m3", 3), rules=rule, at=NOW + timedelta(seconds=3.1)
    )
    engine.execute(
        "e2", market=market(root, "m4", 4, thin=True), at=NOW + timedelta(seconds=4.1)
    )
    assert engine.status()["stop_reason"] == "terminal_close"
    assert "unmatched_hedge" in engine.status()["quality_failures"]
    engine.decide(
        "d3", market=market(root, "m5", 5), rules=rule, at=NOW + timedelta(seconds=5.1)
    )
    engine.execute("e3", market=market(root, "m6", 6), at=NOW + timedelta(seconds=6.1))
    assert engine.status()["terminal_flat"]
    assert "unmatched_hedge" in engine.status()["quality_failures"]


def test_different_settlement_times_cannot_net_away_an_intermediate_risk_breach(
    tmp_path,
):
    root = tmp_path / "paper"
    engine, rule, _, _, _ = entry(root)
    account = engine.status()["account"]
    quantity = abs(account["quantities"][1])
    mark = 100.5 + (account["equity"] - 9001) / (2 * quantity)
    events = {
        "BTCUSDT": [(NOW + timedelta(seconds=3), 0.01)],
        "ETHUSDT": [(NOW + timedelta(seconds=2), -0.01)],
    }
    paid = result(
        engine.decide(
            "d2",
            market=market(root, "m3", 4, mark=mark, events=events),
            rules=rule,
            at=NOW + timedelta(seconds=4.1),
        )
    )
    assert paid["account"]["equity"] == pytest.approx(9001)
    assert paid["account"]["funding_pnl"] == pytest.approx(0)
    assert paid["account"]["maximum_drawdown"] > 0.1
    assert paid["stop_reason"] == "maximum_drawdown"


@pytest.mark.parametrize("credit_time", [2, 3])
def test_out_of_order_or_fragmented_settlement_receipts_cannot_qualify(
    tmp_path, credit_time
):
    root = tmp_path / "paper"
    engine, rule, _, _, _ = entry(root)
    account = engine.status()["account"]
    quantity = abs(account["quantities"][1])
    mark = 100.5 + (account["equity"] - 9001) / (2 * quantity)
    first = {"BTCUSDT": [(NOW + timedelta(seconds=credit_time), 0.01)], "ETHUSDT": []}
    engine.decide(
        "d2",
        market=market(root, "m3", 3, mark=mark, events=first),
        rules=rule,
        at=NOW + timedelta(seconds=3.1),
    )
    second = dict(first, ETHUSDT=[(NOW + timedelta(seconds=2), -0.01)])
    engine.decide(
        "d3",
        market=market(root, "m4", 4, mark=mark, events=second),
        rules=rule,
        at=NOW + timedelta(seconds=4.1),
    )
    assert engine.status()["account"]["funding_pnl"] == pytest.approx(0)
    assert engine.status()["stop_reason"] == "funding_out_of_order"
    assert any(
        reason.startswith("funding_out_of_order:")
        for reason in engine.status()["quality_failures"]
    )
