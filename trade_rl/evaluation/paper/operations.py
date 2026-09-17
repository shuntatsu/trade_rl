"""Fixed prospective sealing and bounded-cadence public collection operations."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from trade_rl._validation import require_aware_datetime
from trade_rl.evaluation.paper.account import PaperAccount, PaperSettings
from trade_rl.evaluation.paper.store import PaperJournal
from trade_rl.evaluation.paper.supervisor import PaperCollector, seal_collection


def screen_plan() -> dict[str, Any]:
    """Return a fresh copy of the fixed economic screen; no adjustable gates."""
    return dict(
        schema="carry_paper_screen_v1",
        duration_days=90,
        block_days=30,
        block_count=3,
        minimum_seal_notice_seconds=300,
        comparison="cash_zero_interest",
        positive_net_profit=True,
        positive_each_block=True,
        net_profit_exceeds_recorded_fees=True,
        maximum_drawdown_exclusive=0.1,
        nonzero_funding_each_block=True,
        complete_coverage_and_actual_flat=True,
        production_eligible=False,
    )


def seal_paper_study(
    root: str | Path,
    *,
    start_at: datetime,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> str:
    start = require_aware_datetime(start_at, field="paper start").astimezone(UTC)
    now = require_aware_datetime(clock(), field="paper seal").astimezone(UTC)
    if start < now + timedelta(minutes=5):
        raise ValueError("paper study requires at least five minutes before start")
    return seal_collection(
        root,
        settings=PaperSettings(start_at=start, close_at=start + timedelta(days=90)),
        research_plan=screen_plan(),
        clock=lambda: now,
    )


def collect_until_finished(
    collector: PaperCollector,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
    publish: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Do not retry exceptions, fabricate missed slots, or extend the deadline."""
    end = collector.close_at + timedelta(seconds=180)
    previous: datetime | None = None

    def read_clock() -> datetime:
        nonlocal previous
        at = require_aware_datetime(clock(), field="cadence clock")
        if previous is not None and at < previous:
            raise ValueError("cadence clock moved backwards")
        previous = at
        return at

    while True:
        read_clock()
        result = collector.cycle()
        if result["status"]["quality_failures"] and result["status"]["terminal_flat"]:
            result = dict(result, phase="rejected")
        publish(result)
        if result["phase"] in {"finished", "rejected"}:
            return result
        now = read_clock()
        slot = max(0, math.floor((now - collector.start_at).total_seconds() / 60) + 1)
        due = min(end, collector.start_at + timedelta(seconds=slot * 60))
        while (remaining := (due - read_clock()).total_seconds()) > 0:
            sleep(min(60, remaining))


def collection_status(
    root: str | Path, *, expected_protocol_sha256: str
) -> dict[str, Any]:
    """Operational status verifies the chain, not raw-source financial replay."""
    import json

    journal = PaperJournal(root, expected_protocol_sha256=expected_protocol_sha256)
    records = journal.events()
    if records:
        status = records[-1]["event"]["payload"]["result"]
        tip = records[-1]["sha256"]
    else:
        config = json.loads((journal.root / "protocol.json").read_bytes())["protocol"][
            "settings"
        ]
        for key in ("start_at", "close_at"):
            config[key] = datetime.fromisoformat(config[key])
        status = PaperAccount(PaperSettings(**config)).status()
        tip = expected_protocol_sha256
    return dict(
        schema="paper_operational_status_v1",
        protocol_sha256=expected_protocol_sha256,
        verification="journal_chain_only",
        economic_decision="NOT_EVALUATED",
        production_eligible=False,
        collection_failure=(journal.root / "collection-failure.json").exists(),
        events=len(records),
        tip=tip,
        status=status,
    )
