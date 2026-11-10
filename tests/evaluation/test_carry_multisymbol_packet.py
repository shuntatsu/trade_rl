"""The multi-symbol carry packet stays UNREVIEWED, result-blind and bound to the source.

These tests keep the author packet, the specification, the current-status record and
the code in step, and keep the packet from claiming a review, an economic result or
any authorization. They read no market data.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance
from trade_rl.evaluation.carry_multisymbol import protocol

ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = (
    ROOT / "report/issue810-carry-multisymbol-research-assurance-20261110.json"
)
PINNED_DIGEST = "f72e61b911b523b36f31f8a7838450065ef2be2bfc5e26f5f8d9451dc26e6296"
SPEC = ROOT / "docs/specs/funding-carry-multisymbol.md"
DOCUMENTS = (
    SPEC,
    ROOT / "docs/architecture/research-assurance.md",
    ROOT / "docs/research/current-status.md",
)
CODE_PATHS = (
    "trade_rl/evaluation/carry.py",
    "trade_rl/evaluation/carry_multisymbol",
    "trade_rl/integrations/binance/carry.py",
    "trade_rl/integrations/market_store/carry_source.py",
    "trade_rl/strategies/carry.py",
    "tools/carry_multisymbol_screen.py",
)


def packet():
    return json.loads(PACKET_PATH.read_text(encoding="utf-8"))


def test_packet_is_valid_unreviewed_and_pinned() -> None:
    record = packet()
    result = evaluate_assurance(record)
    assert result.status == "UNREVIEWED" and result.errors == ()
    assert result.economic_execution_authorized is False
    assert assurance_digest(record) == PINNED_DIGEST == result.record_digest


def test_packet_claims_nothing_beyond_software_validity() -> None:
    record = packet()
    claims = record["claims"]
    assert claims["claim_level"] == record["evidence"]["evidence_level"]
    assert claims["claim_level"] == "software_validity"
    assert claims["production_eligible"] is False
    assert claims["live_trading_authorized"] is False
    assert claims["next_action_class"] == "request_separate_economic_authorization"
    assert record["thesis"]["stage"] == "development"
    forbidden = " ".join(claims["forbidden"])
    for needle in ("edge", "FS1", "live", "unused"):
        assert needle in forbidden
    evidence = record["evidence"]
    assert evidence["point_in_time"] is False
    assert evidence["realistic_costs"] is False
    assert evidence["fit_development_unused_separated"] is False


def test_packet_states_the_required_g0_to_g2_content() -> None:
    thesis = packet()["thesis"]
    assert "2023-2024 regime effect" in thesis["counter_hypothesis"]
    assert "fees" in thesis["counter_hypothesis"]
    assert "2x-cost" in thesis["cheapest_falsifier"]
    assert "No cell, symbol, window, cost, gross or rule" in thesis["stop_rule"]
    assert "Leveraged long demand" in thesis["economic_mechanism"]
    mechanism = packet()["mechanism"]
    assert set(mechanism["independent_oracles"]) == {"time", "accounting", "evidence"}
    oracle = mechanism["independent_oracles"]["accounting"]["oracle"]
    assert "5992.7" in oracle and "hand-computed" in oracle


def test_documents_name_the_packet_digest_and_the_protocol_digest() -> None:
    for path in DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        assert PINNED_DIGEST in text, path.name
        assert "UNREVIEWED" in text, path.name
    for path in (SPEC, ROOT / "docs/research/current-status.md"):
        assert protocol.protocol_digest() in path.read_text(encoding="utf-8")
    assert "Status: Active" in SPEC.read_text(encoding="utf-8")


def test_spec_states_the_registered_universe_window_and_thresholds() -> None:
    text = " ".join(SPEC.read_text(encoding="utf-8").split())
    for symbol in protocol.CANDIDATES:
        assert symbol.removesuffix("USDT") in text
    for needle in (
        "[2022-01-01, 2026-08-01)",
        "1,000,000 USDT",
        "92-day lookback",
        "K = ceil(20 x admitted / 32)",
        "K = 4, 5, 5, 6",
        "Binance Vision archives with official `.CHECKSUM` verification",
        "--g4-authorization-url",
        "G4 authorization",
    ):
        assert needle in text, needle
    assert protocol.MIN_MEDIAN_HOURLY_QUOTE_VOLUME == 1_000_000.0
    assert (
        protocol.LOOKBACK_HOURS
        == (protocol.WINDOW_START - protocol.LOOKBACK_START).total_seconds() // 3600
    )


def test_packet_heads_are_one_commit_and_the_bound_code_is_unchanged() -> None:
    identity = packet()["identity"]
    head = identity["protocol_head"]
    assert head == identity["implementation_head"]
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{head}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if exists.returncode != 0:
        pytest.skip("packet head is not in this clone (shallow checkout)")
    changed = subprocess.run(
        ["git", "diff", "--name-only", head, "HEAD", "--", *CODE_PATHS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert changed == [], "re-pin the packet heads after changing the bound code"
