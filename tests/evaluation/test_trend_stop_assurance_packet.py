"""The revision-1 trend-stop packet is kept as frozen, historical, UNREVIEWED evidence.

Review 5870390989 of PR #1201 blocked this packet (digest ``894fedb0...``); revision 2
(``report/trend-stop-research-assurance-20261113.json``, tested in
``test_trend_stop_v2_packet.py``) supersedes it. These tests keep the historical packet
byte-stable and valid, and keep the specification pointing at it as superseded; the
packet claims no economic result, review, authorization or promotion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance
from trade_rl.evaluation.trend_stop_study.protocol import (
    SUPERSEDES_PROTOCOL_DIGEST,
    protocol_digest,
)

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "report" / "trend-stop-research-assurance-20261110.json"
SPEC = ROOT / "docs" / "specs" / "trend-following-stops.md"
PINNED_PACKET_DIGEST = (
    "894fedb0b88dd872039fdfbc546f872d485d1d6130ae5e096c79ceebf141bb64"
)
REVISION_1_PROTOCOL_DIGEST = (
    "f57f2d6dedd9a46b4d4ffc57c70dc495f60e517d8cd14f873c1cdab47d02bb9d"
)
HEAD_RE = re.compile(r"^[0-9a-f]{40}$")


def _packet() -> dict[str, Any]:
    return json.loads(PACKET.read_text(encoding="utf-8"))


def test_packet_digest_is_pinned_and_its_protocol_is_superseded() -> None:
    assert assurance_digest(_packet()) == PINNED_PACKET_DIGEST
    # The frozen protocol moved on: revision 1's digest is only a recorded lineage.
    assert SUPERSEDES_PROTOCOL_DIGEST == REVISION_1_PROTOCOL_DIGEST
    assert protocol_digest() != REVISION_1_PROTOCOL_DIGEST


def test_packet_is_a_valid_unreviewed_record_that_authorizes_nothing() -> None:
    evaluation = evaluate_assurance(_packet())
    assert evaluation.status == "UNREVIEWED"
    assert evaluation.economic_execution_authorized is False
    assert evaluation.errors == ()


def test_packet_heads_are_full_commit_ids_and_claims_stay_software_level() -> None:
    packet = _packet()
    identity = packet["identity"]
    assert HEAD_RE.fullmatch(identity["protocol_head"])
    assert HEAD_RE.fullmatch(identity["implementation_head"])
    claims = packet["claims"]
    assert claims["claim_level"] == "software_validity"
    assert packet["evidence"]["evidence_level"] == "software_validity"
    assert claims["production_eligible"] is False
    assert claims["live_trading_authorized"] is False
    assert claims["next_action_class"] == "request_separate_economic_authorization"
    assert packet["thesis"]["stage"] == "development"


def test_packet_names_the_g0_to_g3_elements_and_the_stop_rule() -> None:
    packet = _packet()
    thesis = packet["thesis"]
    assert "arbitraged" in thesis["counter_hypothesis"]
    assert "liquidation" in thesis["economic_mechanism"]
    assert "NO_WINNER" in thesis["stop_rule"]
    assert set(packet["mechanism"]["independent_oracles"]) == {
        "time",
        "state",
        "sign",
        "risk",
        "execution",
        "accounting",
    }
    assert "known-used" in packet["claims"]["decision_rule"]
    assert "reinforcement" in " ".join(packet["claims"]["forbidden"])


def test_spec_is_active_links_the_packet_and_states_the_boundary() -> None:
    text = SPEC.read_text(encoding="utf-8")
    assert "Status: Active" in text
    assert "report/trend-stop-research-assurance-20261110.json" in text
    assert "2026-07-31" in text
    assert "UNREVIEWED" in text
    assert "DEVELOPMENT_PROMISING" in text


def test_packet_discloses_prior_exposure_and_does_not_claim_result_blindness() -> None:
    packet = _packet()
    text = json.dumps(packet)
    for needle in (
        "realdata",
        "legacy GBM signal metadata",
        "Phase 2 trial T1",
        "M34",
        "not excluded",
        "haircut-adjusted",
        "HTTP 404",
    ):
        assert needle in text, needle
    limitations = packet["thesis"]["limitations"]
    assert "not result-blind about its universe and window" in limitations
    assert "No economic result, real price or return was read" not in text
    assert any(
        "result-blind about its universe" in item
        for item in packet["claims"]["forbidden"]
    )
