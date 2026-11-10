"""The trend-stop pre-registration packet stays UNREVIEWED and bound to the protocol.

The packet is result-blind: it claims no economic result, review, authorization or
promotion. These tests keep it, the frozen protocol and the specification in step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance
from trade_rl.evaluation.trend_stop_study.protocol import protocol_digest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "report" / "trend-stop-research-assurance-20261110.json"
SPEC = ROOT / "docs" / "specs" / "trend-following-stops.md"
PINNED_PACKET_DIGEST = (
    "92a4f31b11cee5040056b49964939bde3ee30c6b88321d640f9ef3e45e913f29"
)
PINNED_PROTOCOL_DIGEST = (
    "0e1d3015d46f152ffdd5f16356980b1aa137c27b68cebe03a3615f5ce88b5693"
)
HEAD_RE = re.compile(r"^[0-9a-f]{40}$")


def _packet() -> dict[str, Any]:
    return json.loads(PACKET.read_text(encoding="utf-8"))


def test_packet_digest_and_protocol_digest_are_pinned() -> None:
    assert assurance_digest(_packet()) == PINNED_PACKET_DIGEST
    assert protocol_digest() == PINNED_PROTOCOL_DIGEST


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
