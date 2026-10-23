"""Public readback of the immutable artificial Issue 888 release fixture."""

import gzip
import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_global_execution import (
    GlobalAllocationExecutionPlan,
)
from trade_rl.evaluation.allocation_terminal_evidence import (
    TerminalAllocationExecutionReceipt,
)
from trade_rl.evaluation.allocation_terminal_execution import (
    TerminalAllocationExecutionPlan,
)
from trade_rl.evaluation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)

FIXTURES = Path(__file__).with_name("fixtures") / "issue888_release_v1"
_PLAN_PIN = (
    4476,
    "ad4b09aade180fbf69d7c53d4f308fe374835879b18a20684a7ae906c181e9e0",
)
_RECEIPT_PINS = {
    "nonrl": (
        "ARTIFICIAL_H72_A10_A_BASEL2_nonrl.json.gz",
        63520,
        "7c7a967630816479c3ce351a243ef7da8b451975f063a3da515ff465f3b7cb1e",
        "8870dbb4b05c22a55ee27c47f0c35a0de57da8e19f27311a56bca3a1959c7f61",
        "5919a1169fd4e57537099b0d7163ceb0c36a5252decacf02811f5ac8dfb70a07",
        "9baed7f083ee40ba813f9d55618e564867ed175940832d848dfa504e8c94f378",
    ),
    "cash": (
        "ARTIFICIAL_H72_A10_A_BASEL2_cash.json.gz",
        51682,
        "d7bd688d6e3190f116fa0d06385e24f6b44ec2541a69c3f5f66034affee68d4d",
        "e2a3b8dfa174d5e48dabed72606ebca518c16a6deff9cb7c7727157ae772445f",
        "0ab9a50ba2ce6f2bf19fd1623969cf458780296aabb183c8611c2e3bd766f865",
        "f3b3c45e4bcc6c362af3fb410304bce0fabeab74768927b0a1ad262f09a362d4",
    ),
}


@pytest.fixture(scope="module")
def frozen_native_plan() -> dict[str, Any]:
    compressed = (FIXTURES / "native-plan.json.gz").read_bytes()
    expected_length, expected_sha256 = _PLAN_PIN
    assert len(compressed) == expected_length
    assert sha256(compressed).hexdigest() == expected_sha256
    original = gzip.decompress(compressed)
    payload = json.loads(original)
    assert original == canonical_json_bytes(payload)
    return payload


def _terminal_plan(
    payload: dict[str, Any],
) -> tuple[GlobalAllocationExecutionPlan, TerminalAllocationExecutionPlan]:
    native_plan = GlobalAllocationExecutionPlan.from_payload(payload["native_plan"])
    objective_payload = payload["objective"]
    capital_payload = objective_payload["capital"]
    capital = CapitalContract(
        account_mode=capital_payload["mode"],
        currency=capital_payload["currency"],
        initial_equities=tuple(capital_payload["initial_equities"]),
    )
    objective = ObjectiveContract(
        capital=capital,
        evaluation_start=datetime.fromisoformat(objective_payload["evaluation_start"]),
        evaluation_stop_exclusive=datetime.fromisoformat(
            objective_payload["evaluation_stop_exclusive"]
        ),
        terminal_valuation=objective_payload["terminal_valuation"],
        economics_digest=objective_payload["economics_digest"],
        risk_digest=objective_payload["risk_digest"],
        deployment_recipe_digest=objective_payload["deployment_recipe_digest"],
        maximum_drawdown=objective_payload["maximum_drawdown"],
    )
    clock_payload = payload["clock"]
    clock = FinancialClockContract(
        **{key: value for key, value in clock_payload.items() if key != "schema"}
    )
    recipe = payload["recipe"]
    return native_plan, TerminalAllocationExecutionPlan(
        native_plan=native_plan,
        settlement_stop_index=recipe["closing_range"][1],
        closing_clocks_ns=tuple(recipe["closing_clocks_ns"]),
        closing_source_digest=recipe["closing_source_digest"],
        bound=BoundObjectiveClock(objective, clock),
    )


@pytest.mark.parametrize("role", ("nonrl", "cash"))
def test_release_receipt_public_readback_preserves_canonical_plan_and_bytes(
    role: str, frozen_native_plan: dict[str, Any]
) -> None:
    outer = frozen_native_plan
    terminal_payload = outer["groups"][0]["plans"][role]
    native_plan, expected_plan = _terminal_plan(terminal_payload)

    native_plan_input = canonical_json_bytes(terminal_payload["native_plan"])
    terminal_plan_input = canonical_json_bytes(terminal_payload)
    assert canonical_json_bytes(native_plan.payload) == native_plan_input
    assert canonical_json_bytes(expected_plan.payload) == terminal_plan_input

    (
        receipt_name,
        expected_size,
        expected_sha256,
        plan_digest,
        native_digest,
        receipt_digest,
    ) = _RECEIPT_PINS[role]
    compressed = (FIXTURES / receipt_name).read_bytes()
    assert len(compressed) == expected_size
    assert sha256(compressed).hexdigest() == expected_sha256
    original = gzip.decompress(compressed)
    payload = json.loads(original)
    canonical_original = canonical_json_bytes(payload)
    assert original == canonical_original

    assert expected_plan.digest == plan_digest
    assert content_digest(native_plan.payload) == native_digest
    assert payload["plan"] == expected_plan.payload
    assert payload["plan_digest"] == expected_plan.digest
    assert payload["native_receipt"]["plan"] == native_plan.payload
    assert payload["plan"]["recipe"]["closing_source_digest"] == (
        expected_plan.closing_source_digest
    )

    receipt = TerminalAllocationExecutionReceipt.from_payload(
        payload, expected_plan=expected_plan
    )
    roundtrip = canonical_json_bytes(receipt.payload)
    assert roundtrip == canonical_original
    assert receipt.digest == receipt_digest
    assert sha256(roundtrip).hexdigest() == receipt_digest
