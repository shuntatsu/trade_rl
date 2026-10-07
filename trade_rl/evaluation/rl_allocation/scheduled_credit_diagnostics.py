"""Summarize complete scheduled collector targets without another learning call.

Observed terminal boundaries do not imply Monte Carlo targets. Raw chronological
GAE is not shuffled, normalized optimizer minibatch exposure or gradient credit.
Supplied-content consistency does not authenticate historical fitting.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.rl_allocation.scheduled_transition_validation import (
    validate_scheduled_transition_events,
)
from trade_rl.evaluation.rl_allocation.transition_arrays import read_array

__all__ = [
    "ScheduledAllocationCreditDiagnostics",
    "diagnose_scheduled_allocation_credit",
]

_Record = tuple[float, float, float, float, float, float]
_Group = tuple[str, str, int, int, str]


@dataclass(frozen=True, slots=True)
class ScheduledAllocationCreditDiagnostics:
    """Detached immutable content; each payload call returns a fresh tree."""

    _payload_json: str

    def payload(self) -> dict[str, Any]:
        return dict(json.loads(self._payload_json))

    @property
    def digest(self) -> str:
        return content_digest(self.payload())


def _summary(records: list[_Record]) -> dict[str, Any]:
    count = len(records)
    result: dict[str, Any] = {
        "count": count,
        "filled_transition_count": sum(row[4] > 0 for row in records),
        "advantage_counts": {
            "negative": sum(row[0] < 0 for row in records),
            "zero": sum(row[0] == 0 for row in records),
            "positive": sum(row[0] > 0 for row in records),
        },
    }
    for column, name in enumerate(
        (
            "raw_advantage",
            "raw_return",
            "critic_value",
            "collector_reward",
            "filled_notional",
            "interval_cost",
        )
    ):
        if not records:
            result[name] = None
            continue
        values = [row[column] for row in records]
        try:
            mean = math.fsum(values) / count
        except (ValueError, OverflowError) as error:
            raise ValueError(
                "scheduled diagnostic arithmetic must be finite"
            ) from error
        if not math.isfinite(mean):
            raise ValueError("scheduled diagnostic arithmetic must be finite")
        result[name] = {"mean": mean, "min": min(values), "max": max(values)}
    return result


def diagnose_scheduled_allocation_credit(
    events: object,
    bundle: dict[str, Any],
    *,
    learner_diagnostics: str = "none",
) -> ScheduledAllocationCreditDiagnostics:
    """Admit the whole native trace first, then retain every recorded transition.

    POST held sign describes exact retained quantity, not an action's new fill.
    Each target boundary is the first true terminal reachable from this row
    inside its recorded rollout, or the live critic bootstrap if none occurs.
    Earlier critic estimates can still influence terminal-bound GAE at lambda<1.
    """
    admission = validate_scheduled_transition_events(
        events, bundle, learner_diagnostics=learner_diagnostics
    )
    rows = cast(list[dict[str, Any]], events)
    training = bundle["training"]
    windows: dict[str, list[_Record]] = {
        identity: [] for identity in training["schedule"]["train_window_ids"]
    }
    phases: dict[str, list[_Record]] = {
        "first_decision": [],
        "later_decision": [],
    }
    actions: dict[int, list[_Record]] = {code: [] for code in range(4)}
    boundaries: dict[str, int] = {"same_episode_terminal": 0, "live_bootstrap": 0}
    groups: dict[_Group, list[_Record]] = {}
    complete: list[_Record] = []
    batch: list[dict[str, Any]] = []
    for event in rows:
        if event["kind"] == "transition":
            batch.append(event)
        elif event["kind"] == "rollout":
            size = len(batch)
            advantages = read_array(event["advantages"], "<f4", (size, 1))
            returns = read_array(event["returns"], "<f4", (size, 1))
            target_boundaries = ["live_bootstrap"] * size
            terminal_seen = False
            for local in reversed(range(size)):
                terminal_seen = terminal_seen or batch[local]["done"]
                if terminal_seen:
                    target_boundaries[local] = "same_episode_terminal"
            for local, row in enumerate(batch):
                actor, facts = row["actor"], row["facts"]
                record = (
                    float(advantages[local, 0]),
                    float(returns[local, 0]),
                    float(read_array(actor["value"], "<f4", (1, 1))[0, 0]),
                    float(read_array(actor["reward"], "<f4", (1,))[0]),
                    facts["execution"]["filled_notional"],
                    facts["execution"]["interval_cost"],
                )
                window = row["old"]["window_id"]
                phase = "first_decision" if row["episode_start"] else "later_decision"
                action = actor["action_code"]
                quantity = Fraction(
                    facts["book"]["exact_quantities"][facts["symbol_index"]]
                )
                sign = int(quantity > 0) - int(quantity < 0)
                boundary = target_boundaries[local]
                group = (window, phase, action, sign, boundary)
                groups.setdefault(group, []).append(record)
                windows[window].append(record)
                phases[phase].append(record)
                actions[action].append(record)
                boundaries[boundary] += 1
                complete.append(record)
            batch = []
    payload = {
        "schema": "scheduled_allocation_credit_diagnostics_v1",
        "identities": {
            "events_digest": content_digest(rows),
            "bundle_digest": content_digest(bundle),
            "recipe_digest": bundle["recipe_digest"],
            "training_digest": content_digest(training),
            "schedule_digest": training["schedule_digest"],
            "source_records_digest": content_digest(training["sources"]),
            "protocol_digest": training["protocol_digest"],
            "optimization_digest": content_digest(training["optimization"]),
            "learner_diagnostics": learner_diagnostics,
        },
        "admission": admission,
        "transition_count": admission["transition_count"],
        "rollout_count": admission["rollout_count"],
        "phase_counts": {name: len(records) for name, records in phases.items()},
        "raw_action_counts": [len(actions[code]) for code in range(4)],
        "boundary_counts": boundaries,
        "summary": _summary(complete),
        "phases": [
            {"phase": name, **_summary(records)} for name, records in phases.items()
        ],
        "raw_actions": [
            {"raw_action_code": code, **_summary(actions[code])} for code in range(4)
        ],
        "windows": [
            {"window_id": name, **_summary(records)}
            for name, records in windows.items()
        ],
        "groups": [
            {
                "window_id": window,
                "phase": phase,
                "raw_action_code": action,
                "post_held_sign": sign,
                "target_boundary": boundary,
                **_summary(records),
            }
            for (window, phase, action, sign, boundary), records in sorted(
                groups.items()
            )
        ],
        "actual_minibatch_exposure": "NOT_ESTABLISHED",
        "gradient_attribution": "NOT_ESTABLISHED",
    }
    return ScheduledAllocationCreditDiagnostics(canonical_json_bytes(payload).decode())
