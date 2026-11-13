"""Shared pin rule for research assurance packets that bind source code.

An *open* packet (its one authorized run has not happened) is pinned to its head: any
change to the bound code since the head fails ("re-pin the packet"). A *closed* packet
has a recorded run commit, result and protocol digest; its evidence is bound to that
run commit, not to ``HEAD``, so later changes to shared code must not fail it. It is
never re-pinned or re-labelled. Its pin instead verifies the historical binding: the
bound-code closure at the run commit equals the closure at the packet's implementation
head. No further run of a closed protocol is authorized.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_RESULT_URL = re.compile(
    r"https://github\.com/shuntatsu/trade_rl/issues/[0-9]+#issuecomment-[0-9]+"
)


@dataclass(frozen=True)
class ClosedRun:
    """The recorded single authorized run of a closed packet."""

    run_commit: str
    result_url: str
    result_label: str
    protocol_digest: str
    further_runs_authorized: bool = False

    def __post_init__(self) -> None:
        if self.further_runs_authorized:
            raise ValueError("a closed packet authorizes no further run")


def evaluate_pin(
    *,
    head: str,
    closed_run: ClosedRun | None,
    closure_at: Callable[[str], Sequence[str]],
    changed_since_head: Callable[[Sequence[str]], Sequence[str]],
) -> list[str]:
    """Return the violations of one packet's source pin (empty means it holds)."""

    if closed_run is None:
        bound = closure_at(head)
        changed = list(changed_since_head(bound))
        if changed:
            return [
                "re-pin the packet heads after changing the bound code: "
                + ", ".join(sorted(changed))
            ]
        return []

    problems: list[str] = []
    if _COMMIT.fullmatch(closed_run.run_commit) is None:
        problems.append("run_commit must be a full 40-hex commit")
    if _RESULT_URL.fullmatch(closed_run.result_url) is None:
        problems.append("result_url must be an issue 810-style comment URL")
    if not closed_run.result_label.strip():
        problems.append("result_label must be recorded")
    if _DIGEST.fullmatch(closed_run.protocol_digest) is None:
        problems.append("protocol_digest must be a sha256 hex digest")
    if problems:
        return problems
    if list(closure_at(closed_run.run_commit)) != list(closure_at(head)):
        return [
            "bound-code closure at the run commit differs from the closure at the "
            "packet's implementation head"
        ]
    return []


__all__ = ["ClosedRun", "evaluate_pin"]
