"""The closed-packet pin rule, exercised on synthetic closures (no Git, no market data).

An *open* packet (its one authorized run has not happened) stays pinned to its head:
any change to the bound code since the head fails. A *closed* packet has a recorded run
commit and result; its evidence is bound to that run commit, so shared code may change
afterwards, but the bound-code closure at the run commit must equal the closure at the
packet's implementation head, and no further run is authorized.
"""

from __future__ import annotations

import pytest

from tests.evaluation.packet_pin_rule import ClosedRun, evaluate_pin

HEAD = "a" * 40
RUN = "b" * 40
CLOSURE = ["trade_rl/a.py", "trade_rl/b.py"]


def _closure_at(mapping: dict[str, list[str]]):
    return lambda commit: mapping[commit]


def _run(**overrides: object) -> ClosedRun:
    values: dict[str, object] = {
        "run_commit": RUN,
        "result_url": "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
        "result_label": "NO_WINNER",
        "protocol_digest": "c" * 64,
    }
    values.update(overrides)
    return ClosedRun(**values)  # type: ignore[arg-type]


def test_open_packet_fails_on_any_bound_change_since_the_head() -> None:
    problems = evaluate_pin(
        head=HEAD,
        closed_run=None,
        closure_at=_closure_at({HEAD: CLOSURE}),
        changed_since_head=lambda paths: ["trade_rl/a.py"],
    )
    assert problems and "re-pin" in problems[0]


def test_open_packet_passes_when_the_bound_code_is_unchanged() -> None:
    assert (
        evaluate_pin(
            head=HEAD,
            closed_run=None,
            closure_at=_closure_at({HEAD: CLOSURE}),
            changed_since_head=lambda paths: [],
        )
        == []
    )


def test_closed_packet_passes_when_shared_code_changes_after_its_run() -> None:
    # Later PRs change shared bound files; the closed packet's evidence is the run.
    assert (
        evaluate_pin(
            head=HEAD,
            closed_run=_run(),
            closure_at=_closure_at({HEAD: CLOSURE, RUN: CLOSURE}),
            changed_since_head=lambda paths: ["trade_rl/a.py", "trade_rl/b.py"],
        )
        == []
    )


def test_closed_packet_fails_when_the_run_commit_closure_differs() -> None:
    problems = evaluate_pin(
        head=HEAD,
        closed_run=_run(),
        closure_at=_closure_at({HEAD: CLOSURE, RUN: [*CLOSURE, "trade_rl/c.py"]}),
        changed_since_head=lambda paths: [],
    )
    assert problems and "closure" in problems[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_commit", "short"),
        ("result_url", "https://example.com/not-a-result"),
        ("result_label", ""),
        ("protocol_digest", "xyz"),
    ],
)
def test_a_closed_run_record_must_be_complete_and_well_formed(
    field: str, value: str
) -> None:
    problems = evaluate_pin(
        head=HEAD,
        closed_run=_run(**{field: value}),
        closure_at=_closure_at({HEAD: CLOSURE, value: CLOSURE, RUN: CLOSURE}),
        changed_since_head=lambda paths: [],
    )
    assert problems and field in problems[0]


def test_a_closed_run_authorizes_no_further_run() -> None:
    assert _run().further_runs_authorized is False
    with pytest.raises(ValueError, match="further"):
        _run(further_runs_authorized=True)
