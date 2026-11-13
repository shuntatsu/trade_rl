"""The closed-packet pin rule, exercised on synthetic closures (no Git, no market data).

An *open* packet (its one authorized run has not happened) stays pinned to its head:
any change to the bound code since the head fails. A *closed* packet has a recorded run
commit and result; its evidence is bound to that run commit, so shared code may change
afterwards, but the closure AND the content of the bound code at the run commit must
equal those at the implementation head the packet named at the run, the run must be an
ancestor of HEAD, and no further run is authorized.
"""

from __future__ import annotations

import pytest

from tests.evaluation.packet_pin_rule import ClosedRun, content_digest, evaluate_pin

HEAD = "a" * 40
RUN = "b" * 40
CLOSURE = ["trade_rl/a.py", "trade_rl/b.py"]
FILES = {"trade_rl/a.py": b"alpha", "trade_rl/b.py": b"beta"}
PACKET = "report/packet.json"


def _reader(mapping: dict[str, dict[str, bytes]]):
    return lambda commit, path: mapping[commit][path]


SAME = {HEAD: FILES, RUN: FILES}
EXPECTED = content_digest(_reader(SAME), HEAD, CLOSURE)


def _run(**overrides: object) -> ClosedRun:
    values: dict[str, object] = {
        "run_commit": RUN,
        "packet_path": PACKET,
        "result_url": "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
        "result_label": "NO_WINNER",
        "protocol_digest": "c" * 64,
        "content_digest": EXPECTED,
    }
    values.update(overrides)
    return ClosedRun(**values)  # type: ignore[arg-type]


def _pin(*, closed_run, closure=None, blobs=None, ancestor=True, recorded=HEAD, **kw):
    closures = closure or {HEAD: CLOSURE, RUN: CLOSURE}
    return evaluate_pin(
        head=kw.get("head", HEAD),
        closed_run=closed_run,
        closure_at=lambda commit: closures[commit],
        changed_since_head=kw.get("changed", lambda paths: []),
        read_blob=_reader(blobs or SAME),
        is_ancestor_of_head=lambda commit: ancestor,
        packet_head_at=lambda commit, path: recorded,
    )


def test_open_packet_fails_on_any_bound_change_since_the_head() -> None:
    problems = _pin(closed_run=None, changed=lambda paths: ["trade_rl/a.py"])
    assert problems and "re-pin" in problems[0]


def test_open_packet_passes_when_the_bound_code_is_unchanged() -> None:
    assert _pin(closed_run=None) == []


def test_closed_packet_passes_when_shared_code_changes_after_its_run() -> None:
    # Later PRs change shared bound files; the closed packet's evidence is the run.
    assert (
        _pin(
            closed_run=_run(), changed=lambda paths: ["trade_rl/a.py", "trade_rl/b.py"]
        )
        == []
    )


def test_closed_packet_fails_when_the_run_commit_closure_differs() -> None:
    problems = _pin(
        closed_run=_run(),
        closure={HEAD: CLOSURE, RUN: [*CLOSURE, "trade_rl/c.py"]},
    )
    assert problems and "closure" in problems[0]


def test_closed_packet_fails_when_same_files_have_different_content() -> None:
    changed = {HEAD: FILES, RUN: {**FILES, "trade_rl/b.py": b"beta changed"}}
    problems = _pin(closed_run=_run(), blobs=changed)
    assert problems and "content" in problems[0]


def test_closed_packet_fails_when_the_recorded_digest_does_not_match() -> None:
    problems = _pin(closed_run=_run(content_digest="0" * 64))
    assert problems and "recorded in the test" in problems[0]


def test_closed_packet_fails_when_the_run_is_not_an_ancestor_of_head() -> None:
    problems = _pin(closed_run=_run(), ancestor=False)
    assert problems and "ancestor" in problems[0]


def test_closed_packet_fails_when_the_packet_at_the_run_names_no_head() -> None:
    problems = _pin(closed_run=_run(), recorded="")
    assert problems and "implementation head" in problems[0]


def test_content_digest_depends_on_path_bytes_and_order_independent_input() -> None:
    reader = _reader(SAME)
    assert content_digest(reader, HEAD, CLOSURE) == content_digest(
        reader, HEAD, list(reversed(CLOSURE))
    )
    other = _reader({HEAD: {**FILES, "trade_rl/a.py": b"alphA"}})
    assert content_digest(other, HEAD, CLOSURE) != EXPECTED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_commit", "short"),
        ("result_url", "https://example.com/not-a-result"),
        # a comment on another issue is not the issue 810 result
        (
            "result_url",
            "https://github.com/shuntatsu/trade_rl/issues/9#issuecomment-1",
        ),
        ("result_label", ""),
        ("protocol_digest", "xyz"),
        ("content_digest", "xyz"),
    ],
)
def test_a_closed_run_record_must_be_complete_and_well_formed(
    field: str, value: str
) -> None:
    problems = _pin(closed_run=_run(**{field: value}))
    assert problems and field in problems[0]


def test_a_closed_run_authorizes_no_further_run() -> None:
    assert _run().further_runs_authorized is False
    with pytest.raises(ValueError, match="further"):
        _run(further_runs_authorized=True)
