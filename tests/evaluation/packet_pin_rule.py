"""Shared pin rule for research assurance packets that bind source code.

An *open* packet (its one authorized run has not happened) is pinned to its head: any
change to the bound code since the head fails ("re-pin the packet"). A *closed* packet
has a recorded run commit, result and protocol digest; its evidence is bound to that
run commit, not to ``HEAD``, so later changes to shared code must not fail it. It is
never re-pinned or re-labelled. Its pin instead verifies the historical binding:

* the run commit is an ancestor of ``HEAD`` and the recorded result is an issue 810
  comment;
* the packet as it existed at the run commit names an implementation head;
* the bound-code closure at that head equals the closure at the run commit, and the
  SHA-256 content digest over ``(path, blob bytes)`` of every closure file, in sorted
  path order, is the same at both commits and equals the digest recorded in the test.

No further run of a closed protocol is authorized.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_RESULT_URL = re.compile(
    r"https://github\.com/shuntatsu/trade_rl/issues/810#issuecomment-[0-9]+"
)

ReadBlob = Callable[[str, str], bytes]


@dataclass(frozen=True)
class ClosedRun:
    """The recorded single authorized run of a closed packet."""

    run_commit: str
    packet_path: str  # the packet file as it existed at the run commit
    result_url: str
    result_label: str
    protocol_digest: str
    content_digest: str  # bound-code content digest at the implementation head
    further_runs_authorized: bool = False

    def __post_init__(self) -> None:
        if self.further_runs_authorized:
            raise ValueError("a closed packet authorizes no further run")


def content_digest(read_blob: ReadBlob, commit: str, closure: Sequence[str]) -> str:
    """SHA-256 over ``(path, blob bytes)`` of every closure file in sorted order."""

    digest = hashlib.sha256()
    for path in sorted(closure):
        data = read_blob(commit, path)
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
    return digest.hexdigest()


def git_blob_reader(root: Path) -> ReadBlob:
    """Read a blob's exact bytes at a commit (never the working tree)."""

    def read(commit: str, path: str) -> bytes:
        result = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise FileNotFoundError(f"{commit}:{path}")
        return result.stdout

    return read


def evaluate_pin(
    *,
    head: str | None,
    closed_run: ClosedRun | None,
    closure_at: Callable[[str], Sequence[str]],
    changed_since_head: Callable[[Sequence[str]], Sequence[str]],
    read_blob: ReadBlob,
    is_ancestor_of_head: Callable[[str], bool],
    packet_head_at: Callable[[str, str], str],
) -> list[str]:
    """Return the violations of one packet's source pin (empty means it holds).

    ``head`` is the packet's implementation head for an open packet; a closed packet
    derives its head from the packet file at the run commit.
    """

    if closed_run is None:
        assert head is not None, "an open packet needs its implementation head"
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
        problems.append("result_url must be an issue 810 comment URL")
    if not closed_run.result_label.strip():
        problems.append("result_label must be recorded")
    if _DIGEST.fullmatch(closed_run.protocol_digest) is None:
        problems.append("protocol_digest must be a sha256 hex digest")
    if _DIGEST.fullmatch(closed_run.content_digest) is None:
        problems.append("content_digest must be a sha256 hex digest")
    if problems:
        return problems
    if not is_ancestor_of_head(closed_run.run_commit):
        return ["run_commit is not an ancestor of HEAD"]
    recorded_head = packet_head_at(closed_run.run_commit, closed_run.packet_path)
    if _COMMIT.fullmatch(recorded_head) is None:
        return ["the packet at the run commit records no implementation head"]
    closure = list(closure_at(recorded_head))
    if closure != list(closure_at(closed_run.run_commit)):
        return [
            "bound-code closure at the run commit differs from the closure at the "
            "packet's implementation head"
        ]
    at_head = content_digest(read_blob, recorded_head, closure)
    at_run = content_digest(read_blob, closed_run.run_commit, closure)
    if at_head != at_run:
        return [
            "bound-code content at the run commit differs from the content at the "
            "packet's implementation head"
        ]
    if at_run != closed_run.content_digest:
        return [
            "bound-code content digest differs from the digest recorded in the test"
        ]
    return []


__all__ = [
    "ClosedRun",
    "content_digest",
    "evaluate_pin",
    "git_blob_reader",
]
