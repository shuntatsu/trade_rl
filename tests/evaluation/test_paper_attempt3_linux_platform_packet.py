"""The attempt-3 Linux platform packet stays UNREVIEWED and bound to the source.

The three earlier attempt-3 packets are frozen. This packet claims only that the
operating platform of attempt 3 can be a Linux host (SIGHUP handling, run layout,
systemd user units, read-only clock report) with every economic and protocol parameter
of the seal-candidate packet unchanged. The tests bind the packet, the specification and
the source, and keep the packet from claiming a review, a seal or any economic result.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation.test_paper_attempt2_assurance_packet import (
    paper_code_dependency_paths,
)
from tests.evaluation.test_paper_attempt3_assurance_packet import (
    PENDING,
    POLICY_PREFIX,
    TIMING_PREFIX,
    _mismatches,
    _parameters,
)
from tests.evaluation.test_paper_attempt3_implementation_packet import (
    _policy_mismatches,
)
from tests.evaluation.test_paper_attempt3_seal_candidate_packet import (
    IMPL_DIGEST,
    IMPL_PACKET,
    PREREG_DIGEST,
    PREREG_PACKET,
)
from tests.evaluation.test_paper_attempt3_seal_candidate_packet import (
    PINNED_DIGEST as SEAL_CANDIDATE_DIGEST,
)
from tests.evaluation.test_paper_attempt3_seal_candidate_packet import (
    RELATIVE_PACKET as SEAL_CANDIDATE_PACKET,
)
from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance

ROOT = Path(__file__).resolve().parents[2]
RELATIVE_PACKET = (
    "report/issue810-carry-attempt3-linux-platform-research-assurance-20261119.json"
)
PACKET = ROOT / RELATIVE_PACKET
PINNED_DIGEST = "702698adbcf18831933a1222a5eadebb2579df9d7d242f2497a01441c539abc3"
SEAL_CANDIDATE_HEAD = "d0a76538f77a5a99825fc0c698a27153d8181933"
SPEC = "docs/specs/funding-carry-paper.md"
ARCHITECTURE = "docs/architecture/research-assurance.md"
STATUS = "docs/research/current-status.md"
BOUNDARIES = "docs/architecture/package-boundaries.md"
PLAN = "docs/plans/funding-carry-paper.md"
PLATFORM_FILES = (
    "trade_rl/evaluation/paper/linux_service.py",
    "trade_rl/evaluation/paper/watchdog.py",
)


def _packet() -> Any:
    return json.loads(PACKET.read_text(encoding="utf-8"))


def _text(packet: Any) -> str:
    return json.dumps(packet, ensure_ascii=False)


def _raw(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _spec_section() -> str:
    document = _raw(SPEC)
    start = document.index("#### Attempt 3 Linux operating platform")
    return " ".join(document[start:].split())


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_repo", "assurance", *args, RELATIVE_PACKET],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_packet_is_unreviewed_valid_and_pins_its_digest() -> None:
    packet = _packet()
    result = evaluate_assurance(packet)
    assert result.status == "UNREVIEWED"
    assert result.errors == ()
    assert result.economic_execution_authorized is False
    assert result.record_digest == assurance_digest(packet) == PINNED_DIGEST
    assert PINNED_DIGEST not in {
        SEAL_CANDIDATE_DIGEST,
        IMPL_DIGEST,
        PREREG_DIGEST,
    }
    assert b"\r" not in PACKET.read_bytes()
    identity = packet["identity"]
    assert identity["protocol_head"] == identity["implementation_head"]
    assert re.fullmatch(r"[0-9a-f]{40}", identity["implementation_head"])
    assert identity["implementation_head"] != SEAL_CANDIDATE_HEAD


def test_cli_check_inspect_and_digest_agree_and_report_unreviewed() -> None:
    check, inspect_, digest = _cli("check"), _cli("inspect"), _cli("digest")
    assert check.returncode == 1, check.stderr
    assert json.loads(check.stdout)["status"] == "UNREVIEWED"
    assert json.loads(inspect_.stdout)["record_digest"] == PINNED_DIGEST
    assert json.loads(inspect_.stdout)["economic_execution_authorized"] is False
    assert json.loads(digest.stdout) == {"record_digest": PINNED_DIGEST}


def test_the_earlier_attempt3_packets_are_untouched_and_not_re_pinned() -> None:
    for relative, digest in (
        (PREREG_PACKET, PREREG_DIGEST),
        (IMPL_PACKET, IMPL_DIGEST),
        (SEAL_CANDIDATE_PACKET, SEAL_CANDIDATE_DIGEST),
    ):
        frozen = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        assert assurance_digest(frozen) == digest, relative


def test_the_economic_and_protocol_content_equals_the_seal_candidate_packet() -> None:
    new = _packet()
    old = json.loads((ROOT / SEAL_CANDIDATE_PACKET).read_text(encoding="utf-8"))
    assert new["mechanism"]["chain"] == old["mechanism"]["chain"]
    assert new["mechanism"]["authorities"] == old["mechanism"]["authorities"]
    assert new["mechanism"]["changed_boundaries"] == ["evidence"]
    assert new["evidence"] == old["evidence"]
    for name in (
        "hypothesis",
        "economic_mechanism",
        "information_gain",
        "metric_proxy_rationale",
        "stage",
    ):
        assert new["thesis"][name] == old["thesis"][name], name
    # The new text only extends the old text; nothing registered is removed.
    for name in ("counter_hypothesis", "stop_rule", "limitations"):
        assert old["thesis"][name] in new["thesis"][name], name
    assert old["thesis"]["cheapest_falsifier"] in new["thesis"]["cheapest_falsifier"]
    assert old["claims"]["decision_rule"] in new["claims"]["decision_rule"]
    for name in ("permitted", "forbidden"):
        assert new["claims"][name][: len(old["claims"][name])] == old["claims"][name]
    text = _text(new)
    assert _policy_mismatches(_parameters(text, POLICY_PREFIX)) == []
    assert _mismatches(_parameters(text, TIMING_PREFIX)) == []


def test_packet_grants_no_authorization_and_claims_no_review_or_seal() -> None:
    packet = _packet()
    claims, thesis = packet["claims"], packet["thesis"]
    limitations = " ".join(thesis["limitations"].split())
    assert claims["production_eligible"] is False
    assert claims["live_trading_authorized"] is False
    assert claims["claim_level"] == "software_validity"
    assert claims["next_action_class"] == "documentation_only"
    assert packet["evidence"]["evidence_level"] == "software_validity"
    assert thesis["stage"] == "prospective_paper"
    assert "No PASS is claimed" in limitations
    assert "no seal, launch, probe or economic run" in " ".join(
        claims["next_authorized_action"].split()
    )
    assert PENDING in _text(packet)
    assert "Do not seal attempt 3 under this packet alone" in claims["decision_rule"]
    assert "does not extend to attempt 3" in " ".join(thesis["stop_rule"].split())
    forbidden = " ".join(claims["forbidden"])
    for required in (
        "changing any economic, tolerance, budget or criterion value",
        "without the owner's explicit decision",
        "sealing on one host and launching on another",
        "reusing the attempt-2 seal",
    ):
        assert required in forbidden, required
    cited = set(re.findall(r"tests/[\w/]+\.py", _text(packet)))
    assert cited
    modules = set(re.findall(r"trade_rl/[\w/]+\.py", _text(packet)))
    assert set(PLATFORM_FILES) <= modules

    # A packet cites its frozen implementation, not the evolving main tree.
    # Only explicitly archived paths may disappear from the current checkout.
    manifest = json.loads(
        (ROOT / "report/legacy-research-retirement-20261126.json").read_text(
            encoding="utf-8"
        )
    )
    archived = {row["path"]: row for row in manifest["records"]}
    head = packet["identity"]["implementation_head"]
    for path in sorted(cited | modules):
        assert _git("cat-file", "-t", f"{head}:{path}").stdout.strip() == "blob", path
        if not (ROOT / path).is_file():
            row = archived[path]
            original = subprocess.check_output(
                ["git", "show", f"{row['source_commit']}:{path}"], cwd=ROOT
            )
            assert hashlib.sha256(original).hexdigest() == row["sha256"], path


def test_heads_bind_the_packet_to_the_platform_source_and_the_listed_delta() -> None:
    head = _packet()["identity"]["implementation_head"]
    if _git("cat-file", "-e", f"{head}^{{commit}}").returncode != 0:
        pytest.skip("implementation_head is not available in this checkout")
    if _git("cat-file", "-e", f"{SEAL_CANDIDATE_HEAD}^{{commit}}").returncode != 0:
        pytest.skip("the seal-candidate head is not available in this checkout")
    linux = _git("show", f"{head}:{PLATFORM_FILES[0]}")
    assert linux.returncode == 0, linux.stderr
    for name in ("build_layout", "render_supervisor_unit", "clock_report"):
        assert name in linux.stdout, name
    watchdog = _git("show", f"{head}:{PLATFORM_FILES[1]}")
    assert '"SIGHUP"' in watchdog.stdout
    published = _git("log", "-1", "--format=%H", "--", RELATIVE_PACKET)
    packet_commit = published.stdout.strip()
    if (
        published.returncode != 0
        or not re.fullmatch(r"[0-9a-f]{40}", packet_commit)
        or _git("merge-base", "--is-ancestor", head, packet_commit).returncode != 0
    ):
        pytest.skip("packet source commit is unavailable in a shallow checkout")
    paths = paper_code_dependency_paths(head)
    assert set(PLATFORM_FILES) <= set(paths)
    changed = _git("diff", "--name-only", head, packet_commit, "--", *paths)
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.strip() == "", (
        "packet-bound code changed after implementation_head; a new packet and digest "
        "are required: " + changed.stdout
    )
    everything = sorted(
        set(paths) | set(paper_code_dependency_paths(SEAL_CANDIDATE_HEAD))
    )
    delta = _git(
        "diff", "--name-status", SEAL_CANDIDATE_HEAD, head, "--", *everything
    ).stdout
    listed = " ".join(_packet()["thesis"]["limitations"].split())
    own = set(PLATFORM_FILES)
    rows = [line.split("\t") for line in delta.splitlines() if line.strip()]
    assert {path for _status, path in rows} >= own
    for status, path in rows:
        assert f"{path} ({status})" in listed, path
    names = re.findall(r"(trade_rl/[\w/]+\.py) \([AMD]\)", listed)
    assert sorted(names) == sorted(path for _status, path in rows)


def test_the_documents_point_to_the_packet_digest_heads_and_unreviewed_status() -> None:
    head = _packet()["identity"]["implementation_head"]
    for relative in (SPEC, ARCHITECTURE, STATUS):
        document = _raw(relative)
        assert RELATIVE_PACKET in document, relative
        assert PINNED_DIGEST in document, relative
        assert "UNREVIEWED" in document, relative
        assert "{{" not in document and "LINUXDIGEST" not in document, relative
        assert "HEADPLACE" not in document, relative
    assert head in _raw(SPEC)
    assert head[:12] in _raw(ARCHITECTURE)
    assert RELATIVE_PACKET in _raw(BOUNDARIES)
    plan = _raw(PLAN)
    assert RELATIVE_PACKET in plan and "Status: Active" in plan[:200]
    section = _spec_section()
    assert "PENDING_OWNER_AUTHORIZATION" in section
    assert "nothing is sealed, launched, probed or collected" in section.lower()


def test_the_specification_keeps_the_windows_recipe_and_states_the_linux_procedure() -> (
    None
):
    document = _raw(SPEC)
    for required in (
        "Register-ScheduledTask -TaskName carry-a3-supervisor",
        "Register-ScheduledTask -TaskName carry-a3-check",
        "C:\\trade_rl_runs\\attempt3\\root-<YYYYMMDD>",
        "C:\\trade_rl_frozen\\attempt3-<first 8 hex of LAUNCH_COMMIT>",
    ):
        assert required in document, required
    section = _spec_section()
    for required in (
        "Windows assumptions and their handling",
        "SetConsoleCtrlHandler",
        "CollectorConsoleClosed",
        "Task Scheduler",
        "loginctl enable-linger",
        "systemctl --user enable --now carry-a3-supervisor.service carry-a3-check.timer",
        "systemctl --user status",
        "journalctl --user",
        "git -C ~/trade_rl_src worktree add --detach",
        "uv sync --frozen",
        "linux_service clock",
        "storage_probe",
        "~/trade_rl_runs/attempt3/",
        "~/trade_rl_frozen/attempt3-",
        "~/trade_rl_runs/attempt2/root-20261107",
        "What still makes the attempt INVALID on Linux",
        "KillMode=control-group",
        "RestartPreventExitStatus=1 3 4",
        "a reboot",
    ):
        assert required in section, required


def test_every_windows_assumption_named_in_the_specification_exists_in_the_source() -> (
    None
):
    from trade_rl.evaluation.paper import watchdog

    source = Path(watchdog.__file__).read_text(encoding="utf-8")
    for name in (
        "SetConsoleCtrlHandler",
        "CollectorConsoleClosed",
        "CTRL_BREAK_EVENT",
        "CREATE_NEW_PROCESS_GROUP",
        "PermissionError",
    ):
        assert name in source, name
    assert hasattr(watchdog.ExitGuard, "console_event")
    assert hasattr(watchdog, "PopenChild")


def test_new_text_is_result_blind() -> None:
    forbidden = (
        re.compile(
            r"\bpass(?:ed|es)\b[^.;]{0,60}\b(?:screens?|criteria|preregistered)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bpast development profit", re.IGNORECASE),
        re.compile(
            r"\bdevelopment (?:replay|run|result)s?[^.;]{0,40}\b"
            r"(?:profit|passed|positive)",
            re.IGNORECASE,
        ),
        re.compile(r"\+\s?\d+(?:\.\d+)?\s?%"),
        re.compile(r"observed consume latency", re.IGNORECASE),
    )
    sections = {
        "packet": _text(_packet()),
        "spec": _spec_section(),
    }
    for pattern in forbidden:
        for name, section in sections.items():
            assert pattern.search(section) is None, (name, pattern.pattern)
    assert (
        "no attempt-2 timing metadata"
        in " ".join(_packet()["thesis"]["limitations"].split()).lower()
    )
