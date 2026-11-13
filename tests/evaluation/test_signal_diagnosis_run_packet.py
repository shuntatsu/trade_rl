"""The SD2 packet stays UNREVIEWED, result-blind and bound to the runner code.

``report/sd2-signal-diagnosis-research-assurance-20261114-after-bounded-posix-main-sync.json`` is the ``research_assurance_v1``
author packet of the SD2 runner. These tests keep it valid, unreviewed and pinned, keep
the documents that name it in step, keep it from claiming a review, a result or an
authorization, and pin the code it covers: the static import closure of the runner,
resolved from the immutable packet head. They read no market data.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance
from trade_rl.evaluation.signal_diagnosis.protocol import (
    default_signal_diagnosis_protocol,
    protocol_digest,
)

ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261114-after-bounded-posix-main-sync.json"
)
SPEC = ROOT / "docs" / "specs" / "signal-diagnosis-run.md"
ARCHITECTURE = ROOT / "docs" / "architecture" / "research-assurance.md"
STATUS = ROOT / "docs" / "research" / "current-status.md"
PINNED_DIGEST = "c5404d14da6bd7493f05c218dd52cb2f6ae8577224da176d1d26d132b5a3d736"
PREVIOUS_NOFOLLOW_MAIN_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261114-after-nofollow-main-sync.json"
)
PREVIOUS_NOFOLLOW_MAIN_PACKET_DIGEST = (
    "b81b269938ae1e348bac4db6312fd1f04434851b27f381ade3c774e180a53211"
)
PREVIOUS_NOFOLLOW_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261114-after-nofollow.json"
)
PREVIOUS_NOFOLLOW_PACKET_DIGEST = (
    "2b68a4b7b751dea113ecec54dcadf09e00c613e23e9bb6d8c0592e4b73de11b7"
)
PREVIOUS_MAIN_SYNC_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261113-after-main-sync.json"
)
PREVIOUS_MAIN_SYNC_PACKET_DIGEST = (
    "4059f82b0708014065349d7956f7c96cab9e5d6647a911d3def794dd616e435e"
)
PREVIOUS_CODEC_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261113-after-codec-isolation.json"
)
PREVIOUS_CODEC_PACKET_DIGEST = (
    "10acf9db00b7ea594f174f0a16e7a978974ed4d483c437f1785b4afd97a094f4"
)
PREVIOUS_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261113-after-manifest-gate.json"
)
PREVIOUS_PINNED_DIGEST = (
    "bc0163631694f42cc6fbc52ca51678e9bebe9e86efb3e9d5ca46607971b30368"
)
HISTORICAL_PACKET_PATH = (
    ROOT / "report" / "sd2-signal-diagnosis-research-assurance-20261113.json"
)
HISTORICAL_PINNED_DIGEST = (
    "ff5c8301491b9809515d8fb00b4b8395a0e065ea6bccd91a9b3172b5db986279"
)
PINNED_PROTOCOL_DIGEST = (
    "1f48b065b03bc38299a751bef89456b57d5afc6a7d8b917668b1b461f852d0b3"
)
# The bound code is the static import closure of these entry points (module-level and
# lazy imports, plus the ``__init__`` of every ancestor package), resolved from the
# immutable packet head, never from the working tree.
ENTRY_POINTS = ("trade_rl/evaluation/signal_diagnosis_run/",)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _pinned_python_files(head: str) -> set[str]:
    listing = _git("ls-tree", "-r", "--name-only", head, "trade_rl", "tools")
    assert listing.returncode == 0, listing.stderr
    return {path for path in listing.stdout.split() if path.endswith(".py")}


def _module_name(path: str) -> str:
    return path[: -len(".py")].replace("/", ".").removesuffix(".__init__")


def _module_file(module: str, files: set[str]) -> str | None:
    base = module.replace(".", "/")
    for candidate in (f"{base}.py", f"{base}/__init__.py"):
        if candidate in files:
            return candidate
    return None


def _imported_modules(path: str, source: str) -> set[str]:
    package = (
        _module_name(path)
        if path.endswith("/__init__.py")
        else _module_name(path).rpartition(".")[0]
    )
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - (node.level - 1)]
                module = ".".join([*parts, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return {name for name in found if name.split(".")[0] in {"trade_rl", "tools"}}


def bound_code_paths(head: str) -> list[str]:
    """Every file the runner statically imports, transitively, at ``head``."""
    files = _pinned_python_files(head)
    pending = [
        path
        for path in files
        if any(path == entry or path.startswith(entry) for entry in ENTRY_POINTS)
    ]
    assert pending, "no entry point found at the packet head"
    seen: set[str] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        parts = _module_name(path).split(".")
        for depth in range(1, len(parts)):
            init = "/".join(parts[:depth]) + "/__init__.py"
            if init in files:
                pending.append(init)
        source = _git("show", f"{head}:{path}")
        assert source.returncode == 0, source.stderr
        for module in _imported_modules(path, source.stdout):
            target = _module_file(module, files)
            if target is not None:
                pending.append(target)
    return sorted(seen)


def packet() -> dict[str, Any]:
    return json.loads(PACKET_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _head_or_skip() -> str:
    head = str(packet()["identity"]["implementation_head"])
    if _git("cat-file", "-e", f"{head}^{{commit}}").returncode != 0:
        pytest.skip("packet head is not in this clone (shallow checkout)")
    return head


def _flat(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_packet_is_valid_unreviewed_and_pinned() -> None:
    record = packet()
    result = evaluate_assurance(record)
    assert result.status == "UNREVIEWED" and result.errors == ()
    assert result.economic_execution_authorized is False
    assert assurance_digest(record) == PINNED_DIGEST == result.record_digest
    assert b"\r" not in PACKET_PATH.read_bytes()


def test_pre_bounded_no_follow_packet_is_immutable_and_unreviewed() -> None:
    """Technical resource admission changes never rewrite earlier source evidence."""

    prior = json.loads(PREVIOUS_NOFOLLOW_MAIN_PACKET_PATH.read_text(encoding="utf-8"))
    result = evaluate_assurance(prior)
    assert result.status == "UNREVIEWED" and result.errors == ()
    assert result.economic_execution_authorized is False
    assert assurance_digest(prior) == PREVIOUS_NOFOLLOW_MAIN_PACKET_DIGEST
    assert prior["identity"]["implementation_head"] == (
        "3a3c959350fe31d4a05d9540d460c3187be3a772"
    )


def test_pre_main_sync_nofollow_packet_is_immutable_and_unreviewed() -> None:
    """Source rebind preserves the no-follow author packet from before the merge."""

    previous = json.loads(PREVIOUS_NOFOLLOW_PACKET_PATH.read_text(encoding="utf-8"))
    result = evaluate_assurance(previous)
    assert result.status == "UNREVIEWED" and result.errors == ()
    assert result.economic_execution_authorized is False
    assert assurance_digest(previous) == PREVIOUS_NOFOLLOW_PACKET_DIGEST
    assert previous["identity"]["implementation_head"] == (
        "7756984b9c5513612c8cf3852feea3c8e35a17e6"
    )


def test_pre_nofollow_packet_stays_immutable_and_unreviewed() -> None:
    """No source rebind may rewrite the predecessor's author-only evidence."""

    prior = json.loads(PREVIOUS_MAIN_SYNC_PACKET_PATH.read_text(encoding="utf-8"))
    result = evaluate_assurance(prior)
    assert result.status == "UNREVIEWED" and result.errors == ()
    assert result.economic_execution_authorized is False
    assert assurance_digest(prior) == PREVIOUS_MAIN_SYNC_PACKET_DIGEST
    assert prior["identity"]["implementation_head"] == (
        "2939e33aebee4008a03a2026c105c8797a41826f"
    )


def test_pre_main_sync_packet_stays_immutable_and_unreviewed() -> None:
    """The new source binding must not retroactively change the previous packet."""

    prior = json.loads(PREVIOUS_CODEC_PACKET_PATH.read_text(encoding="utf-8"))
    result = evaluate_assurance(prior)
    assert result.status == "UNREVIEWED" and result.errors == ()
    assert result.economic_execution_authorized is False
    assert assurance_digest(prior) == PREVIOUS_CODEC_PACKET_DIGEST
    assert prior["identity"]["implementation_head"] == (
        "dd4dc168070055d890b380d62f745054f88e83aa"
    )


def test_preceding_manifest_first_packet_is_immutable() -> None:
    """The unreviewed predecessor does not change when source closure is repaired."""

    predecessor = json.loads(PREVIOUS_PACKET_PATH.read_text(encoding="utf-8"))
    assert assurance_digest(predecessor) == PREVIOUS_PINNED_DIGEST
    result = evaluate_assurance(predecessor)
    assert result.status == "UNREVIEWED" and result.errors == ()
    assert result.economic_execution_authorized is False
    assert predecessor["identity"]["implementation_head"] == (
        "fb67d040838616d20a34c0daca3834d9d7b293e0"
    )


def test_historical_sd2_packet_remains_immutable_and_unreviewed() -> None:
    """A rebind creates a new packet, never retroactively changes the old one."""

    original = json.loads(HISTORICAL_PACKET_PATH.read_text(encoding="utf-8"))
    result = evaluate_assurance(original)
    assert result.status == "UNREVIEWED" and result.errors == ()
    assert assurance_digest(original) == HISTORICAL_PINNED_DIGEST
    assert original["identity"] == {
        "protocol_head": "4eb79f7ef6d648c67d6d5e3b0fb411df7129cbb1",
        "implementation_head": "4eb79f7ef6d648c67d6d5e3b0fb411df7129cbb1",
    }
    assert result.economic_execution_authorized is False


def test_packet_names_the_frozen_protocol_and_claims_nothing_beyond_software() -> None:
    assert (
        protocol_digest(default_signal_diagnosis_protocol()) == PINNED_PROTOCOL_DIGEST
    )
    record = packet()
    text = json.dumps(record)
    assert PINNED_PROTOCOL_DIGEST in text
    claims = record["claims"]
    assert claims["claim_level"] == record["evidence"]["evidence_level"]
    assert claims["claim_level"] == "software_validity"
    assert claims["production_eligible"] is False
    assert claims["live_trading_authorized"] is False
    assert claims["next_action_class"] == "request_separate_economic_authorization"
    assert record["thesis"]["stage"] == "discovery"
    forbidden = " ".join(claims["forbidden"])
    for needle in (
        "dirty working tree",
        "G4 authorization",
        "FS1",
        "beta-adjusted",
        "live-trading",
    ):
        assert needle in forbidden, needle
    evidence = record["evidence"]
    for key in ("point_in_time", "realistic_costs", "fit_development_unused_separated"):
        assert evidence[key] is False


def test_packet_states_the_g0_to_g2_content_of_the_runner() -> None:
    record = packet()
    thesis = record["thesis"]
    assert "known-used" in thesis["limitations"]
    assert "single run" in thesis["stop_rule"]
    assert "INVALID" in thesis["cheapest_falsifier"]
    assert "planted signal" in thesis["cheapest_falsifier"]
    assert set(record["mechanism"]["independent_oracles"]) >= {
        "time",
        "sign",
        "execution",
        "evidence",
    }
    interpretations = thesis["limitations"]
    for needle in (
        "matures exactly at the refit instant",
        "pooled",
        "secondary horizons",
        "stop_when_upper_layer_stops",
        "UNREVIEWED",
    ):
        assert needle in interpretations, needle
    assert "No economic result, real price or return was read" in json.dumps(record)


def test_documents_name_the_packet_and_the_protocol_digest() -> None:
    spec, status, architecture = _flat(SPEC), _flat(STATUS), _flat(ARCHITECTURE)
    assert "Status: Active" in SPEC.read_text(encoding="utf-8")
    assert PINNED_PROTOCOL_DIGEST in spec
    for text in (status, architecture):
        assert PINNED_DIGEST in text
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261114-after-bounded-posix-main-sync.json"
            in text
        )
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261114-after-nofollow-main-sync.json"
            in text
        )
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261114-after-nofollow.json"
            in text
        )
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261113-after-main-sync.json"
            in text
        )
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261113-after-codec-isolation.json"
            in text
        )
        assert "UNREVIEWED" in text
    assert "tests/evaluation/test_signal_diagnosis_run_packet.py" in architecture


def test_packet_heads_are_one_commit_and_the_bound_code_is_unchanged() -> None:
    identity = packet()["identity"]
    head = identity["protocol_head"]
    assert head == identity["implementation_head"]
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    _head_or_skip()
    assert _git("merge-base", "--is-ancestor", head, "HEAD").returncode == 0
    bound = bound_code_paths(head)
    changed = _git("diff", "--name-only", head, "HEAD", "--", *bound)
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.split() == [], (
        "re-pin the packet heads after changing the bound code"
    )


def test_the_static_closure_covers_every_shared_dependency_of_the_runner() -> None:
    bound = set(bound_code_paths(_head_or_skip()))
    for needed in (
        "trade_rl/evaluation/signal_diagnosis_run/runner.py",
        "trade_rl/evaluation/signal_diagnosis_run/fitting.py",
        "trade_rl/evaluation/signal_diagnosis_run/measure.py",
        "trade_rl/evaluation/signal_diagnosis/protocol.py",
        "trade_rl/evaluation/signal_diagnosis/verdict.py",
        "trade_rl/evaluation/signal_measurement/significance.py",
        "trade_rl/evaluation/signal_measurement/payoff.py",
        "trade_rl/strategies/forecasts/prequential.py",
        "trade_rl/strategies/forecasts/ridge.py",
        "trade_rl/strategies/forecasts/lightgbm.py",
        "trade_rl/data/artifacts/publication.py",
        "trade_rl/data/features/core.py",
    ):
        assert needed in bound, needed
    assert not any(path.startswith("tests/") for path in bound)
