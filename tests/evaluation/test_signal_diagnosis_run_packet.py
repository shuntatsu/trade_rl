"""The SD2 packet stays UNREVIEWED, result-blind and bound to the runner code.

``report/sd2-signal-diagnosis-research-assurance-20261125-calendar-feature.json`` is the ``research_assurance_v1``
author packet of the SD2 runner. These tests keep it valid, unreviewed and pinned, keep
the documents that name it in step, keep it from claiming a review, a result or an
authorization, and pin the code it covers: the static import closure of the runner,
resolved from the immutable packet head. They read no market data.
"""

from __future__ import annotations

import ast
import hashlib
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
    / "sd2-signal-diagnosis-research-assurance-20261125-calendar-feature.json"
)
SPEC = ROOT / "docs" / "specs" / "signal-diagnosis-run.md"
ARCHITECTURE = ROOT / "docs" / "architecture" / "research-assurance.md"
STATUS = ROOT / "docs" / "research" / "current-status.md"
PINNED_DIGEST = "687e870e69d64f6972da234e965be6415e8a50e40f2b38281c507be3678622da"
PREVIOUS_CORE7_M2_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261121-after-core7-m2-main.json"
)
PREVIOUS_CORE7_M2_PACKET_DIGEST = (
    "a8fee68557bf2bf5c2fce326fc4955820079bea12cba5b4aedc7aba348ed3e91"
)
PREVIOUS_MAIN1288_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261120-after-main1288-consistent.json"
)
PREVIOUS_MAIN1288_PACKET_DIGEST = (
    "863d96c40b47084c8702f1520cf62724cd5d74ccd0cf3236da37e05e42486350"
)

PREVIOUS_TRACKD_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261120-after-trackd-main.json"
)
PREVIOUS_TRACKD_PACKET_DIGEST = (
    "8175d4a11a2f910cb903056486ea1d40809fc187e7f9051021ade65ca1bea233"
)

PREVIOUS_CONSISTENT_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261120-after-m2-consistent-oracles.json"
)
PREVIOUS_CONSISTENT_PACKET_DIGEST = (
    "0e7457fc150abd8d99871abe70febd264c960c01361b0bef46ff26981cac6458"
)

PREVIOUS_POST_M2_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261120-after-m2-merged-main.json"
)
PREVIOUS_POST_M2_PACKET_DIGEST = (
    "6fadf622741fcc3df68bd2354b8dd9e5e9c0e689dfc7d77e2311bfdd5d0e0f1e"
)

PREVIOUS_UNIFIED_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261120-safe-unified-runner.json"
)
PREVIOUS_UNIFIED_PACKET_DIGEST = (
    "b9565ffebdbff644702aa6cf166cd2a5fc9d69fa1eea57d0a3a8c6b2387eac8d"
)

PREVIOUS_BOUNDED_PACKET_PATH = (
    ROOT
    / "report"
    / "sd2-signal-diagnosis-research-assurance-20261114-after-bounded-posix-main-sync.json"
)
PREVIOUS_BOUNDED_PACKET_DIGEST = (
    "c5404d14da6bd7493f05c218dd52cb2f6ae8577224da176d1d26d132b5a3d736"
)
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


def test_previous_core7_m2_packet_is_immutable_unreviewed_history() -> None:
    previous = json.loads(PREVIOUS_CORE7_M2_PACKET_PATH.read_text(encoding="utf-8"))
    checked = evaluate_assurance(previous)
    assert checked.status == "UNREVIEWED" and checked.errors == ()
    assert checked.economic_execution_authorized is False
    assert assurance_digest(previous) == PREVIOUS_CORE7_M2_PACKET_DIGEST
    assert (
        previous["identity"]["implementation_head"]
        == "7a09a1fbaa3aa3ea542ab41696415248e1ebaca4"
    )
    assert (
        previous["identity"]["protocol_head"]
        == "7a09a1fbaa3aa3ea542ab41696415248e1ebaca4"
    )
    # Byte identity is independent of clone depth; the predecessor remains
    # immutable even in a shallow source checkout.
    assert hashlib.sha256(PREVIOUS_CORE7_M2_PACKET_PATH.read_bytes()).hexdigest() == (
        "d5810d286318415c7872e48441152f4baf5439c014461ec13b85bce5b342bfcb"
    )


def test_prior_main1288_packet_is_immutable_unreviewed_history() -> None:
    """M2 source re-pin cannot rewrite prior exact-source author evidence."""
    previous = json.loads(PREVIOUS_MAIN1288_PACKET_PATH.read_text(encoding="utf-8"))
    checked = evaluate_assurance(previous)
    assert checked.status == "UNREVIEWED" and checked.errors == ()
    assert checked.economic_execution_authorized is False
    assert assurance_digest(previous) == PREVIOUS_MAIN1288_PACKET_DIGEST
    assert previous["identity"]["implementation_head"] == (
        "04d7f22701b3dd0067e5d72cebe6db6d11a60c69"
    )


def test_prior_trackd_packet_is_immutable_unreviewed_history() -> None:
    """The prior published Track D author packet cannot be rewritten."""
    previous = json.loads(PREVIOUS_TRACKD_PACKET_PATH.read_text(encoding="utf-8"))
    checked = evaluate_assurance(previous)
    assert checked.status == "UNREVIEWED" and checked.errors == ()
    assert checked.economic_execution_authorized is False
    assert assurance_digest(previous) == PREVIOUS_TRACKD_PACKET_DIGEST
    assert (
        previous["identity"]["implementation_head"]
        == "3e206e9620a4eeb0970eecbcceb550faa8275c15"
    )


def test_prior_consistent_oracles_packet_is_immutable_unreviewed_history() -> None:
    prior = json.loads(PREVIOUS_CONSISTENT_PACKET_PATH.read_text(encoding="utf-8"))
    assert assurance_digest(prior) == PREVIOUS_CONSISTENT_PACKET_DIGEST
    assert prior["identity"]["implementation_head"] == (
        "698ebb10095bad5cf1773fc524bf2adae836fa0f"
    )
    result = evaluate_assurance(prior)
    assert result.status == "UNREVIEWED"
    assert result.errors == ()
    assert result.economic_execution_authorized is False


def test_previous_post_m2_packet_is_immutable_and_unreviewed() -> None:
    """New author records cannot rewrite the earlier post-M2 snapshot."""
    previous = json.loads(PREVIOUS_POST_M2_PACKET_PATH.read_text(encoding="utf-8"))
    checked = evaluate_assurance(previous)
    assert checked.status == "UNREVIEWED" and checked.errors == ()
    assert not checked.economic_execution_authorized
    assert assurance_digest(previous) == PREVIOUS_POST_M2_PACKET_DIGEST
    assert previous["identity"]["implementation_head"] == (
        "698ebb10095bad5cf1773fc524bf2adae836fa0f"
    )


def test_previous_unified_packet_is_immutable_and_unreviewed() -> None:
    """An M2 source re-pin cannot overwrite the preceding SD2 author packet."""
    previous = json.loads(PREVIOUS_UNIFIED_PACKET_PATH.read_text(encoding="utf-8"))
    verified = evaluate_assurance(previous)
    assert verified.status == "UNREVIEWED" and verified.errors == ()
    assert not verified.economic_execution_authorized
    assert assurance_digest(previous) == PREVIOUS_UNIFIED_PACKET_DIGEST
    assert previous["identity"]["implementation_head"] == (
        "2f725160b4f585deb74eb431d846814676de996d"
    )


def test_previous_bounded_packet_is_immutable_and_unreviewed() -> None:
    """The combined code never rewrites an earlier strict POSIX author packet."""
    prior = json.loads(PREVIOUS_BOUNDED_PACKET_PATH.read_text(encoding="utf-8"))
    result = evaluate_assurance(prior)
    assert result.status == "UNREVIEWED" and result.errors == ()
    assert not result.economic_execution_authorized
    assert assurance_digest(prior) == PREVIOUS_BOUNDED_PACKET_DIGEST
    assert prior["identity"]["implementation_head"] == (
        "93d94145c7809c7fc0f1819a500f910b28eb2a0f"
    )


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


def test_current_packet_reconciles_post_m2_code_lineage_and_oracles() -> None:
    """Release-facing author evidence cannot contradict its frozen source SHA."""
    record = packet()
    text = record["evidence"]["independent_reconstruction"]
    assert "Existing shared verified_file code remains unchanged" not in text
    assert "revises shared verified_file.py" in text
    cheapest = record["thesis"]["cheapest_falsifier"]
    assert "POST-M2 PREVIEW" not in cheapest
    assert "eventual exact post-M2 main" not in cheapest
    assert (
        "Additional preview"
        not in record["mechanism"]["independent_oracles"]["time"]["oracle"]
    )
    authority = record["mechanism"]["authorities"]["state"]
    assert "no account or book state" in authority.lower()
    assert "Per-vintage" in authority
    assert (
        record["identity"]["implementation_head"]
        == (record["identity"]["protocol_head"])
    )


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
        "state",
        "sign",
        "execution",
        "evidence",
    }
    reconstruction = record["evidence"]["independent_reconstruction"]
    assert "duplicated two-symbol-style timestamps" in reconstruction
    assert "1 GiB+1" in reconstruction
    assert "test_grouped_ic_omits_exactly_constant_float_groups" in reconstruction
    assert "calendar" in reconstruction
    assert "before the first scheduled refit" in reconstruction
    assert "state" in record["mechanism"]["changed_boundaries"]
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


def test_current_packet_falsifier_and_reconstruction_bind_exact_source() -> None:
    """An old code head may be historical, never the source binding in an active falsifier."""
    record = packet()
    source = record["identity"]["implementation_head"]
    assert source == record["identity"]["protocol_head"]
    falsifier = record["thesis"]["cheapest_falsifier"]
    assert f"implementation_head {source}" in falsifier
    assert (
        "implementation_head 698ebb10095bad5cf1773fc524bf2adae836fa0f" not in falsifier
    )
    reconstruction = record["evidence"]["independent_reconstruction"]
    assert "#1287" in reconstruction and "xa_" in reconstruction


def test_architecture_has_only_one_current_sd2_author_packet_pointer() -> None:
    """Prevent duplicate Current pointers from misdirecting an owner G4 request."""
    contents = ARCHITECTURE.read_text(encoding="utf-8")
    current_lines = [
        line
        for line in contents.splitlines()
        if line.startswith("- **Current SD2 UNREVIEWED author packet")
    ]
    assert len(current_lines) == 1, current_lines
    assert PACKET_PATH.name in current_lines[0]


def test_documents_name_the_packet_and_the_protocol_digest() -> None:
    spec, status, architecture = _flat(SPEC), _flat(STATUS), _flat(ARCHITECTURE)
    assert "Status: Active" in SPEC.read_text(encoding="utf-8")
    assert PINNED_PROTOCOL_DIGEST in spec
    for text in (status, architecture):
        assert PINNED_DIGEST in text
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261121-after-core7-m2-main.json"
            in text
        )
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261124-after-current-main.json"
            in text
        )
        assert PREVIOUS_CORE7_M2_PACKET_DIGEST in text
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261120-after-main1288-consistent.json"
            in text
        )
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261120-after-trackd-main.json"
            in text
        )
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261120-after-m2-consistent-oracles.json"
            in text
        )
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261120-after-m2-merged-main.json"
            in text
        )
        assert (
            "report/sd2-signal-diagnosis-research-assurance-20261120-safe-unified-runner.json"
            in text
        )
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


def test_previous_current_main_packet_is_immutable_after_real_request_fix() -> None:
    path = (
        ROOT
        / "report/sd2-signal-diagnosis-research-assurance-20261124-after-current-main.json"
    )
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "7cf568426b1450fb518b214a11a119b1e013f41414fd9a68d22dc20a286fb9cd"
    )
    prior = json.loads(path.read_text(encoding="utf-8"))
    assert (
        assurance_digest(prior)
        == "921a28a50d4051181ae049c55356e5a5cfaae6943259ab839e2f5e566f09906a"
    )
    checked = evaluate_assurance(prior)
    assert checked.status == "UNREVIEWED" and checked.errors == ()
    assert not checked.economic_execution_authorized
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261124-real-request-boundary.json"
        ).read_bytes()
    )
    assert current["identity"] == {
        "implementation_head": "6879447ccd17f8a3ff9516c18ddcd9836b8055bd",
        "protocol_head": "6879447ccd17f8a3ff9516c18ddcd9836b8055bd",
    }
    assert current["mechanism"] == prior["mechanism"]
    assert (
        current["evidence"]
        | {
            "independent_reconstruction": prior["evidence"][
                "independent_reconstruction"
            ]
        }
        == prior["evidence"]
    )
    assert (
        "921a28a50d4051181ae049c55356e5a5cfaae6943259ab839e2f5e566f09906a"
        in current["thesis"]["limitations"]
    )
    for doc in (STATUS, ARCHITECTURE):
        assert (
            "921a28a50d4051181ae049c55356e5a5cfaae6943259ab839e2f5e566f09906a"
            in _flat(doc)
        )


def test_integrated_core7_packet_preserves_real_request_predecessor() -> None:
    path = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261124-real-request-boundary.json"
    )
    raw = path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "aeb1366ecbac19d4e808c3f7ca2ade4dda6b60705fe1b683b28fa1815a9b1a26"
    )
    prior = json.loads(raw)
    assert (
        assurance_digest(prior)
        == "8c46efca4cd9c112816d3d8e5082f08df5f468ff4c22bce2696b05cea0228686"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261124-integrated-core7.json"
        ).read_bytes()
    )
    for record in (prior, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and verdict.errors == ()
        assert not verdict.economic_execution_authorized
    assert current["mechanism"] == prior["mechanism"]
    for field in prior["claims"]:
        if field != "next_authorized_action":
            assert current["claims"][field] == prior["claims"][field]
    for field in prior["thesis"]:
        if field not in ("limitations", "cheapest_falsifier"):
            assert current["thesis"][field] == prior["thesis"][field]
    assert current["thesis"]["cheapest_falsifier"] == prior["thesis"][
        "cheapest_falsifier"
    ].replace(
        "6879447ccd17f8a3ff9516c18ddcd9836b8055bd",
        "d691f3b5eee72be1ee58e907213640b103aab0a7",
    )
    for field in prior["evidence"]:
        if field != "independent_reconstruction":
            assert current["evidence"][field] == prior["evidence"][field]
    assert (
        "8c46efca4cd9c112816d3d8e5082f08df5f468ff4c22bce2696b05cea0228686"
        in current["thesis"]["limitations"]
    )
    assert current["identity"] == {
        "implementation_head": "d691f3b5eee72be1ee58e907213640b103aab0a7",
        "protocol_head": "d691f3b5eee72be1ee58e907213640b103aab0a7",
    }
    assert bound_code_paths(
        "6879447ccd17f8a3ff9516c18ddcd9836b8055bd"
    ) == bound_code_paths("d691f3b5eee72be1ee58e907213640b103aab0a7")
    changed = _git(
        "diff",
        "--name-only",
        "6879447ccd17f8a3ff9516c18ddcd9836b8055bd",
        "d691f3b5eee72be1ee58e907213640b103aab0a7",
        "--",
        *bound_code_paths("6879447ccd17f8a3ff9516c18ddcd9836b8055bd"),
    )
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.splitlines() == [
        "trade_rl/data/features/multitimeframe.py",
        "trade_rl/evaluation/decision_value_dev_study.py",
        "trade_rl/evaluation/decision_value_ledger_oracle.py",
        "trade_rl/evaluation/replay.py",
        "trade_rl/evaluation/runs/artifact.py",
        "trade_rl/simulation/accounting.py",
        "trade_rl/simulation/native_transition.py",
        "trade_rl/simulation/stateful/portfolio_session.py",
        "trade_rl/strategies/forecasts/ridge.py",
        "trade_rl/strategies/forecasts/supervised.py",
    ]
    for doc in (STATUS, ARCHITECTURE):
        assert (
            "sd2-signal-diagnosis-research-assurance-20261124-real-request-boundary.json"
            in _flat(doc)
        )
        assert (
            "8c46efca4cd9c112816d3d8e5082f08df5f468ff4c22bce2696b05cea0228686"
            in _flat(doc)
        )


def test_calendar_feature_source_preserves_integrated_core7_packet() -> None:
    previous_path = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261124-integrated-core7.json"
    )
    raw = previous_path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "2a8408117fd74b914a3e007937c9dd657a5e63a16df8a69d8b4141a6940dade3"
    )
    previous = json.loads(raw)
    current = packet()
    assert (
        assurance_digest(previous)
        == "e89bcb29086cd4e00543a041e47e7019cb04dca87fde4bfc9fa4bef5d69a3d7c"
    )
    assert current.keys() == previous.keys()
    for field in previous:
        if field not in ("identity", "thesis", "evidence", "claims"):
            assert current[field] == previous[field]
    for field in previous["thesis"]:
        if field not in ("limitations", "cheapest_falsifier"):
            assert current["thesis"][field] == previous["thesis"][field]
    for field in previous["evidence"]:
        if field != "independent_reconstruction":
            assert current["evidence"][field] == previous["evidence"][field]
    for field in previous["claims"]:
        if field != "next_authorized_action":
            assert current["claims"][field] == previous["claims"][field]
    assert current["thesis"]["limitations"].startswith(
        previous["thesis"]["limitations"]
    )
    assert current["evidence"]["independent_reconstruction"].startswith(
        previous["evidence"]["independent_reconstruction"]
    )
    assert current["identity"] == {
        "implementation_head": "a542bcb38bc3118c4419bf7497e74c1d670d8075",
        "protocol_head": "a542bcb38bc3118c4419bf7497e74c1d670d8075",
    }
    old_head = previous["identity"]["implementation_head"]
    new_head = current["identity"]["implementation_head"]
    assert current["thesis"]["cheapest_falsifier"] == previous["thesis"][
        "cheapest_falsifier"
    ].replace(old_head, new_head)
    old = set(bound_code_paths(old_head))
    new = set(bound_code_paths(new_head))
    assert new - old == {"trade_rl/data/native_clock.py"}
    assert not old - new
    changed = _git("diff", "--name-only", old_head, new_head, "--", *sorted(old | new))
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.splitlines() == ["trade_rl/data/features/multitimeframe.py"]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized
    for doc in (STATUS, ARCHITECTURE):
        assert previous_path.name in _flat(doc)
        assert (
            "e89bcb29086cd4e00543a041e47e7019cb04dca87fde4bfc9fa4bef5d69a3d7c"
            in _flat(doc)
        )
