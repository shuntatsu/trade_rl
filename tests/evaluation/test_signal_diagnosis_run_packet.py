"""The SD2 packet stays UNREVIEWED, result-blind and bound to the runner code.

``report/sd2-signal-diagnosis-research-assurance-20261126-fill-td8-node-union.json`` is the ``research_assurance_v1``
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
    / "sd2-signal-diagnosis-research-assurance-20261126-fill-td8-node-union.json"
)
SPEC = ROOT / "docs" / "specs" / "signal-diagnosis-run.md"
ARCHITECTURE = ROOT / "docs" / "architecture" / "research-assurance.md"
STATUS = ROOT / "docs" / "research" / "current-status.md"
PINNED_DIGEST = "9afe855813642edc646c5c63732c833315c99206250dcee0eb8066aff09851d9"
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
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-calendar-feature.json"
        ).read_bytes()
    )
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


def test_elapsed_dataset_preserves_calendar_feature_packet() -> None:
    previous_path = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-calendar-feature.json"
    )
    raw = previous_path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "389cf75cf22a3dc3ad0e765285988097b9c38907da08b31578ccdd108235a7e1"
    )
    previous = json.loads(raw)
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-dataset-elapsed.json"
        ).read_bytes()
    )
    assert (
        assurance_digest(previous)
        == "687e870e69d64f6972da234e965be6415e8a50e40f2b38281c507be3678622da"
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
        "implementation_head": "1dc565ad09e379b0fc77f24d68561ce52896a348",
        "protocol_head": "1dc565ad09e379b0fc77f24d68561ce52896a348",
    }
    old_head = previous["identity"]["implementation_head"]
    new_head = current["identity"]["implementation_head"]
    assert current["thesis"]["cheapest_falsifier"] == previous["thesis"][
        "cheapest_falsifier"
    ].replace(old_head, new_head)
    old = set(bound_code_paths(old_head))
    new = set(bound_code_paths(new_head))
    assert new == old
    changed = _git("diff", "--name-only", old_head, new_head, "--", *sorted(old))
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.splitlines() == ["trade_rl/data/market.py"]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized
    for doc in (STATUS, ARCHITECTURE):
        assert previous_path.name in _flat(doc)
        assert (
            "687e870e69d64f6972da234e965be6415e8a50e40f2b38281c507be3678622da"
            in _flat(doc)
        )


def test_native_grid_preserves_elapsed_dataset_packet() -> None:
    previous_path = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-dataset-elapsed.json"
    )
    raw = previous_path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "4b19d91233737d3f892279f7e54e1b9601547f17316e359bcb4862f3cb2f6956"
    )
    previous = json.loads(raw)
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-native-grid.json"
        ).read_bytes()
    )
    assert (
        assurance_digest(previous)
        == "84ccdd78a87d2d0f471b8dba988e1893426e054a6f47ed0e52c5a7b94c80e77a"
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
        "implementation_head": "da5bee160c0a37b4c2dbe5130aa95c99f20e1313",
        "protocol_head": "da5bee160c0a37b4c2dbe5130aa95c99f20e1313",
    }
    old_head = previous["identity"]["implementation_head"]
    new_head = current["identity"]["implementation_head"]
    assert current["thesis"]["cheapest_falsifier"] == previous["thesis"][
        "cheapest_falsifier"
    ].replace(old_head, new_head)
    old = set(bound_code_paths(old_head))
    new = set(bound_code_paths(new_head))
    assert new == old
    changed = _git("diff", "--name-only", old_head, new_head, "--", *sorted(old))
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.splitlines() == ["trade_rl/data/native_clock.py"]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized
    for doc in (STATUS, ARCHITECTURE):
        assert previous_path.name in _flat(doc)
        assert (
            "84ccdd78a87d2d0f471b8dba988e1893426e054a6f47ed0e52c5a7b94c80e77a"
            in _flat(doc)
        )


def test_rsi_scratch_preserves_native_grid_packet() -> None:
    previous_path = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-native-grid.json"
    )
    raw = previous_path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "9d5f9caf31273dbb256dab7e90b3885b8c865dabff60109f54613afa439b7af2"
    )
    previous = json.loads(raw)
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-rsi-scratch.json"
        ).read_bytes()
    )
    assert (
        assurance_digest(previous)
        == "d18bb91dee5ea551ce8129a1189afeafc45b3ae50e8dfceb575f87c712166fab"
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
        "implementation_head": "c8f09fe5f5c80665e6c3e9ddf45c6ed0c3498473",
        "protocol_head": "c8f09fe5f5c80665e6c3e9ddf45c6ed0c3498473",
    }
    old_head = previous["identity"]["implementation_head"]
    new_head = current["identity"]["implementation_head"]
    assert current["thesis"]["cheapest_falsifier"] == previous["thesis"][
        "cheapest_falsifier"
    ].replace(old_head, new_head)
    old = set(bound_code_paths(old_head))
    new = set(bound_code_paths(new_head))
    assert new == old
    changed = _git("diff", "--name-only", old_head, new_head, "--", *sorted(old))
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.splitlines() == ["trade_rl/data/features/core.py"]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized
    for doc in (STATUS, ARCHITECTURE):
        assert previous_path.name in _flat(doc)
        assert (
            "d18bb91dee5ea551ce8129a1189afeafc45b3ae50e8dfceb575f87c712166fab"
            in _flat(doc)
        )


def test_raw_mask_preserves_rsi_scratch_packet() -> None:
    previous_path = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-rsi-scratch.json"
    )
    raw = previous_path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "77bdbb53137cdb06b9d50c59355e5ce4521e6bb63fc36d11f974b9c1cefafece"
    )
    previous = json.loads(raw)
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-raw-mask.json"
        ).read_bytes()
    )
    assert (
        assurance_digest(previous)
        == "8629596c1c1c95104023fea98b04004f2a7e320795df2702ff208f62136e4526"
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
        "implementation_head": "28dae2a341f04e30abfc954890e32e47ba2501bc",
        "protocol_head": "28dae2a341f04e30abfc954890e32e47ba2501bc",
    }
    old_head = previous["identity"]["implementation_head"]
    new_head = current["identity"]["implementation_head"]
    assert current["thesis"]["cheapest_falsifier"] == previous["thesis"][
        "cheapest_falsifier"
    ].replace(old_head, new_head)
    old = set(bound_code_paths(old_head))
    new = set(bound_code_paths(new_head))
    assert new == old
    changed = _git("diff", "--name-only", old_head, new_head, "--", *sorted(old))
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.splitlines() == [
        "trade_rl/data/native_clock.py",
        "trade_rl/data/source.py",
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized
    for doc in (STATUS, ARCHITECTURE):
        assert previous_path.name in _flat(doc)
        assert (
            "8629596c1c1c95104023fea98b04004f2a7e320795df2702ff208f62136e4526"
            in _flat(doc)
        )


def test_strict_spot_cash_preserves_rsi_scratch_packet() -> None:
    previous_path = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-rsi-scratch.json"
    )
    raw = previous_path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "77bdbb53137cdb06b9d50c59355e5ce4521e6bb63fc36d11f974b9c1cefafece"
    )
    previous = json.loads(raw)
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-strict-spot-cash.json"
        ).read_bytes()
    )
    assert (
        assurance_digest(previous)
        == "8629596c1c1c95104023fea98b04004f2a7e320795df2702ff208f62136e4526"
    )
    assert (
        assurance_digest(current)
        == "f7939f9b3fdc195f9d560eef029c4e3ef249545ba7dcca830ea573432f4c3a77"
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
        "implementation_head": "811eeb6284bad90219773691d3e283bacb5640cf",
        "protocol_head": "811eeb6284bad90219773691d3e283bacb5640cf",
    }
    old_head = previous["identity"]["implementation_head"]
    new_head = current["identity"]["implementation_head"]
    assert current["thesis"]["cheapest_falsifier"] == previous["thesis"][
        "cheapest_falsifier"
    ].replace(old_head, new_head)
    old = set(bound_code_paths(old_head))
    new = set(bound_code_paths(new_head))
    assert old == new
    changed = _git("diff", "--name-only", old_head, new_head, "--", *sorted(old))
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.splitlines() == [
        "trade_rl/evaluation/replay.py",
        "trade_rl/simulation/accounting.py",
        "trade_rl/simulation/execution.py",
        "trade_rl/simulation/stateful/execution.py",
        "trade_rl/simulation/stateful/portfolio_session.py",
        "trade_rl/simulation/stateful/symbol_fills.py",
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized
    for doc in (STATUS, ARCHITECTURE):
        assert previous_path.name in _flat(doc)
        assert (
            "8629596c1c1c95104023fea98b04004f2a7e320795df2702ff208f62136e4526"
            in _flat(doc)
        )


def test_native_cash_union_retains_both_predecessors_and_exact_source() -> None:
    a_path = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-raw-mask.json"
    )
    b_path = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-strict-spot-cash.json"
    )
    previous = json.loads(a_path.read_bytes())
    parallel = json.loads(b_path.read_bytes())
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-native-spot-cash.json"
        ).read_bytes()
    )
    assert (
        hashlib.sha256(a_path.read_bytes()).hexdigest()
        == "562697d44e15bc95949dba4a24984ea10290c4dd22e27394da6a71feb6ac029c"
    )
    assert (
        hashlib.sha256(b_path.read_bytes()).hexdigest()
        == "937459231a50c32d90a3dfb84527f66cfd4f2985c7e901381b8645576d26c37d"
    )
    assert (
        assurance_digest(previous)
        == "bd13a6a48701c7473a32aeaed1a6b5eaafb7620c90ae707196e5896a0fbfa11a"
    )
    assert (
        assurance_digest(parallel)
        == "f7939f9b3fdc195f9d560eef029c4e3ef249545ba7dcca830ea573432f4c3a77"
    )
    assert current["identity"] == {
        "implementation_head": "31180c14867aaab2caeb56976090cee99c882591",
        "protocol_head": "31180c14867aaab2caeb56976090cee99c882591",
    }
    assert current.keys() == previous.keys()
    for key in previous:
        if key not in ("identity", "thesis", "claims", "evidence"):
            assert current[key] == previous[key] == parallel[key]
    for key in previous["thesis"]:
        if key not in ("limitations", "cheapest_falsifier"):
            assert (
                current["thesis"][key]
                == previous["thesis"][key]
                == parallel["thesis"][key]
            )
    for key in previous["evidence"]:
        if key != "independent_reconstruction":
            assert (
                current["evidence"][key]
                == previous["evidence"][key]
                == parallel["evidence"][key]
            )
    for key in previous["claims"]:
        if key != "next_authorized_action":
            assert (
                current["claims"][key]
                == previous["claims"][key]
                == parallel["claims"][key]
            )
    assert current["thesis"]["limitations"].startswith(
        previous["thesis"]["limitations"]
    )
    a = set(bound_code_paths(previous["identity"]["implementation_head"]))
    b = set(bound_code_paths(current["identity"]["implementation_head"]))
    assert a == b and len(b) == 176
    diff = _git(
        "diff",
        "--name-only",
        previous["identity"]["implementation_head"],
        current["identity"]["implementation_head"],
        "--",
        *sorted(a),
    )
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.splitlines() == [
        "trade_rl/evaluation/replay.py",
        "trade_rl/simulation/accounting.py",
        "trade_rl/simulation/execution.py",
        "trade_rl/simulation/stateful/execution.py",
        "trade_rl/simulation/stateful/portfolio_session.py",
        "trade_rl/simulation/stateful/symbol_fills.py",
    ]
    for record in (previous, parallel, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized
    assert current["thesis"]["cheapest_falsifier"] == previous["thesis"][
        "cheapest_falsifier"
    ].replace(
        "28dae2a341f04e30abfc954890e32e47ba2501bc",
        "31180c14867aaab2caeb56976090cee99c882591",
    )
    assert current["evidence"]["independent_reconstruction"].startswith(
        previous["evidence"]["independent_reconstruction"]
    )
    for doc in (STATUS, ARCHITECTURE):
        assert (
            a_path.name in _flat(doc)
            and "bd13a6a48701c7473a32aeaed1a6b5eaafb7620c90ae707196e5896a0fbfa11a"
            in _flat(doc)
        )
        assert (
            b_path.name in _flat(doc)
            and "f7939f9b3fdc195f9d560eef029c4e3ef249545ba7dcca830ea573432f4c3a77"
            in _flat(doc)
        )


def test_native_risk_dataset_successor_preserves_immutable_cash_union() -> None:
    previous_path = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-native-spot-cash.json"
    )
    previous_bytes = previous_path.read_bytes()
    assert (
        hashlib.sha256(previous_bytes).hexdigest()
        == "a68478097f4e2c37771568c1b78a2db9dd4a2e6ff697dc058acbd7afe7d8e871"
    )
    previous = json.loads(previous_bytes)
    assert (
        assurance_digest(previous)
        == "114adc053cd041c26ac1c5ce67df3a47b605b84a331fe8c26122be5c3a353272"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-native-risk-dataset.json"
        ).read_bytes()
    )
    assert current["identity"] == {
        "implementation_head": "814580ea4857390b607e5729e0cb9f96cf4ce735",
        "protocol_head": "814580ea4857390b607e5729e0cb9f96cf4ce735",
    }
    assert current.keys() == previous.keys()
    for key in previous:
        if key not in ("identity", "thesis", "claims", "evidence"):
            assert current[key] == previous[key]
    for key in previous["thesis"]:
        if key == "limitations":
            assert current["thesis"][key].startswith(previous["thesis"][key])
        elif key == "cheapest_falsifier":
            assert current["thesis"][key] == previous["thesis"][key].replace(
                "31180c14867aaab2caeb56976090cee99c882591",
                "814580ea4857390b607e5729e0cb9f96cf4ce735",
            )
        else:
            assert current["thesis"][key] == previous["thesis"][key]
    for key in previous["claims"]:
        assert current["claims"][key] == (
            previous["claims"][key].replace(
                "31180c14867aaab2caeb56976090cee99c882591",
                "814580ea4857390b607e5729e0cb9f96cf4ce735",
            )
            if key == "next_authorized_action"
            else previous["claims"][key]
        )
    for key in previous["evidence"]:
        if key == "independent_reconstruction":
            assert current["evidence"][key].startswith(previous["evidence"][key])
        else:
            assert current["evidence"][key] == previous["evidence"][key]
    a = set(bound_code_paths("31180c14867aaab2caeb56976090cee99c882591"))
    b = set(bound_code_paths("814580ea4857390b607e5729e0cb9f96cf4ce735"))
    assert a == b and len(b) == 176
    diff = _git(
        "diff",
        "--name-only",
        "31180c14867aaab2caeb56976090cee99c882591",
        "814580ea4857390b607e5729e0cb9f96cf4ce735",
        "--",
        *sorted(a),
    )
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.splitlines() == [
        "trade_rl/data/market.py",
        "trade_rl/risk/inputs.py",
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_execution_identity_successor_preserves_native_risk_dataset() -> None:
    previous_path = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-native-risk-dataset.json"
    )
    previous_bytes = previous_path.read_bytes()
    assert (
        hashlib.sha256(previous_bytes).hexdigest()
        == "ca761e39afc7fa2c19d6dc918cec728ab38fffb99adc7095e1430024aaf96ac6"
    )
    previous = json.loads(previous_bytes)
    assert (
        assurance_digest(previous)
        == "7c08e19aa3f544e0470427c25f4f652f05bf8841d857e5ff3b3b77d38c350251"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-native-execution-identity.json"
        ).read_bytes()
    )
    assert current["identity"] == {
        "implementation_head": "5139465355fce3d54fb509a246122a1161c96306",
        "protocol_head": "5139465355fce3d54fb509a246122a1161c96306",
    }
    assert current.keys() == previous.keys()
    for key in previous:
        if key not in ("identity", "thesis", "claims", "evidence"):
            assert current[key] == previous[key]
    for key in previous["thesis"]:
        if key == "limitations":
            assert current["thesis"][key].startswith(previous["thesis"][key])
        elif key == "cheapest_falsifier":
            assert current["thesis"][key] == previous["thesis"][key].replace(
                "814580ea4857390b607e5729e0cb9f96cf4ce735",
                "5139465355fce3d54fb509a246122a1161c96306",
            )
        else:
            assert current["thesis"][key] == previous["thesis"][key]
    for key in previous["claims"]:
        assert current["claims"][key] == (
            previous["claims"][key].replace(
                "814580ea4857390b607e5729e0cb9f96cf4ce735",
                "5139465355fce3d54fb509a246122a1161c96306",
            )
            if key == "next_authorized_action"
            else previous["claims"][key]
        )
    for key in previous["evidence"]:
        if key == "independent_reconstruction":
            assert current["evidence"][key].startswith(previous["evidence"][key])
        else:
            assert current["evidence"][key] == previous["evidence"][key]
    a = set(bound_code_paths("814580ea4857390b607e5729e0cb9f96cf4ce735"))
    b = set(bound_code_paths("5139465355fce3d54fb509a246122a1161c96306"))
    assert a == b and len(b) == 176
    diff = _git(
        "diff",
        "--name-only",
        "814580ea4857390b607e5729e0cb9f96cf4ce735",
        "5139465355fce3d54fb509a246122a1161c96306",
        "--",
        *sorted(a),
    )
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.splitlines() == [
        "trade_rl/simulation/__init__.py",
        "trade_rl/simulation/execution.py",
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_native_admission_union_preserves_execution_identity_packet() -> None:
    previous_bytes = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-native-execution-identity.json"
    ).read_bytes()
    assert (
        hashlib.sha256(previous_bytes).hexdigest()
        == "104475f189dfa6044bfea017df96c0aaeb99e2b72cc7a15b979a04c979ecc8f7"
    )
    previous = json.loads(previous_bytes)
    assert (
        assurance_digest(previous)
        == "1c411e5a122deb30a961be6a77e0a307da0fef53485dccbb602d3c0c834aaf23"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-native-admission-union.json"
        ).read_bytes()
    )
    assert current["identity"] == {
        "implementation_head": "018b5a5d3a4f6d08212e1efc6f78d1ee17f24212",
        "protocol_head": "018b5a5d3a4f6d08212e1efc6f78d1ee17f24212",
    }
    assert current.keys() == previous.keys()
    for key in previous:
        if key not in ("identity", "thesis", "claims", "evidence"):
            assert current[key] == previous[key]
    assert current["thesis"].keys() == previous["thesis"].keys()
    for key in previous["thesis"]:
        if key == "limitations":
            assert current["thesis"][key].startswith(previous["thesis"][key])
        elif key == "cheapest_falsifier":
            assert current["thesis"][key] == previous["thesis"][key].replace(
                "5139465355fce3d54fb509a246122a1161c96306",
                "018b5a5d3a4f6d08212e1efc6f78d1ee17f24212",
            )
        else:
            assert current["thesis"][key] == previous["thesis"][key]
    assert current["claims"].keys() == previous["claims"].keys()
    for key in previous["claims"]:
        assert current["claims"][key] == (
            previous["claims"][key].replace(
                "5139465355fce3d54fb509a246122a1161c96306",
                "018b5a5d3a4f6d08212e1efc6f78d1ee17f24212",
            )
            if key == "next_authorized_action"
            else previous["claims"][key]
        )
    assert current["evidence"].keys() == previous["evidence"].keys()
    for key in previous["evidence"]:
        if key == "independent_reconstruction":
            assert current["evidence"][key].startswith(previous["evidence"][key])
        else:
            assert current["evidence"][key] == previous["evidence"][key]
    a = set(bound_code_paths("5139465355fce3d54fb509a246122a1161c96306"))
    b = set(bound_code_paths("018b5a5d3a4f6d08212e1efc6f78d1ee17f24212"))
    assert a == b and len(b) == 176
    diff = _git(
        "diff",
        "--name-only",
        "5139465355fce3d54fb509a246122a1161c96306",
        "018b5a5d3a4f6d08212e1efc6f78d1ee17f24212",
        "--",
        *sorted(a),
    )
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.splitlines() == [
        "trade_rl/integrations/binance/dataset.py",
        "trade_rl/integrations/binance/derived_klines.py",
        "trade_rl/integrations/binance/source_clock.py",
        "trade_rl/integrations/binance/transport.py",
        "trade_rl/integrations/binance/vision.py",
        "trade_rl/risk/inputs.py",
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_native_exposure_successor_preserves_admission_union_packet() -> None:
    previous_bytes = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-native-admission-union.json"
    ).read_bytes()
    assert (
        hashlib.sha256(previous_bytes).hexdigest()
        == "8e9c72fa7921a4753544b8f449bd0fcf34801a1191518b9e9a1db7b299c14725"
    )
    previous = json.loads(previous_bytes)
    assert (
        assurance_digest(previous)
        == "13669a5653f92528f479a057b5bebe2d53c53df0fd6a0d2c5df16aa23a3c2034"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-native-exposure.json"
        ).read_bytes()
    )
    assert current["identity"] == {
        "implementation_head": "0b982c8caf6c4fca2880b4b4cf4b39bece756f44",
        "protocol_head": "0b982c8caf6c4fca2880b4b4cf4b39bece756f44",
    }
    assert current.keys() == previous.keys()
    for key in previous:
        if key not in ("identity", "thesis", "claims", "evidence"):
            assert current[key] == previous[key]
    assert current["thesis"].keys() == previous["thesis"].keys()
    for key in previous["thesis"]:
        if key == "limitations":
            assert current["thesis"][key].startswith(previous["thesis"][key])
        elif key == "cheapest_falsifier":
            assert current["thesis"][key] == previous["thesis"][key].replace(
                "018b5a5d3a4f6d08212e1efc6f78d1ee17f24212",
                "0b982c8caf6c4fca2880b4b4cf4b39bece756f44",
            )
        else:
            assert current["thesis"][key] == previous["thesis"][key]
    assert current["claims"].keys() == previous["claims"].keys()
    for key in previous["claims"]:
        assert current["claims"][key] == (
            previous["claims"][key].replace(
                "018b5a5d3a4f6d08212e1efc6f78d1ee17f24212",
                "0b982c8caf6c4fca2880b4b4cf4b39bece756f44",
            )
            if key == "next_authorized_action"
            else previous["claims"][key]
        )
    assert current["evidence"].keys() == previous["evidence"].keys()
    for key in previous["evidence"]:
        if key == "independent_reconstruction":
            assert current["evidence"][key].startswith(previous["evidence"][key])
        else:
            assert current["evidence"][key] == previous["evidence"][key]
    a = set(bound_code_paths("018b5a5d3a4f6d08212e1efc6f78d1ee17f24212"))
    b = set(bound_code_paths("0b982c8caf6c4fca2880b4b4cf4b39bece756f44"))
    assert len(a) == 176 and len(b) == 177
    assert b == a | {"trade_rl/evaluation/replay_exposure.py"}
    diff = _git(
        "diff",
        "--name-only",
        "018b5a5d3a4f6d08212e1efc6f78d1ee17f24212",
        "0b982c8caf6c4fca2880b4b4cf4b39bece756f44",
        "--",
        *sorted(a | b),
    )
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.splitlines() == [
        "trade_rl/evaluation/replay.py",
        "trade_rl/evaluation/replay_exposure.py",
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_m2_td8_reader_successor_preserves_native_admission_packet() -> None:
    previous_bytes = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-native-admission-union.json"
    ).read_bytes()
    assert (
        hashlib.sha256(previous_bytes).hexdigest()
        == "8e9c72fa7921a4753544b8f449bd0fcf34801a1191518b9e9a1db7b299c14725"
    )
    previous = json.loads(previous_bytes)
    assert (
        assurance_digest(previous)
        == "13669a5653f92528f479a057b5bebe2d53c53df0fd6a0d2c5df16aa23a3c2034"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-m2-td8-readers.json"
        ).read_bytes()
    )
    assert current["identity"] == {
        "implementation_head": "05733c8f8fe94b7d08e1108d024c28f6fc86bab8",
        "protocol_head": "05733c8f8fe94b7d08e1108d024c28f6fc86bab8",
    }
    expected = json.loads(previous_bytes)
    expected["identity"] = current["identity"]
    for section, key in (
        ("thesis", "limitations"),
        ("evidence", "independent_reconstruction"),
    ):
        assert current[section][key].startswith(previous[section][key])
        expected[section][key] = current[section][key]
    for section, key in (
        ("thesis", "cheapest_falsifier"),
        ("claims", "next_authorized_action"),
    ):
        expected[section][key] = previous[section][key].replace(
            "018b5a5d3a4f6d08212e1efc6f78d1ee17f24212",
            "05733c8f8fe94b7d08e1108d024c28f6fc86bab8",
        )
    assert current == expected
    a = set(bound_code_paths("018b5a5d3a4f6d08212e1efc6f78d1ee17f24212"))
    b = set(bound_code_paths("05733c8f8fe94b7d08e1108d024c28f6fc86bab8"))
    assert a == b and len(b) == 176
    diff = _git(
        "diff",
        "--name-only",
        "018b5a5d3a4f6d08212e1efc6f78d1ee17f24212",
        "05733c8f8fe94b7d08e1108d024c28f6fc86bab8",
        "--",
        *sorted(a),
    )
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.splitlines() == [
        "trade_rl/evaluation/decision_value_dev_study.py",
        "trade_rl/evaluation/runs/artifact.py",
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_exposure_m2_td8_union_preserves_both_v37_lineages() -> None:
    common_bytes = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-native-admission-union.json"
    ).read_bytes()
    exposure_bytes = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-native-exposure.json"
    ).read_bytes()
    reader_bytes = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-m2-td8-readers.json"
    ).read_bytes()
    assert (
        hashlib.sha256(exposure_bytes).hexdigest()
        == "ef45f6ab87aa8d9c2ee9de68b0082a2a780131ddc19faa73d0ba42a42d785c7e"
    )
    assert (
        hashlib.sha256(reader_bytes).hexdigest()
        == "b3eca6d21b534496878fdf3defe00c65ae9ada7747c962aad527111266686c1d"
    )
    origin = json.loads(common_bytes)
    exposure = json.loads(exposure_bytes)
    reader = json.loads(reader_bytes)
    assert (
        assurance_digest(exposure)
        == "1f86f608ca57bd062e41bb06b9e424832104f873d9a242059e40c1b179e46c01"
    )
    assert (
        assurance_digest(reader)
        == "7d8fec57badb5c75567c918e7aefc0903311329f7dda5757474f50afebe4d06a"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-exposure-m2-td8-union.json"
        ).read_bytes()
    )
    expected = json.loads(exposure_bytes)
    assert current["identity"] == {
        "implementation_head": "0f8946eafd0b305c8bd72fab478e2ac516667b3b",
        "protocol_head": "0f8946eafd0b305c8bd72fab478e2ac516667b3b",
    }
    expected["identity"] = current["identity"]
    for section, key in (
        ("thesis", "limitations"),
        ("evidence", "independent_reconstruction"),
    ):
        prefix = origin[section][key]
        assert exposure[section][key].startswith(prefix)
        assert reader[section][key].startswith(prefix)
        combined = exposure[section][key] + reader[section][key][len(prefix) :]
        assert current[section][key].startswith(combined)
        expected[section][key] = current[section][key]
    for section, key in (
        ("thesis", "cheapest_falsifier"),
        ("claims", "next_authorized_action"),
    ):
        replacement = origin[section][key].replace(
            "018b5a5d3a4f6d08212e1efc6f78d1ee17f24212",
            "0f8946eafd0b305c8bd72fab478e2ac516667b3b",
        )
        assert (
            exposure[section][key].replace(
                "0b982c8caf6c4fca2880b4b4cf4b39bece756f44",
                "0f8946eafd0b305c8bd72fab478e2ac516667b3b",
            )
            == replacement
        )
        assert (
            reader[section][key].replace(
                "05733c8f8fe94b7d08e1108d024c28f6fc86bab8",
                "0f8946eafd0b305c8bd72fab478e2ac516667b3b",
            )
            == replacement
        )
        expected[section][key] = replacement
    assert current == expected
    exposure_paths = set(bound_code_paths("0b982c8caf6c4fca2880b4b4cf4b39bece756f44"))
    reader_paths = set(bound_code_paths("05733c8f8fe94b7d08e1108d024c28f6fc86bab8"))
    union_paths = set(bound_code_paths("0f8946eafd0b305c8bd72fab478e2ac516667b3b"))
    assert len(exposure_paths) == 177
    assert len(reader_paths) == 176
    assert (
        union_paths
        == exposure_paths
        == reader_paths | {"trade_rl/evaluation/replay_exposure.py"}
    )
    for previous_head, changes in (
        (
            "0b982c8caf6c4fca2880b4b4cf4b39bece756f44",
            [
                "trade_rl/evaluation/decision_value_dev_study.py",
                "trade_rl/evaluation/runs/artifact.py",
            ],
        ),
        (
            "05733c8f8fe94b7d08e1108d024c28f6fc86bab8",
            ["trade_rl/evaluation/replay.py", "trade_rl/evaluation/replay_exposure.py"],
        ),
    ):
        diff = _git(
            "diff",
            "--name-only",
            previous_head,
            "0f8946eafd0b305c8bd72fab478e2ac516667b3b",
            "--",
            *sorted(union_paths),
        )
        assert diff.returncode == 0, diff.stderr
        assert diff.stdout.splitlines() == changes
    for record in (origin, exposure, reader, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_td8_partitioned_decoder_successor_preserves_union_packet() -> None:
    previous_bytes = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-exposure-m2-td8-union.json"
    ).read_bytes()
    assert (
        hashlib.sha256(previous_bytes).hexdigest()
        == "243ebf9bc9558da7c2e1480550fb8afc84c16bce2dfb786cdb6c96eceedfc1d7"
    )
    previous = json.loads(previous_bytes)
    assert (
        assurance_digest(previous)
        == "d5234f153e355d063657c44979cbedcfde9cb7e6619bbe26b764fa2d096bc4ab"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-td8-partitioned-decode.json"
        ).read_bytes()
    )
    assert current["identity"] == {
        "implementation_head": "81a47fc7b3df3c5b6f3833fc0c31a2f5a7892e36",
        "protocol_head": "81a47fc7b3df3c5b6f3833fc0c31a2f5a7892e36",
    }
    expected = json.loads(previous_bytes)
    expected["identity"] = current["identity"]
    for section, key in (
        ("thesis", "limitations"),
        ("evidence", "independent_reconstruction"),
    ):
        assert current[section][key].startswith(previous[section][key])
        expected[section][key] = current[section][key]
    for section, key in (
        ("thesis", "cheapest_falsifier"),
        ("claims", "next_authorized_action"),
    ):
        expected[section][key] = previous[section][key].replace(
            "0f8946eafd0b305c8bd72fab478e2ac516667b3b",
            "81a47fc7b3df3c5b6f3833fc0c31a2f5a7892e36",
        )
    assert current == expected
    a = set(bound_code_paths("0f8946eafd0b305c8bd72fab478e2ac516667b3b"))
    b = set(bound_code_paths("81a47fc7b3df3c5b6f3833fc0c31a2f5a7892e36"))
    assert a == b and len(b) == 177
    diff = _git(
        "diff",
        "--name-only",
        "0f8946eafd0b305c8bd72fab478e2ac516667b3b",
        "81a47fc7b3df3c5b6f3833fc0c31a2f5a7892e36",
        "--",
        *sorted(a),
    )
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.splitlines() == [
        "trade_rl/evaluation/decision_value_dev_study.py"
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_native_fill_cash_successor_preserves_td8_packet() -> None:
    previous_bytes = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-td8-partitioned-decode.json"
    ).read_bytes()
    assert (
        hashlib.sha256(previous_bytes).hexdigest()
        == "a7f5cc757ce8a8eb8879002caa35bd5a44dc4c17f3c2c68953b19386fcdda346"
    )
    previous = json.loads(previous_bytes)
    assert (
        assurance_digest(previous)
        == "138062e2a9004f6d343cad737a313bb9af508c9d0f24cf662417813bebcf6649"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-native-fill-cash.json"
        ).read_bytes()
    )
    assert current["identity"] == {
        "implementation_head": "787140c94c75a96cda3a42e9b8309bc485783d0e",
        "protocol_head": "787140c94c75a96cda3a42e9b8309bc485783d0e",
    }
    expected = json.loads(previous_bytes)
    expected["identity"] = current["identity"]
    expected["thesis"]["limitations"] += (
        " Source-only native fill-cash observer successor (#1306): separate opt-in order-fill observations capture actual cash, cost and exact quantity immediately before/after BookState.execute_fill, before margin handling; default execution and legacy ledger bytes unchanged. Other cash flows, venue authentication, and full independent cash reconciliation remain outside this boundary. No new economic or live authorization."
    )
    expected["evidence"]["independent_reconstruction"] += (
        " Source-only successor 787140c94c75a96cda3a42e9b8309bc485783d0e: 1114 related synthetic simulation, shared ledger and dependency tests passed on the identical source tree, including 13 new observer tests. Literal two-sale case: cash 0 to 98 to 9898, actual costs 2 and 200, conservative first-fill interval-cost reservation incorrectly gives -102. Strict rejected fill has no cash record, partial and terminal orders are observed, funding and non-order margin flattening remain separate, default and observed legacy ledger bytes agree. Static closure grows from 177 to 178 paths; six existing replay/stateful execution paths change and simulation/diagnostics/fill_cash.py is added. No market data or economic execution."
    )
    for section, key in (
        ("thesis", "cheapest_falsifier"),
        ("claims", "next_authorized_action"),
    ):
        expected[section][key] = previous[section][key].replace(
            "81a47fc7b3df3c5b6f3833fc0c31a2f5a7892e36",
            "787140c94c75a96cda3a42e9b8309bc485783d0e",
        )
    assert current == expected
    a = set(bound_code_paths("81a47fc7b3df3c5b6f3833fc0c31a2f5a7892e36"))
    b = set(bound_code_paths("787140c94c75a96cda3a42e9b8309bc485783d0e"))
    assert len(a) == 177
    assert b == a | {"trade_rl/simulation/diagnostics/fill_cash.py"}
    diff = _git(
        "diff",
        "--name-only",
        "81a47fc7b3df3c5b6f3833fc0c31a2f5a7892e36",
        "787140c94c75a96cda3a42e9b8309bc485783d0e",
        "--",
        *sorted(a),
    )
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.splitlines() == [
        "trade_rl/evaluation/replay.py",
        "trade_rl/simulation/execution.py",
        "trade_rl/simulation/stateful/execution.py",
        "trade_rl/simulation/stateful/portfolio_session.py",
        "trade_rl/simulation/stateful/runtime.py",
        "trade_rl/simulation/stateful/symbol_fills.py",
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_td8_atomic_publication_successor_preserves_previous_packet() -> None:
    previous_bytes = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-td8-partitioned-decode.json"
    ).read_bytes()
    assert (
        hashlib.sha256(previous_bytes).hexdigest()
        == "a7f5cc757ce8a8eb8879002caa35bd5a44dc4c17f3c2c68953b19386fcdda346"
    )
    previous = json.loads(previous_bytes)
    assert (
        assurance_digest(previous)
        == "138062e2a9004f6d343cad737a313bb9af508c9d0f24cf662417813bebcf6649"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261126-td8-atomic-publication.json"
        ).read_bytes()
    )
    old_head = "81a47fc7b3df3c5b6f3833fc0c31a2f5a7892e36"
    new_head = "bd21022612d772ae9c3ff15353fb5e871f0d25da"
    assert current["identity"] == {
        "implementation_head": new_head,
        "protocol_head": new_head,
    }
    expected = json.loads(previous_bytes)
    expected["identity"] = current["identity"]
    for section, key in (
        ("thesis", "limitations"),
        ("evidence", "independent_reconstruction"),
    ):
        assert current[section][key].startswith(previous[section][key])
        expected[section][key] = current[section][key]
    for section, key in (
        ("thesis", "cheapest_falsifier"),
        ("claims", "next_authorized_action"),
    ):
        expected[section][key] = previous[section][key].replace(old_head, new_head)
    assert current == expected
    a = set(bound_code_paths(old_head))
    b = set(bound_code_paths(new_head))
    assert a == b and len(b) == 177
    diff = _git("diff", "--name-only", old_head, new_head, "--", *sorted(b))
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.splitlines() == [
        "trade_rl/evaluation/decision_value_dev_study.py"
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_fill_td8_publication_union_preserves_both_predecessors() -> None:
    common = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-td8-partitioned-decode.json"
        ).read_bytes()
    )
    a_raw = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261125-native-fill-cash.json"
    ).read_bytes()
    b_raw = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261126-td8-atomic-publication.json"
    ).read_bytes()
    assert (
        hashlib.sha256(a_raw).hexdigest()
        == "89b5287c723434912fb6a69b6277589a684befe53dac5a09fb4d8edf7cf57ed7"
    )
    assert (
        hashlib.sha256(b_raw).hexdigest()
        == "e20e13547fb9f1a4108a44e7e9836de6a9a00f34455f4f67f954d32698a414a9"
    )
    a, b = json.loads(a_raw), json.loads(b_raw)
    assert (
        assurance_digest(a)
        == "537656baa28a648f6cfb0a4de83383e54b2bb8b77be7caf548b13962eaf58048"
    )
    assert (
        assurance_digest(b)
        == "6288c55738ecbd7c4dfa785e162173f80451da76d9cccee2f1cbb2faa5c14826"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261126-fill-td8-publication-union.json"
        ).read_bytes()
    )
    head = "f4b29a5b02f9943ee937896f61d3a753acc7db55"
    assert current["identity"] == {"implementation_head": head, "protocol_head": head}
    expected = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261125-td8-partitioned-decode.json"
        ).read_bytes()
    )
    expected["identity"] = current["identity"]
    for section, key, extra in (
        (
            "thesis",
            "limitations",
            " Source-only union of both immutable predecessors: retain opt-in native fill cash observations and atomic TD8 failure-receipt directory publication; no change to frozen economic protocol, historical evidence or authorization. No general cash reconciliation, source authenticity, power-loss durability or aggregate resource guarantee.",
        ),
        (
            "evidence",
            "independent_reconstruction",
            " Source-only union at f4b29a5b02f9943ee937896f61d3a753acc7db55. Static closure verified: 178 paths, exactly the union of the 178-path fill predecessor and 177-path TD8 predecessor; one shared TD8 publication module changes relative to the fill predecessor and seven replay/native-fill modules relative to the TD8 predecessor. No economic run, market data access or research approval.",
        ),
    ):
        prefix = common[section][key]
        assert a[section][key].startswith(prefix) and b[section][key].startswith(prefix)
        expected[section][key] = (
            a[section][key] + b[section][key][len(prefix) :] + extra
        )
    for section, key in (
        ("thesis", "cheapest_falsifier"),
        ("claims", "next_authorized_action"),
    ):
        wanted = common[section][key].replace(
            common["identity"]["implementation_head"], head
        )
        for predecessor in (a, b):
            assert (
                predecessor[section][key].replace(
                    predecessor["identity"]["implementation_head"], head
                )
                == wanted
            )
        expected[section][key] = wanted
    assert current == expected
    a_head, b_head = (
        a["identity"]["implementation_head"],
        b["identity"]["implementation_head"],
    )
    a_paths, b_paths, union = (set(bound_code_paths(x)) for x in (a_head, b_head, head))
    assert len(a_paths) == 178 and len(b_paths) == 177
    assert len(union) == 178 and union == a_paths | b_paths
    for previous, paths in (
        (a_head, ["trade_rl/evaluation/decision_value_dev_study.py"]),
        (
            b_head,
            [
                "trade_rl/evaluation/replay.py",
                "trade_rl/simulation/diagnostics/fill_cash.py",
                "trade_rl/simulation/execution.py",
                "trade_rl/simulation/stateful/execution.py",
                "trade_rl/simulation/stateful/portfolio_session.py",
                "trade_rl/simulation/stateful/runtime.py",
                "trade_rl/simulation/stateful/symbol_fills.py",
            ],
        ),
    ):
        diff = _git("diff", "--name-only", previous, head, "--", *sorted(union))
        assert diff.returncode == 0, diff.stderr
        assert diff.stdout.splitlines() == paths
    for record in (common, a, b, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_td8_predecode_pin_successor_preserves_previous_packet() -> None:
    previous_bytes = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261126-td8-atomic-publication.json"
    ).read_bytes()
    assert (
        hashlib.sha256(previous_bytes).hexdigest()
        == "e20e13547fb9f1a4108a44e7e9836de6a9a00f34455f4f67f954d32698a414a9"
    )
    previous = json.loads(previous_bytes)
    assert (
        assurance_digest(previous)
        == "6288c55738ecbd7c4dfa785e162173f80451da76d9cccee2f1cbb2faa5c14826"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261126-td8-predecode-pin.json"
        ).read_bytes()
    )
    old_head = "bd21022612d772ae9c3ff15353fb5e871f0d25da"
    new_head = "296d0adaeeb4c078e0888ad68b28107cbbb35af1"
    assert current["identity"] == {
        "implementation_head": new_head,
        "protocol_head": new_head,
    }
    expected = json.loads(previous_bytes)
    expected["identity"] = current["identity"]
    for section, key in (
        ("thesis", "limitations"),
        ("evidence", "independent_reconstruction"),
    ):
        assert current[section][key].startswith(previous[section][key])
        expected[section][key] = current[section][key]
    for section, key in (
        ("thesis", "cheapest_falsifier"),
        ("claims", "next_authorized_action"),
    ):
        expected[section][key] = previous[section][key].replace(old_head, new_head)
    assert current == expected
    a = set(bound_code_paths(old_head))
    b = set(bound_code_paths(new_head))
    assert a == b and len(b) == 177
    diff = _git("diff", "--name-only", old_head, new_head, "--", *sorted(b))
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.splitlines() == [
        "trade_rl/evaluation/decision_value_dev_study.py"
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_td8_node_envelope_successor_preserves_previous_packet() -> None:
    previous_bytes = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261126-td8-predecode-pin.json"
    ).read_bytes()
    assert (
        hashlib.sha256(previous_bytes).hexdigest()
        == "87846fce9d34018976736cd13fe4994b8f9b11d2d7fb81761df8c48967b3a54d"
    )
    previous = json.loads(previous_bytes)
    assert (
        assurance_digest(previous)
        == "42a990f0b5feeb0824c8795e4beec47353ed8ecab8e2d66a0544a02f8ae237f1"
    )
    current = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261126-td8-node-envelope.json"
        ).read_bytes()
    )
    old_head = "296d0adaeeb4c078e0888ad68b28107cbbb35af1"
    new_head = "96db36f48f12ed7e4d91c24f7dc3633406d99801"
    assert current["identity"] == {
        "implementation_head": new_head,
        "protocol_head": new_head,
    }
    expected = json.loads(previous_bytes)
    expected["identity"] = current["identity"]
    for section, key in (
        ("thesis", "limitations"),
        ("evidence", "independent_reconstruction"),
    ):
        assert current[section][key].startswith(previous[section][key])
        expected[section][key] = current[section][key]
    for section, key in (
        ("thesis", "cheapest_falsifier"),
        ("claims", "next_authorized_action"),
    ):
        expected[section][key] = previous[section][key].replace(old_head, new_head)
    assert current == expected
    a = set(bound_code_paths(old_head))
    b = set(bound_code_paths(new_head))
    assert a == b and len(b) == 177
    diff = _git("diff", "--name-only", old_head, new_head, "--", *sorted(b))
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.splitlines() == [
        "trade_rl/evaluation/decision_value_dev_study.py"
    ]
    for record in (previous, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized


def test_fill_td8_node_union_preserves_both_predecessors() -> None:
    common = json.loads(
        (
            ROOT
            / "report"
            / "sd2-signal-diagnosis-research-assurance-20261126-td8-atomic-publication.json"
        ).read_bytes()
    )
    a_raw = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261126-fill-td8-publication-union.json"
    ).read_bytes()
    b_raw = (
        ROOT
        / "report"
        / "sd2-signal-diagnosis-research-assurance-20261126-td8-node-envelope.json"
    ).read_bytes()
    assert (
        hashlib.sha256(a_raw).hexdigest()
        == "caeeae6591d814601f99704ba5c65132d85eb4dfaef9bf4a46b472421863f385"
    )
    assert (
        hashlib.sha256(b_raw).hexdigest()
        == "ddd3d24be5e028af9867009c3edafd1b1e5f799cb06dc1fcf316d5927e160eb7"
    )
    a, b = json.loads(a_raw), json.loads(b_raw)
    assert (
        assurance_digest(a)
        == "17eab857200007515c35a729278e2fe8a7babd4966cba5e24f598fa0367953cb"
    )
    assert (
        assurance_digest(b)
        == "daa946b09c931270a68af5feb3507a5f5dee0525928ec6e03a8347bb68c491aa"
    )
    current = json.loads(PACKET_PATH.read_bytes())
    head = "25e7f8998d296a07526cb7ed07fd0d488c0bea49"
    assert current["identity"] == {"implementation_head": head, "protocol_head": head}
    expected = json.loads(a_raw)
    expected["identity"] = current["identity"]
    for section, key, extra in (
        (
            "thesis",
            "limitations",
            " Source-only integration of immutable fill/TD8 publication union and TD8 pre-decode pin/node-envelope successors. Optional reader node admission is not a producer or absolute resource guarantee. Strict spot cash verification and POSIX path quoting are outside this bound closure. No economic protocol, evidence or authority change.",
        ),
        (
            "evidence",
            "independent_reconstruction",
            " Source-only exact integration anchor 25e7f8998d296a07526cb7ed07fd0d488c0bea49; verified 178-path union of the prior 178-path combined closure and 177-path TD8 closure. Relative to the prior combined closure only decision_value_dev_study.py changes. Relative to TD8 only seven replay/native-fill paths differ. No market data or economic execution.",
        ),
    ):
        prefix = common[section][key]
        assert b[section][key].startswith(prefix)
        expected[section][key] = (
            a[section][key] + b[section][key][len(prefix) :] + extra
        )
    a_head, b_head = (
        a["identity"]["implementation_head"],
        b["identity"]["implementation_head"],
    )
    for section, key in (
        ("thesis", "cheapest_falsifier"),
        ("claims", "next_authorized_action"),
    ):
        wanted = a[section][key].replace(a_head, head)
        assert b[section][key].replace(b_head, head) == wanted
        expected[section][key] = wanted
    assert current == expected
    a_paths, b_paths, union = (set(bound_code_paths(x)) for x in (a_head, b_head, head))
    assert len(a_paths) == 178 and len(b_paths) == 177
    assert len(union) == 178 and union == a_paths | b_paths
    for previous, paths in (
        (a_head, ["trade_rl/evaluation/decision_value_dev_study.py"]),
        (
            b_head,
            [
                "trade_rl/evaluation/replay.py",
                "trade_rl/simulation/diagnostics/fill_cash.py",
                "trade_rl/simulation/execution.py",
                "trade_rl/simulation/stateful/execution.py",
                "trade_rl/simulation/stateful/portfolio_session.py",
                "trade_rl/simulation/stateful/runtime.py",
                "trade_rl/simulation/stateful/symbol_fills.py",
            ],
        ),
    ):
        diff = _git("diff", "--name-only", previous, head, "--", *sorted(union))
        assert diff.returncode == 0, diff.stderr
        assert diff.stdout.splitlines() == paths
    for record in (common, a, b, current):
        verdict = evaluate_assurance(record)
        assert verdict.status == "UNREVIEWED" and not verdict.errors
        assert not verdict.economic_execution_authorized
