"""The revision-2 trend-stop packet stays UNREVIEWED, result-blind and bound to the code.

Revision 1 (``report/trend-stop-research-assurance-20261110.json``) was blocked by
review 5870390989 of PR #1201. These tests keep revision 2's packet, protocol digest,
specification and documents in step, keep the packet from claiming a review, an
economic result or any authorization, and pin the code the packet covers: the static
import closure of the study entry points, resolved from the immutable packet head. They
read no market data.
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
from trade_rl.evaluation.trend_stop_study.protocol import (
    OWNER_ROSTER_DECISION_URL,
    PROTOCOL,
    SUPERSEDES_PROTOCOL_DIGEST,
    protocol_digest,
)

ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = ROOT / "report" / "trend-stop-research-assurance-20261113-after1203.json"
REVISION_1_PACKET_PATH = ROOT / "report" / "trend-stop-research-assurance-20261110.json"
PRIOR_V2_PACKET_PATH = ROOT / "report" / "trend-stop-research-assurance-20261113.json"
SPEC = ROOT / "docs" / "specs" / "trend-following-stops.md"
ARCHITECTURE = ROOT / "docs" / "architecture" / "research-assurance.md"
STATUS = ROOT / "docs" / "research" / "current-status.md"
PINNED_DIGEST = "8fa94eba74e70147b9a2123aae50907e878457ed7c5d3e4008b97524a0c34e64"
PRIOR_V2_PACKET_DIGEST = (
    "8178da9c823feec126ff65cb9cc53fe23fcffc085a544444b7c285b834fb077a"
)
REVISION_1_PACKET_DIGEST = (
    "894fedb0b88dd872039fdfbc546f872d485d1d6130ae5e096c79ceebf141bb64"
)
PINNED_PROTOCOL_DIGEST = (
    "2e5ceb6bdad76deaab503ebf5be714cff1b7b8611b0c44048bd562897f707fda"
)
DECISION_ID = "6818495567"
# The bound code is the static import closure of these entry points (module-level and
# lazy imports, plus the ``__init__`` of every ancestor package), resolved from the
# immutable packet head, never from the working tree.
ENTRY_POINTS = (
    "trade_rl/evaluation/trend_stop_study/",
    "trade_rl/strategies/rules/trend_stop.py",
)


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
    """Every file the study statically imports, transitively, at ``head``.

    A deliberate superset: package ``__init__`` re-exports pull in whole subpackages.
    Non-Python package data is not covered.
    """
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


def _require_pinned_head() -> str:
    head = packet()["identity"]["implementation_head"]
    if _git("cat-file", "-e", f"{head}^{{commit}}").returncode != 0:
        pytest.fail(
            "packet head unavailable: fetch full Git history before checking source closure"
        )
    return head


def packet() -> dict[str, Any]:
    return json.loads(PACKET_PATH.read_text(encoding="utf-8"))


def _flat(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_missing_trend_packet_head_fails_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing history is a failed source proof even in a shallow checkout."""

    def missing_history(*args: str) -> subprocess.CompletedProcess[str]:
        assert args == (
            "cat-file",
            "-e",
            f"{packet()['identity']['implementation_head']}^{{commit}}",
        )
        return subprocess.CompletedProcess(args, 1, "", "object unavailable")

    monkeypatch.setattr(
        "tests.evaluation.test_trend_stop_v2_packet._git", missing_history
    )
    with pytest.raises(pytest.fail.Exception, match="packet head unavailable"):
        _require_pinned_head()


def test_prior_v2_trend_packet_is_unchanged_and_unreviewed() -> None:
    """A new source binding must not rewrite already reviewed historical bytes."""
    old = json.loads(PRIOR_V2_PACKET_PATH.read_text(encoding="utf-8"))
    assert assurance_digest(old) == PRIOR_V2_PACKET_DIGEST
    assert old["identity"]["implementation_head"] == (
        "b433ea8dd78f79859a09bc862b93d2459b6c86a3"
    )
    verified = evaluate_assurance(old)
    assert verified.status == "UNREVIEWED"
    assert verified.economic_execution_authorized is False


def test_prior_after1216_source_reissue_is_immutable_and_unreviewed() -> None:
    """Keep the earlier source-bound author packet as immutable history."""
    path = ROOT / "report" / "trend-stop-research-assurance-20261113-after1216.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    assert assurance_digest(old) == (
        "94f9af8af37bbc022f8550c3a0ad1b642fb4b50fa51d75fb2682d93fc59714f7"
    )
    expected_head = "81f5f7f3d8f7d25b67cc03556986bedc9d23cbb9"
    assert old["identity"]["protocol_head"] == expected_head
    assert old["identity"]["implementation_head"] == expected_head
    verified = evaluate_assurance(old)
    assert verified.status == "UNREVIEWED"
    assert verified.economic_execution_authorized is False


def test_packet_is_valid_unreviewed_and_pinned() -> None:
    record = packet()
    result = evaluate_assurance(record)
    assert result.status == "UNREVIEWED" and result.errors == ()
    assert result.economic_execution_authorized is False
    assert assurance_digest(record) == PINNED_DIGEST == result.record_digest
    assert PINNED_DIGEST != REVISION_1_PACKET_DIGEST
    assert PINNED_DIGEST != PRIOR_V2_PACKET_DIGEST
    assert b"\r" not in PACKET_PATH.read_bytes()
    # The blocked revision stays untouched beside it.
    assert assurance_digest(json.loads(REVISION_1_PACKET_PATH.read_text("utf-8"))) == (
        REVISION_1_PACKET_DIGEST
    )


def test_the_protocol_digest_is_pinned_and_supersedes_revision_1() -> None:
    assert protocol_digest() == PINNED_PROTOCOL_DIGEST
    assert SUPERSEDES_PROTOCOL_DIGEST != PINNED_PROTOCOL_DIGEST
    assert PROTOCOL.roster_decision_url == OWNER_ROSTER_DECISION_URL
    assert OWNER_ROSTER_DECISION_URL.endswith(f"#issuecomment-{DECISION_ID}")
    assert len(PROTOCOL.symbols) == 11
    text = " ".join(PROTOCOL.deviations)
    assert "no exclusion path" in text and "every (timeframe, variant)" in text


def test_packet_claims_nothing_beyond_software_validity() -> None:
    record = packet()
    claims = record["claims"]
    assert claims["claim_level"] == record["evidence"]["evidence_level"]
    assert claims["claim_level"] == "software_validity"
    assert claims["production_eligible"] is False
    assert claims["live_trading_authorized"] is False
    assert claims["next_action_class"] == "request_separate_economic_authorization"
    assert record["thesis"]["stage"] == "development"
    forbidden = " ".join(claims["forbidden"])
    for needle in (
        "reduced roster",
        "dirty working tree",
        "principal independence",
        "re-tuning",
        "FS1",
        "live trading",
    ):
        assert needle in forbidden, needle
    evidence = record["evidence"]
    assert evidence["point_in_time"] is False
    assert evidence["realistic_costs"] is False
    assert evidence["fit_development_unused_separated"] is False
    assert (
        "future_perturbation_probe_per_variant_and_timeframe"
        in (evidence["robustness"])
    )


def test_packet_states_the_g0_to_g2_content_of_revision_2() -> None:
    record = packet()
    thesis = record["thesis"]
    assert "arbitraged" in thesis["counter_hypothesis"]
    assert "reduced or reselected roster" in thesis["counter_hypothesis"]
    assert "every (timeframe, variant) pair" in thesis["cheapest_falsifier"]
    assert "a dirty Git working tree" in thesis["cheapest_falsifier"]
    assert "NO_WINNER" in thesis["stop_rule"]
    assert "INVALID" in thesis["stop_rule"]
    assert f"#issuecomment-{DECISION_ID}" in thesis["stop_rule"]
    assert "no run registry exists" in thesis["stop_rule"]
    assert set(record["mechanism"]["independent_oracles"]) == {
        "time",
        "state",
        "sign",
        "risk",
        "execution",
        "accounting",
    }
    assert (
        "every (timeframe, variant) pair"
        in (record["mechanism"]["independent_oracles"]["time"]["oracle"])
    )
    assert "-62" in record["mechanism"]["independent_oracles"]["state"]["oracle"]
    assert "known-used" in record["claims"]["decision_rule"]


def test_packet_carries_the_prior_exposure_disclosures_and_the_review_route() -> None:
    text = json.dumps(packet())
    for needle in (
        "realdata_20260713_signal_metadata",
        "legacy GBM signal metadata",
        "Phase 2 trial T1",
        "M34",
        "not excluded",
        "haircut-adjusted",
        "894fedb0b88dd872039fdfbc546f872d485d1d6130ae5e096c79ceebf141bb64",
        "review 5870390989",
        "Claude Haiku 5.5",
        "not principal independence",
        f"issues/810#issuecomment-{DECISION_ID}",
        "No price path, return or PnL",
    ):
        assert needle in text, needle
    limitations = packet()["thesis"]["limitations"]
    assert "not result-blind about its universe and window" in limitations
    assert "UNREVIEWED for the tooling" in limitations
    assert "No economic result, real price or return was read" not in text
    # A 404 only ever makes the run INVALID; nothing in the packet allows an exclusion.
    assert "HTTP 404" in text and "fewer than 8" not in text
    stop_rule = packet()["thesis"]["stop_rule"]
    assert "halts, writes no result and does not consume the run" in stop_rule
    assert "can never produce a reduced roster" in limitations
    assert "5879836463" in limitations and "5879837718" in limitations
    assert "5880408681" in limitations and "5880409010" in limitations
    assert "unexpected cache member" in limitations


def test_documents_name_the_packet_and_the_protocol_digest() -> None:
    spec = _flat(SPEC)
    status = _flat(STATUS)
    architecture = _flat(ARCHITECTURE)
    assert "Status: Active" in SPEC.read_text(encoding="utf-8")
    for text in (spec, status):
        assert PINNED_PROTOCOL_DIGEST in text
        assert "report/trend-stop-research-assurance-20261113-after1203.json" in text
        assert "UNREVIEWED" in text
    assert PINNED_DIGEST in architecture
    assert "UNREVIEWED" in architecture and DECISION_ID in architecture
    assert "tests/evaluation/test_trend_stop_v2_packet.py" in spec
    for text in (spec, status):
        assert "fewer than 8" not in text.lower() and "recorded 404" not in text


def test_spec_states_the_registered_rules_of_revision_2() -> None:
    text = _flat(SPEC)
    for needle in (
        "no exclusion path, no minimum roster size and no CLI option for either",
        "one probe per (timeframe, variant) pair",
        "0.625% in each tail",
        'one-sided test of "mean above zero" at the 0.625% level',
        "the channel lengths N and M, the ATR stop multiples",
        "A fired stop keeps working until the position is flat",
        "The stop exit takes precedence over a reversal",
        "no entry is made in either direction",
        "HTTP 404 evidence",
        "binance_vision_missing_archive_v1",
        "missing without that evidence (not yet downloaded",
        "never ends in a traceback",
        "evidence pass over every archive of both timeframes and funding",
        "whatever the order of the timeframes",
        "does not ignore the `.missing.json` sidecar",
        "dirty Git working tree",
        "There is no registry of runs",
        "same-account route",
        "not principal independence",
    ):
        assert needle in text, needle
    assert " k," not in text.split("## 5.")[0]
    assert "docs/research/trend-stop-protocol.md" not in text
    for path in ("trade_rl/evaluation/trend_stop_study/protocol.py",):
        assert "docs/research/trend-stop-protocol.md" not in (ROOT / path).read_text(
            encoding="utf-8"
        )


def test_packet_heads_are_one_commit_and_the_bound_code_is_unchanged() -> None:
    identity = packet()["identity"]
    head = identity["protocol_head"]
    assert head == identity["implementation_head"]
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    _require_pinned_head()
    assert _git("merge-base", "--is-ancestor", head, "HEAD").returncode == 0
    bound = bound_code_paths(head)
    changed = _git("diff", "--name-only", head, "HEAD", "--", *bound)
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.split() == [], (
        "re-pin the packet heads after changing the bound code"
    )


def test_the_static_closure_covers_every_shared_dependency_of_the_study() -> None:
    head = _require_pinned_head()
    bound = set(bound_code_paths(head))
    for needed in (
        "trade_rl/__init__.py",
        "trade_rl/evaluation/trend_stop_study/runner.py",
        "trade_rl/evaluation/trend_stop_study/data.py",
        "trade_rl/evaluation/trend_stop_study/protocol.py",
        "trade_rl/evaluation/trend_stop_study/metrics.py",
        "trade_rl/evaluation/trend_stop_study/cli.py",
        "trade_rl/strategies/rules/trend_stop.py",
        # imported by trend_stop.py and the runner (the files review B3 named)
        "trade_rl/strategies/portfolio.py",
        "trade_rl/data/build/economics.py",
        "trade_rl/data/build/config.py",
        "trade_rl/data/build/__init__.py",
        "trade_rl/data/market.py",
        "trade_rl/evaluation/replay.py",
        "trade_rl/evaluation/dev_realdata_study/runner.py",
        "trade_rl/simulation/__init__.py",
        "trade_rl/risk/pretrade.py",
        "trade_rl/integrations/binance/dataset.py",
        "trade_rl/integrations/binance/cache.py",
        "trade_rl/integrations/binance/transport.py",
        "trade_rl/artifacts/canonical.py",
    ):
        assert needed in bound, needed
    # a facade pulls in its submodules (lazy and package imports are followed)
    assert any(path.startswith("trade_rl/simulation/") for path in bound)
    assert any(path.startswith("trade_rl/integrations/binance/") for path in bound)
    assert "trade_rl/evaluation/paper/engine.py" not in bound
    assert not any(path.startswith("tests/") for path in bound)


def test_the_closure_is_computed_not_listed_and_tracks_a_new_import() -> None:
    files = {"trade_rl/a.py", "trade_rl/b.py", "trade_rl/pkg/__init__.py"}
    assert _imported_modules("trade_rl/a.py", "from trade_rl import b\n") == {
        "trade_rl",
        "trade_rl.b",
    }
    lazy = "def f():\n    import trade_rl.pkg\n"
    assert "trade_rl.pkg" in _imported_modules("trade_rl/a.py", lazy)
    relative = "from .b import x\n"
    assert "trade_rl.b" in _imported_modules("trade_rl/a.py", relative)
    assert _module_file("trade_rl.pkg", files) == "trade_rl/pkg/__init__.py"
    assert _module_file("trade_rl.nothing", files) is None
    # an ancestor package ``__init__`` and a package facade's submodules are bound
    assert _imported_modules("trade_rl/pkg/__init__.py", "from . import c\n") == {
        "trade_rl.pkg",
        "trade_rl.pkg.c",
    }
