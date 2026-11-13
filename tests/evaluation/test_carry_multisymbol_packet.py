"""The multi-symbol carry packet stays UNREVIEWED, result-blind and bound to the source.

These tests keep the author packet, the specification, the current-status record and
the code in step, and keep the packet from claiming a review, an economic result or
any authorization. They read no market data.
"""

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

from tools.agent_repo.research_assurance import assurance_digest, evaluate_assurance
from trade_rl.evaluation.carry_multisymbol import protocol

ROOT = Path(__file__).resolve().parents[2]
_PACKET_STEM = "issue810-carry-multisymbol-research-assurance-20261113"
PACKET_PATH = ROOT / "report" / f"{_PACKET_STEM}-after1203.json"
PINNED_DIGEST = "f3a7c06ebd824b33498d5cd120337cf62491a1770e1bed73efb1b1b450535fea"
SPEC = ROOT / "docs/specs/funding-carry-multisymbol.md"
DOCUMENTS = (
    SPEC,
    ROOT / "docs/architecture/research-assurance.md",
    ROOT / "docs/research/current-status.md",
)
# The bound code is the static import closure of these entry points (module-level and
# lazy imports, plus the ``__init__`` of every ancestor package), resolved from the
# immutable packet head, never from the working tree.
ENTRY_POINTS = (
    "trade_rl/evaluation/carry_multisymbol/",
    "trade_rl/evaluation/carry.py",
    "tools/carry_multisymbol_screen.py",
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
    """Every file the screen statically imports, transitively, at ``head``.

    A deliberate superset: package ``__init__`` re-exports pull in whole subpackages
    (for example the Binance facade and its submodules). Non-Python package data is
    not covered.
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
            "packet head is unavailable: fetch full Git history before checking source closure"
        )
    return head


def packet():
    return json.loads(PACKET_PATH.read_text(encoding="utf-8"))


def test_missing_packet_head_fails_instead_of_silently_skipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required history loss must be a failed proof, not a green pytest run."""

    def missing_history(*args: str) -> subprocess.CompletedProcess[str]:
        assert args == (
            "cat-file",
            "-e",
            f"{packet()['identity']['implementation_head']}^{{commit}}",
        )
        return subprocess.CompletedProcess(args, 1, "", "object unavailable")

    monkeypatch.setattr(
        "tests.evaluation.test_carry_multisymbol_packet._git", missing_history
    )
    try:
        _require_pinned_head()
    except pytest.skip.Exception as error:
        pytest.fail(f"source-closure check skipped without the pinned commit: {error}")
    except pytest.fail.Exception as error:
        assert "packet head" in str(error)
    else:
        pytest.fail("missing immutable source must not satisfy the packet check")


def test_historical_carry_author_packet_remains_unchanged_and_unreviewed() -> None:
    """A software reissue does not overwrite the original author evidence."""
    old_path = (
        ROOT / "report/issue810-carry-multisymbol-research-assurance-20261110.json"
    )
    historical = json.loads(old_path.read_text(encoding="utf-8"))
    assert assurance_digest(historical) == (
        "baf18575794470c4ba2328a6fac93393bcef31e43f16aeed339531bdbe00844f"
    )
    assert historical["identity"]["implementation_head"] == (
        "46760f4948a55f37cb79bc56b89c3b477c54d6de"
    )
    assert evaluate_assurance(historical).status == "UNREVIEWED"


def test_prior_20261113_source_reissue_is_immutable_and_unreviewed() -> None:
    """Do not rewrite a preceding source-bound author packet after another merge."""
    path = ROOT / "report/issue810-carry-multisymbol-research-assurance-20261113.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    assert assurance_digest(old) == (
        "65099fe95da04565f6429a4252ff9657d19acfed83c589ff1d2173abf9855afd"
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
    for needle in ("edge", "FS1", "live", "unused"):
        assert needle in forbidden
    evidence = record["evidence"]
    assert evidence["point_in_time"] is False
    assert evidence["realistic_costs"] is False
    assert evidence["fit_development_unused_separated"] is False


def test_packet_states_the_required_g0_to_g2_content() -> None:
    thesis = packet()["thesis"]
    assert "2023-2024 regime effect" in thesis["counter_hypothesis"]
    assert "fees" in thesis["counter_hypothesis"]
    assert "2x-cost" in thesis["cheapest_falsifier"]
    assert "No cell, symbol, window, cost, gross or rule" in thesis["stop_rule"]
    assert "Leveraged long demand" in thesis["economic_mechanism"]
    mechanism = packet()["mechanism"]
    assert set(mechanism["independent_oracles"]) == {"time", "accounting", "evidence"}
    oracle = mechanism["independent_oracles"]["accounting"]["oracle"]
    assert "5992.7" in oracle and "hand-computed" in oracle


def test_documents_name_the_packet_digest_and_the_protocol_digest() -> None:
    for path in DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        assert PINNED_DIGEST in text, path.name
        assert "UNREVIEWED" in text, path.name
    for path in (SPEC, ROOT / "docs/research/current-status.md"):
        assert protocol.protocol_digest() in path.read_text(encoding="utf-8")
    assert "Status: Active" in SPEC.read_text(encoding="utf-8")


def test_spec_states_the_registered_universe_window_and_thresholds() -> None:
    text = " ".join(SPEC.read_text(encoding="utf-8").split())
    for symbol in protocol.CANDIDATES:
        assert symbol.removesuffix("USDT") in text
    for needle in (
        "[2022-01-01, 2026-08-01)",
        "1,000,000 USDT",
        "92-day lookback",
        "K = ceil(20 x admitted / 32)",
        "K = 4, 5, 5, 6",
        "Binance Vision archives with official `.CHECKSUM` verification",
        "--g4-authorization-url",
        "G4 authorization",
    ):
        assert needle in text, needle
    assert protocol.MIN_MEDIAN_HOURLY_QUOTE_VOLUME == 1_000_000.0
    assert (
        protocol.LOOKBACK_HOURS
        == (protocol.WINDOW_START - protocol.LOOKBACK_START).total_seconds() // 3600
    )


def test_packet_heads_are_one_commit_and_the_bound_code_is_unchanged() -> None:
    identity = packet()["identity"]
    head = identity["protocol_head"]
    assert head == identity["implementation_head"]
    assert re.fullmatch(r"[0-9a-f]{40}", head)
    _require_pinned_head()
    bound = bound_code_paths(head)
    changed = _git("diff", "--name-only", head, "HEAD", "--", *bound)
    assert changed.returncode == 0, changed.stderr
    assert changed.stdout.split() == [], (
        "re-pin the packet heads after changing the bound code"
    )


def test_the_static_closure_covers_every_shared_dependency_of_the_screen() -> None:
    head = _require_pinned_head()
    bound = set(bound_code_paths(head))
    for needed in (
        "trade_rl/__init__.py",
        "trade_rl/_validation.py",
        "trade_rl/strategies/carry.py",
        "trade_rl/evaluation/carry.py",
        "trade_rl/evaluation/carry_multisymbol/runner.py",
        "tools/carry_multisymbol_screen.py",
        "trade_rl/integrations/binance/__init__.py",
        "trade_rl/integrations/binance/dataset.py",
        "trade_rl/integrations/binance/carry.py",
        "trade_rl/integrations/binance/vision.py",
        "trade_rl/integrations/market_store/carry_source.py",
        "trade_rl/integrations/market_store/parse.py",
        "trade_rl/integrations/market_store/checksum.py",
        "trade_rl/integrations/market_store/fetch.py",
        "trade_rl/integrations/market_store/rules.py",
        "trade_rl/integrations/market_store/store.py",
        "trade_rl/data/market.py",
        "trade_rl/data/build/config.py",
        "trade_rl/data/build/economics.py",
        "trade_rl/artifacts/verified_file.py",
        "trade_rl/simulation/__init__.py",
    ):
        assert needed in bound, needed
    # a facade pulls in its submodules, and nothing outside the closure is bound
    assert any(path.startswith("trade_rl/simulation/") for path in bound)
    assert "trade_rl/evaluation/paper/engine.py" not in bound


def test_the_closure_is_computed_not_listed_and_tracks_a_new_import() -> None:
    files = {"trade_rl/a.py", "trade_rl/b.py", "trade_rl/pkg/__init__.py"}
    assert _imported_modules("trade_rl/a.py", "from trade_rl import b\n") == {
        "trade_rl",
        "trade_rl.b",
    }
    lazy = "def f():\n    import trade_rl.pkg\n"
    assert "trade_rl.pkg" in _imported_modules("trade_rl/a.py", lazy)
    assert _module_file("trade_rl.pkg", files) == "trade_rl/pkg/__init__.py"
    assert _module_file("trade_rl.nothing", files) is None


def test_exposure_candidate_provenance_and_bch_are_disclosed_everywhere() -> None:
    fixture = "tests/fixtures/legacy/realdata_20260713_signal_metadata.json"
    assert (ROOT / fixture).is_file()
    spec = " ".join(SPEC.read_text(encoding="utf-8").split())
    assert fixture in spec and "withdrawn" in spec
    assert "BCH" in spec and "included" in spec
    assert "416,667 USDT of hourly volume" in spec
    assert "read no market data in this task" in spec
    assert "not a controlled error rate" in spec
    thesis = packet()["thesis"]["limitations"]
    for needle in (
        "realdata_20260713_signal_metadata.json",
        "BCHUSDT",
        "carry-outcome-blind, not data-blind",
        "not a controlled error rate",
    ):
        assert needle in thesis, needle
    forbidden = " ".join(packet()["claims"]["forbidden"])
    assert "data-blind" in forbidden and "controlled error rate" in forbidden
    text = json.dumps(packet())
    assert "mutation" not in text


def test_hypothesis_prose_states_the_pinned_cell_rule_exactly() -> None:
    thesis = packet()["thesis"]
    hypothesis = thesis["hypothesis"]
    # decision.cell_passes: the every-sub-period clause is the base cell's alone
    assert "base cell only" in hypothesis
    assert "full-period net return" in hypothesis
    for stress in ("2x cost", "one hour of entry delay", "10x lower participation"):
        assert stress in hypothesis
    assert "in every calendar sub-period 2022 to 2026 and under" not in hypothesis
    spec = " ".join(SPEC.read_text(encoding="utf-8").split())
    assert "base cell only" in spec
    assert "the stress cells are not required to be positive in every" in spec


def test_inconclusive_universe_is_terminal_in_the_packet_the_spec_and_the_code() -> (
    None
):
    thesis = packet()["thesis"]
    assert "INCONCLUSIVE_UNIVERSE" in thesis["stop_rule"]
    assert "terminal for this protocol" in thesis["stop_rule"]
    assert "terminal for this protocol" in packet()["claims"]["decision_rule"]
    spec = " ".join(SPEC.read_text(encoding="utf-8").split())
    assert "`INCONCLUSIVE_UNIVERSE`, which is terminal for this protocol" in spec
    assert "never rerun" in spec


def test_one_authorization_covers_the_coverage_stage_and_every_resume() -> None:
    sentence = (
        "The coverage-only stage and any resume on the same output directory "
        "are part of that one run."
    )
    assert sentence in protocol.lease_text(code_commit="a" * 40)
    spec = " ".join(SPEC.read_text(encoding="utf-8").split())
    assert sentence in spec
    action = packet()["claims"]["next_authorized_action"]
    assert "coverage-only stage" in action and "resume" in action


def test_spec_states_the_exclusive_funding_end_and_the_oversize_halt() -> None:
    spec = " ".join(SPEC.read_text(encoding="utf-8").split())
    assert "a funding event at exactly 2026-08-01T00:00Z is excluded" in spec
    assert "larger than the fetcher's byte bound" in spec
