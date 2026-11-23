"""Source-bound documentation consistency probes (synthetic, no market data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.docs_source_consistency import (
    ContractViolation,
    ProtocolBinding,
    check_all,
    check_historical_agy_status,
    check_protocol_digest,
)

ROOT = Path(__file__).resolve().parents[2]


def _fixture_agy(root: Path) -> None:
    for name in (
        "docs/README.md",
        "docs/specs/agy-anchor-realistic.md",
        "docs/research/current-status.md",
        "tools/agy_anchor_oos.py",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / "docs/README.md").write_text(
        "- Ported agy anchor OOS: historical BACKCAST_2020 INVALID; "
        "economic execution HOLD, FORWARD not authorized "
        "(specs/agy-anchor-realistic.md).\n"
    )
    (root / "docs/specs/agy-anchor-realistic.md").write_text(
        "Status: **Historical v1 / economic execution HOLD**. "
        "Historical INVALID BACKCAST_2020.\n"
    )
    (root / "docs/research/current-status.md").write_text(
        "AGY v1 is a separate historical INVALID result.\n"
    )
    (root / "tools/agy_anchor_oos.py").write_text(
        "def main(argv, *, rows_factory=None):\n"
        "    args = build_parser().parse_args(argv)\n"
        '    if args.command in {"run", "coverage"} and rows_factory is None:\n'
        "        return EXIT_REFUSED\n"
        "    moment = now()\n"
    )


def test_current_repo_is_source_doc_consistent() -> None:
    check_all(ROOT)


def test_agy_historical_status_matches_source_and_spec(tmp_path: Path) -> None:
    _fixture_agy(tmp_path)
    check_historical_agy_status(tmp_path)


def test_old_readme_does_not_erase_historical_invalid(tmp_path: Path) -> None:
    _fixture_agy(tmp_path)
    (tmp_path / "docs/README.md").write_text(
        "- Realistic agy OOS (DEV, BACKCAST_2020 then FORWARD, not yet run): "
        "(specs/agy-anchor-realistic.md)\n"
    )
    with pytest.raises(ContractViolation, match="README.*BACKCAST_2020 INVALID"):
        check_historical_agy_status(tmp_path)


@pytest.mark.parametrize(
    "path,change,match",
    [
        (
            "docs/specs/agy-anchor-realistic.md",
            "Historical study has not run",
            "historical",
        ),
        ("docs/research/current-status.md", "No study status", "historical"),
        (
            "tools/agy_anchor_oos.py",
            "def main(argv, *, rows_factory=None):\n    moment = now()\n",
            "source",
        ),
    ],
)
def test_historical_source_or_status_drift_fails_closed(
    tmp_path: Path, path: str, change: str, match: str
) -> None:
    _fixture_agy(tmp_path)
    (tmp_path / path).write_text(change)
    with pytest.raises(ContractViolation, match=match):
        check_historical_agy_status(tmp_path)


def test_commented_out_agy_refusal_does_not_satisfy_source_contract(
    tmp_path: Path,
) -> None:
    _fixture_agy(tmp_path)
    p = tmp_path / "tools/agy_anchor_oos.py"
    p.write_text(
        "def main(argv, *, rows_factory=None):\n"
        "    args = build_parser().parse_args(argv)\n"
        '    # if args.command in {"run", "coverage"} and rows_factory is None:\n'
        "    #     return EXIT_REFUSED\n"
        "    moment = now()\n"
    )
    with pytest.raises(ContractViolation, match="source requires"):
        check_historical_agy_status(tmp_path)


def test_agy_guard_after_real_data_read_fails_even_before_clock(
    tmp_path: Path,
) -> None:
    _fixture_agy(tmp_path)
    p = tmp_path / "tools/agy_anchor_oos.py"
    p.write_text(
        "def main(argv, *, rows_factory=None):\n"
        "    args = build_parser().parse_args(argv)\n"
        "    rows = read_market_data()\n"
        '    if args.command in {"run", "coverage"} and rows_factory is None:\n'
        "        return EXIT_REFUSED\n"
        "    moment = now()\n"
    )
    with pytest.raises(ContractViolation, match="pre-data"):
        check_historical_agy_status(tmp_path)


def test_agy_guard_must_not_read_before_refusing_inside_guard(tmp_path: Path) -> None:
    _fixture_agy(tmp_path)
    p = tmp_path / "tools/agy_anchor_oos.py"
    p.write_text(
        "def main(argv, *, rows_factory=None):\n"
        "    args = build_parser().parse_args(argv)\n"
        '    if args.command in {"run", "coverage"} and rows_factory is None:\n'
        "        read_market_data()\n"
        "        return EXIT_REFUSED\n"
        "    moment = now()\n"
    )
    with pytest.raises(ContractViolation, match="pre-data"):
        check_historical_agy_status(tmp_path)


def test_agy_guard_after_clock_does_not_satisfy_source_contract(tmp_path: Path) -> None:
    _fixture_agy(tmp_path)
    p = tmp_path / "tools/agy_anchor_oos.py"
    p.write_text(
        "def main(argv, *, rows_factory=None):\n"
        "    args = build_parser().parse_args(argv)\n"
        "    moment = now()\n"
        '    if args.command in {"run", "coverage"} and rows_factory is None:\n'
        "        return EXIT_REFUSED\n"
    )
    with pytest.raises(ContractViolation, match="pre-data"):
        check_historical_agy_status(tmp_path)


def test_optional_protocol_digest_and_spec_are_bound(tmp_path: Path) -> None:
    binding = ProtocolBinding(
        protocol_id="trend_breadth_v1",
        source="tools/trend_breadth.py",
        spec="docs/specs/trend-breadth.md",
    )
    assert (
        check_protocol_digest(
            tmp_path, binding, lambda _p: ("trend_breadth_v1", "a" * 64)
        )
        is False
    )
    p = tmp_path / binding.spec
    p.parent.mkdir(parents=True)
    p.write_text(
        "# Trend breadth (`trend_breadth_v1`)\nProtocol digest: `" + "a" * 64 + "`\n"
    )
    with pytest.raises(ContractViolation, match="missing source"):
        check_protocol_digest(
            tmp_path, binding, lambda _p: ("trend_breadth_v1", "a" * 64)
        )
    s = tmp_path / binding.source
    s.parent.mkdir(parents=True)
    s.write_text("PROTOCOL_ID = 'trend_breadth_v1'\n")
    assert check_protocol_digest(
        tmp_path, binding, lambda _p: ("trend_breadth_v1", "a" * 64)
    )
    with pytest.raises(ContractViolation, match="digest mismatch"):
        check_protocol_digest(
            tmp_path, binding, lambda _p: ("trend_breadth_v1", "b" * 64)
        )
    with pytest.raises(ContractViolation, match="protocol_id"):
        check_protocol_digest(tmp_path, binding, lambda _p: ("unrelated_v1", "a" * 64))


def test_active_lab_docs_must_be_indexed_on_the_same_head(tmp_path: Path) -> None:
    from tools.docs_source_consistency import check_learning_lab_links

    readme = tmp_path / "docs/README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("index\n")
    check_learning_lab_links(tmp_path)
    for name in ("learning-lab-registry.md", "anchor-rl-execution-dev.md"):
        (readme.parent / "specs").mkdir(exist_ok=True)
        (readme.parent / "specs" / name).write_text("Status: Active\n")
    with pytest.raises(ContractViolation, match="Learning Lab index"):
        check_learning_lab_links(tmp_path)
    readme.write_text(
        "method registry (specs/learning-lab-registry.md)\n"
        "execution overlay (specs/anchor-rl-execution-dev.md)\n"
    )
    source = tmp_path / "tools/learning_lab.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic-only Lab CLI\n")
    check_learning_lab_links(tmp_path)


def test_ci_checks_semantics_on_fast_and_full() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert ci.count("Docs/source semantic contract") == 2
    assert ci.count("python -m tools.docs_source_consistency --check") == 2
    assert ci.count("tools/docs_source_consistency.py") == 2
    assert "tests/tools/test_docs_source_consistency.py" in ci


def test_project_documentation_describes_checked_scope_and_limits() -> None:
    root_agents = (ROOT / "AGENTS.md").read_text()
    docs_agents = (ROOT / "docs/AGENTS.md").read_text()
    docs_index = (ROOT / "docs/README.md").read_text()
    assert "tools.docs_source_consistency --check" in root_agents
    assert "tools.docs_source_consistency --check" in docs_agents
    assert "tools.docs_source_consistency --check" in docs_index
    assert "semantic" in docs_agents
    assert "部分" in docs_index


def test_unregistered_new_harness_is_never_silently_unchecked(tmp_path: Path) -> None:
    from tools.docs_source_consistency import check_registered_protocol_coverage

    path = tmp_path / "tools/new_research.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "PROTOCOL_ID = 'new_v1'\ndef protocol_digest():\n    return 'a' * 64\n"
    )
    with pytest.raises(ContractViolation, match="unregistered protocol_digest harness"):
        check_registered_protocol_coverage(tmp_path)
    path.unlink()
    check_registered_protocol_coverage(tmp_path)


def test_asof_equity_oracle_rejects_only_future_dependent_marks() -> None:
    from tools.docs_source_consistency import _verify_asof_equity

    _verify_asof_equity([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], observed_rows=2)
    with pytest.raises(ContractViolation, match="noncausal no-notice"):
        _verify_asof_equity([1.0, 1.0, 0.975], [1.0, 1.0, 1.0], 2)
    with pytest.raises(ContractViolation, match="noncausal no-notice"):
        _verify_asof_equity([1.0, 1.0], [1.0, 1.0, 1.0], 2)


def test_no_notice_exemption_requires_actual_invalid_decision() -> None:
    from types import SimpleNamespace

    from tools.docs_source_consistency import _verified_no_notice_refusal

    good = SimpleNamespace(
        LABEL_INVALID="INVALID",
        decide=lambda _cells, _problems: {"label": "INVALID"},
    )
    assert _verified_no_notice_refusal(good, ["no notice at delisting"])
    assert not _verified_no_notice_refusal(good, ["other data gap"])
    bad = SimpleNamespace(
        LABEL_INVALID="INVALID",
        decide=lambda _cells, _problems: {"label": "SUPPORTED"},
    )
    with pytest.raises(ContractViolation, match="must yield INVALID"):
        _verified_no_notice_refusal(bad, ["no notice at delisting"])


@pytest.mark.parametrize("future_changes_past", [True, False])
def test_no_notice_invalid_on_empty_cells_never_exempts_asof_equity_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    future_changes_past: bool,
) -> None:
    """An INVALID label from empty cells is not a real stage/refusal oracle."""
    import sys
    from types import ModuleType, SimpleNamespace

    import numpy as np

    from tools.docs_source_consistency import check_breadth_asof

    source = tmp_path / "tools/trend_breadth.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic stage stub\n", encoding="utf-8")
    spec = tmp_path / "docs/specs/trend-breadth.md"
    spec.parent.mkdir(parents=True)
    spec.write_text("synthetic stub\n", encoding="utf-8")

    fake = ModuleType("tools.trend_breadth")
    fake.__file__ = str(source)
    fake.LABEL_INVALID = "INVALID"  # type: ignore[attr-defined]
    fake.Stage = lambda *_: None  # type: ignore[attr-defined]

    def build_panel(bars, *_):
        # The Stage has 1h inputs but the portfolio oracle has 4h marks.
        rows = len(bars["ZZZUSDT"]["close"]) // 4
        panel = SimpleNamespace(bars=rows, times=tuple(range(rows)))
        for field in ("close", "mark_px", "fill_qv", "regime", "vol", "fund", "forced"):
            setattr(panel, field, np.ones(rows))
        return panel, {}

    fake.build_panel = build_panel  # type: ignore[attr-defined]
    fake.panel_problems = lambda *_: ["no notice at delisting"]  # type: ignore[attr-defined]

    def decide(cells, _problems):
        return {"label": "INVALID" if not cells else "SUPPORTED"}

    fake.decide = decide  # type: ignore[attr-defined]
    assert decide({}, ["no notice at delisting"])["label"] == "INVALID"
    assert (
        decide({"asof": "eligible"}, ["no notice at delisting"])["label"] == "SUPPORTED"
    )

    simulated: list[int] = []

    def simulate_weights(panel, *_):
        simulated.append(panel.bars)
        equity = np.full(panel.bars + 1, 1_000.0)
        if future_changes_past and panel.bars == 18:
            equity[12] -= 25.0  # future observation retroactively changes past cash
        return {"valid": True, "equity": equity}

    fake.simulate_weights = simulate_weights  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.trend_breadth", fake)

    if future_changes_past:
        with pytest.raises(ContractViolation, match="noncausal no-notice"):
            check_breadth_asof(tmp_path)
    else:
        assert check_breadth_asof(tmp_path) is True
    # Both histories must be inspected before no-notice can be called safe.
    assert simulated == [18, 12]


def test_empty_registered_spec_is_rejected_as_contract_violation(
    tmp_path: Path,
) -> None:
    binding = ProtocolBinding(
        protocol_id="trend_breadth_v1",
        source="tools/trend_breadth.py",
        spec="docs/specs/trend-breadth.md",
    )
    src = tmp_path / binding.source
    src.parent.mkdir(parents=True)
    src.write_text("PROTOCOL_ID = 'trend_breadth_v1'\n")
    spec = tmp_path / binding.spec
    spec.parent.mkdir(parents=True)
    spec.write_bytes(b"")
    with pytest.raises(ContractViolation, match="protocol_id heading mismatch"):
        check_protocol_digest(
            tmp_path, binding, lambda _b: ("trend_breadth_v1", "a" * 64)
        )


def test_installed_digest_rejects_other_checkout_or_spec_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import ModuleType

    from tools.docs_source_consistency import _installed_digest

    binding = ProtocolBinding(
        "trend_breadth_v1",
        "tools/trend_breadth.py",
        "docs/specs/trend-breadth.md",
    )
    src = tmp_path / binding.source
    src.parent.mkdir(parents=True)
    src.write_text("PROTOCOL_ID = 'trend_breadth_v1'\n")
    fake = ModuleType("tools.trend_breadth")
    fake.__file__ = str(tmp_path / "other-checkout.py")
    fake.PROTOCOL_ID = binding.protocol_id  # type: ignore[attr-defined]
    fake.SPEC_RELPATH = binding.spec  # type: ignore[attr-defined]
    fake.protocol_digest = lambda: "a" * 64  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.trend_breadth", fake)
    with pytest.raises(ContractViolation, match="outside checkout"):
        _installed_digest(tmp_path, binding)
    fake.__file__ = str(src)
    fake.SPEC_RELPATH = "docs/specs/other.md"  # type: ignore[attr-defined]
    with pytest.raises(ContractViolation, match="spec path"):
        _installed_digest(tmp_path, binding)
    fake.SPEC_RELPATH = binding.spec  # type: ignore[attr-defined]
    assert _installed_digest(tmp_path, binding) == (binding.protocol_id, "a" * 64)


def test_breadth_zero_length_asof_prefix_cannot_vacuously_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    import numpy as np

    from tools.docs_source_consistency import check_breadth_asof

    src = tmp_path / "tools/trend_breadth.py"
    src.parent.mkdir(parents=True)
    src.write_text("# synthetic zero-prefix stub\n")
    spec = tmp_path / "docs/specs/trend-breadth.md"
    spec.parent.mkdir(parents=True)
    spec.write_text("synthetic stub\n")
    fake = ModuleType("tools.trend_breadth")
    fake.__file__ = str(src)
    fake.Stage = lambda *_: None  # type: ignore[attr-defined]

    def build_panel(bars, *_):
        original_length = len(bars["ZZZUSDT"]["close"])
        n = 72 if original_length == 72 else 0
        panel = SimpleNamespace(bars=n, times=tuple(range(n)))
        for field in ("close", "mark_px", "fill_qv", "regime", "vol", "fund", "forced"):
            setattr(panel, field, np.ones(n))
        return panel, {}

    fake.build_panel = build_panel  # type: ignore[attr-defined]
    fake.panel_problems = lambda *_: []  # type: ignore[attr-defined]
    fake.simulate_weights = lambda p, *_: {  # type: ignore[attr-defined]
        "valid": True,
        "equity": np.ones(p.bars + 1),
    }
    monkeypatch.setitem(sys.modules, "tools.trend_breadth", fake)
    with pytest.raises(ContractViolation, match="nonempty"):
        check_breadth_asof(tmp_path)


@pytest.mark.parametrize(
    ("observed_rows", "shift_asof_clock", "future_changes_past"),
    [
        (2, False, True),
        (12, True, False),
        (12, False, False),
    ],
)
def test_asof_expected_clock_prefix_is_never_truncated_or_relabelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed_rows: int,
    shift_asof_clock: bool,
    future_changes_past: bool,
) -> None:
    """The oracle must compare *all* 48 prior hours, not a convenient subset."""
    import sys
    from types import ModuleType, SimpleNamespace

    import numpy as np

    from tools.docs_source_consistency import check_breadth_asof

    source = tmp_path / "tools/trend_breadth.py"
    source.parent.mkdir(parents=True)
    source.write_text("# as-of clock stub\n", encoding="utf-8")
    spec = tmp_path / "docs/specs/trend-breadth.md"
    spec.parent.mkdir(parents=True)
    spec.write_text("synthetic source clock\n", encoding="utf-8")

    fake = ModuleType("tools.trend_breadth")
    fake.__file__ = str(source)
    fake.Stage = lambda *_: None  # type: ignore[attr-defined]

    def build_panel(bars, *_):
        is_full = len(bars["ZZZUSDT"]["close"]) == 72
        n = 18 if is_full else observed_rows
        times = np.arange(n, dtype=int)
        if not is_full and shift_asof_clock:
            times += 1
        panel = SimpleNamespace(times=times, bars=n - 1)
        for field in ("close", "mark_px", "fill_qv", "regime", "vol", "fund", "forced"):
            setattr(panel, field, np.ones(n))
        return panel, {}

    fake.build_panel = build_panel  # type: ignore[attr-defined]
    fake.panel_problems = lambda *_: []  # type: ignore[attr-defined]

    def simulate_weights(panel, *_):
        equity = np.full(panel.bars + 2, 1_000.0)
        if future_changes_past and len(panel.times) == 18:
            equity[5] = 975.0
        return {"valid": True, "equity": equity}

    fake.simulate_weights = simulate_weights  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.trend_breadth", fake)
    if observed_rows != 12 or shift_asof_clock:
        with pytest.raises(ContractViolation, match="as-of.*(rows|clock|times)"):
            check_breadth_asof(tmp_path)
    else:
        assert check_breadth_asof(tmp_path) is True
