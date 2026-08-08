from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.workflows.stage_a_nautilus_economic_comparison import (
    StageANautilusHistoricalEconomicClosure,
    compare_stage_a_nautilus_historical_economics,
)
from trade_rl.workflows.stage_a_nautilus_historical_differential import (
    StageANautilusHistoricalDifferentialEvidence,
)
from trade_rl.workflows.stage_a_nautilus_representative_evidence import (
    RepresentativeNautilusWindowEvidence,
    build_representative_nautilus_evidence,
    load_representative_nautilus_evidence,
    write_representative_nautilus_evidence,
)


def _structural(*, replay_digest: str, passed: bool = True):
    return StageANautilusHistoricalDifferentialEvidence(
        replay_digest=replay_digest,
        request_digest="b" * 64,
        dataset_id="c" * 64,
        candidate_runtime_version="1.230.0",
        legacy_terminal_position_lots=0,
        candidate_terminal_position_lots=0,
        terminal_position_matches=passed,
        terminal_open_orders_passed=True,
        funding_matches=True,
        structural_passed=passed,
    )


def _window(time_quantile: float, replay_digest: str):
    structural = _structural(replay_digest=replay_digest)
    closure = StageANautilusHistoricalEconomicClosure(
        final_equity_minor=10_000_000_000_000,
        execution_cost_minor=1_000,
    )
    economic = compare_stage_a_nautilus_historical_economics(
        structural=structural,
        legacy=closure,
        candidate=closure,
    )
    return RepresentativeNautilusWindowEvidence(
        time_quantile=time_quantile,
        structural=structural,
        economic=economic,
    )


def test_representative_evidence_round_trips_and_requires_all_three_windows(
    tmp_path,
) -> None:
    evidence = build_representative_nautilus_evidence(
        source_digest="a" * 64,
        windows=(
            _window(0.1, "1" * 64),
            _window(0.5, "2" * 64),
            _window(0.9, "3" * 64),
        ),
    )

    assert evidence.time_quantiles == (0.1, 0.5, 0.9)
    assert evidence.exact_parity_passed is True
    path = write_representative_nautilus_evidence(tmp_path / "evidence.json", evidence)
    assert load_representative_nautilus_evidence(path) == evidence


def test_representative_evidence_fails_closed_on_one_window_mismatch() -> None:
    good = _window(0.1, "1" * 64)
    bad_structural = _structural(replay_digest="2" * 64, passed=False)
    closure = StageANautilusHistoricalEconomicClosure(
        final_equity_minor=10_000_000_000_000,
        execution_cost_minor=1_000,
    )
    bad = RepresentativeNautilusWindowEvidence(
        time_quantile=0.5,
        structural=bad_structural,
        economic=compare_stage_a_nautilus_historical_economics(
            structural=bad_structural,
            legacy=closure,
            candidate=closure,
        ),
    )
    evidence = build_representative_nautilus_evidence(
        source_digest="a" * 64,
        windows=(good, bad, _window(0.9, "3" * 64)),
    )

    assert evidence.exact_parity_passed is False


def test_representative_evidence_rejects_wrong_quantiles_and_identity_drift() -> None:
    with pytest.raises(ValueError, match="unsupported representative time quantile"):
        _window(0.4, "4" * 64)

    with pytest.raises(ValueError, match="representative time quantiles"):
        build_representative_nautilus_evidence(
            source_digest="a" * 64,
            windows=(
                _window(0.1, "1" * 64),
                _window(0.9, "2" * 64),
                _window(0.5, "3" * 64),
            ),
        )

    window = _window(0.1, "1" * 64)
    with pytest.raises(ValueError, match="economic replay identity mismatch"):
        RepresentativeNautilusWindowEvidence(
            time_quantile=0.1,
            structural=window.structural,
            economic=replace(window.economic, replay_digest="f" * 64),
        )
