from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.workflows import stage_a_nautilus_representative_batch as runner
from trade_rl.workflows.stage_a_nautilus_economic_comparison import (
    StageANautilusHistoricalEconomicEvidence,
)
from trade_rl.workflows.stage_a_nautilus_historical_differential import (
    StageANautilusHistoricalDifferentialEvidence,
)
from trade_rl.workflows.stage_a_nautilus_representative_evidence import (
    RepresentativeNautilusWindowEvidence,
    load_representative_nautilus_evidence,
)


def _window(time_quantile: float) -> RepresentativeNautilusWindowEvidence:
    replay_digest = f"{int(time_quantile * 10):064x}"
    structural = StageANautilusHistoricalDifferentialEvidence(
        replay_digest=replay_digest,
        request_digest="a" * 64,
        dataset_id="b" * 64,
        candidate_runtime_version="1.230.0",
        legacy_terminal_position_lots=0,
        candidate_terminal_position_lots=0,
        terminal_position_matches=True,
        terminal_open_orders_passed=True,
        funding_matches=True,
        structural_passed=True,
    )
    economic = StageANautilusHistoricalEconomicEvidence(
        replay_digest=replay_digest,
        funding_matches=True,
        structural_passed=True,
        legacy_cost_neutral_equity_minor=10_000,
        candidate_cost_neutral_equity_minor=10_000,
        execution_cost_representation_delta_minor=0,
        normalized_equity_delta_minor=0,
        economic_passed=True,
    )
    return RepresentativeNautilusWindowEvidence(
        time_quantile=time_quantile,
        structural=structural,
        economic=economic,
    )


def test_run_and_persist_representative_evidence_uses_all_quantiles_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def fake_window_runner(**kwargs: object) -> RepresentativeNautilusWindowEvidence:
        quantile = float(kwargs["time_quantile"])
        calls.append(quantile)
        return _window(quantile)

    monkeypatch.setattr(
        runner,
        "run_representative_nautilus_window",
        fake_window_runner,
    )
    output_path = tmp_path / "representative-evidence.json"

    evidence = runner.run_and_persist_representative_nautilus_evidence(
        markets={0.9: object(), 0.1: object(), 0.5: object()},
        source_digest="f" * 64,
        store_root=tmp_path / "stage-a",
        output_path=output_path,
        target_exposure=0.10,
    )

    assert calls == [0.1, 0.5, 0.9]
    assert evidence.time_quantiles == (0.1, 0.5, 0.9)
    assert evidence.exact_parity_passed is True
    assert load_representative_nautilus_evidence(output_path) == evidence


def test_run_and_persist_representative_evidence_rejects_missing_window_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_window_runner(**kwargs: object) -> RepresentativeNautilusWindowEvidence:
        nonlocal called
        called = True
        return _window(float(kwargs["time_quantile"]))

    monkeypatch.setattr(
        runner,
        "run_representative_nautilus_window",
        fake_window_runner,
    )

    with pytest.raises(ValueError, match="exactly the 0.1, 0.5, and 0.9 windows"):
        runner.run_and_persist_representative_nautilus_evidence(
            markets={0.1: object(), 0.9: object()},
            source_digest="f" * 64,
            store_root=tmp_path / "stage-a",
            output_path=tmp_path / "representative-evidence.json",
        )

    assert called is False
