from __future__ import annotations

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaCostAwareConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaCandidateConfig,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_selection import (
    _causal_alpha_target_for_contract,
    _cost_aware_causal_alpha_target_for_contract,
    causal_alpha_one_way_cost_rates,
)


def _samples() -> CausalAlphaSymbolSamples:
    # Decision 12 is intentionally absent: it represents an inactive/non-tradable
    # decision. Decision 11 is tradable but has one unavailable feature.
    decisions = np.asarray([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14], dtype=np.int64)
    features = np.column_stack(
        (decisions.astype(np.float64), 0.5 * decisions.astype(np.float64))
    )
    available = np.ones_like(features, dtype=np.bool_)
    available[decisions.tolist().index(11), 1] = False
    return CausalAlphaSymbolSamples(
        symbol="AAAUSDT",
        dataset_id=content_digest("dataset"),
        feature_names=("fast", "slow"),
        feature_schema_digest=content_digest("feature-schema"),
        context_digest=content_digest("context"),
        reference_equity_mode="initial_capital",
        reference_equity=1_000.0,
        decision_indices=decisions,
        features=features,
        feature_available=available,
        labels_24h=0.01 * decisions.astype(np.float64),
        label_end_indices_24h=decisions + 1,
        labels_72h=0.02 * decisions.astype(np.float64),
        label_end_indices_72h=decisions + 2,
    )


def test_target_generation_zero_imputes_missing_features_and_holds_nontradable() -> (
    None
):
    samples = _samples()
    contract = OracleEpisodeContract(
        dataset_id=samples.dataset_id,
        episode_index=0,
        start=10,
        stop=15,
        initial_state_mode="cash",
        initial_weights=np.asarray([0.0], dtype=np.float64),
    )
    candidate = CausalAlphaCandidateConfig(
        name="availability-contract",
        ridge=CausalAlphaRidgeConfig(ridge_strength=0.1),
        controller=CausalAlphaControllerConfig(
            horizon_mix=CausalAlphaHorizonMix.EQUAL,
            score_scale=10.0,
            entry_threshold=0.01,
            exit_threshold=0.005,
            no_trade_band=0.0,
            max_target_delta=2.0,
        ),
    )

    targets = _causal_alpha_target_for_contract(
        symbol="AAAUSDT",
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": samples},
        contract=contract,
        candidate=candidate,
    ).reshape(-1)

    assert targets.shape == (4,)
    assert np.isfinite(targets).all()
    # Decision sequence is 10, 11, 12, 13. Decision 12 is not actionable,
    # therefore its target must equal the previous decision's target.
    assert targets[2] == pytest.approx(targets[1])


class _CostDataset:
    n_bars = 32

    def __init__(self) -> None:
        rows = np.arange(self.n_bars, dtype=np.float64).reshape(-1, 1)
        self._values = {
            "fee_rate": rows * 0.0001,
            "maker_fee_rate": rows * 0.0001 + 0.0002,
            "taker_fee_rate": rows * 0.0001 + 0.001,
            "spread_rate": rows * 0.0001 + 0.002,
            "max_participation_rate": np.full_like(rows, 0.36),
        }

    def resolved_array(self, name: str) -> np.ndarray:
        return self._values[name]


def test_one_way_cost_rate_uses_first_executable_row_after_signal_delay() -> None:
    config = ExecutionCostConfig(
        fee_rate=0.0005,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0004,
        spread_rate=0.0002,
        impact_rate=0.0001,
        max_participation_rate=0.25,
        order_type="market",
    )

    rates = causal_alpha_one_way_cost_rates(
        _CostDataset(),
        config,
        decision_indices=np.asarray([10, 14]),
        signal_delay_decisions=1,
        decision_bars=4,
    )

    assert rates.tolist() == pytest.approx(
        [
            0.0005 + 0.0015 + 0.0004 + 0.0025 + 0.0002 + 0.0035 + 0.00005,
            0.0005 + 0.0019 + 0.0004 + 0.0029 + 0.0002 + 0.0039 + 0.00005,
        ]
    )


def test_cost_aware_contract_targets_bind_signal_diagnostics_and_cost_path() -> None:
    samples = _samples()
    contract = OracleEpisodeContract(
        dataset_id=samples.dataset_id,
        episode_index=0,
        start=10,
        stop=15,
        initial_state_mode="cash",
        initial_weights=np.asarray([0.0], dtype=np.float64),
    )
    candidate = CausalAlphaCandidateConfig(
        name="cost-aware",
        ridge=CausalAlphaRidgeConfig(ridge_strength=0.1),
        controller=CausalAlphaControllerConfig(
            horizon_mix=CausalAlphaHorizonMix.EQUAL,
            score_scale=25.0,
            entry_threshold=0.001,
            exit_threshold=0.0005,
            no_trade_band=0.0,
            max_target_delta=0.125,
        ),
        economic_controller=CausalAlphaCostAwareConfig(
            execution_cost_multiplier=1.5,
            edge_margin=0.001,
            confirmation_count=2,
            strong_reversal_threshold=0.02,
            max_abs_target=0.5,
        ),
    )

    result = _cost_aware_causal_alpha_target_for_contract(
        symbol="AAAUSDT",
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": samples},
        contract=contract,
        candidate=candidate,
        dataset=_CostDataset(),
        execution_cost=ExecutionCostConfig(),
        signal_delay_decisions=1,
        decision_bars=1,
    )

    assert result.actions.shape == (4, 1)
    assert result.signal_24h.sample_count == 2
    assert result.signal_72h.sample_count == 2
    assert result.target_path.targets[2] == pytest.approx(result.target_path.targets[1])
    assert result.target_path.digest
