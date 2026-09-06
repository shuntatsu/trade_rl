from __future__ import annotations

import math
from dataclasses import replace

import pytest

from tests.integrations.test_universal_trade_rl_u2_replay import (
    ReplayIntegrationFixture,
)
from tests.integrations.test_universal_trade_rl_u2_replay_runtime import (
    DeterministicModelSpy,
    _request,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_trade_rl_u2_development_closure import (
    UniversalTradeRLU2FinalCheckpointClosure,
)
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayEvidence,
    UniversalTradeRLU2ReplayVariant,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import U2_DECISION_STEP_NS

pytest_plugins = ("tests.integrations.test_universal_trade_rl_u2_replay",)


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _checkpoint_digest(seed: int) -> str:
    return content_digest({"fixture": "candidate-checkpoint", "seed": seed})


def _checkpoint_closure(
    fixture: ReplayIntegrationFixture,
) -> UniversalTradeRLU2FinalCheckpointClosure:
    return UniversalTradeRLU2FinalCheckpointClosure(
        predevelopment_contract_digest=content_digest(
            {"fixture": "pairing-predevelopment"}
        ),
        universe_manifest_digest=fixture.manifest.digest,
        u2_contract_digest=fixture.u2_contract.digest,
        source_closure_digest=content_digest({"fixture": "pairing-source-closure"}),
        u1_contract_digest=fixture.u2_contract.u1_contract_digest,
        normalizer_digest=fixture.u2_contract.u1_normalizer_digest,
        time_partition_digest=fixture.u2_contract.time_partition_digest,
        training_config_digest=fixture.u2_contract.training_config_digest,
        training_plan_digests=tuple(
            (seed, content_digest({"fixture": "pairing-plan", "seed": seed}))
            for seed in (0, 1, 2)
        ),
        checkpoint_digests=tuple(
            (seed, _checkpoint_digest(seed)) for seed in (0, 1, 2)
        ),
        environment_digests=tuple(
            (seed, content_digest({"fixture": "pairing-env", "seed": seed}))
            for seed in (0, 1, 2)
        ),
    )


@pytest.fixture(scope="module")
def candidate_cash_pair(
    replay_fixture: ReplayIntegrationFixture,
) -> tuple[UniversalTradeRLU2ReplayEvidence, UniversalTradeRLU2ReplayEvidence]:
    candidate = replay_fixture.session.replay(
        _request(
            replay_fixture,
            variant=UniversalTradeRLU2ReplayVariant.CANDIDATE,
            seed=0,
        ),
        model=DeterministicModelSpy(action=[0.0]),
    )
    cash = replay_fixture.session.replay(
        _request(
            replay_fixture,
            variant=UniversalTradeRLU2ReplayVariant.CASH,
            seed=0,
        )
    )
    return candidate, cash


def test_u2_paired_replay_scope_is_derived_from_exact_candidate_cash_evidence(
    replay_fixture: ReplayIntegrationFixture,
    candidate_cash_pair: tuple[
        UniversalTradeRLU2ReplayEvidence,
        UniversalTradeRLU2ReplayEvidence,
    ],
) -> None:
    module = _module()
    candidate, cash = candidate_cash_pair
    closure = _checkpoint_closure(replay_fixture)

    paired = module.build_universal_trade_rl_u2_paired_replay_scope_evidence(
        candidate_replay=candidate,
        cash_replay=cash,
        u2_contract=replay_fixture.u2_contract,
        time_partition=replay_fixture.partition,
        checkpoint_closure=closure,
    )

    assert paired.training_seed == 0
    assert paired.candidate_replay_evidence_digest == candidate.digest
    assert paired.cash_replay_evidence_digest == cash.digest
    assert paired.scope_digest == candidate.scope_digest
    assert paired.evaluation_dataset_digest == candidate.evaluation_dataset_digest
    assert paired.paired_candidate_checkpoint_digest == _checkpoint_digest(0)
    assert len(paired.decision_timestamps_ns) == candidate.observed_decision_count
    assert len(paired.candidate_minus_cash_net_log_excess) == (
        candidate.observed_decision_count
    )
    expected_first_timestamp = (
        replay_fixture.partition.common_first_timestamp_ns
        + candidate.step_evidence[0].decision_bar_index * U2_DECISION_STEP_NS
    )
    assert paired.decision_timestamps_ns[0] == expected_first_timestamp
    expected_excess = tuple(
        math.log1p(candidate_return) - math.log1p(cash_return)
        for candidate_return, cash_return in zip(
            candidate.net_simple_returns,
            cash.net_simple_returns,
            strict=True,
        )
    )
    assert paired.candidate_minus_cash_net_log_excess == pytest.approx(expected_excess)


def test_u2_paired_replay_scope_rejects_dataset_or_checkpoint_pair_drift(
    replay_fixture: ReplayIntegrationFixture,
    candidate_cash_pair: tuple[
        UniversalTradeRLU2ReplayEvidence,
        UniversalTradeRLU2ReplayEvidence,
    ],
) -> None:
    module = _module()
    candidate, cash = candidate_cash_pair
    closure = _checkpoint_closure(replay_fixture)

    wrong_dataset = replace(
        cash,
        evaluation_dataset_digest=content_digest({"drift": "evaluation-dataset"}),
        digest="",
    )
    with pytest.raises(ValueError, match="dataset|scope|pair|identity"):
        module.build_universal_trade_rl_u2_paired_replay_scope_evidence(
            candidate_replay=candidate,
            cash_replay=wrong_dataset,
            u2_contract=replay_fixture.u2_contract,
            time_partition=replay_fixture.partition,
            checkpoint_closure=closure,
        )

    wrong_checkpoint = replace(
        cash,
        paired_candidate_checkpoint_digest=content_digest(
            {"drift": "candidate-checkpoint"}
        ),
        digest="",
    )
    with pytest.raises(ValueError, match="checkpoint|pair|identity"):
        module.build_universal_trade_rl_u2_paired_replay_scope_evidence(
            candidate_replay=candidate,
            cash_replay=wrong_checkpoint,
            u2_contract=replay_fixture.u2_contract,
            time_partition=replay_fixture.partition,
            checkpoint_closure=closure,
        )


def _paired_scope(
    *,
    seed: int,
    source_window: str,
    cell: str,
    symbol: str,
    values: tuple[float, ...],
):
    module = _module()
    timestamps = tuple(1_000_000_000 + index * U2_DECISION_STEP_NS for index in range(len(values)))
    return module.UniversalTradeRLU2PairedReplayScopeEvidence(
        training_seed=seed,
        source_window=source_window,
        cell=cell,
        concrete_symbol=symbol,
        scope_digest=content_digest(
            {"fixture": "pairing-scope", "window": source_window, "symbol": symbol}
        ),
        evaluation_dataset_digest=content_digest(
            {"fixture": "pairing-dataset", "symbol": symbol}
        ),
        paired_candidate_checkpoint_digest=_checkpoint_digest(seed),
        candidate_replay_evidence_digest=content_digest(
            {
                "fixture": "pairing-candidate",
                "seed": seed,
                "window": source_window,
                "symbol": symbol,
            }
        ),
        cash_replay_evidence_digest=content_digest(
            {
                "fixture": "pairing-cash",
                "seed": seed,
                "window": source_window,
                "symbol": symbol,
            }
        ),
        decision_timestamps_ns=timestamps,
        candidate_minus_cash_net_log_excess=values,
    )


def test_u2_pairing_reducer_binds_complete_seed_symbol_window_provenance() -> None:
    module = _module()
    symbols = ("DEV_A", "DEV_B")
    pairs = tuple(
        _paired_scope(
            seed=seed,
            source_window=window,
            cell=cell,
            symbol=symbol,
            values=(0.01 + seed * 0.001, 0.02 + seed * 0.001),
        )
        for seed in (0, 1, 2)
        for window, cell in (
            ("development_future_1", "D1"),
            ("development_future_2", "D2"),
        )
        for symbol in symbols
    )

    segments = module.reduce_universal_trade_rl_u2_paired_replay_evidence(
        pairs=pairs,
        expected_symbols=symbols,
    )

    assert tuple(segment.source_window for segment in segments) == (
        "development_future_1",
        "development_future_2",
    )
    assert all(segment.paired_scope_evidence_digests for segment in segments)
    assert tuple(
        len(segment.paired_scope_evidence_digests) for segment in segments
    ) == (6, 6)

    with pytest.raises(ValueError, match="complete|closure|seed|symbol|window"):
        module.reduce_universal_trade_rl_u2_paired_replay_evidence(
            pairs=pairs[:-1],
            expected_symbols=symbols,
        )
