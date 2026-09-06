from __future__ import annotations

import pytest

from tests.integrations.test_universal_trade_rl_u2_replay import (
    ReplayIntegrationFixture,
)
from tests.integrations.test_universal_trade_rl_u2_replay_runtime import (
    DeterministicModelSpy,
    _request,
)
from tests.integrations.test_universal_trade_rl_u2_selection_pairing import (
    _checkpoint_closure,
)
from tests.workflows.test_universal_trade_rl_u2_selection_final import _final_fixture
from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows import universal_trade_rl_u2_selection_pairing as pairing_module
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayVariant,
)

pytest_plugins = ("tests.integrations.test_universal_trade_rl_u2_replay",)


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _candidate_cash(replay_fixture: ReplayIntegrationFixture):
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
    return candidate, cash, _checkpoint_closure(replay_fixture)


def _build_final(bundle):
    module = _module()
    (
        u2_contract,
        base_lock,
        development_lock,
        checkpoint_closure,
        primary,
        robustness,
    ) = bundle
    return module.build_universal_trade_rl_u2_development_selection_evidence(
        u2_contract=u2_contract,
        base_lock=base_lock,
        development_lock=development_lock,
        checkpoint_closure=checkpoint_closure,
        primary_cell_gates=primary,
        seed_robustness_gates=robustness,
    )


def _assert_stale_digest(artifact) -> None:
    assert artifact.digest != content_digest(artifact.to_payload(include_digest=False))


def _drift_replay_digest(candidate) -> None:
    object.__setattr__(
        candidate,
        "digest",
        content_digest({"drift": "candidate-replay-artifact-digest"}),
    )
    _assert_stale_digest(candidate)


def test_u2_selection_pairing_rejects_source_replay_with_stale_artifact_digest(
    replay_fixture: ReplayIntegrationFixture,
) -> None:
    module = _module()
    candidate, cash, closure = _candidate_cash(replay_fixture)
    _drift_replay_digest(candidate)

    with pytest.raises(ValueError, match="digest|artifact|candidate|replay"):
        module.build_universal_trade_rl_u2_paired_replay_scope_evidence(
            candidate_replay=candidate,
            cash_replay=cash,
            u2_contract=replay_fixture.u2_contract,
            time_partition=replay_fixture.partition,
            checkpoint_closure=closure,
        )


def test_u2_pairing_module_rejects_source_replay_with_stale_artifact_digest(
    replay_fixture: ReplayIntegrationFixture,
) -> None:
    candidate, cash, closure = _candidate_cash(replay_fixture)
    _drift_replay_digest(candidate)

    with pytest.raises(ValueError, match="digest|artifact|candidate|replay"):
        pairing_module.build_universal_trade_rl_u2_paired_replay_scope_evidence(
            candidate_replay=candidate,
            cash_replay=cash,
            u2_contract=replay_fixture.u2_contract,
            time_partition=replay_fixture.partition,
            checkpoint_closure=closure,
        )


def test_u2_final_selection_rejects_nested_pair_with_stale_artifact_digest() -> None:
    bundle = _final_fixture()
    robustness = bundle[-1]
    pair = robustness[0].paired_scope_evidence[0]
    object.__setattr__(
        pair,
        "candidate_replay_evidence_digest",
        content_digest({"drift": "nested-candidate-evidence"}),
    )
    _assert_stale_digest(pair)

    with pytest.raises(ValueError, match="digest|artifact|pair|evidence|identity"):
        _build_final(bundle)


def test_u2_final_selection_rejects_primary_summary_with_stale_artifact_digest() -> (
    None
):
    bundle = _final_fixture()
    primary = bundle[-2]
    summary = primary[0].summary
    object.__setattr__(
        summary,
        "symbol_balanced_net_log_growth",
        summary.symbol_balanced_net_log_growth + 0.123,
    )
    _assert_stale_digest(summary)

    with pytest.raises(ValueError, match="digest|artifact|summary|primary|identity"):
        _build_final(bundle)


def test_u2_final_selection_rejects_base_lock_with_stale_artifact_digest() -> None:
    bundle = _final_fixture()
    base_lock = bundle[1]
    object.__setattr__(
        base_lock,
        "source_tree_digest",
        content_digest({"drift": "base-lock-source-tree"}),
    )
    _assert_stale_digest(base_lock)

    with pytest.raises(ValueError, match="digest|artifact|lock|identity"):
        _build_final(bundle)
