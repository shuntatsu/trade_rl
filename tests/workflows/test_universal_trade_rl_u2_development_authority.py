from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.workflows.test_universal_trade_rl_u2_contract import _u1_contract
from tests.workflows.test_universal_trade_rl_u2_development_closure import (
    _checkpoint_members,
    _exposure_rows,
    _predevelopment_bundle,
    _role_symbols,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.workflows.universal_trade_rl_u2_evaluation import (
    UniversalTradeRLU2DevelopmentScopeClosure,
    build_universal_trade_rl_u2_development_scope_closure,
)
from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    UniversalTradeRLU2DevelopmentLock,
    build_universal_trade_rl_u2_development_lock,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    build_universal_trade_rl_u2_time_partition,
)


def _canonical_evaluation_dataset_digests(
    scope_closure: UniversalTradeRLU2DevelopmentScopeClosure,
) -> tuple[tuple[str, str], ...]:
    by_symbol: dict[str, str] = {}
    for scope in scope_closure.scopes:
        previous = by_symbol.setdefault(
            scope.concrete_symbol,
            scope.evaluation_dataset_digest,
        )
        if previous != scope.evaluation_dataset_digest:
            raise AssertionError(
                "fixture scope closure has inconsistent dataset identity"
            )
    return tuple(sorted(by_symbol.items()))


def _development_bundle():
    from trade_rl.workflows import universal_trade_rl_u2_development_closure as closure

    manifest, u2_contract, predevelopment = _predevelopment_bundle()
    partition = build_universal_trade_rl_u2_time_partition(manifest=manifest)
    scope_closure = build_universal_trade_rl_u2_development_scope_closure(
        manifest=manifest,
        time_partition=partition,
        u2_contract=u2_contract,
    )
    checkpoint_closure = closure.build_universal_trade_rl_u2_final_checkpoint_closure(
        predevelopment_contract=predevelopment,
        u2_contract=u2_contract,
        members=_checkpoint_members(u2_contract=u2_contract),
    )
    train_symbols = _role_symbols(manifest, UniversalTradeRLSymbolRole.TRAIN)
    exposure = closure.build_universal_trade_rl_u2_training_exposure_evidence(
        predevelopment_contract=predevelopment,
        manifest=manifest,
        u2_contract=u2_contract,
        rows=_exposure_rows(train_symbols=train_symbols),
    )
    base_lock = build_universal_trade_rl_u2_development_lock(
        predevelopment_contract=predevelopment,
        u1_contract_digest=u2_contract.u1_contract_digest,
        u1_normalizer_digest=u2_contract.u1_normalizer_digest,
        checkpoint_digests=checkpoint_closure.checkpoint_digests,
        development_scope_closure_digest=scope_closure.digest,
        evaluation_dataset_digests=_canonical_evaluation_dataset_digests(scope_closure),
        source_tree_digest=content_digest({"fixture": "authority-source-tree"}),
        lockfile_digest=content_digest({"fixture": "authority-lockfile"}),
        evaluation_runtime_identity_digest=content_digest(
            {"fixture": "authority-runtime"}
        ),
        development_numeric_open_count=0,
        admission_numeric_open_count=0,
    )
    return (
        closure,
        manifest,
        partition,
        u2_contract,
        predevelopment,
        scope_closure,
        checkpoint_closure,
        exposure,
        base_lock,
    )


def _authoritative_lock():
    (
        closure,
        manifest,
        partition,
        u2_contract,
        predevelopment,
        scope_closure,
        checkpoint_closure,
        exposure,
        base_lock,
    ) = _development_bundle()
    authoritative_lock = (
        closure.build_authoritative_universal_trade_rl_u2_development_lock(
            base_lock=base_lock,
            checkpoint_closure=checkpoint_closure,
            training_exposure_evidence=exposure,
            manifest=manifest,
            u2_contract=u2_contract,
            scope_closure=scope_closure,
        )
    )
    return (
        manifest,
        partition,
        u2_contract,
        predevelopment,
        scope_closure,
        base_lock,
        authoritative_lock,
    )


def test_u2_authoritative_lock_binds_exact_scope_and_dataset_identities() -> None:
    (
        closure,
        manifest,
        _partition,
        u2_contract,
        _predevelopment,
        scope_closure,
        checkpoint_closure,
        exposure,
        base_lock,
    ) = _development_bundle()

    wrong_dataset_mapping = list(base_lock.evaluation_dataset_digests)
    symbol, _digest = wrong_dataset_mapping[0]
    wrong_dataset_mapping[0] = (
        symbol,
        content_digest({"fixture": "wrong-evaluation-view", "symbol": symbol}),
    )
    drifted_dataset_lock = replace(
        base_lock,
        evaluation_dataset_digests=tuple(wrong_dataset_mapping),
        digest="",
    )
    with pytest.raises(ValueError, match="evaluation|dataset|digest|identity|scope"):
        closure.build_authoritative_universal_trade_rl_u2_development_lock(
            base_lock=drifted_dataset_lock,
            checkpoint_closure=checkpoint_closure,
            training_exposure_evidence=exposure,
            manifest=manifest,
            u2_contract=u2_contract,
            scope_closure=scope_closure,
        )

    drifted_scope_lock = replace(
        base_lock,
        development_scope_closure_digest=content_digest(
            {"fixture": "wrong-development-scope"}
        ),
        digest="",
    )
    with pytest.raises(ValueError, match="scope|closure|digest|identity"):
        closure.build_authoritative_universal_trade_rl_u2_development_lock(
            base_lock=drifted_scope_lock,
            checkpoint_closure=checkpoint_closure,
            training_exposure_evidence=exposure,
            manifest=manifest,
            u2_contract=u2_contract,
            scope_closure=scope_closure,
        )


def test_u2_authoritative_session_gate_blocks_before_numeric_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.workflows import (
        universal_trade_rl_u2_development_authority as authority,
    )

    (
        manifest,
        partition,
        u2_contract,
        predevelopment,
        scope_closure,
        base_lock,
        authoritative_lock,
    ) = _authoritative_lock()
    numeric_opens: list[object] = []
    delegated: list[dict[str, object]] = []

    def source_loader(locator: object):
        numeric_opens.append(locator)
        return object()

    def fake_lower_builder(**kwargs: object):
        delegated.append(dict(kwargs))
        loader = kwargs["source_loader"]
        assert callable(loader)
        loader("fixture://authorized-development")
        return object()

    monkeypatch.setattr(
        authority,
        "build_universal_trade_rl_u2_development_replay_session",
        fake_lower_builder,
    )

    drifted_base_lock: UniversalTradeRLU2DevelopmentLock = replace(
        base_lock,
        development_scope_closure_digest=content_digest(
            {"fixture": "pre-open-scope-drift"}
        ),
        digest="",
    )
    with pytest.raises(ValueError, match="lock|scope|closure|digest|identity"):
        authority.build_authoritative_universal_trade_rl_u2_development_replay_session(
            predevelopment_contract=predevelopment,
            base_lock=drifted_base_lock,
            development_lock=authoritative_lock,
            manifest=manifest,
            time_partition=partition,
            u2_contract=u2_contract,
            u1_contract=SimpleNamespace(digest=u2_contract.u1_contract_digest),
            policy_contract=object(),
            normalizer=SimpleNamespace(digest=u2_contract.u1_normalizer_digest),
            supplied_scope_closure=scope_closure,
            artifact_locators={},
            source_loader=source_loader,
            environment_factory=lambda _dataset: object(),
            source_tree_digest=drifted_base_lock.source_tree_digest,
            lockfile_digest=drifted_base_lock.lockfile_digest,
            evaluation_runtime_identity_digest=(
                drifted_base_lock.evaluation_runtime_identity_digest
            ),
        )

    assert delegated == []
    assert numeric_opens == []


def test_u2_authoritative_session_gate_delegates_only_after_exact_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.workflows import (
        universal_trade_rl_u2_development_authority as authority,
    )

    (
        manifest,
        partition,
        u2_contract,
        predevelopment,
        scope_closure,
        base_lock,
        authoritative_lock,
    ) = _authoritative_lock()
    u1_contract = _u1_contract(manifest=manifest, fit_end_ns=partition.fit_end_ns)
    numeric_opens: list[object] = []
    delegated: list[dict[str, object]] = []
    sentinel_session = object()

    def source_loader(locator: object):
        numeric_opens.append(locator)
        return object()

    def fake_lower_builder(**kwargs: object):
        delegated.append(dict(kwargs))
        loader = kwargs["source_loader"]
        assert callable(loader)
        loader("fixture://authorized-development")
        return sentinel_session

    monkeypatch.setattr(
        authority,
        "build_universal_trade_rl_u2_development_replay_session",
        fake_lower_builder,
    )

    result = (
        authority.build_authoritative_universal_trade_rl_u2_development_replay_session(
            predevelopment_contract=predevelopment,
            base_lock=base_lock,
            development_lock=authoritative_lock,
            manifest=manifest,
            time_partition=partition,
            u2_contract=u2_contract,
            u1_contract=u1_contract,
            policy_contract=object(),
            normalizer=SimpleNamespace(digest=u2_contract.u1_normalizer_digest),
            supplied_scope_closure=scope_closure,
            artifact_locators={},
            source_loader=source_loader,
            environment_factory=lambda _dataset: object(),
            source_tree_digest=base_lock.source_tree_digest,
            lockfile_digest=base_lock.lockfile_digest,
            evaluation_runtime_identity_digest=(
                base_lock.evaluation_runtime_identity_digest
            ),
        )
    )

    assert result is sentinel_session
    assert len(delegated) == 1
    assert numeric_opens == ["fixture://authorized-development"]
