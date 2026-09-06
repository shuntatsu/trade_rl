from __future__ import annotations

from dataclasses import replace

import pytest

from tests.workflows.test_universal_trade_rl_u2_development_closure import (
    _checkpoint_members,
    _exposure_rows,
    _predevelopment_bundle,
    _role_symbols,
)
from tests.workflows.test_universal_trade_rl_u2_selection_gate import (
    _inclusive_boundary_summary,
    _leaf,
    _summary,
)
from tests.workflows.test_universal_trade_rl_u2_selection_robustness import (
    _scope_leaves,
    _segments,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.workflows.universal_trade_rl_u2_development_closure import (
    build_authoritative_universal_trade_rl_u2_development_lock,
    build_universal_trade_rl_u2_final_checkpoint_closure,
    build_universal_trade_rl_u2_training_exposure_evidence,
)
from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    build_universal_trade_rl_u2_development_lock,
)

_MANDATORY_CELLS = ("B", "C1", "C2", "D1", "D2")
_ROBUSTNESS_SCOPES = ("D1", "D2", "D1+D2")


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _lock_bundle():
    manifest, u2_contract, predevelopment = _predevelopment_bundle()
    checkpoint_closure = build_universal_trade_rl_u2_final_checkpoint_closure(
        predevelopment_contract=predevelopment,
        u2_contract=u2_contract,
        members=_checkpoint_members(u2_contract=u2_contract),
    )
    train_symbols = _role_symbols(manifest, UniversalTradeRLSymbolRole.TRAIN)
    development_symbols = _role_symbols(
        manifest,
        UniversalTradeRLSymbolRole.DEVELOPMENT,
    )
    exposure = build_universal_trade_rl_u2_training_exposure_evidence(
        predevelopment_contract=predevelopment,
        manifest=manifest,
        u2_contract=u2_contract,
        rows=_exposure_rows(train_symbols=train_symbols),
    )
    evaluation_symbols = tuple(sorted((*train_symbols, *development_symbols)))
    base_lock = build_universal_trade_rl_u2_development_lock(
        predevelopment_contract=predevelopment,
        u1_contract_digest=u2_contract.u1_contract_digest,
        u1_normalizer_digest=u2_contract.u1_normalizer_digest,
        checkpoint_digests=checkpoint_closure.checkpoint_digests,
        development_scope_closure_digest=content_digest(
            {"fixture": "selection-scope-closure"}
        ),
        evaluation_dataset_digests=tuple(
            (symbol, content_digest({"fixture": "selection-view", "symbol": symbol}))
            for symbol in evaluation_symbols
        ),
        source_tree_digest=content_digest({"fixture": "selection-source-tree"}),
        lockfile_digest=content_digest({"fixture": "selection-lockfile"}),
        evaluation_runtime_identity_digest=content_digest(
            {"fixture": "selection-runtime"}
        ),
        development_numeric_open_count=0,
        admission_numeric_open_count=0,
    )
    authoritative_lock = build_authoritative_universal_trade_rl_u2_development_lock(
        base_lock=base_lock,
        checkpoint_closure=checkpoint_closure,
        training_exposure_evidence=exposure,
        manifest=manifest,
        u2_contract=u2_contract,
    )
    return checkpoint_closure, base_lock, authoritative_lock


def _primary_gates():
    module = _module()
    return tuple(
        module.evaluate_universal_trade_rl_u2_primary_cell_gate(
            summary=_inclusive_boundary_summary(cell=cell)
        )
        for cell in _MANDATORY_CELLS
    )


def _failed_primary_gate(*, cell: str):
    module = _module()
    summary = _summary(
        _leaf(
            label=f"{cell}-final-fail-a",
            cell=cell,
            net_log_growth=0.0,
            gross_log_growth=0.02,
        ),
        _leaf(
            label=f"{cell}-final-fail-b",
            cell=cell,
            net_log_growth=0.0,
            gross_log_growth=0.02,
        ),
    )
    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)
    assert result.passed is False
    return result


def _robustness_results():
    module = _module()
    segments = _segments()
    return (
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D1",
            leaves=_scope_leaves("D1"),
            segments=segments,
        ),
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D2",
            leaves=_scope_leaves("D2"),
            segments=segments,
        ),
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D1+D2",
            leaves=_scope_leaves("D1") + _scope_leaves("D2"),
            segments=segments,
        ),
    )


def _failed_robustness(*, scope: str):
    module = _module()
    if scope == "D1":
        segments = _segments(d1_values=(0.0, 0.0, 0.0, 0.0))
        leaves = _scope_leaves("D1")
    elif scope == "D2":
        segments = _segments(d2_values=(0.0, 0.0, 0.0, 0.0))
        leaves = _scope_leaves("D2")
    else:
        segments = _segments(
            d1_values=(0.0, 0.0, 0.0, 0.0),
            d2_values=(0.0, 0.0, 0.0, 0.0),
        )
        leaves = _scope_leaves("D1") + _scope_leaves("D2")
    result = module.evaluate_universal_trade_rl_u2_seed_robustness(
        scope=scope,
        leaves=leaves,
        segments=segments,
    )
    assert result.passed is False
    return result


def _build(*, primary_cell_gates=None, seed_robustness=None, lock_bundle=None):
    module = _module()
    checkpoint_closure, base_lock, authoritative_lock = lock_bundle or _lock_bundle()
    return module.build_universal_trade_rl_u2_development_selection_evidence(
        primary_cell_gates=(
            _primary_gates() if primary_cell_gates is None else primary_cell_gates
        ),
        seed_robustness=(
            _robustness_results() if seed_robustness is None else seed_robustness
        ),
        checkpoint_closure=checkpoint_closure,
        base_lock=base_lock,
        authoritative_lock=authoritative_lock,
    )


def test_u2_final_selection_all_pass_selects_only_primary_exact_final_checkpoint() -> (
    None
):
    checkpoint_closure, base_lock, authoritative_lock = _lock_bundle()
    evidence = _build(lock_bundle=(checkpoint_closure, base_lock, authoritative_lock))

    seed0_checkpoint = dict(checkpoint_closure.checkpoint_digests)[0]
    assert evidence.primary_cells == _MANDATORY_CELLS
    assert evidence.robustness_scopes == _ROBUSTNESS_SCOPES
    assert evidence.passed is True
    assert evidence.admission_eligible is True
    assert evidence.selected_checkpoint_digest == seed0_checkpoint
    assert evidence.promotion_eligible is False
    assert evidence.base_lock_digest == base_lock.digest
    assert evidence.authoritative_lock_digest == authoritative_lock.digest
    assert evidence.final_checkpoint_closure_digest == checkpoint_closure.digest
    assert len(evidence.digest) == 64


@pytest.mark.parametrize("failed_cell", _MANDATORY_CELLS)
def test_u2_final_selection_ands_every_primary_cell_gate(failed_cell: str) -> None:
    gates = tuple(
        _failed_primary_gate(cell=cell) if cell == failed_cell else gate
        for cell, gate in zip(_MANDATORY_CELLS, _primary_gates(), strict=True)
    )

    evidence = _build(primary_cell_gates=gates)

    assert evidence.passed is False
    assert evidence.admission_eligible is False
    assert evidence.selected_checkpoint_digest is None
    assert evidence.promotion_eligible is False


@pytest.mark.parametrize("failed_scope", _ROBUSTNESS_SCOPES)
def test_u2_final_selection_ands_every_seed_robustness_gate(failed_scope: str) -> None:
    robustness = tuple(
        _failed_robustness(scope=scope) if scope == failed_scope else result
        for scope, result in zip(
            _ROBUSTNESS_SCOPES,
            _robustness_results(),
            strict=True,
        )
    )

    evidence = _build(seed_robustness=robustness)

    assert evidence.passed is False
    assert evidence.admission_eligible is False
    assert evidence.selected_checkpoint_digest is None
    assert evidence.promotion_eligible is False


def test_u2_final_selection_rejects_missing_or_reordered_primary_cells() -> None:
    gates = _primary_gates()
    with pytest.raises(ValueError, match="cell|B|C1|C2|D1|D2|closure|order"):
        _build(primary_cell_gates=gates[:-1])
    with pytest.raises(ValueError, match="cell|B|C1|C2|D1|D2|closure|order"):
        _build(primary_cell_gates=(gates[1], gates[0], *gates[2:]))


def test_u2_final_selection_rejects_missing_or_reordered_robustness_scopes() -> None:
    robustness = _robustness_results()
    with pytest.raises(ValueError, match="D1|D2|scope|closure|order"):
        _build(seed_robustness=robustness[:-1])
    with pytest.raises(ValueError, match="D1|D2|scope|closure|order"):
        _build(
            seed_robustness=(
                robustness[1],
                robustness[0],
                robustness[2],
            )
        )


def test_u2_final_selection_rejects_seed0_checkpoint_substitution() -> None:
    checkpoint_closure, base_lock, authoritative_lock = _lock_bundle()
    substituted = replace(
        checkpoint_closure,
        checkpoint_digests=(
            (
                0,
                content_digest({"fixture": "substituted-primary-final-checkpoint"}),
            ),
            *checkpoint_closure.checkpoint_digests[1:],
        ),
        digest="",
    )

    with pytest.raises(ValueError, match="checkpoint|closure|mapping|identity|seed"):
        _build(lock_bundle=(substituted, base_lock, authoritative_lock))


def test_u2_final_selection_rejects_admission_open_drift_before_selection() -> None:
    checkpoint_closure, base_lock, authoritative_lock = _lock_bundle()
    object.__setattr__(base_lock, "admission_numeric_open_count", 1)

    with pytest.raises(ValueError, match="Admission|admission|open|lock|sealed"):
        _build(lock_bundle=(checkpoint_closure, base_lock, authoritative_lock))


def test_u2_final_selection_rejects_authoritative_lock_child_identity_drift() -> None:
    checkpoint_closure, base_lock, authoritative_lock = _lock_bundle()
    object.__setattr__(
        authoritative_lock,
        "final_checkpoint_closure_digest",
        content_digest({"fixture": "wrong-final-checkpoint-closure"}),
    )

    with pytest.raises(
        ValueError, match="authoritative|checkpoint|closure|digest|identity"
    ):
        _build(lock_bundle=(checkpoint_closure, base_lock, authoritative_lock))
