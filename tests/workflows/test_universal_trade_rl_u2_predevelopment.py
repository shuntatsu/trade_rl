from __future__ import annotations

import inspect
from hashlib import sha256

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLUniverseConfig
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)

_STEP_NS = 15 * 60 * 1_000_000_000
_FIRST_NS = 1_000_000 * _STEP_NS
_U2_DIGEST = "a" * 64
_U1_DIGEST = "b" * 64
_NORMALIZER_DIGEST = "c" * 64
_SCOPE_DIGEST = "d" * 64
_SOURCE_TREE_DIGEST = "e" * 64
_LOCKFILE_DIGEST = "f" * 64
_RUNTIME_DIGEST = sha256(b"runtime").hexdigest()


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_predevelopment

    return universal_trade_rl_u2_predevelopment


def _symbols(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{index:02d}" for index in range(1, count + 1))


def _manifest(
    *,
    train_count: int = 9,
    development_count: int = 3,
    admission_count: int = 3,
) -> UniversalTradeRLUniverseManifest:
    if train_count <= 0:
        raise ValueError("test manifest requires at least one Train symbol")
    train = tuple(sorted(("BTCUSDT", *_symbols("TRN", train_count - 1))))
    development = _symbols("DEV", development_count)
    admission = _symbols("ADM", admission_count)
    config = UniversalTradeRLUniverseConfig(
        train_symbols=train,
        development_symbols=development,
        admission_symbols=admission,
    )
    sources = tuple(
        UniversalTradeRLSymbolSource(
            symbol=symbol,
            dataset_digest=sha256(symbol.encode("utf-8")).hexdigest(),
            first_timestamp_ns=_FIRST_NS,
            last_timestamp_ns=_FIRST_NS + _STEP_NS,
            row_count=2,
        )
        for symbol in sorted((*train, *development, *admission))
    )
    return build_universal_trade_rl_universe_manifest(config=config, sources=sources)


def _predevelopment_contract():
    module = _module()
    manifest = _manifest()
    return module.build_universal_trade_rl_u2_predevelopment_contract(
        manifest=manifest,
        u2_contract_digest=_U2_DIGEST,
        u2_universe_manifest_digest=manifest.digest,
    )


@pytest.mark.parametrize(
    ("train_count", "development_count", "admission_count", "match"),
    (
        (8, 3, 3, "Train|train|9"),
        (9, 2, 3, "Development|development|3"),
        (9, 3, 2, "Admission|admission|3"),
    ),
)
def test_u2_predevelopment_rejects_under_cardinality_roles(
    train_count: int,
    development_count: int,
    admission_count: int,
    match: str,
) -> None:
    module = _module()
    manifest = _manifest(
        train_count=train_count,
        development_count=development_count,
        admission_count=admission_count,
    )

    with pytest.raises(ValueError, match=match):
        module.build_universal_trade_rl_u2_predevelopment_contract(
            manifest=manifest,
            u2_contract_digest=_U2_DIGEST,
            u2_universe_manifest_digest=manifest.digest,
        )


def test_u2_predevelopment_rejects_u2_universe_identity_drift() -> None:
    module = _module()
    manifest = _manifest()

    with pytest.raises(ValueError, match="universe|manifest|identity"):
        module.build_universal_trade_rl_u2_predevelopment_contract(
            manifest=manifest,
            u2_contract_digest=_U2_DIGEST,
            u2_universe_manifest_digest="9" * 64,
        )


def test_u2_evaluation_seed_is_scope_common_and_has_no_training_seed_surface() -> None:
    module = _module()
    function = module.universal_trade_rl_u2_evaluation_seed

    assert tuple(inspect.signature(function).parameters) == (
        "u2_contract_digest",
        "scope_digest",
    )
    first = function(
        u2_contract_digest=_U2_DIGEST,
        scope_digest=_SCOPE_DIGEST,
    )
    second = function(
        u2_contract_digest=_U2_DIGEST,
        scope_digest=_SCOPE_DIGEST,
    )

    assert first == second
    assert first in (0, 1, 2)

    material = {
        "schema_version": "universal_trade_rl_u2_evaluation_crn_v1",
        "u2_contract_digest": _U2_DIGEST,
        "scope_digest": _SCOPE_DIGEST,
        "allowed_evaluation_seeds": (0, 1, 2),
    }
    expected = (0, 1, 2)[int(content_digest(material)[:8], 16) % 3]
    assert first == expected


def test_u2_predevelopment_freezes_exact_metric_bootstrap_resume_and_exposure() -> None:
    contract = _predevelopment_contract()

    metric_payload = {
        "schema_version": "universal_trade_rl_u2_selection_metric_contract_v1",
        "leaf_identity": (
            "training_seed",
            "cell",
            "concrete_symbol",
            "tile_identity",
        ),
        "return_domain": "log1p_simple_returns",
        "gross_net_same_replay_required": True,
        "symbol_log_growth_reduction": "sum_leaves",
        "symbol_balanced_log_growth_reduction": "equal_weight_mean_symbols",
        "wealth_transform": "exp_log_growth",
        "median_symbol_net_wealth": "statistics.median",
        "minimum_symbol_net_wealth": "min",
        "positive_scope_rule": "leaf_net_log_growth_strictly_greater_than_zero",
        "cvar10_count_rule": "max_1_ceil_0.10_times_leaf_count",
        "cvar10_value_rule": "mean_worst_leaf_net_log_growth",
        "turnover_per_day_rule": "turnover_total/(decision_count*0.25/24)",
        "turnover_p95_quantile": 0.95,
        "turnover_p95_method": "linear",
        "meaningful_execution_rule": (
            "executed_change_count_gt_0_or_turnover_total_gt_1e-6"
        ),
        "meaningful_execution_turnover_tolerance": 1e-6,
        "positive_gross_retention_rule": (
            "symbol_balanced_net_log_growth/symbol_balanced_gross_log_growth"
        ),
    }
    bootstrap_payload = {
        "schema_version": "universal_trade_rl_u2_bootstrap_panel_contract_v1",
        "paired_quantity": "candidate_minus_cash_net_log_excess",
        "symbol_reduction": "equal_weight_mean_per_timestamp",
        "seed_reduction": "median_per_timestamp",
        "time_order": "chronological",
        "aggregate_segments": ("development_future_1", "development_future_2"),
        "blocks_may_cross_segment_boundary": False,
        "bootstrap_method": "moving_block_mean_test",
        "confidence_level": 0.95,
        "resamples": 2_000,
        "bootstrap_seed": 0,
        "block_length_rule": "ceil_sqrt_segment_length_capped",
        "quantile_method": "linear",
        "pass_rule": "lower_95pct_ci_strictly_greater_than_zero",
    }
    resume_payload = {
        "schema_version": "universal_trade_rl_u2_resume_contract_v1",
        "exact_mid_episode_resume_supported": False,
        "restart_from_timestep_zero_required": True,
        "intermediate_checkpoint_role": "recovery_debug_evidence_only",
        "selection_checkpoint_rule": "exact_final_fixed_budget_only",
    }
    exposure_payload = {
        "schema_version": "universal_trade_rl_u2_training_exposure_contract_v1",
        "required_fields": (
            "training_seed",
            "worker_index",
            "concrete_symbol",
            "completed_episode_count",
            "decision_step_count",
            "partial_final_episode_step_count",
            "routing_cycle_count",
        ),
        "posthoc_reweighting_allowed": False,
        "required_before_development_open": True,
    }

    assert contract.selection_metric_contract_digest == content_digest(metric_payload)
    assert contract.bootstrap_panel_contract_digest == content_digest(bootstrap_payload)
    assert contract.resume_contract_digest == content_digest(resume_payload)
    assert contract.training_exposure_contract_digest == content_digest(
        exposure_payload
    )
    assert contract.production_status == "NO-GO"
    assert contract.admission_status == "SEALED"

    restored = type(contract).from_payload(contract.to_payload())
    assert restored == contract
    assert restored.digest == contract.digest


def _lock_kwargs() -> dict[str, object]:
    contract = _predevelopment_contract()
    return {
        "predevelopment_contract": contract,
        "u1_contract_digest": _U1_DIGEST,
        "u1_normalizer_digest": _NORMALIZER_DIGEST,
        "checkpoint_digests": (
            (0, sha256(b"seed0").hexdigest()),
            (1, sha256(b"seed1").hexdigest()),
            (2, sha256(b"seed2").hexdigest()),
        ),
        "development_scope_closure_digest": _SCOPE_DIGEST,
        "evaluation_dataset_digests": (
            ("DEV01", sha256(b"DEV01-view").hexdigest()),
            ("DEV02", sha256(b"DEV02-view").hexdigest()),
            ("DEV03", sha256(b"DEV03-view").hexdigest()),
        ),
        "source_tree_digest": _SOURCE_TREE_DIGEST,
        "lockfile_digest": _LOCKFILE_DIGEST,
        "evaluation_runtime_identity_digest": _RUNTIME_DIGEST,
        "development_numeric_open_count": 0,
        "admission_numeric_open_count": 0,
    }


def test_u2_development_lock_binds_exact_final_checkpoint_and_runtime_closure() -> None:
    module = _module()
    lock = module.build_universal_trade_rl_u2_development_lock(**_lock_kwargs())

    assert tuple(seed for seed, _ in lock.checkpoint_digests) == (0, 1, 2)
    assert lock.development_numeric_open_count == 0
    assert lock.admission_numeric_open_count == 0
    assert lock.admission_status == "SEALED"
    assert lock.production_status == "NO-GO"

    restored = type(lock).from_payload(lock.to_payload())
    assert restored == lock
    assert restored.digest == lock.digest


@pytest.mark.parametrize(
    "checkpoint_digests",
    (
        (
            (0, sha256(b"seed0").hexdigest()),
            (2, sha256(b"seed2").hexdigest()),
        ),
        (
            (1, sha256(b"seed1").hexdigest()),
            (0, sha256(b"seed0").hexdigest()),
            (2, sha256(b"seed2").hexdigest()),
        ),
        (
            (0, sha256(b"seed0").hexdigest()),
            (1, sha256(b"seed1").hexdigest()),
            (1, sha256(b"seed1b").hexdigest()),
        ),
    ),
)
def test_u2_development_lock_rejects_noncanonical_checkpoint_closure(
    checkpoint_digests: tuple[tuple[int, str], ...],
) -> None:
    module = _module()
    kwargs = _lock_kwargs()
    kwargs["checkpoint_digests"] = checkpoint_digests

    with pytest.raises(ValueError, match="checkpoint|seed|canonical|exact"):
        module.build_universal_trade_rl_u2_development_lock(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("development_numeric_open_count", 1),
        ("admission_numeric_open_count", 1),
    ),
)
def test_u2_development_lock_rejects_any_prelock_numeric_open(
    field: str,
    value: int,
) -> None:
    module = _module()
    kwargs = _lock_kwargs()
    kwargs[field] = value

    with pytest.raises(ValueError, match="numeric|open|zero|sealed"):
        module.build_universal_trade_rl_u2_development_lock(**kwargs)


def test_u2_development_lock_rejects_unsorted_or_duplicate_dataset_mapping() -> None:
    module = _module()
    kwargs = _lock_kwargs()
    kwargs["evaluation_dataset_digests"] = (
        ("DEV02", sha256(b"DEV02-view").hexdigest()),
        ("DEV01", sha256(b"DEV01-view").hexdigest()),
    )
    with pytest.raises(ValueError, match="dataset|mapping|sorted|canonical"):
        module.build_universal_trade_rl_u2_development_lock(**kwargs)

    kwargs = _lock_kwargs()
    kwargs["evaluation_dataset_digests"] = (
        ("DEV01", sha256(b"DEV01-view").hexdigest()),
        ("DEV01", sha256(b"DEV01-other-view").hexdigest()),
    )
    with pytest.raises(ValueError, match="dataset|mapping|unique|canonical"):
        module.build_universal_trade_rl_u2_development_lock(**kwargs)
