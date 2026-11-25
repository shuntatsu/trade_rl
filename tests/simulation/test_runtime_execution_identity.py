"""Source-free execution identity contract; no market reads or replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from tests.integrations.test_binance_market_order_profile import _dataset, _profile
from trade_rl.simulation import (
    ExecutionCostConfig,
    MarketExecutor,
    runtime_execution_policy_digest,
)
from trade_rl.simulation.execution import ExecutionRuleStress


def _sha(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


@pytest.mark.parametrize("funding", [False, True])
@pytest.mark.parametrize("insolvency", ["floor_zero", "retain_debt"])
@pytest.mark.parametrize(
    "variant",
    [
        "normal",
        "double_cost",
        "latency",
        "stress",
        "profile",
        "profile_stress",
        "strict_spot",
    ],
)
def test_source_free_identity_matches_native_and_literal_wrappers(
    funding: bool, insolvency: Any, variant: str
) -> None:
    data = _dataset()
    if funding:
        data = replace(
            data,
            dataset_id="d" * 64,
            identity_payload_json=None,
            funding_price_rate=np.zeros_like(data.close),
        ).with_content_identity()
    cost = ExecutionCostConfig()
    stress = ExecutionRuleStress()
    profile = None
    strict = variant == "strict_spot"
    if variant == "double_cost":
        cost = replace(
            cost,
            fee_rate=2 * cost.fee_rate,
            spread_rate=2 * cost.spread_rate,
            impact_rate=2 * cost.impact_rate,
        )
    elif variant == "latency":
        cost = replace(cost, order_latency_bars=2)
    elif variant in ("stress", "profile_stress"):
        stress = ExecutionRuleStress(name="lot", lot_size_factor=2.0)
    if variant in ("profile", "profile_stress"):
        profile = _profile(data)
    if strict:
        cost = replace(cost, allow_short=False)
    profile_digest = profile.digest if profile is not None else None
    expected = cost.execution_policy_digest
    if profile_digest is not None:
        expected = _sha(
            {
                "schema_version": "profile_market_execution_v1",
                "base_policy_digest": expected,
                "market_order_profile_digest": profile_digest,
                "rule_stress": stress.digest_payload(),
            }
        )
    elif stress.enabled:
        expected = _sha(
            {
                "schema_version": "stressed_execution_policy_v1",
                "base_policy_digest": expected,
                "rule_stress": stress.digest_payload(),
            }
        )
    if insolvency == "retain_debt":
        expected = _sha(
            {
                "schema_version": "insolvency_execution_policy_v2",
                "base_policy_digest": expected,
                "insolvency_valuation": insolvency,
                "loss_metric_semantics": "unbounded_simple_return_drawdown_v1",
            }
        )
    if funding:
        expected = _sha(
            {
                "schema_version": "funding_execution_policy_v1",
                "base_policy_digest": expected,
                "funding_semantics": "dataset_funding_price_rate_v1",
            }
        )
    if strict:
        expected = _sha(
            {
                "schema_version": "strict_spot_cash_execution_v1",
                "base_policy_digest": expected,
                "unaffordable_fill": "no_fill",
            }
        )
    actual = runtime_execution_policy_digest(
        cost,
        funding_price_rate_present=funding,
        market_order_profile_digest=profile_digest,
        rule_stress=stress,
        insolvency_valuation=insolvency,
        strict_spot_cash=strict,
    )
    executor = MarketExecutor(
        data,
        cost,
        rule_stress=stress,
        market_order_profile=profile,
        insolvency_valuation=insolvency,
        strict_spot_cash=strict,
    )
    assert actual == executor.execution_policy_digest == expected
    assert executor.execution_policy_digest == expected  # cache hit


@pytest.mark.parametrize(
    "changes",
    [
        {"funding_price_rate_present": None},
        {"funding_price_rate_present": 1},
        {"funding_price_rate_present": np.bool_(True)},
        {"strict_spot_cash": 1},
        {"insolvency_valuation": "unknown"},
        {"market_order_profile_digest": "A" * 64},
        {"market_order_profile_digest": "0" * 64},
        {"market_order_profile_digest": "not-a-digest"},
        {"rule_stress": object()},
    ],
)
def test_source_free_metadata_rejects_ambiguous_or_malformed_values(
    changes: dict[str, Any],
) -> None:
    fields = {"funding_price_rate_present": False, **changes}
    with pytest.raises(ValueError):
        runtime_execution_policy_digest(ExecutionCostConfig(), **fields)


def test_source_free_identity_requires_funding_presence_and_binds_scenario() -> None:
    with pytest.raises(TypeError):
        runtime_execution_policy_digest(ExecutionCostConfig())  # type: ignore[call-arg]
    cost = ExecutionCostConfig()
    doubled = replace(
        cost,
        fee_rate=2 * cost.fee_rate,
        spread_rate=2 * cost.spread_rate,
        impact_rate=2 * cost.impact_rate,
    )
    normal = runtime_execution_policy_digest(cost, funding_price_rate_present=False)
    other = runtime_execution_policy_digest(doubled, funding_price_rate_present=False)
    assert normal != other
    assert normal == cost.execution_policy_digest
    assert normal != runtime_execution_policy_digest(
        cost, funding_price_rate_present=True
    )


def test_source_free_rejects_incompatible_nondata_execution_settings() -> None:
    with pytest.raises(ValueError, match="long-only"):
        runtime_execution_policy_digest(
            ExecutionCostConfig(),
            funding_price_rate_present=False,
            strict_spot_cash=True,
        )
    with pytest.raises(ValueError, match="MARKET"):
        runtime_execution_policy_digest(
            ExecutionCostConfig(order_type="limit"),
            funding_price_rate_present=False,
            market_order_profile_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="ExecutionCostConfig"):
        runtime_execution_policy_digest(object(), funding_price_rate_present=False)  # type: ignore[arg-type]


def test_runtime_cache_reuses_owner_after_metadata_changes() -> None:
    data = _dataset()
    executor = MarketExecutor(data)
    assert executor.execution_policy_digest == runtime_execution_policy_digest(
        executor.cost, funding_price_rate_present=False
    )
    executor.cost = replace(executor.cost, fee_rate=0.004)
    executor.insolvency_valuation = "retain_debt"
    executor.rule_stress = ExecutionRuleStress(name="lot", lot_size_factor=2)
    assert executor.execution_policy_digest == runtime_execution_policy_digest(
        executor.cost,
        funding_price_rate_present=False,
        rule_stress=executor.rule_stress,
        insolvency_valuation="retain_debt",
    )
