from __future__ import annotations

import math

import pytest

from tests.rl.universal_trade_test_support import make_u1_feature_specs
from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.rl.universal_trade_contract import (
    UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
    UniversalTradePolicyContract,
)


def test_contract_freezes_windows() -> None:
    assert UNIVERSAL_TRADE_SEQUENCE_WINDOWS == (
        ("15m", 96),
        ("1h", 168),
        ("4h", 120),
        ("1d", 60),
    )


def test_contract_digest_is_sha256() -> None:
    contract = UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    assert len(contract.digest) == 64
    assert set(contract.digest) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    "kind",
    (
        FeatureKind.RELATIVE_RETURN_TO_BTC,
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
        FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
        FeatureKind.CROSS_ASSET_DISPERSION,
    ),
)
def test_contract_rejects_cross_asset_feature(kind: FeatureKind) -> None:
    with pytest.raises(ValueError, match="U1 feature"):
        UniversalTradePolicyContract(
            feature_specs=(FeatureSpec(name="15m__forbidden", kind=kind),)
        )


def test_contract_requires_prefix_matching_resolved_timeframe() -> None:
    bad = FeatureSpec(
        name="15m__ret",
        kind=FeatureKind.LOG_RETURN,
        timeframe="1h",
    )
    with pytest.raises(ValueError, match="prefix|timeframe"):
        UniversalTradePolicyContract(feature_specs=(bad,))


def test_contract_binds_feature_order() -> None:
    specs = make_u1_feature_specs()
    normal = UniversalTradePolicyContract(feature_specs=specs)
    reversed_contract = UniversalTradePolicyContract(
        feature_specs=tuple(reversed(specs))
    )
    assert normal.digest != reversed_contract.digest


@pytest.mark.parametrize(
    "scale",
    (0.0, -0.1, 1.01, math.inf, math.nan),
)
def test_contract_rejects_invalid_policy_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="policy_weight_scale"):
        UniversalTradePolicyContract(
            feature_specs=make_u1_feature_specs(),
            policy_weight_scale=scale,
        )


def test_contract_uses_fixed_reward_scale() -> None:
    with pytest.raises(ValueError, match="reward_scale"):
        UniversalTradePolicyContract(
            feature_specs=make_u1_feature_specs(),
            reward_scale=99.0,
        )
