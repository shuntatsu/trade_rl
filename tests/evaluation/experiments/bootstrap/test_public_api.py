from __future__ import annotations

import trade_rl.evaluation.experiments as experiments
import trade_rl.evaluation.experiments.bootstrap as bootstrap

EXPECTED_BOOTSTRAP_API = {
    "CanonicalM2BootstrapConfig",
    "CanonicalM2BootstrapResult",
    "bootstrap_canonical_m2_study",
    "inspect_canonical_m2_bootstrap",
}


def test_bootstrap_package_has_exact_public_api() -> None:
    assert set(bootstrap.__all__) == EXPECTED_BOOTSTRAP_API
    for name in EXPECTED_BOOTSTRAP_API:
        assert hasattr(bootstrap, name), name


def test_experiments_facade_exposes_only_high_level_bootstrap_contract() -> None:
    for name in EXPECTED_BOOTSTRAP_API:
        assert name in experiments.__all__
        assert hasattr(experiments, name), name
    for private_name in (
        "FrozenBinanceSource",
        "_freeze_binance_source",
        "_inspect_frozen_binance_source",
    ):
        assert private_name not in experiments.__all__
        assert not hasattr(experiments, private_name)
