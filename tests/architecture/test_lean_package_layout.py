from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def test_transitional_domain_package_is_absent() -> None:
    assert not (PACKAGE / "domain").exists()


def test_validation_and_canonical_owners_exist() -> None:
    assert (PACKAGE / "_validation.py").is_file()
    assert (PACKAGE / "artifacts" / "canonical.py").is_file()


def test_evaluation_gate_package_exists() -> None:
    gates = PACKAGE / "evaluation" / "gates"
    assert (gates / "__init__.py").is_file()
    assert (gates / "models.py").is_file()
    assert (gates / "resolve.py").is_file()


def test_forwarding_artifact_codec_is_absent() -> None:
    assert not (PACKAGE / "artifacts" / "codec.py").exists()


def test_strategy_families_have_explicit_packages() -> None:
    assert (PACKAGE / "strategies" / "rules" / "trend.py").is_file()
    assert (PACKAGE / "strategies" / "forecasts" / "ridge.py").is_file()
    assert (PACKAGE / "strategies" / "rl" / "ppo.py").is_file()


def test_binance_is_a_package_not_a_god_module() -> None:
    assert not (PACKAGE / "integrations" / "binance.py").exists()
    for name in ("transport.py", "cache.py", "vision.py", "metadata.py", "dataset.py"):
        assert (PACKAGE / "integrations" / "binance" / name).is_file()
