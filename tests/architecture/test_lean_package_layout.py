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
