from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "trade_rl" / "data"


def test_data_lifecycle_packages_exist() -> None:
    for relative in (
        "artifacts/__init__.py",
        "artifacts/codec.py",
        "artifacts/publication.py",
        "build/__init__.py",
        "build/config.py",
        "build/builder.py",
        "features/__init__.py",
        "features/core.py",
        "features/cross_asset.py",
        "features/economic.py",
        "features/multitimeframe.py",
        "view.py",
    ):
        assert (DATA / relative).is_file(), relative


def test_flat_legacy_data_modules_are_absent() -> None:
    for name in (
        "artifact.py",
        "artifact_codec.py",
        "artifacts.py",
        "builder.py",
        "config.py",
        "features.py",
        "cross_asset_features.py",
        "economic_semantics.py",
        "multitimeframe.py",
    ):
        assert not (DATA / name).exists(), name
