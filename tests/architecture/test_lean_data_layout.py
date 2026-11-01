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
        "features/price_channels.py",
        "view.py",
        "market_order_rules.py",
        "_timestamps.py",
    ):
        assert (DATA / relative).is_file(), relative


def test_exact_datetime_normalizer_has_one_internal_data_owner() -> None:
    import ast

    import trade_rl.data as data

    module_path = DATA / "_timestamps.py"
    assert module_path.is_file()
    owners: list[Path] = []
    for source_path in (ROOT / "trade_rl").rglob("*.py"):
        module = ast.parse(
            source_path.read_text(encoding="utf-8"), filename=str(source_path)
        )
        owners.extend(
            source_path
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_datetime64_values_to_ns_exact"
        )
    assert owners == [module_path]
    assert "_datetime64_values_to_ns_exact" not in data.__all__
    assert not hasattr(data, "_datetime64_values_to_ns_exact")

    expected_callers = (
        ROOT / "trade_rl" / "data" / "market.py",
        ROOT / "trade_rl" / "data" / "source.py",
        ROOT / "trade_rl" / "evaluation" / "runs" / "candidate_suite.py",
        ROOT / "trade_rl" / "evaluation" / "runs" / "config.py",
        ROOT / "trade_rl" / "strategies" / "forecasts" / "supervised.py",
    )
    for path in expected_callers:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert any(
            isinstance(node, ast.ImportFrom)
            and node.module == "trade_rl.data._timestamps"
            and any(
                alias.name == "_datetime64_values_to_ns_exact" for alias in node.names
            )
            for node in ast.walk(tree)
        ), path


def test_verified_dataset_identity_pair_is_public_through_data_facades() -> None:
    import trade_rl.data as data
    import trade_rl.data.artifacts as artifacts
    from trade_rl.data.artifacts.publication import (
        load_published_market_dataset_artifact,
    )

    assert artifacts.load_published_market_dataset_artifact is (
        load_published_market_dataset_artifact
    )
    assert data.load_published_market_dataset_artifact is (
        load_published_market_dataset_artifact
    )


def test_source_build_package_is_explicitly_unignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!trade_rl/data/build/" in gitignore
    assert "!trade_rl/data/build/*.py" in gitignore


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
