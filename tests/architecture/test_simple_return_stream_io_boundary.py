"""The filesystem owner reuses the frozen reader without a trading runtime."""

import ast
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

PACKAGE = Path(__file__).resolve().parents[2] / "trade_rl"
OWNER = PACKAGE / "strategies" / "forecasts" / "simple_stream_io.py"


def test_stream_filesystem_owner_does_not_import_data_build_or_trading_runtime():
    imports = ImportCollector(PACKAGE).collect(OWNER)
    assert not any(
        within_module(module, prefix)
        for module in imports
        for prefix in (
            "trade_rl.data",
            "trade_rl.evaluation",
            "trade_rl.risk",
            "trade_rl.simulation",
            "trade_rl.strategies.rl",
            "trade_rl.strategies.forecasts.simple_prequential",
        )
    )
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(ast.parse(OWNER.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert {
        "canonical_json_bytes",
        "open_regular_binary",
        "from_payload",
        "link",
    } <= calls
    assert calls.isdisjoint(
        {
            "fit",
            "predict",
            "fit_prequential_simple_ridge",
            "replace",
            "rename",
            "execute_interval",
            "apply_fill",
        }
    )


def test_stream_file_exports_are_family_local_and_use_the_direct_owner():
    from trade_rl import strategies
    from trade_rl.strategies import forecasts
    from trade_rl.strategies.forecasts import simple_stream_io

    assert set(simple_stream_io.__all__) == {
        "PublishedSimpleReturnStreamArtifact",
        "load_simple_return_stream_artifact",
        "publish_simple_return_stream_artifact",
    }
    for name in simple_stream_io.__all__:
        assert getattr(forecasts, name) is getattr(simple_stream_io, name)
        assert not hasattr(strategies, name)
