from __future__ import annotations

import importlib
from pathlib import Path


def test_runtime_factory_descriptor_binds_implementation_source_bytes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from trade_rl.integrations.runtime_factory import (
        describe_runtime_factory,
        load_runtime_factory,
    )

    module_path = tmp_path / "runtime_fixture.py"
    module_path.write_text(
        "def build_runtime(**kwargs):\n    return kwargs\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    factory = load_runtime_factory("runtime_fixture:build_runtime")
    first = describe_runtime_factory(
        "runtime_fixture:build_runtime",
        factory=factory,
    )

    module_path.write_text(
        "def build_runtime(**kwargs):\n    return dict(kwargs)\n",
        encoding="utf-8",
    )
    second = describe_runtime_factory(
        "runtime_fixture:build_runtime",
        factory=factory,
    )

    assert first.spec == "runtime_fixture:build_runtime"
    assert first.module == "runtime_fixture"
    assert first.callable_name == "build_runtime"
    assert first.implementation_digest != second.implementation_digest
    assert first.digest != second.digest


def test_full_research_cli_passes_runtime_factory_evidence() -> None:
    source = Path("scripts/run_universal_full_research.py").read_text(encoding="utf-8")

    assert "describe_runtime_factory" in source
    assert "runtime_factory_descriptor=" in source
