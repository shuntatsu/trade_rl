from __future__ import annotations

import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _bundle_budget_module():
    return importlib.import_module("guide.tools.bundle_budget")


def test_bundle_budget_rejects_javascript_chunk_above_limit(tmp_path: Path) -> None:
    bundle_budget = _bundle_budget_module()
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (assets / "too-large.js").write_bytes(b"x" * 501)

    try:
        bundle_budget.check_javascript_bundle_budget(dist, max_bytes=500)
    except bundle_budget.BundleBudgetError as exc:
        assert "too-large.js" in str(exc)
        assert "501" in str(exc)
        assert "500" in str(exc)
    else:
        raise AssertionError("expected BundleBudgetError for oversized JavaScript chunk")


def test_bundle_budget_accepts_javascript_chunks_at_or_below_limit(tmp_path: Path) -> None:
    bundle_budget = _bundle_budget_module()
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (assets / "one.js").write_bytes(b"x" * 500)
    (assets / "two.js").write_bytes(b"x" * 127)

    bundle_budget.check_javascript_bundle_budget(dist, max_bytes=500)


def test_guide_build_uses_compact_runtime_index_and_enforces_budget() -> None:
    package = json.loads((ROOT / "guide" / "package.json").read_text(encoding="utf-8"))
    scripts = package["scripts"]
    assert isinstance(scripts, dict)

    code_index = scripts["code-index"]
    assert ".generated/code-symbols.json" in code_index
    assert ".generated/code-symbols-runtime.json" in code_index
    assert "--topics content/topics" in code_index

    postbuild = scripts["postbuild"]
    assert "tools/bundle_budget.py" in postbuild
    assert "--max-bytes 500000" in postbuild

    browser_index = (ROOT / "guide" / "src" / "content" / "codeSymbols.ts").read_text(
        encoding="utf-8"
    )
    assert 'from "../../.generated/code-symbols-runtime.json"' in browser_index
