from __future__ import annotations

import importlib
from pathlib import Path


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
