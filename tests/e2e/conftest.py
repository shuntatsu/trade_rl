from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _bind_research_to_serving_declared_git_identity(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep the serving E2E focused on its signed pipeline, not host checkout data."""

    if request.node.name != "test_research_training_to_attested_runtime_prediction":
        yield
        return

    def matching_git_identity(_root: Path, *args: str) -> str | None:
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(f"unexpected Git provenance query: {args!r}")

    monkeypatch.setattr("trade_rl.artifacts.provenance._git", matching_git_identity)
    yield
