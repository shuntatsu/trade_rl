from __future__ import annotations

import pytest

from trade_rl.studio.resource_ids import require_resource_id, resource_id


def test_resource_ids_bind_kind_path_and_canonical_identity() -> None:
    first = resource_id("run", "research-a/runs/run-001", "a" * 64)
    second = resource_id("run", "research-b/runs/run-001", "a" * 64)

    assert first.startswith("run-")
    assert second.startswith("run-")
    assert first != second
    assert require_resource_id(first, kind="run") == first


def test_resource_id_rejects_unknown_kinds_and_malformed_values() -> None:
    with pytest.raises(ValueError, match="resource kind"):
        resource_id("worker", "path", "identity")
    with pytest.raises(ValueError, match="resource id"):
        require_resource_id("../run-001", kind="run")
    with pytest.raises(ValueError, match="resource id kind"):
        require_resource_id(
            resource_id("config", "configs/a.json", "b" * 64), kind="run"
        )
