from __future__ import annotations

import json

from trade_rl.workflows.universal_causal_alpha_v3_store import CausalAlphaV3RecordStore


def test_exact_artifact_reuse_is_json_semantic_not_python_container_specific(tmp_path) -> None:
    store = CausalAlphaV3RecordStore(
        tmp_path,
        run_manifest_digest="1" * 64,
    )
    payload = {
        "schema_version": "test_v1",
        "items": ("alpha", "beta"),
    }

    first = store.write_exact_artifact("meta.json", payload)
    second = store.write_exact_artifact("meta.json", payload)

    assert second == first
    assert json.loads(first.read_text(encoding="utf-8"))["items"] == ["alpha", "beta"]
