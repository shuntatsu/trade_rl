from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v4_artifact_store import (
    CausalAlphaV4ArtifactStore,
    CausalAlphaV4RunLock,
)


def _digest(char: str) -> str:
    return char * 64


def _payload(*, run: str = "a" * 64, net: float = 0.01) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "causal_alpha_v4_test_leaf_v1",
        "run_manifest_digest": run,
        "v4_context_manifest_digest": _digest("b"),
        "config_digest": _digest("c"),
        "contract_digest": _digest("d"),
        "fit_digest": _digest("e"),
        "forecast_digest": _digest("f"),
        "target_path_digest": _digest("1"),
        "net_return": net,
    }
    return {**body, "artifact_digest": content_digest(body)}


def _store(tmp_path: Path) -> CausalAlphaV4ArtifactStore:
    return CausalAlphaV4ArtifactStore(
        tmp_path,
        run_manifest_digest=_digest("a"),
        v4_context_manifest_digest=_digest("b"),
        config_digest=_digest("c"),
        generator_code_digest=_digest("9"),
    )


def test_v4_store_allows_identical_reuse_but_rejects_different_content(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    relative = Path("selection/BTCUSDT/0.json")

    first = store.write_leaf(relative, _payload())
    second = store.write_leaf(relative, _payload())

    assert first == second
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError, match="different content"):
        store.write_leaf(relative, _payload(net=0.02))


def test_v4_store_missing_scope_is_resumable(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert (
        store.load_leaf(
            Path("signal/fast/BTCUSDT/0.json"),
            expected_schema="causal_alpha_v4_test_leaf_v1",
        )
        is None
    )


def test_v4_store_rejects_corruption_and_wrong_run_identity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    relative = Path("signal/fast/BTCUSDT/0.json")
    path = store.write_leaf(relative, _payload())

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["net_return"] = 99.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        store.load_leaf(
            relative,
            expected_schema="causal_alpha_v4_test_leaf_v1",
        )

    relative_wrong = Path("signal/slow/BTCUSDT/0.json")
    wrong = tmp_path / relative_wrong
    wrong.parent.mkdir(parents=True, exist_ok=True)
    wrong_payload = _payload(run=_digest("8"))
    wrong.write_text(json.dumps(wrong_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="run manifest identity mismatch"):
        store.load_leaf(
            relative_wrong,
            expected_schema="causal_alpha_v4_test_leaf_v1",
        )


def test_v4_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="relative artifact path"):
        store.write_leaf(Path("../escape.json"), _payload())


def test_v4_run_lock_excludes_second_writer_and_releases_on_exit(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / ".causal-alpha-v4.lock"

    with CausalAlphaV4RunLock(tmp_path):
        assert lock_path.is_file()
        with pytest.raises(RuntimeError, match="active or unrecovered writer"):
            CausalAlphaV4RunLock(tmp_path).acquire()

    assert not lock_path.exists()
