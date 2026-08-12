from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_universal_training_generation as module


def test_launcher_builds_clean_digest_bound_image_and_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"lock")
    (tmp_path / "artifacts" / "universal").mkdir(parents=True)
    (tmp_path / "artifacts" / "universal" / "runtime-manifest.json").write_text("{}")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(module, "_git", lambda *_args: "a" * 40)
    monkeypatch.setattr(module, "_git_status", lambda _root: "")
    monkeypatch.setattr(module, "source_tree_digest", lambda _root: "b" * 64)
    monkeypatch.setattr(module, "_sha256_file", lambda _path: "c" * 64)
    monkeypatch.setattr(
        module,
        "load_universal_runtime_manifest",
        lambda _path: SimpleNamespace(manifest_digest="d" * 64),
    )
    monkeypatch.setattr(
        module, "_run", lambda command, **_kwargs: calls.append(tuple(command))
    )
    monkeypatch.setattr(module, "_container_exists", lambda _name: False)

    result = module.launch_generation(
        project_root=tmp_path,
        generation="universal-u6-20260812T120000Z",
        compose_file=tmp_path / "compose.universal-training.yaml",
        runtime_manifest=tmp_path / "artifacts/universal/runtime-manifest.json",
    )

    build = next(call for call in calls if call[:2] == ("docker", "build"))
    joined = " ".join(build)
    assert "TRADE_RL_GIT_COMMIT" in joined
    assert "TRADE_RL_SOURCE_TREE_DIGEST" in joined
    assert "TRADE_RL_LOCKFILE_DIGEST" in joined
    assert "TRADE_RL_RUNTIME_MANIFEST_DIGEST" in joined
    assert result.container_name == "trade-rl-universal-u6-20260812T120000Z"


def test_launcher_rejects_dirty_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(module, "_git_status", lambda _root: " M trade_rl/x.py")
    with pytest.raises(RuntimeError, match="clean Git tree"):
        module.launch_generation(
            project_root=tmp_path,
            generation="generation-a",
            compose_file=tmp_path / "compose.universal-training.yaml",
            runtime_manifest=tmp_path / "runtime.json",
        )
