from __future__ import annotations

from pathlib import Path

from trade_rl.artifacts.provenance import capture_runtime_provenance


def test_runtime_provenance_is_content_addressed_and_path_independent(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "uv.lock").write_text("same-lock", encoding="utf-8")
    (right / "uv.lock").write_text("same-lock", encoding="utf-8")

    first = capture_runtime_provenance(
        left,
        git_commit="a" * 40,
        git_dirty=False,
        deterministic_seed_config={"seeds": (1, 2)},
        package_versions={"numpy": "2.0"},
        python_version="3.12.0",
        platform_name="test-platform",
        hardware_name="test-hardware",
    )
    second = capture_runtime_provenance(
        right,
        git_commit="a" * 40,
        git_dirty=False,
        deterministic_seed_config={"seeds": (1, 2)},
        package_versions={"numpy": "2.0"},
        python_version="3.12.0",
        platform_name="test-platform",
        hardware_name="test-hardware",
    )

    assert first == second
    assert first.digest == second.digest
    assert first.lockfile_digest is not None
    assert "/left" not in repr(first)
