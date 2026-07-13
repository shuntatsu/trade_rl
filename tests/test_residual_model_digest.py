from __future__ import annotations

from pathlib import Path

from mars_lite.pipeline.residual_candidates import digest_model_artifact


def test_model_digest_is_content_bound_for_file(tmp_path: Path) -> None:
    model = tmp_path / "model.zip"
    model.write_bytes(b"first")
    first = digest_model_artifact(model)
    model.write_bytes(b"second")
    second = digest_model_artifact(model)

    assert first != second
    assert len(first) == 64
    assert len(second) == 64


def test_ensemble_digest_is_deterministic_and_name_bound(tmp_path: Path) -> None:
    ensemble = tmp_path / "ensemble"
    ensemble.mkdir()
    (ensemble / "seed_1.zip").write_bytes(b"one")
    (ensemble / "seed_0.zip").write_bytes(b"zero")

    first = digest_model_artifact(ensemble)
    second = digest_model_artifact(ensemble)
    (ensemble / "seed_0.zip").rename(ensemble / "renamed.zip")
    renamed = digest_model_artifact(ensemble)

    assert first == second
    assert first != renamed
