from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OBSOLETE = ROOT / "trade_rl/serving/observations.py"


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_obsolete_serving_observation_pipeline_is_absent() -> None:
    assert not OBSOLETE.exists()


def test_production_sources_do_not_reference_obsolete_observation_authority() -> None:
    for path in (ROOT / "trade_rl").rglob("*.py"):
        if path == OBSOLETE:
            continue
        source = path.read_text(encoding="utf-8")
        assert "trade_rl.serving.observations" not in source, path
        assert "ServingObservationPipeline" not in source, path

    serving_init = _source("trade_rl/serving/__init__.py")
    assert "NORMALIZER_FILE" not in serving_init
    assert "ServingObservationPipeline" not in serving_init


def test_bundle_loader_and_runtime_remain_the_single_observation_authority() -> None:
    bundle = _source("trade_rl/serving/bundle.py")
    runtime = _source("trade_rl/serving/runtime.py")

    assert "load_observation_normalizer(root)" in bundle
    assert "serving bundle normalizer digest mismatch" in bundle
    assert "vector.shape != (snapshot.observation_size,)" in runtime
    assert "not np.isfinite(vector).all()" in runtime
    assert "normalizer.transform(vector)" in runtime
    assert "structured observation violates the active schema" in runtime
