from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import trade_rl.strategies.forecasts.ridge_artifact as ridge_artifact_module
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.strategies.forecasts import (
    RidgeForecastArtifactBinding as FacadeBinding,
)
from trade_rl.strategies.forecasts import (
    load_ridge_forecast_artifact as facade_load,
)
from trade_rl.strategies.forecasts import (
    publish_ridge_forecast_artifact as facade_publish,
)
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
)
from trade_rl.strategies.forecasts.ridge_artifact import (
    RidgeForecastArtifactBinding,
    load_ridge_forecast_artifact,
    publish_ridge_forecast_artifact,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent


def _binding(model: RidgeForecastModel) -> RidgeForecastArtifactBinding:
    cutoff_ns = int(model.fit_cutoff.astype("int64"))
    return RidgeForecastArtifactBinding(
        dataset_id="1" * 64,
        dataset_schema_version="market_dataset_v3",
        dataset_artifact_sha256="2" * 64,
        fit_symbol_indices=(0,),
        fit_symbol_names=("BTCUSDT",),
        feature_names=("momentum", "volume"),
        feature_indices=model.feature_indices,
        fit_cutoff_ns=cutoff_ns,
        horizon_hours=model.horizon_hours,
        alpha_hex=model.alpha.hex(),
        n_samples=model.n_samples,
        vintage_id="vintage-2026-01",
        source_commit="a" * 40,
        implementation_sha256="3" * 64,
        runtime_sha256="4" * 64,
    )


def _model() -> RidgeForecastModel:
    return RidgeForecastModel(
        feature_indices=(1, 0),
        feature_mean=np.asarray([1.0, -0.0], dtype=np.float64),
        feature_scale=np.asarray([2.0, 0.5], dtype=np.float64),
        coefficients=np.asarray([0.8, -0.2], dtype=np.float64),
        intercept=0.2,
        horizon_hours=24,
        alpha=1.0,
        n_samples=5,
        fit_cutoff=np.datetime64("2026-01-01T00:00:00", "ns"),
    )


def _published_artifact(
    tmp_path: Path,
) -> tuple[RidgeForecastModel, RidgeForecastArtifactBinding, Path, str]:
    model = _model()
    binding = _binding(model)
    root = tmp_path / "artifact"
    digest = publish_ridge_forecast_artifact(root, model, binding)
    return model, binding, root, digest


def test_ridge_artifact_roundtrip_preserves_model_bits_and_strategy_behavior(
    tmp_path,
) -> None:
    original = _model()
    binding = _binding(original)
    destination = tmp_path / "ridge-model"

    digest = publish_ridge_forecast_artifact(destination, original, binding)
    payload = _read_payload(destination)
    assert set(payload["model"]) == {
        "feature_mean_hex",
        "feature_scale_hex",
        "coefficients_hex",
        "intercept_hex",
    }
    loaded = load_ridge_forecast_artifact(
        destination,
        expected_digest=digest,
        expected_binding=binding,
    )

    assert isinstance(loaded, RidgeForecastModel)
    assert FacadeBinding is RidgeForecastArtifactBinding
    assert facade_publish is publish_ridge_forecast_artifact
    assert facade_load is load_ridge_forecast_artifact
    for original_values, loaded_values in (
        (original.feature_mean, loaded.feature_mean),
        (original.feature_scale, loaded.feature_scale),
        (original.coefficients, loaded.coefficients),
    ):
        np.testing.assert_array_equal(
            original_values.view(np.uint64), loaded_values.view(np.uint64)
        )
        assert loaded_values.flags.writeable is False
    assert struct.pack("<d", original.intercept) == struct.pack("<d", loaded.intercept)
    features = np.asarray([0.5, 1.5])
    expected_forecast = 0.2 + (0.5 / 2.0) * 0.8 + (0.5 / 0.5) * -0.2
    assert original.predict(features) == pytest.approx(expected_forecast)
    assert loaded.predict(features) == pytest.approx(expected_forecast)

    strategy = RidgeForecastStrategy(
        loaded,
        entry_threshold=0.10,
        exit_threshold=0.02,
    )
    observation = StrategyObservation(
        index=0,
        timestamp=np.datetime64("2026-01-01T00:00:00", "ns"),
        symbol="BTCUSDT",
        features=features,
        feature_available=np.asarray([True, True]),
        global_features=np.asarray([0.0]),
        global_feature_available=np.asarray([True]),
        current_intent=PositionIntent.FLAT,
        current_weight=0.0,
    )
    assert strategy.decide(observation) is PositionIntent.LONG


def _artifact_digest(payload: bytes) -> str:
    return hashlib.sha256(
        b"trade_rl.ridge_forecast_model_artifact_v1\0" + payload
    ).hexdigest()


def _read_payload(root: Path) -> dict[str, object]:
    return json.loads((root / "artifact.json").read_text(encoding="utf-8"))


def _load_tampered(root: Path, payload: dict[str, object], binding) -> None:
    encoded = canonical_json_bytes(payload)
    (root / "artifact.json").write_bytes(encoded)
    load_ridge_forecast_artifact(
        root,
        expected_digest=_artifact_digest(encoded),
        expected_binding=binding,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", "5" * 64),
        ("dataset_schema_version", "market_dataset_v4"),
        ("dataset_artifact_sha256", "6" * 64),
        ("fit_symbol_indices", (1,)),
        ("fit_symbol_names", ("ETHUSDT",)),
        ("feature_names", ("volume", "momentum")),
        ("feature_indices", (1,)),
        (
            "fit_cutoff_ns",
            int(np.datetime64("2026-01-02", "ns").astype("int64")),
        ),
        ("horizon_hours", 48),
        ("alpha_hex", float(2.0).hex()),
        ("n_samples", 6),
        ("vintage_id", "vintage-2026-02"),
        ("source_commit", "b" * 40),
        ("implementation_sha256", "7" * 64),
        ("runtime_sha256", "8" * 64),
    ],
)
def test_ridge_artifact_requires_exact_expected_binding(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    _, binding, root, digest = _published_artifact(tmp_path)

    with pytest.raises(ValueError, match="context differs"):
        load_ridge_forecast_artifact(
            root,
            expected_digest=digest,
            expected_binding=replace(binding, **{field: value}),
        )


def test_ridge_artifact_rejects_self_rehashed_tamper_and_unknown_fields(
    tmp_path: Path,
) -> None:
    _, binding, root, digest = _published_artifact(tmp_path)
    payload = _read_payload(root)
    payload["binding"]["vintage_id"] = "tampered-vintage"
    tampered = canonical_json_bytes(payload)
    (root / "artifact.json").write_bytes(tampered)

    with pytest.raises(ValueError, match="externally expected digest"):
        load_ridge_forecast_artifact(
            root,
            expected_digest=digest,
            expected_binding=binding,
        )
    with pytest.raises(ValueError, match="context differs"):
        load_ridge_forecast_artifact(
            root,
            expected_digest=_artifact_digest(tampered),
            expected_binding=binding,
        )

    payload["unexpected"] = "not in schema"
    with pytest.raises(ValueError, match="keys differ"):
        _load_tampered(root, payload, binding)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda p: p["binding"].update(fit_cutoff_ns=-(2**63)),
            "fit_cutoff_ns",
        ),
        (lambda p: p["model"].update(feature_mean_hex=[]), "feature_mean shape"),
        (lambda p: p["model"].update(intercept_hex="inf"), "intercept_hex"),
        (lambda p: p.update(schema="ridge_forecast_model_artifact_v2"), "schema"),
    ],
)
def test_ridge_artifact_rejects_malformed_model_metadata(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    _, binding, root, _ = _published_artifact(tmp_path)
    payload = _read_payload(root)
    mutate(payload)
    encoded = canonical_json_bytes(payload)
    (root / "artifact.json").write_bytes(encoded)

    with pytest.raises(ValueError, match=message):
        load_ridge_forecast_artifact(
            root,
            expected_digest=_artifact_digest(encoded),
            expected_binding=binding,
        )


def test_ridge_artifact_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    _, binding, root, _ = _published_artifact(tmp_path)
    payload = (root / "artifact.json").read_bytes()
    duplicated = payload.replace(b'"schema":', b'"schema":"bad","schema":', 1)
    (root / "artifact.json").write_bytes(duplicated)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_ridge_forecast_artifact(
            root,
            expected_digest=_artifact_digest(duplicated),
            expected_binding=binding,
        )


def test_ridge_artifact_publication_is_idempotent_but_never_overwrites(
    tmp_path: Path,
) -> None:
    model = _model()
    binding = _binding(model)
    destination = tmp_path / "artifact"
    first = publish_ridge_forecast_artifact(destination, model, binding)
    second = publish_ridge_forecast_artifact(destination, model, binding)
    assert second == first
    original_payload = (destination / "artifact.json").read_bytes()

    changed = RidgeForecastModel(
        feature_indices=model.feature_indices,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        coefficients=model.coefficients,
        intercept=0.3,
        horizon_hours=model.horizon_hours,
        alpha=model.alpha,
        n_samples=model.n_samples,
        fit_cutoff=model.fit_cutoff,
    )
    with pytest.raises(FileExistsError, match="already differs"):
        publish_ridge_forecast_artifact(destination, changed, binding)
    assert (destination / "artifact.json").read_bytes() == original_payload


def test_ridge_artifact_failed_publish_cleans_only_its_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    binding = _binding(model)
    destination = tmp_path / "artifact"

    def fail_rename(source: Path, target: Path) -> Path:
        raise OSError("injected atomic rename failure")

    monkeypatch.setattr(ridge_artifact_module, "atomic_rename_directory", fail_rename)
    with pytest.raises(OSError, match="injected"):
        publish_ridge_forecast_artifact(destination, model, binding)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fit_cutoff_ns", -(2**63)),
        ("fit_symbol_indices", (0, 0)),
        ("fit_symbol_names", ("BTCUSDT", "BTCUSDT")),
        ("feature_names", ("same", "same")),
        ("feature_indices", (True,)),
        ("horizon_hours", True),
        ("n_samples", 0),
        ("alpha_hex", "nan"),
    ],
)
def test_ridge_artifact_binding_rejects_invalid_declared_context(
    field: str,
    value: object,
) -> None:
    binding = _binding(_model())
    with pytest.raises((TypeError, ValueError)):
        replace(binding, **{field: value})


def test_ridge_artifact_rejects_nat_model_cutoff(tmp_path: Path) -> None:
    model = _model()
    binding = _binding(model)
    # Simulate corrupted in-memory state without weakening the model constructor.
    object.__setattr__(model, "fit_cutoff", np.datetime64("NaT", "ns"))
    destination = tmp_path / "invalid-cutoff"

    with pytest.raises(ValueError, match="fit cutoff"):
        publish_ridge_forecast_artifact(destination, model, binding)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_ridge_artifact_rejects_model_binding_mismatch_before_publication(
    tmp_path: Path,
) -> None:
    model = _model()
    binding = _binding(model)
    mismatched = RidgeForecastModel(
        feature_indices=model.feature_indices,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        coefficients=model.coefficients,
        intercept=model.intercept,
        horizon_hours=48,
        alpha=model.alpha,
        n_samples=model.n_samples,
        fit_cutoff=model.fit_cutoff,
    )

    with pytest.raises(ValueError, match="horizon"):
        publish_ridge_forecast_artifact(tmp_path / "mismatched", mismatched, binding)
    assert not (tmp_path / "mismatched").exists()


def test_ridge_artifact_rejects_boolean_intercept(tmp_path: Path) -> None:
    model = _model()
    binding = _binding(model)
    boolean_intercept = RidgeForecastModel(
        feature_indices=model.feature_indices,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        coefficients=model.coefficients,
        intercept=True,
        horizon_hours=model.horizon_hours,
        alpha=model.alpha,
        n_samples=model.n_samples,
        fit_cutoff=model.fit_cutoff,
    )

    with pytest.raises(ValueError, match="intercept"):
        publish_ridge_forecast_artifact(
            tmp_path / "invalid-intercept", boolean_intercept, binding
        )
    assert not (tmp_path / "invalid-intercept").exists()
