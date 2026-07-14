from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

import trade_rl.artifacts.signals as signals
from trade_rl.artifacts.hashing import content_digest

SHA = "a" * 64


def _write_signal(root: Path) -> None:
    signals.write_signal_artifact(
        root,
        kind="alpha",
        dataset_id=SHA,
        fit_start=0,
        fit_stop=1,
        names=("BTC",),
        values=np.zeros((4, 1)),
    )


def _rewrite_signal_manifest(root: Path, **changes: object) -> None:
    path = root / signals.SIGNAL_MANIFEST_NAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.update(changes)
    digest_payload = {
        "arrays_digest": raw["arrays_digest"],
        "arrays_file": signals.SIGNAL_ARRAYS_NAME,
        "available_at_dtype": raw["available_at_dtype"],
        "dataset_id": raw["dataset_id"],
        "dtype": raw["dtype"],
        "fit_start": raw["fit_start"],
        "fit_stop": raw["fit_stop"],
        "generator_code_digest": raw["generator_code_digest"],
        "generator_config_digest": raw["generator_config_digest"],
        "kind": raw["kind"],
        "names": tuple(raw["names"]),
        "prediction_start": raw["prediction_start"],
        "prediction_stop": raw["prediction_stop"],
        "schema_version": raw["schema_version"],
        "shape": tuple(raw["shape"]),
    }
    raw["artifact_digest"] = content_digest(digest_payload)
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_signal_loader_rejects_manifest_and_array_tampering(tmp_path: Path) -> None:
    nonmapping = tmp_path / "nonmapping"
    _write_signal(nonmapping)
    (nonmapping / signals.SIGNAL_MANIFEST_NAME).write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        signals.load_signal_artifact(nonmapping)

    invalid = tmp_path / "invalid"
    _write_signal(invalid)
    (invalid / signals.SIGNAL_MANIFEST_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        signals.load_signal_artifact(invalid)

    wrong_kind = tmp_path / "kind"
    _write_signal(wrong_kind)
    with pytest.raises(ValueError, match="kind"):
        signals.load_signal_artifact(wrong_kind, expected_kind="factor")

    digest = tmp_path / "digest"
    _write_signal(digest)
    (digest / signals.SIGNAL_ARRAYS_NAME).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="arrays digest"):
        signals.load_signal_artifact(digest)

    invalid_zip = tmp_path / "invalid-zip"
    _write_signal(invalid_zip)
    payload = b"not-a-zip"
    (invalid_zip / signals.SIGNAL_ARRAYS_NAME).write_bytes(payload)
    _rewrite_signal_manifest(invalid_zip, arrays_digest=signals._sha256(payload))
    with pytest.raises(ValueError, match="arrays are invalid"):
        signals.load_signal_artifact(invalid_zip)

    allowlist = tmp_path / "allowlist"
    _write_signal(allowlist)
    output = io.BytesIO()
    np.savez(output, values=np.zeros((4, 1)))
    payload = output.getvalue()
    (allowlist / signals.SIGNAL_ARRAYS_NAME).write_bytes(payload)
    _rewrite_signal_manifest(allowlist, arrays_digest=signals._sha256(payload))
    with pytest.raises(ValueError, match="arrays are invalid"):
        signals.load_signal_artifact(allowlist)

    wrong_shape = tmp_path / "shape"
    _write_signal(wrong_shape)
    _rewrite_signal_manifest(wrong_shape, shape=[4, 2])
    with pytest.raises(ValueError, match="shape or dtype"):
        signals.load_signal_artifact(wrong_shape)

    wrong_availability = tmp_path / "availability"
    _write_signal(wrong_availability)
    _rewrite_signal_manifest(wrong_availability, available_at_dtype="<f8")
    with pytest.raises(ValueError, match="availability dtype"):
        signals.load_signal_artifact(wrong_availability)
