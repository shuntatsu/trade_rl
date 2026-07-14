from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.serving.helpers import OBSERVATION_SIZE, create_bundle, runtime_identity_contract
from trade_rl.serving.runtime import ServingRuntime


def test_runtime_loads_and_applies_bundle_normalizer(tmp_path: Path) -> None:
    runtime = ServingRuntime(identity_contract=runtime_identity_contract())
    runtime.activate(create_bundle(tmp_path / "bundle"))
    action = runtime.predict(np.arange(OBSERVATION_SIZE, dtype=np.float32))
    np.testing.assert_array_equal(action, np.zeros(3, dtype=np.float32))


def test_runtime_rejects_normalizer_tampering_before_swap(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "bundle")
    path = root / "normalizer.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mean"][0] = 999.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    runtime = ServingRuntime(identity_contract=runtime_identity_contract())
    with pytest.raises(ValueError, match="digest mismatch|bundle artifact (?:digest|size)"):
        runtime.activate(root)
