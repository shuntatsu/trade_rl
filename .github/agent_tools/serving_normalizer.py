from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "tests/serving/test_runtime_normalizer.py"


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def write_tests() -> None:
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.write_text(
        '''from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.serving.helpers import OBSERVATION_SIZE, create_bundle, runtime_identity_contract
from trade_rl.domain.selection import PolicyMode
from trade_rl.serving.bundle import ServingBundle
from trade_rl.serving.runtime import LoadedPolicy, ServingRuntime


class RecordingPolicy:
    def __init__(self, action: np.ndarray) -> None:
        self.action = action
        self.observations: list[np.ndarray] = []

    def predict(self, observation: np.ndarray) -> np.ndarray:
        self.observations.append(np.asarray(observation).copy())
        return self.action.copy()


class Loader:
    def __init__(self, policy: RecordingPolicy) -> None:
        self.policy = policy

    def load(self, bundle: ServingBundle) -> LoadedPolicy:
        return self.policy


def test_runtime_applies_bundle_normalizer_before_policy(tmp_path: Path) -> None:
    policy = RecordingPolicy(np.zeros(3, dtype=np.float32))
    runtime = ServingRuntime(
        policy_loader=Loader(policy),
        identity_contract=runtime_identity_contract(normalizer_mean=2.0, normalizer_scale=4.0),
    )
    runtime.activate(
        create_bundle(
            tmp_path / "normalized",
            policy_mode=PolicyMode.RESIDUAL_POLICY,
            normalizer_mean=2.0,
            normalizer_scale=4.0,
        )
    )
    policy.observations.clear()

    runtime.predict(np.full(OBSERVATION_SIZE, 6.0, dtype=np.float32))

    np.testing.assert_allclose(
        policy.observations[-1], np.ones(OBSERVATION_SIZE, dtype=np.float32)
    )


def test_activation_probes_policy_before_replacing_live_state(tmp_path: Path) -> None:
    runtime = ServingRuntime(identity_contract=runtime_identity_contract())
    original = runtime.activate(create_bundle(tmp_path / "baseline"))
    bad_policy = RecordingPolicy(np.array([0.0, np.nan, 0.0], dtype=np.float32))
    runtime.policy_loader = Loader(bad_policy)

    with pytest.raises(ValueError, match="action schema"):
        runtime.activate(
            create_bundle(
                tmp_path / "bad",
                policy_mode=PolicyMode.RESIDUAL_POLICY,
            )
        )

    assert runtime.snapshot() == original


def test_bundle_rejects_missing_normalizer_sidecar(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "missing")
    (root / "normalizer.json").unlink()
    with pytest.raises(ValueError, match="normalizer|missing"):
        from trade_rl.serving.bundle import load_serving_bundle

        load_serving_bundle(root)
''',
        encoding="utf-8",
    )


def apply_implementation() -> None:
    normalizer_module = ROOT / "trade_rl/serving/normalizer.py"
    normalizer_module.write_text(
        '''"""Canonical observation-normalizer sidecars for serving bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.rl.normalization import ObservationNormalizer

NORMALIZER_ARTIFACT_NAME = "normalizer.json"


def write_observation_normalizer(root: Path, normalizer: ObservationNormalizer) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / NORMALIZER_ARTIFACT_NAME
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(
        canonical_json_bytes({"digest": normalizer.digest, **normalizer.digest_payload()})
    )
    temporary.replace(path)
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer or null")
    return value


def load_observation_normalizer(root: Path) -> ObservationNormalizer:
    path = Path(root) / NORMALIZER_ARTIFACT_NAME
    if not path.is_file():
        raise ValueError("serving bundle normalizer sidecar is missing")
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), field="normalizer")
    try:
        mean = np.asarray(raw["mean"], dtype=np.float64)
        scale = np.asarray(raw["scale"], dtype=np.float64)
        passthrough = tuple(int(value) for value in raw["passthrough_indices"])
        normalizer = ObservationNormalizer(
            mean=mean,
            scale=scale,
            train_start=int(raw["train_start"]),
            train_end=int(raw["train_end"]),
            clip=float(raw["clip"]),
            epsilon=float(raw["epsilon"]),
            passthrough_indices=passthrough,
            dataset_id=_optional_string(raw.get("dataset_id"), field="dataset_id"),
            source_dataset_id=_optional_string(
                raw.get("source_dataset_id"), field="source_dataset_id"
            ),
            source_dataset_artifact_digest=_optional_string(
                raw.get("source_dataset_artifact_digest"),
                field="source_dataset_artifact_digest",
            ),
            absolute_train_start=_optional_int(
                raw.get("absolute_train_start"), field="absolute_train_start"
            ),
            absolute_train_end=_optional_int(
                raw.get("absolute_train_end"), field="absolute_train_end"
            ),
            observation_schema=str(raw["observation_schema"]),
            observation_schema_digest=_optional_string(
                raw.get("observation_schema_digest"),
                field="observation_schema_digest",
            ),
            action_spec_digest=_optional_string(
                raw.get("action_spec_digest"), field="action_spec_digest"
            ),
            alpha_artifact_digest=_optional_string(
                raw.get("alpha_artifact_digest"), field="alpha_artifact_digest"
            ),
            factor_artifact_digest=_optional_string(
                raw.get("factor_artifact_digest"), field="factor_artifact_digest"
            ),
            candidate_config_digest=_optional_string(
                raw.get("candidate_config_digest"), field="candidate_config_digest"
            ),
            schema_version=str(raw["schema_version"]),
            digest=str(raw["digest"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("serving normalizer sidecar is invalid") from error
    return normalizer


__all__ = [
    "NORMALIZER_ARTIFACT_NAME",
    "load_observation_normalizer",
    "write_observation_normalizer",
]
''',
        encoding="utf-8",
    )

    replace_once(
        "trade_rl/serving/bundle.py",
        '''from trade_rl.serving.release import (
    RELEASE_ATTESTATION_NAME,
    load_release_attestation,
)
''',
        '''from trade_rl.serving.normalizer import (
    NORMALIZER_ARTIFACT_NAME,
    load_observation_normalizer,
)
from trade_rl.serving.release import (
    RELEASE_ATTESTATION_NAME,
    load_release_attestation,
)
''',
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '''class ServingBundle:
    root: Path
    manifest: ServingBundleManifest
    release: ReleaseManifest | None = None
''',
        '''class ServingBundle:
    root: Path
    manifest: ServingBundleManifest
    release: ReleaseManifest | None = None
    normalizer: object | None = None
''',
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '''    release: ReleaseManifest | None = None
    if manifest.schema_version == "serving_bundle_v4" and manifest.release_digest is not None:
''',
        '''    release: ReleaseManifest | None = None
    normalizer = None
    if manifest.normalizer_digest is not None:
        if NORMALIZER_ARTIFACT_NAME not in {item.path for item in manifest.files}:
            raise ValueError("serving bundle does not declare its normalizer sidecar")
        normalizer = load_observation_normalizer(root)
        if normalizer.digest != manifest.normalizer_digest:
            raise ValueError("serving bundle normalizer digest mismatch")
    elif NORMALIZER_ARTIFACT_NAME in {item.path for item in manifest.files}:
        raise ValueError("serving bundle declares an unbound normalizer sidecar")
    if manifest.schema_version == "serving_bundle_v4" and manifest.release_digest is not None:
''',
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '    return ServingBundle(root=root, manifest=manifest, release=release)\n',
        '''    return ServingBundle(
        root=root,
        manifest=manifest,
        release=release,
        normalizer=normalizer,
    )
''',
    )

    replace_once(
        "trade_rl/serving/runtime.py",
        "from trade_rl.rl.observations import OBSERVATION_SCHEMA\n",
        "from trade_rl.rl.normalization import ObservationNormalizer\nfrom trade_rl.rl.observations import OBSERVATION_SCHEMA\n",
    )
    replace_once(
        "trade_rl/serving/runtime.py",
        '''        self._snapshot: RuntimeSnapshot | None = None
        self._policy: LoadedPolicy | None = None
''',
        '''        self._snapshot: RuntimeSnapshot | None = None
        self._policy: LoadedPolicy | None = None
        self._normalizer: ObservationNormalizer | None = None
''',
    )
    replace_once(
        "trade_rl/serving/runtime.py",
        '''    def activate(self, root: Path) -> RuntimeSnapshot:
''',
        '''    @staticmethod
    def _predict_action(
        policy: LoadedPolicy,
        snapshot: RuntimeSnapshot,
        normalizer: ObservationNormalizer | None,
        observation: np.ndarray,
    ) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.shape != (snapshot.observation_size,) or not np.isfinite(vector).all():
            raise ValueError("observation violates the active observation schema")
        policy_input = vector if normalizer is None else normalizer.transform(vector)
        raw_action = np.asarray(
            policy.predict(policy_input),
            dtype=np.float32,
        ).reshape(-1)
        if (
            raw_action.shape != (snapshot.action_size,)
            or not np.isfinite(raw_action).all()
        ):
            raise ValueError("policy output violates the residual action schema")
        if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
            raise ValueError("policy output violates the residual action schema bounds")
        return raw_action.copy()

    def activate(self, root: Path) -> RuntimeSnapshot:
''',
    )
    replace_once(
        "trade_rl/serving/runtime.py",
        '''        candidate_snapshot = self._snapshot_for(bundle)
        with self._lock:
            self._policy = candidate_policy
            self._snapshot = candidate_snapshot
''',
        '''        candidate_snapshot = self._snapshot_for(bundle)
        candidate_normalizer = bundle.normalizer
        if candidate_normalizer is not None and not isinstance(
            candidate_normalizer, ObservationNormalizer
        ):
            raise ValueError("serving bundle normalizer type is invalid")
        self._predict_action(
            candidate_policy,
            candidate_snapshot,
            candidate_normalizer,
            np.zeros(candidate_snapshot.observation_size, dtype=np.float32),
        )
        with self._lock:
            self._policy = candidate_policy
            self._snapshot = candidate_snapshot
            self._normalizer = candidate_normalizer
''',
    )
    replace_once(
        "trade_rl/serving/runtime.py",
        '''        with self._lock:
            policy = self._policy
            snapshot = self._snapshot
        if policy is None or snapshot is None:
            raise RuntimeError("serving runtime has no active policy")
        if vector.shape != (snapshot.observation_size,):
            raise ValueError("observation violates the active observation schema")
        raw_action = np.asarray(
            policy.predict(vector),
            dtype=np.float32,
        ).reshape(-1)
        if (
            raw_action.shape != (snapshot.action_size,)
            or not np.isfinite(raw_action).all()
        ):
            raise ValueError("policy output violates the residual action schema")
        if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
            raise ValueError("policy output violates the residual action schema bounds")
        return raw_action.copy()
''',
        '''        with self._lock:
            policy = self._policy
            snapshot = self._snapshot
            normalizer = self._normalizer
        if policy is None or snapshot is None:
            raise RuntimeError("serving runtime has no active policy")
        return self._predict_action(policy, snapshot, normalizer, vector)
''',
    )

    helper = ROOT / "tests/serving/helpers.py"
    helper.write_text(
        '''from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.releases import ReleaseManifest
from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import ServingBundleManifest, write_serving_bundle_manifest
from trade_rl.serving.normalizer import write_observation_normalizer
from trade_rl.serving.release import write_release_attestation
from trade_rl.serving.runtime import RuntimeIdentityContract

OBSERVATION_SIZE = 5
ACTION_NAMES = ("fast_tilt", "slow_tilt", "risk_tilt")
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})
INITIAL_CAPITAL = 250_000.0
_CREATED_AT = datetime(2026, 7, 13, tzinfo=UTC)


def _normalizer(
    *,
    observation_size: int = OBSERVATION_SIZE,
    mean: float = 0.0,
    scale: float = 1.0,
    action_spec_digest: str = ACTION_SPEC_DIGEST,
) -> ObservationNormalizer:
    return ObservationNormalizer(
        mean=np.full(observation_size, mean, dtype=np.float64),
        scale=np.full(observation_size, scale, dtype=np.float64),
        train_start=0,
        train_end=1,
        dataset_id="a" * 64,
        source_dataset_id="a" * 64,
        observation_schema=OBSERVATION_SCHEMA,
        action_spec_digest=action_spec_digest,
    )


NORMALIZER_DIGEST = _normalizer().digest


def create_bundle(
    root: Path,
    *,
    policy_mode: PolicyMode = PolicyMode.BASELINE_ONLY,
    release_digest: str | None = "released",
    observation_size: int = OBSERVATION_SIZE,
    action_names: tuple[str, ...] = ACTION_NAMES,
    action_spec_digest: str = ACTION_SPEC_DIGEST,
    environment_digest: str = "d" * 64,
    normalizer_digest: str | None = NORMALIZER_DIGEST,
    normalizer_mean: float = 0.0,
    normalizer_scale: float = 1.0,
) -> Path:
    root.mkdir(parents=True)
    artifact_paths = ["dataset.json", "signal.json", "selection.json"]
    (root / "dataset.json").write_text('{"dataset":"a"}', encoding="utf-8")
    (root / "signal.json").write_text('{"signal":"rejected"}', encoding="utf-8")
    (root / "selection.json").write_text(
        f'{{"selection":"{policy_mode.value}"}}', encoding="utf-8"
    )
    resolved_normalizer = None
    if normalizer_digest is not None:
        resolved_normalizer = _normalizer(
            observation_size=observation_size,
            mean=normalizer_mean,
            scale=normalizer_scale,
            action_spec_digest=action_spec_digest,
        )
        normalizer_digest = resolved_normalizer.digest
        write_observation_normalizer(root, resolved_normalizer)
        artifact_paths.append("normalizer.json")
    policy_digest: str | None = None
    if policy_mode is PolicyMode.RESIDUAL_POLICY:
        (root / "policy.zip").write_bytes(b"residual-policy")
        artifact_paths.append("policy.zip")
        policy_digest = "e" * 64
    candidate = ServingBundleManifest.build(
        root=root,
        dataset_id="a" * 64,
        action_schema=ACTION_SCHEMA,
        action_size=len(action_names),
        action_names=action_names,
        action_spec_digest=action_spec_digest,
        observation_schema=OBSERVATION_SCHEMA,
        observation_size=observation_size,
        environment_digest=environment_digest,
        initial_capital=INITIAL_CAPITAL,
        policy_mode=policy_mode,
        policy_digest=policy_digest,
        signal_digest="b" * 64,
        selection_digest="c" * 64,
        release_digest=None,
        normalizer_digest=normalizer_digest,
        artifact_paths=tuple(artifact_paths),
        created_at=_CREATED_AT,
    )
    manifest = candidate
    if release_digest is not None:
        release = ReleaseManifest(
            version="2026.07.13",
            git_commit="e" * 40,
            dataset_id=candidate.dataset_id,
            signal_digest=candidate.signal_digest,
            selection_digest=candidate.selection_digest,
            selection_evaluation_digest="1" * 64,
            gate_evaluation_digest="2" * 64,
            selected_policy_digest=candidate.policy_digest,
            bundle_digest=candidate.bundle_digest,
            created_at=_CREATED_AT,
        )
        manifest = candidate.with_release(release)
        write_release_attestation(root, release)
    write_serving_bundle_manifest(root, manifest)
    return root


def runtime_identity_contract(
    *,
    environment_digest: str = "d" * 64,
    action_names: tuple[str, ...] = ACTION_NAMES,
    action_spec_digest: str = ACTION_SPEC_DIGEST,
    normalizer_digest: str | None = NORMALIZER_DIGEST,
    normalizer_mean: float = 0.0,
    normalizer_scale: float = 1.0,
    alpha_artifact_digest: str | None = None,
    factor_artifact_digest: str | None = None,
) -> RuntimeIdentityContract:
    if normalizer_digest is not None:
        normalizer_digest = _normalizer(
            mean=normalizer_mean,
            scale=normalizer_scale,
            action_spec_digest=action_spec_digest,
        ).digest
    return RuntimeIdentityContract(
        environment_digest=environment_digest,
        action_names=action_names,
        action_spec_digest=action_spec_digest,
        normalizer_digest=normalizer_digest,
        alpha_artifact_digest=alpha_artifact_digest,
        factor_artifact_digest=factor_artifact_digest,
    )
''',
        encoding="utf-8",
    )

    replace_once(
        "tests/serving/test_bundle.py",
        '''    INITIAL_CAPITAL,
    OBSERVATION_SIZE,
''',
        '''    INITIAL_CAPITAL,
    NORMALIZER_DIGEST,
    OBSERVATION_SIZE,
''',
    )
    replace_once(
        "tests/serving/test_bundle.py",
        '    assert manifest.normalizer_digest == "9" * 64\n',
        '    assert manifest.normalizer_digest == NORMALIZER_DIGEST\n',
    )
    replace_once(
        "tests/serving/test_runtime.py",
        '''        runtime.activate(
            create_bundle(
                tmp_path / name,
                policy_mode=PolicyMode.RESIDUAL_POLICY,
            )
        )
        with pytest.raises(ValueError, match="action schema"):
            runtime.predict(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
''',
        '''        with pytest.raises(ValueError, match="action schema"):
            runtime.activate(
                create_bundle(
                    tmp_path / name,
                    policy_mode=PolicyMode.RESIDUAL_POLICY,
                )
            )
''',
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: serving_normalizer.py tests|implementation")
    if sys.argv[1] == "tests":
        write_tests()
    else:
        apply_implementation()


if __name__ == "__main__":
    main()
