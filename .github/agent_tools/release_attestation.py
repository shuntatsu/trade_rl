from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "tests/serving/test_release_attestation.py"


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

import json
from pathlib import Path

import pytest

from tests.serving.helpers import create_bundle
from trade_rl.serving.bundle import load_serving_bundle


def test_release_attestation_binds_candidate_without_changing_bundle_digest(
    tmp_path: Path,
) -> None:
    candidate = load_serving_bundle(
        create_bundle(tmp_path / "candidate", release_digest=None)
    )
    released = load_serving_bundle(create_bundle(tmp_path / "released"))

    assert released.manifest.bundle_digest == candidate.manifest.bundle_digest
    assert released.release is not None
    assert released.manifest.release_digest == released.release.digest
    assert released.release.bundle_digest == released.manifest.bundle_digest


def test_fake_release_digest_without_attestation_is_rejected(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "candidate", release_digest=None)
    path = root / "bundle.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["release_digest"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="release attestation"):
        load_serving_bundle(root)


def test_tampered_release_attestation_is_rejected(tmp_path: Path) -> None:
    root = create_bundle(tmp_path / "released")
    path = root / "release.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bundle_digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="release"):
        load_serving_bundle(root)
''',
        encoding="utf-8",
    )


def apply_implementation() -> None:
    releases = ROOT / "trade_rl/domain/releases.py"
    releases.write_text(
        '''"""Production release attestation identity and fail-closed construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trade_rl.domain.common import (
    domain_content_digest,
    require_aware_datetime,
    require_git_sha,
    require_non_empty,
    require_sha256,
)
from trade_rl.domain.datasets import DatasetManifest
from trade_rl.domain.evaluation import GateDecision
from trade_rl.domain.selection import SelectionDecision
from trade_rl.domain.signals import SignalArtifactManifest


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    """Immutable attestation created after a candidate bundle and gates exist."""

    version: str
    git_commit: str
    dataset_id: str
    signal_digest: str
    selection_digest: str
    selection_evaluation_digest: str
    gate_evaluation_digest: str
    selected_policy_digest: str | None
    bundle_digest: str
    created_at: datetime
    schema_version: str = "release_manifest_v3"

    def __post_init__(self) -> None:
        require_non_empty(self.version, field="version")
        require_git_sha(self.git_commit)
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.signal_digest, field="signal_digest")
        require_sha256(self.selection_digest, field="selection_digest")
        require_sha256(
            self.selection_evaluation_digest,
            field="selection_evaluation_digest",
        )
        require_sha256(
            self.gate_evaluation_digest,
            field="gate_evaluation_digest",
        )
        if self.selected_policy_digest is not None:
            require_sha256(
                self.selected_policy_digest,
                field="selected_policy_digest",
            )
        require_sha256(self.bundle_digest, field="bundle_digest")
        require_aware_datetime(self.created_at, field="created_at")
        if self.schema_version != "release_manifest_v3":
            raise ValueError("unsupported release manifest schema")

    @property
    def digest(self) -> str:
        return domain_content_digest(self.digest_payload())

    def digest_payload(self) -> dict[str, object]:
        return {
            "bundle_digest": self.bundle_digest,
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "gate_evaluation_digest": self.gate_evaluation_digest,
            "git_commit": self.git_commit,
            "schema_version": self.schema_version,
            "selected_policy_digest": self.selected_policy_digest,
            "selection_digest": self.selection_digest,
            "selection_evaluation_digest": self.selection_evaluation_digest,
            "signal_digest": self.signal_digest,
            "version": self.version,
        }

    def validate_bundle_identity(
        self,
        *,
        bundle_digest: str,
        dataset_id: str,
        signal_digest: str,
        selection_digest: str,
        selected_policy_digest: str | None,
    ) -> None:
        comparisons = (
            (self.bundle_digest, bundle_digest, "bundle"),
            (self.dataset_id, dataset_id, "dataset"),
            (self.signal_digest, signal_digest, "signal"),
            (self.selection_digest, selection_digest, "selection"),
            (self.selected_policy_digest, selected_policy_digest, "policy"),
        )
        for attested, observed, label in comparisons:
            if attested != observed:
                raise ValueError(f"release attestation {label} identity mismatch")

    @classmethod
    def create(
        cls,
        *,
        version: str,
        git_commit: str,
        dataset: DatasetManifest,
        signal: SignalArtifactManifest,
        selection: SelectionDecision,
        gate: GateDecision,
        bundle_digest: str,
        created_at: datetime,
    ) -> ReleaseManifest:
        """Build a release only after bundle identity and mandatory gates exist."""

        if gate.failed_mandatory_checks:
            failed = ", ".join(check.name for check in gate.failed_mandatory_checks)
            raise ValueError(f"mandatory gate checks failed: {failed}")
        if not gate.passed:
            raise ValueError("mandatory gate decision did not pass")
        if dataset.dataset_id != signal.dataset_id:
            raise ValueError("dataset identity mismatch between dataset and signal")
        if dataset.dataset_id != selection.dataset_id:
            raise ValueError("dataset identity mismatch between dataset and selection")
        if signal.digest != selection.signal_digest:
            raise ValueError("signal digest mismatch between signal and selection")
        if gate.dataset_id != selection.dataset_id:
            raise ValueError("gate dataset identity mismatch")
        if gate.selected_policy_digest != selection.selected_policy_digest:
            raise ValueError("gate selected policy identity mismatch")
        return cls(
            version=version,
            git_commit=git_commit,
            dataset_id=dataset.dataset_id,
            signal_digest=signal.digest,
            selection_digest=selection.digest,
            selection_evaluation_digest=selection.evaluation_digest,
            gate_evaluation_digest=gate.evaluation_digest,
            selected_policy_digest=selection.selected_policy_digest,
            bundle_digest=bundle_digest,
            created_at=created_at,
        )
''',
        encoding="utf-8",
    )

    release_io = ROOT / "trade_rl/serving/release.py"
    release_io.write_text(
        '''"""Canonical release-attestation sidecars for serving bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.domain.releases import ReleaseManifest

RELEASE_ATTESTATION_NAME = "release.json"


def write_release_attestation(root: Path, release: ReleaseManifest) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / RELEASE_ATTESTATION_NAME
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(
        canonical_json_bytes(
            {"release_digest": release.digest, **release.digest_payload()}
        )
    )
    temporary.replace(path)
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def load_release_attestation(root: Path) -> ReleaseManifest:
    path = Path(root) / RELEASE_ATTESTATION_NAME
    if not path.is_file():
        raise ValueError("serving bundle release attestation is missing")
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), field="release")
    selected = raw.get("selected_policy_digest")
    if selected is not None and not isinstance(selected, str):
        raise ValueError("selected_policy_digest must be a string or null")
    release = ReleaseManifest(
        version=_string(raw.get("version"), field="version"),
        git_commit=_string(raw.get("git_commit"), field="git_commit"),
        dataset_id=_string(raw.get("dataset_id"), field="dataset_id"),
        signal_digest=_string(raw.get("signal_digest"), field="signal_digest"),
        selection_digest=_string(
            raw.get("selection_digest"), field="selection_digest"
        ),
        selection_evaluation_digest=_string(
            raw.get("selection_evaluation_digest"),
            field="selection_evaluation_digest",
        ),
        gate_evaluation_digest=_string(
            raw.get("gate_evaluation_digest"), field="gate_evaluation_digest"
        ),
        selected_policy_digest=selected,
        bundle_digest=_string(raw.get("bundle_digest"), field="bundle_digest"),
        created_at=datetime.fromisoformat(
            _string(raw.get("created_at"), field="created_at").replace("Z", "+00:00")
        ),
        schema_version=_string(raw.get("schema_version"), field="schema_version"),
    )
    declared = _string(raw.get("release_digest"), field="release_digest")
    if release.digest != declared:
        raise ValueError("release attestation digest mismatch")
    return release
''',
        encoding="utf-8",
    )

    replace_once(
        "trade_rl/serving/bundle.py",
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass, replace\n",
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        "from trade_rl.domain.selection import PolicyMode\n",
        "from trade_rl.domain.releases import ReleaseManifest\nfrom trade_rl.domain.selection import PolicyMode\nfrom trade_rl.serving.release import (\n    RELEASE_ATTESTATION_NAME,\n    load_release_attestation,\n)\n",
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '    schema_version: str = "serving_bundle_v3"\n',
        '    schema_version: str = "serving_bundle_v4"\n',
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '        if self.schema_version == "serving_bundle_v3":\n',
        '        if self.schema_version in {"serving_bundle_v3", "serving_bundle_v4"}:\n',
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '''        return {
            "action_names": self.action_names,
''',
        '''        payload = {
            "action_names": self.action_names,
''',
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '''            "policy_mode": self.policy_mode,
            "release_digest": self.release_digest,
            "schema_version": self.schema_version,
            "selection_digest": self.selection_digest,
            "signal_digest": self.signal_digest,
        }
''',
        '''            "policy_mode": self.policy_mode,
            "schema_version": self.schema_version,
            "selection_digest": self.selection_digest,
            "signal_digest": self.signal_digest,
        }
        if self.schema_version == "serving_bundle_v3":
            payload["release_digest"] = self.release_digest
        return payload
''',
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '            "release_digest": release_digest,\n            "schema_version": "serving_bundle_v3",\n',
        '            "schema_version": "serving_bundle_v4",\n',
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '''            normalizer_digest=normalizer_digest,
        )


@dataclass(frozen=True, slots=True)
class ServingBundle:
    root: Path
    manifest: ServingBundleManifest
''',
        '''            normalizer_digest=normalizer_digest,
            schema_version="serving_bundle_v4",
        )

    def with_release(self, release: ReleaseManifest) -> ServingBundleManifest:
        if self.schema_version != "serving_bundle_v4":
            raise ValueError("release attestation binding requires serving bundle v4")
        release.validate_bundle_identity(
            bundle_digest=self.bundle_digest,
            dataset_id=self.dataset_id,
            signal_digest=self.signal_digest,
            selection_digest=self.selection_digest,
            selected_policy_digest=self.policy_digest,
        )
        return replace(self, release_digest=release.digest)


@dataclass(frozen=True, slots=True)
class ServingBundle:
    root: Path
    manifest: ServingBundleManifest
    release: ReleaseManifest | None = None
''',
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '''    declared = {BUNDLE_MANIFEST_NAME}
    for file in manifest.files:
''',
        '''    declared = {BUNDLE_MANIFEST_NAME}
    release: ReleaseManifest | None = None
    if manifest.schema_version == "serving_bundle_v4" and manifest.release_digest is not None:
        release = load_release_attestation(root)
        if release.digest != manifest.release_digest:
            raise ValueError("release attestation pointer mismatch")
        release.validate_bundle_identity(
            bundle_digest=manifest.bundle_digest,
            dataset_id=manifest.dataset_id,
            signal_digest=manifest.signal_digest,
            selection_digest=manifest.selection_digest,
            selected_policy_digest=manifest.policy_digest,
        )
        declared.add(RELEASE_ATTESTATION_NAME)
    for file in manifest.files:
''',
    )
    replace_once(
        "trade_rl/serving/bundle.py",
        '    return ServingBundle(root=root, manifest=manifest)\n',
        '    return ServingBundle(root=root, manifest=manifest, release=release)\n',
    )

    replace_once(
        "trade_rl/serving/runtime.py",
        '''            release_digest=manifest.release_digest,
''',
        '''            release_digest=(
                None if bundle.release is None else bundle.release.digest
            ),
''',
    )
    replace_once(
        "trade_rl/serving/runtime.py",
        '''        if manifest.release_digest is None and not self.allow_unreleased:
            raise ValueError("serving bundle requires an approved release identity")
''',
        '''        if bundle.release is None and not self.allow_unreleased:
            raise ValueError("serving bundle requires a verified release attestation")
''',
    )
    replace_once(
        "trade_rl/serving/registry.py",
        '''        if bundle.manifest.release_digest is None and not self.allow_unreleased:
            raise ValueError("serving bundle requires an approved release identity")
''',
        '''        if bundle.release is None and not self.allow_unreleased:
            raise ValueError("serving bundle requires a verified release attestation")
''',
    )
    replace_once(
        "trade_rl/serving/registry.py",
        '            installed = staged\n',
        '            installed = load_serving_bundle(destination)\n',
    )

    helper = ROOT / "tests/serving/helpers.py"
    helper.write_text(
        '''from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.releases import ReleaseManifest
from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import ServingBundleManifest, write_serving_bundle_manifest
from trade_rl.serving.release import write_release_attestation
from trade_rl.serving.runtime import RuntimeIdentityContract

OBSERVATION_SIZE = 5
ACTION_NAMES = ("fast_tilt", "slow_tilt", "risk_tilt")
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})
INITIAL_CAPITAL = 250_000.0
_CREATED_AT = datetime(2026, 7, 13, tzinfo=UTC)


def create_bundle(
    root: Path,
    *,
    policy_mode: PolicyMode = PolicyMode.BASELINE_ONLY,
    release_digest: str | None = "released",
    observation_size: int = OBSERVATION_SIZE,
    action_names: tuple[str, ...] = ACTION_NAMES,
    action_spec_digest: str = ACTION_SPEC_DIGEST,
    environment_digest: str = "d" * 64,
    normalizer_digest: str | None = "9" * 64,
) -> Path:
    root.mkdir(parents=True)
    artifact_paths = ["dataset.json", "signal.json", "selection.json"]
    (root / "dataset.json").write_text('{"dataset":"a"}', encoding="utf-8")
    (root / "signal.json").write_text('{"signal":"rejected"}', encoding="utf-8")
    (root / "selection.json").write_text(
        f'{{"selection":"{policy_mode.value}"}}', encoding="utf-8"
    )
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
    normalizer_digest: str | None = "9" * 64,
    alpha_artifact_digest: str | None = None,
    factor_artifact_digest: str | None = None,
) -> RuntimeIdentityContract:
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


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: release_attestation.py tests|implementation")
    if sys.argv[1] == "tests":
        write_tests()
    else:
        apply_implementation()


if __name__ == "__main__":
    main()
