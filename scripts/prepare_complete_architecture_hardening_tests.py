from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "tests/architecture/test_complete_architecture_hardening.py"
text = path.read_text(encoding="utf-8")
text = text.replace(
    "from tests.serving.helpers import create_bundle\n",
    '''from datetime import UTC, datetime

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.selection import PolicyMode
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import ServingBundleManifest, write_serving_bundle_manifest
''',
    1,
)
marker = "\n\ndef dataset(\n"
helper = '''

ACTION_NAMES = ("fast_tilt", "slow_tilt", "risk_tilt")
ACTION_SPEC_DIGEST = content_digest({"names": ACTION_NAMES})


def create_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    artifact_paths = ("dataset.json", "signal.json", "selection.json", "release.json")
    (root / "dataset.json").write_text('{"dataset":"a"}', encoding="utf-8")
    (root / "signal.json").write_text('{"signal":"rejected"}', encoding="utf-8")
    (root / "selection.json").write_text(
        '{"selection":"baseline_only"}', encoding="utf-8"
    )
    (root / "release.json").write_text('{"release":"approved"}', encoding="utf-8")
    manifest = ServingBundleManifest.build(
        root=root,
        dataset_id="a" * 64,
        action_schema=ACTION_SCHEMA,
        action_size=len(ACTION_NAMES),
        action_names=ACTION_NAMES,
        action_spec_digest=ACTION_SPEC_DIGEST,
        observation_schema=OBSERVATION_SCHEMA,
        observation_size=5,
        environment_digest="d" * 64,
        initial_capital=250_000.0,
        policy_mode=PolicyMode.BASELINE_ONLY,
        policy_digest=None,
        signal_digest="b" * 64,
        selection_digest="c" * 64,
        release_digest="f" * 64,
        normalizer_digest="9" * 64,
        artifact_paths=artifact_paths,
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    write_serving_bundle_manifest(root, manifest)
    return root
'''
if marker not in text:
    raise RuntimeError("hardening test dataset marker is missing")
text = text.replace(marker, helper + marker, 1)
path.write_text(text, encoding="utf-8")
