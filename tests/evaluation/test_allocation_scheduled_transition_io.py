"""Backend-free external-pin and relational sidecar discriminator controls."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256

import pytest

from tests.evaluation.test_allocation_scheduled_transition_trace import (
    fake_bundle,
    fake_capture,
)
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.rl_allocation.scheduled_transition_trace_io import (
    _manifest,
    read_scheduled_allocation_transition_trace,
)


def fixture(tmp_path):
    env, _, events, _ = fake_capture()
    bundle = fake_bundle(env)
    bundle["policy_sha256"] = sha256(b"inert-backend-free-fixture").hexdigest()
    pin = content_digest(bundle)
    saved = tmp_path / "bundle"
    saved.mkdir()
    (saved / "policy.zip").write_bytes(b"inert-backend-free-fixture")
    (saved / "manifest.json").write_bytes(canonical_json_bytes(bundle))
    root = tmp_path / "trace"
    root.mkdir()
    raw = b"".join(canonical_json_bytes(event) + b"\n" for event in events)
    manifest = _manifest(bundle, pin, events, raw, "none")
    (root / "transitions.jsonl").write_bytes(raw)
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return root, saved, content_digest(manifest), pin, manifest


def read(root, saved, pin, bundle_pin, **kwargs):
    return read_scheduled_allocation_transition_trace(
        root,
        expected_digest=pin,
        bundle_root=saved,
        expected_bundle_digest=bundle_pin,
        **kwargs,
    )


def test_closed_reader_external_bundle_and_required_diagnostics(tmp_path):
    root, saved, pin, bundle_pin, manifest = fixture(tmp_path)
    assert read(root, saved, pin, bundle_pin) == manifest
    with pytest.raises(ValueError, match="diagnostics"):
        read(root, saved, pin, bundle_pin, require_learner_diagnostics=True)
    with pytest.raises(ValueError):
        read(root, saved, pin, "f" * 64)
    with pytest.raises(ValueError):
        read(root, saved, "f" * 64, bundle_pin)


@pytest.mark.parametrize(
    "mutation",
    ["rows", "policy", "extra", "canonical", "manifest", "rehashed_relation"],
)
def test_trace_io_rejects_bytes_and_closed_rehashed_links(tmp_path, mutation):
    root, saved, pin, bundle_pin, manifest = fixture(tmp_path)
    if mutation == "rows":
        (root / "transitions.jsonl").write_bytes(b"{}\n")
    elif mutation == "policy":
        (saved / "policy.zip").write_bytes(b"changed")
    elif mutation == "extra":
        (root / "extra").write_bytes(b"extra")
    elif mutation == "canonical":
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    elif mutation == "manifest":
        altered = deepcopy(manifest)
        altered["input_summary"]["windows"][0]["actor_count"] += 1
        (root / "manifest.json").write_bytes(canonical_json_bytes(altered))
        pin = content_digest(altered)
    else:
        rows = [
            json.loads(line)
            for line in (root / "transitions.jsonl").read_bytes().splitlines()
        ]
        rows[4]["next"]["window_ordinal"] = 0
        raw = b"".join(canonical_json_bytes(event) + b"\n" for event in rows)
        manifest["transitions_sha256"] = sha256(raw).hexdigest()
        (root / "transitions.jsonl").write_bytes(raw)
        (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        pin = content_digest(manifest)
    with pytest.raises(ValueError):
        read(root, saved, pin, bundle_pin)


def test_symlink_rows_rejected(tmp_path):
    root, saved, pin, bundle_pin, _ = fixture(tmp_path)
    actual = tmp_path / "actual-rows"
    (root / "transitions.jsonl").rename(actual)
    try:
        os.symlink(actual, root / "transitions.jsonl")
    except OSError as error:
        pytest.skip(f"symlink permission unavailable: {error}")
    with pytest.raises(ValueError):
        read(root, saved, pin, bundle_pin)


def test_public_scheduled_reader_imports_with_optional_backend_blocked(tmp_path):
    root, saved, pin, bundle_pin, _ = fixture(tmp_path)
    code = """
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'stable_baselines3'}:
            raise RuntimeError('optional backend import reached: '+fullname)
sys.meta_path.insert(0,Deny())
from trade_rl.evaluation.rl_allocation.scheduled_transition_trace_io import read_scheduled_allocation_transition_trace
from pathlib import Path
value=read_scheduled_allocation_transition_trace(Path(sys.argv[1]),bundle_root=Path(sys.argv[2]),expected_digest=sys.argv[3],expected_bundle_digest=sys.argv[4])
assert value['transition_count']==12
assert 'torch' not in sys.modules and 'stable_baselines3' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(root), str(saved), pin, bundle_pin],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ordinary_single_source_reader_does_not_accept_scheduled_v5(tmp_path):
    from trade_rl.evaluation.rl_allocation.transition_validation import (
        validate_transition_events,
    )

    root, saved, pin, bundle_pin, _ = fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in (root / "transitions.jsonl").read_bytes().splitlines()
    ]
    bundle = json.loads((saved / "manifest.json").read_bytes())
    with pytest.raises(ValueError, match="v3 or v4"):
        validate_transition_events(rows, bundle)
    assert read(root, saved, pin, bundle_pin)["transition_count"] == 12
