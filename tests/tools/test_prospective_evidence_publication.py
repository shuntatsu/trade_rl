"""Synthetic, future-only immutable publication contract; not an agy v1 rerun."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from tools import prospective_evidence_publication as future
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.errors import (
    ArtifactIntegrityError,
    InvalidExperimentStateError,
    PublishedArtifactCleanupError,
)


def test_canonical_json_and_digest_are_verified_after_actual_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "prospective"
    payload = {"z": [2, 1], "a": {"finite": 0.3}}
    before = canonical_json_bytes(payload)
    receipt = future.publish_json_evidence_once(
        root=root, name="interim-synthetic.json", payload=payload
    )
    assert receipt.path == root / "interim-synthetic.json"
    assert receipt.sha256 == hashlib.sha256(before).hexdigest()
    assert receipt.size_bytes == len(before)
    assert receipt.path.read_bytes() == before
    assert stat.S_ISREG(os.stat(receipt.path).st_mode)


@pytest.mark.parametrize(
    "name",
    [
        "../escape.json",
        "parent/child.json",
        "",
        ".hidden.json",
        "X.json",
        "a.txt",
        "a.json\n",
        "a" * 81 + ".json",
    ],
)
def test_bad_names_are_rejected_without_creating_artifacts(
    tmp_path: Path, name: str
) -> None:
    root = tmp_path / "future"
    with pytest.raises(ValueError, match="name"):
        future.publish_json_evidence_once(root=root, name=name, payload={"x": 1})
    assert not root.exists()
    assert not (tmp_path / "escape.json").exists()


@pytest.mark.parametrize("payload", [float("nan"), {"x": math.inf}, {"x": object()}])
def test_invalid_payload_refused_before_any_file_open(
    tmp_path: Path, payload: object
) -> None:
    root = tmp_path / "future"
    with pytest.raises((TypeError, ValueError)):
        future.publish_json_evidence_once(
            root=root, name="result.json", payload=payload
        )
    assert not root.exists()


def test_oversized_payload_refused_before_touching_output(tmp_path: Path) -> None:
    root = tmp_path / "future"
    with pytest.raises(ValueError, match="size"):
        future.publish_json_evidence_once(
            root=root,
            name="huge.json",
            payload={"text": "z" * future.MAX_JSON_EVIDENCE_BYTES},
        )
    assert not root.exists()


def test_second_write_never_overwrites_original_even_with_identical_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "future"
    first = future.publish_json_evidence_once(
        root=root, name="result.json", payload={"v": 1}
    )
    original = first.path.read_bytes()
    for value in ({"v": 1}, {"v": 100}):
        with pytest.raises(InvalidExperimentStateError, match="already exists"):
            future.publish_json_evidence_once(
                root=root, name="result.json", payload=value
            )
        assert first.path.read_bytes() == original


def test_dangling_leaf_symlink_never_creates_external_target(tmp_path: Path) -> None:
    root = tmp_path / "future"
    root.mkdir()
    outside = tmp_path / "outside.json"
    leaf = root / "result.json"
    leaf.symlink_to(outside)
    with pytest.raises(InvalidExperimentStateError, match="already exists"):
        future.publish_json_evidence_once(
            root=root, name="result.json", payload={"v": 1}
        )
    assert leaf.is_symlink() and not outside.exists()


def test_ancestor_symlink_never_writes_outside(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "future"
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises((ArtifactIntegrityError, OSError, ValueError)):
        future.publish_json_evidence_once(
            root=root, name="result.json", payload={"v": 1}
        )
    assert not (outside / "result.json").exists()


def test_two_concurrent_thread_publishers_get_one_winner_and_no_overwrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "future"
    barrier = Barrier(2)

    def writer(value: int) -> str:
        barrier.wait(timeout=10)
        try:
            future.publish_json_evidence_once(
                root=root, name="result.json", payload={"seed": value}
            )
        except InvalidExperimentStateError:
            return "conflict"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            x.result(timeout=30)
            for x in (pool.submit(writer, 1), pool.submit(writer, 2))
        ]
    assert sorted(results) == ["conflict", "published"]
    assert (root / "result.json").read_bytes() in (
        canonical_json_bytes({"seed": 1}),
        canonical_json_bytes({"seed": 2}),
    )


def test_two_real_processes_cannot_both_publish_same_evidence(tmp_path: Path) -> None:
    """Both processes must be alive and barrier-ready before either may publish."""
    root = tmp_path / "future"
    release = tmp_path / "release"
    ready_a, ready_b = tmp_path / "ready-a", tmp_path / "ready-b"
    script = """
from pathlib import Path
import sys
import time
from tools.prospective_evidence_publication import publish_json_evidence_once
from trade_rl.evaluation.experiments.errors import InvalidExperimentStateError
ready, release = Path(sys.argv[3]), Path(sys.argv[4])
ready.touch()
deadline = time.monotonic() + 20
while not release.exists():
    if time.monotonic() > deadline:
        raise SystemExit('cross-process release barrier timed out')
    time.sleep(0.01)
try:
    publish_json_evidence_once(root=Path(sys.argv[1]), name='same.json', payload={'seed':sys.argv[2]})
except InvalidExperimentStateError:
    print('conflict')
else:
    print('published')
"""
    cwd = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-c", script, str(root)]
    first = subprocess.Popen(
        [*command, "A", str(ready_a), str(release)],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    workers = [first]
    try:
        second = subprocess.Popen(
            [*command, "B", str(ready_b), str(release)],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        workers.append(second)
        deadline = time.monotonic() + 20
        while not (ready_a.is_file() and ready_b.is_file()):
            assert time.monotonic() < deadline, "both workers must reach the barrier"
            assert first.poll() is None and second.poll() is None, (
                "a worker exited before the publication barrier"
            )
            time.sleep(0.01)
        assert first.poll() is None and second.poll() is None
        assert not (root / "same.json").exists()
        release.touch()
        a, aerr = first.communicate(timeout=30)
        b, berr = second.communicate(timeout=30)
    finally:
        for child in workers:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
    assert (first.returncode, second.returncode) == (0, 0), (aerr, berr)
    assert sorted((a.strip(), b.strip())) == ["conflict", "published"]
    assert (root / "same.json").read_bytes() in (
        canonical_json_bytes({"seed": "A"}),
        canonical_json_bytes({"seed": "B"}),
    )


def test_nested_payload_mutation_between_validation_and_publication_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish exact prevalidated bytes even when a caller mutates nested values."""
    from trade_rl.evaluation.experiments.store import StudyStore

    root = tmp_path / "future"
    source = {"nested": {"value": "before"}}
    original = canonical_json_bytes(source)
    actual_publish = StudyStore.publish_json_once

    def mutate_during_write(self: StudyStore, name: str, payload: object) -> Path:
        source["nested"]["value"] = "after"
        return actual_publish(self, name, payload)

    monkeypatch.setattr(StudyStore, "publish_json_once", mutate_during_write)
    result = future.publish_json_evidence_once(
        root=root, name="result.json", payload=source
    )
    assert result.path.read_bytes() == original
    assert result.sha256 == hashlib.sha256(original).hexdigest()
    assert source["nested"]["value"] == "after"


def test_mutation_past_size_budget_does_not_commit_oversized_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A postcheck race must not bypass the advertised 1 MiB admission budget."""
    from trade_rl.evaluation.experiments.store import StudyStore

    root = tmp_path / "future"
    source = {"value": "ok"}
    original = canonical_json_bytes(source)
    actual_publish = StudyStore.publish_json_once

    def mutate_during_write(self: StudyStore, name: str, payload: object) -> Path:
        source["value"] = "a" * (future.MAX_JSON_EVIDENCE_BYTES + 256)
        return actual_publish(self, name, payload)

    monkeypatch.setattr(StudyStore, "publish_json_once", mutate_during_write)
    result = future.publish_json_evidence_once(
        root=root, name="receipt.json", payload=source
    )
    assert result.path.read_bytes() == original
    assert result.size_bytes == len(original)
    assert len(result.path.read_bytes()) <= future.MAX_JSON_EVIDENCE_BYTES


def test_postpublication_readback_failure_never_claims_success_or_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "future"

    def unavailable(*args: object, **kwargs: object) -> bytes:
        raise OSError("readback failed after committed publication")

    monkeypatch.setattr(future, "read_verified_bytes", unavailable)
    with pytest.raises(PublishedArtifactCleanupError, match="do not retry"):
        future.publish_json_evidence_once(
            root=root, name="result.json", payload={"complete": True}
        )
    actual = (root / "result.json").read_bytes()
    assert actual == canonical_json_bytes({"complete": True})
    with pytest.raises(InvalidExperimentStateError):
        future.publish_json_evidence_once(
            root=root, name="result.json", payload={"complete": False}
        )
    assert (root / "result.json").read_bytes() == actual
