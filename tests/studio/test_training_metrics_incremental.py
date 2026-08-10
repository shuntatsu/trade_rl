from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.studio.contracts import JobSummary
from trade_rl.studio.errors import ArtifactInvalid
from trade_rl.studio.training_metrics import StudioTrainingMetricsReader

from .test_catalog import settings


def _job(run_id: str) -> JobSummary:
    return JobSummary(
        id=f"job-{run_id}",
        status="running",
        run_id=run_id,
        config_resource_id="config-resource",
        dataset_resource_id="dataset-resource",
        config_digest="c" * 64,
        dataset_id="d" * 64,
        config_path="configs/training.json",
        dataset_path="datasets/btc",
        artifact_root="research",
        submitted_at="2026-07-27T00:00:00+00:00",
        owner_instance_id="owner",
    )


def _event_file(tmp_path: Path, *, run_id: str, suffix: str = "0") -> Path:
    root = (
        tmp_path
        / "research"
        / ".staging"
        / run_id
        / "members"
        / "member-000"
        / "tensorboard"
        / "seed-3-ppo"
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"events.out.tfevents.{suffix}"
    path.touch()
    return path


def _write_scalar(
    path: Path,
    *,
    step: int,
    value: float,
    tag: str = "train/learning_rate",
    wall_time: float | None = None,
    append: bool = True,
) -> None:
    payload = {
        "step": step,
        "tag": tag,
        "value": value,
        "wall_time": float(step if wall_time is None else wall_time),
    }
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class _FakeAccumulator:
    def __init__(self, path: Path, owner: _AccumulatorFactory) -> None:
        self.path = path
        self.owner = owner
        self.events: dict[str, list[SimpleNamespace]] = {}

    def Reload(self) -> _FakeAccumulator:
        self.owner.reloads[self.path] += 1
        if self.owner.reload_delay:
            time.sleep(self.owner.reload_delay)
        if self.owner.fail_next[self.path]:
            self.owner.fail_next[self.path] = False
            raise RuntimeError("injected reload failure")
        by_tag: dict[str, list[SimpleNamespace]] = defaultdict(list)
        for line in self.path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            by_tag[str(raw["tag"])].append(
                SimpleNamespace(
                    step=int(raw["step"]),
                    wall_time=float(raw["wall_time"]),
                    value=float(raw["value"]),
                )
            )
        self.events = dict(by_tag)
        return self

    def Tags(self) -> dict[str, tuple[str, ...]]:
        return {"scalars": tuple(sorted(self.events))}

    def Scalars(self, tag: str) -> tuple[SimpleNamespace, ...]:
        self.owner.scalar_reads[(self.path, tag)] += 1
        return tuple(self.events.get(tag, ()))


class _AccumulatorFactory:
    def __init__(self, *, reload_delay: float = 0.0) -> None:
        self.creations: dict[Path, int] = defaultdict(int)
        self.reloads: dict[Path, int] = defaultdict(int)
        self.scalar_reads: dict[tuple[Path, str], int] = defaultdict(int)
        self.fail_next: dict[Path, bool] = defaultdict(bool)
        self.reload_delay = reload_delay

    def __call__(self, path: Path) -> _FakeAccumulator:
        self.creations[path] += 1
        return _FakeAccumulator(path, self)


def _reader(
    tmp_path: Path,
    factory: _AccumulatorFactory,
    *,
    max_cached_sources: int = 32,
) -> StudioTrainingMetricsReader:
    return StudioTrainingMetricsReader(
        settings(tmp_path),
        accumulator_factory=factory,
        max_cached_sources=max_cached_sources,
    )


def test_unchanged_status_and_scalars_share_one_loaded_snapshot(tmp_path: Path) -> None:
    path = _event_file(tmp_path, run_id="run-cache")
    _write_scalar(path, step=100, value=1.2e-4)
    factory = _AccumulatorFactory()
    reader = _reader(tmp_path, factory)
    job = _job("run-cache")

    status = reader.status(job, seed=3)
    page = reader.scalars(
        job,
        seed=3,
        tags=("train/learning_rate",),
        after_step=0,
        limit=512,
        generation=status.generation,
    )
    reader.status(job, seed=3)

    assert [point.step for point in page.series[0].points] == [100]
    assert factory.creations[path.resolve()] == 1
    assert factory.reloads[path.resolve()] == 1
    assert factory.scalar_reads[(path.resolve(), "train/learning_rate")] == 1


def test_reward_components_are_allowlisted_for_intermediate_audit(
    tmp_path: Path,
) -> None:
    path = _event_file(tmp_path, run_id="run-reward-components")
    tag = "trade_rl/reward_absolute_component_mean"
    _write_scalar(path, step=100, value=0.002, tag=tag)
    factory = _AccumulatorFactory()
    reader = _reader(tmp_path, factory)
    job = _job("run-reward-components")

    status = reader.status(job, seed=3)
    page = reader.scalars(
        job,
        seed=3,
        tags=(tag,),
        after_step=0,
        limit=512,
        generation=status.generation,
    )

    assert tag in status.available_tags
    assert page.series[0].display_name == "Mean absolute reward component"
    assert page.series[0].points[0].value == pytest.approx(0.002)


def test_append_reloads_once_and_preserves_generation(tmp_path: Path) -> None:
    path = _event_file(tmp_path, run_id="run-append")
    _write_scalar(path, step=100, value=1.2e-4)
    factory = _AccumulatorFactory()
    reader = _reader(tmp_path, factory)
    job = _job("run-append")
    initial = reader.status(job, seed=3)

    _write_scalar(path, step=200, value=1.0e-4)
    updated = reader.status(job, seed=3)
    page = reader.scalars(
        job,
        seed=3,
        tags=("train/learning_rate",),
        after_step=100,
        limit=512,
        generation=initial.generation,
    )

    assert updated.generation == initial.generation
    assert [point.step for point in page.series[0].points] == [200]
    assert factory.creations[path.resolve()] == 1
    assert factory.reloads[path.resolve()] == 2
    assert factory.scalar_reads[(path.resolve(), "train/learning_rate")] == 2


def test_only_changed_event_file_is_reloaded_and_reparsed(tmp_path: Path) -> None:
    first = _event_file(tmp_path, run_id="run-two-files", suffix="1")
    second = _event_file(tmp_path, run_id="run-two-files", suffix="2")
    _write_scalar(first, step=100, value=1.2e-4)
    _write_scalar(second, step=200, value=1.0e-4)
    factory = _AccumulatorFactory()
    reader = _reader(tmp_path, factory)
    job = _job("run-two-files")
    reader.status(job, seed=3)

    _write_scalar(second, step=300, value=8.0e-5)
    reader.status(job, seed=3)

    assert factory.reloads[first.resolve()] == 1
    assert factory.reloads[second.resolve()] == 2
    assert factory.scalar_reads[(first.resolve(), "train/learning_rate")] == 1
    assert factory.scalar_reads[(second.resolve(), "train/learning_rate")] == 2


def test_truncated_event_file_uses_fresh_accumulator(tmp_path: Path) -> None:
    path = _event_file(tmp_path, run_id="run-truncated")
    _write_scalar(path, step=100, value=1.2e-4)
    _write_scalar(path, step=200, value=1.0e-4)
    factory = _AccumulatorFactory()
    reader = _reader(tmp_path, factory)
    job = _job("run-truncated")
    reader.status(job, seed=3)

    _write_scalar(path, step=50, value=2.0e-4, append=False)
    reader.status(job, seed=3)

    assert factory.creations[path.resolve()] == 2
    assert factory.reloads[path.resolve()] == 2


def test_failed_reload_discards_cache_before_retry(tmp_path: Path) -> None:
    path = _event_file(tmp_path, run_id="run-failure")
    _write_scalar(path, step=100, value=1.2e-4)
    factory = _AccumulatorFactory()
    reader = _reader(tmp_path, factory)
    job = _job("run-failure")
    reader.status(job, seed=3)

    _write_scalar(path, step=200, value=1.0e-4)
    factory.fail_next[path.resolve()] = True
    with pytest.raises(ArtifactInvalid, match="malformed"):
        reader.status(job, seed=3)

    recovered = reader.status(job, seed=3)

    assert recovered.last_step == 200
    assert factory.creations[path.resolve()] == 2
    assert factory.reloads[path.resolve()] == 3


def test_source_cache_is_bounded_and_lru_evicted(tmp_path: Path) -> None:
    first = _event_file(tmp_path, run_id="run-lru-a")
    second = _event_file(tmp_path, run_id="run-lru-b")
    _write_scalar(first, step=100, value=1.2e-4)
    _write_scalar(second, step=100, value=1.1e-4)
    factory = _AccumulatorFactory()
    reader = _reader(tmp_path, factory, max_cached_sources=1)

    reader.status(_job("run-lru-a"), seed=3)
    reader.status(_job("run-lru-b"), seed=3)
    reader.status(_job("run-lru-a"), seed=3)

    assert factory.creations[first.resolve()] == 2
    assert factory.creations[second.resolve()] == 1


def test_concurrent_status_and_scalars_share_one_refresh(tmp_path: Path) -> None:
    path = _event_file(tmp_path, run_id="run-concurrent")
    _write_scalar(path, step=100, value=1.2e-4)
    factory = _AccumulatorFactory(reload_delay=0.05)
    reader = _reader(tmp_path, factory)
    job = _job("run-concurrent")
    start = threading.Barrier(2)

    def load_status() -> object:
        start.wait()
        return reader.status(job, seed=3)

    def load_scalars() -> object:
        start.wait()
        return reader.scalars(
            job,
            seed=3,
            tags=("train/learning_rate",),
            after_step=0,
            limit=512,
            generation=None,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda call: call(), (load_status, load_scalars)))

    assert len(results) == 2
    assert factory.creations[path.resolve()] == 1
    assert factory.reloads[path.resolve()] == 1
