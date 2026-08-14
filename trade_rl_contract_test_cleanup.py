#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

EXPECTED_BRANCH = "integration/cost-aware-causal-teacher-final2"

STUDIO_SUPPORT = r'''from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from trade_rl.studio.api import create_app
from trade_rl.studio.catalog import StudioCatalog
from trade_rl.studio.contracts import ConfigSummary, DatasetSummary, TrainingJobRequest
from trade_rl.studio.errors import ResourceNotFound
from trade_rl.studio.jobs import JobSupervisor
from trade_rl.studio.resource_ids import resource_id
from trade_rl.studio.settings import StudioSettings

from .helpers import write_dataset, write_run


def settings(
    tmp_path: Path,
    *,
    dataset_roots: tuple[Path, ...] | None = None,
    run_roots: tuple[Path, ...] | None = None,
) -> StudioSettings:
    return StudioSettings(
        project_root=tmp_path,
        dataset_roots=dataset_roots or (tmp_path / "datasets",),
        run_roots=run_roots or (tmp_path / "research",),
        config_roots=(tmp_path / "configs",),
        job_root=tmp_path / "jobs",
    )


class FakeProcess:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.exit_code: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.exit_code = -15

    def kill(self) -> None:
        self.exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0 if self.exit_code is None else self.exit_code


class FakeFactory:
    def __init__(self, *, pid: int = 4242) -> None:
        self.process = FakeProcess(pid)
        self.commands: list[tuple[str, ...]] = []
        self.logs: list[Path] = []
        self.cwds: list[Path] = []

    def __call__(
        self, command: tuple[str, ...], *, cwd: Path, log_path: Path
    ) -> FakeProcess:
        self.commands.append(command)
        self.logs.append(log_path)
        self.cwds.append(cwd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("started\n", encoding="utf-8")
        return self.process


class FakeCatalog:
    def __init__(self, root: Path) -> None:
        config_path = root / "configs" / "training.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{}", encoding="utf-8")
        dataset_path = root / "datasets" / "btc"
        dataset_path.mkdir(parents=True, exist_ok=True)
        self.config = SimpleNamespace(
            path=config_path,
            summary=ConfigSummary(
                id=resource_id("config", "configs/training.json", "c" * 64),
                config_digest="c" * 64,
                name="training",
                relative_path="configs/training.json",
                algorithm="ppo",
                status="VALID",
            ),
        )
        self.dataset = SimpleNamespace(
            path=dataset_path,
            summary=DatasetSummary(
                id=resource_id("dataset", "datasets/btc", "d" * 64),
                dataset_id="d" * 64,
                name="btc",
                relative_path="datasets/btc",
                market="continuous",
                symbols=("BTCUSDT",),
                timeframes=("1h",),
                range="2026-01-01 — 2026-01-02",
                status="VALID",
                feature_count=1,
                bar_count=12,
                symbol_count=1,
                updated="2026-01-01T00:00:00+00:00",
            ),
        )

    def resolve_config(self, value: str) -> SimpleNamespace:
        if value != self.config.summary.id:
            raise ResourceNotFound(value)
        return self.config

    def resolve_dataset(self, value: str) -> SimpleNamespace:
        if value != self.dataset.summary.id:
            raise ResourceNotFound(value)
        return self.dataset


def request(catalog: FakeCatalog, *, run_id: str = "run-001") -> TrainingJobRequest:
    return TrainingJobRequest(
        config_resource_id=catalog.config.summary.id,
        dataset_resource_id=catalog.dataset.summary.id,
        run_id=run_id,
    )


def studio_client(
    tmp_path: Path,
) -> tuple[TestClient, FakeFactory, FakeCatalog, StudioCatalog]:
    write_dataset(tmp_path / "datasets" / "btc")
    write_run(tmp_path / "research")
    real_catalog = StudioCatalog(settings(tmp_path))
    job_catalog = FakeCatalog(tmp_path)
    factory = FakeFactory()
    supervisor = JobSupervisor(
        settings(tmp_path),
        catalog=job_catalog,
        process_factory=factory,
    )
    return (
        TestClient(
            create_app(
                settings(tmp_path),
                catalog=real_catalog,
                supervisor=supervisor,
            )
        ),
        factory,
        job_catalog,
        real_catalog,
    )
'''

STUDIO_CONFTEST = r'''from __future__ import annotations

import pytest

import trade_rl.studio.jobs as studio_jobs

from .support import FakeProcess


@pytest.fixture(autouse=True)
def isolate_fake_process_tree_termination(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Studio fake workers from crossing into host process-management APIs."""

    original = studio_jobs._terminate_process_tree

    def terminate(process: studio_jobs.ProcessHandle) -> int:
        if isinstance(process, FakeProcess):
            process.terminate()
            return process.wait()
        return original(process)

    monkeypatch.setattr(studio_jobs, "_terminate_process_tree", terminate)
'''

OLD_CAUSAL_TEST = r'''def test_one_way_cost_rate_uses_first_executable_row_after_signal_delay() -> None:
    config = ExecutionCostConfig(
        fee_rate=0.0005,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0004,
        spread_rate=0.0002,
        impact_rate=0.0001,
        max_participation_rate=0.25,
        order_type="market",
    )

    rates = causal_alpha_one_way_cost_rates(
        _CostDataset(),
        config,
        decision_indices=np.asarray([10, 14]),
        signal_delay_decisions=1,
        decision_bars=4,
    )

    assert rates.tolist() == pytest.approx(
        [
            0.0005 + 0.0015 + 0.0004 + 0.0025 + 0.0002 + 0.0035 + 0.00005,
            0.0005 + 0.0019 + 0.0004 + 0.0029 + 0.0002 + 0.0039 + 0.00005,
        ]
    )
'''

NEW_CAUSAL_TEST = r'''def test_one_way_cost_rate_is_causal_and_ignores_future_execution_rows() -> None:
    config = ExecutionCostConfig(
        fee_rate=0.0005,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0004,
        spread_rate=0.0002,
        impact_rate=0.0001,
        max_participation_rate=0.25,
        order_type="market",
    )
    decisions = np.asarray([10, 14])
    dataset = _CostDataset()

    baseline = causal_alpha_one_way_cost_rates(
        dataset,
        config,
        decision_indices=decisions,
        signal_delay_decisions=1,
        decision_bars=4,
    )

    execution_rows = decisions + 1 + 4
    for name in ("fee_rate", "maker_fee_rate", "taker_fee_rate", "spread_rate"):
        dataset._values[name][execution_rows, 0] = 9.0
    dataset._values["max_participation_rate"][execution_rows, 0] = 0.99
    mutated = causal_alpha_one_way_cost_rates(
        dataset,
        config,
        decision_indices=decisions,
        signal_delay_decisions=1,
        decision_bars=4,
    )

    assert baseline.tolist() == pytest.approx([0.00715, 0.00835])
    assert mutated.tolist() == pytest.approx(baseline.tolist())
'''

DUPLICATE_DOCKER_TEST_RE = re.compile(
    r"\n\ndef test_training_image_is_digest_pinned_and_generation_scoped\(\) -> None:\n"
    r".*?"
    r"    assert \"TRADE_RL_CONFIRMATION_KEYS\" not in compose\n",
    re.DOTALL,
)

OLD_BENCHMARK_GUARD = r'''    if not sys.platform.startswith("linux") or not Path("/proc").is_dir():
        raise RuntimeError(
            "benchmark process-tree RSS measurement requires Linux /proc"
        )
    runtime_version = importlib.metadata.version("nautilus_trader")
'''

NEW_BENCHMARK_GUARD = r'''    _require_process_tree_rss_support()
    runtime_version = importlib.metadata.version("nautilus_trader")
'''

BENCHMARK_HELPER = r'''def _require_process_tree_rss_support() -> None:
    if not sys.platform.startswith("linux") or not Path("/proc").is_dir():
        raise RuntimeError(
            "benchmark process-tree RSS measurement requires Linux /proc"
        )


'''

OLD_BENCHMARK_TEST_PATCH = r'''    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda _: "1.230.0")
    monkeypatch.setattr(benchmark, "_run_worker_subprocess", fake_worker)

    evidence = benchmark.run_benchmark(timesteps=(8,), dataset_artifact=root)
'''

NEW_BENCHMARK_TEST_PATCH = r'''    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda _: "1.230.0")
    monkeypatch.setattr(benchmark, "_run_worker_subprocess", fake_worker)
    monkeypatch.setattr(benchmark, "_require_process_tree_rss_support", lambda: None)

    evidence = benchmark.run_benchmark(timesteps=(8,), dataset_artifact=root)
'''

BENCHMARK_GUARD_TEST = r'''def test_process_tree_rss_support_rejects_non_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="requires Linux /proc"):
        benchmark._require_process_tree_rss_support()


'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finish trade_rl contract-first test cleanup."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def run(root: Path, *cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=root, check=True)


def capture(root: Path, *cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=root, text=True).strip()


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, content: str) -> None:
    if not content.endswith("\n"):
        content += "\n"
    current = read(path) if path.exists() else None
    if current != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print("updated", path)


def remove_type_ignores(root: Path) -> None:
    replacements = {
        "trade_rl/studio/system_probe.py": [
            (
                'load = os.getloadavg()[0]  # type: ignore[attr-defined]',
                'load = os.getloadavg()[0]',
            )
        ],
        "trade_rl/studio/jobs.py": [
            (
                'os.killpg(process.pid, signal.SIGTERM)  # type: ignore[attr-defined]',
                'os.killpg(process.pid, signal.SIGTERM)',
            ),
            (
                'os.killpg(process.pid, signal.SIGKILL)  # type: ignore[attr-defined]',
                'os.killpg(process.pid, signal.SIGKILL)',
            ),
        ],
        "scripts/nautilus_training_throughput_benchmark.py": [
            (
                'importlib.metadata.version("nautilus_trader")  # type: ignore[unreachable]',
                'importlib.metadata.version("nautilus_trader")',
            ),
            (
                'resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]',
                'resource.getrusage(resource.RUSAGE_SELF)',
            ),
            (
                'resource.getrusage(resource.RUSAGE_CHILDREN)  # type: ignore[attr-defined]',
                'resource.getrusage(resource.RUSAGE_CHILDREN)',
            ),
        ],
        "trade_rl/workflows/symbol_triplet_stage_orchestrator.py": [
            (
                'fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]',
                'fcntl.flock(handle.fileno(), fcntl.LOCK_EX)',
            ),
            (
                'fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]',
                'fcntl.flock(handle.fileno(), fcntl.LOCK_UN)',
            ),
        ],
        "trade_rl/workflows/symbol_triplet_stage_state.py": [
            (
                'fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]',
                'fcntl.flock(handle.fileno(), fcntl.LOCK_EX)',
            ),
            (
                'fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]',
                'fcntl.flock(handle.fileno(), fcntl.LOCK_UN)',
            ),
        ],
    }
    for relative, pairs in replacements.items():
        path = root / relative
        text = read(path)
        for old, new in pairs:
            text = text.replace(old, new)
        write(path, text)


def restore_telemetry_comment(root: Path) -> None:
    path = root / "trade_rl/telemetry/_indexed_storage.py"
    text = read(path)
    marker = '''                if path_stat.st_size != self._expected_size:
                    index = _refresh_index_unlocked(self.path).index
'''
    restored = '''                if path_stat.st_size != self._expected_size:
                    # Another writer may have appended while this instance was
                    # idle. Reconcile only when the stream size actually moved.
                    index = _refresh_index_unlocked(self.path).index
'''
    if marker in text:
        text = text.replace(marker, restored, 1)
    elif restored not in text:
        raise RuntimeError("telemetry reconciliation location changed unexpectedly")
    write(path, text)


def fix_historical_streaming(root: Path) -> None:
    path = root / "trade_rl/integrations/nautilus/historical_streaming.py"
    text = read(path)
    text = text.replace(
        "from multiprocessing.connection import Connection, PipeConnection",
        "from multiprocessing.connection import Connection",
    )
    text = text.replace("connection: PipeConnection | Connection", "connection: Connection")
    text = text.replace("connection.close()  # type: ignore[unreachable]", "connection.close()")
    if "PipeConnection" in text:
        raise RuntimeError("PipeConnection remains after portability cleanup")
    write(path, text)


def fix_benchmark(root: Path) -> None:
    source = root / "scripts/nautilus_training_throughput_benchmark.py"
    text = read(source)
    if "def _require_process_tree_rss_support()" not in text:
        anchor = "\ndef run_benchmark(\n"
        if anchor not in text:
            raise RuntimeError("run_benchmark anchor not found")
        text = text.replace(anchor, "\n" + BENCHMARK_HELPER + "def run_benchmark(\n", 1)
    if OLD_BENCHMARK_GUARD in text:
        text = text.replace(OLD_BENCHMARK_GUARD, NEW_BENCHMARK_GUARD, 1)
    elif NEW_BENCHMARK_GUARD not in text:
        raise RuntimeError("benchmark guard shape changed unexpectedly")
    write(source, text)

    test = root / "tests/test_nautilus_training_throughput_benchmark.py"
    text = read(test)
    if OLD_BENCHMARK_TEST_PATCH in text:
        text = text.replace(OLD_BENCHMARK_TEST_PATCH, NEW_BENCHMARK_TEST_PATCH, 1)
    elif NEW_BENCHMARK_TEST_PATCH not in text:
        raise RuntimeError("benchmark orchestration test shape changed unexpectedly")
    if "test_process_tree_rss_support_rejects_non_linux" not in text:
        anchor = "def test_run_benchmark_binds_persisted_source_to_workers_and_evidence("
        if anchor not in text:
            raise RuntimeError("benchmark test insertion anchor not found")
        text = text.replace(anchor, BENCHMARK_GUARD_TEST + anchor, 1)
    write(test, text)


def fix_license_contract(root: Path) -> None:
    path = root / "tests/test_license_contract.py"
    text = read(path)
    old = '''    assert set(project["license-files"]) >= {
        "LICENSE",
        "LICENSES/*",
        "THIRD_PARTY_NOTICES.md",
    }
'''
    new = '''    assert set(project["license-files"]) == {"LICENSE", "LICENSES/*"}
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError("license-files test shape changed unexpectedly")
    text = text.replace(
        'notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")',
        'notices = (ROOT / "LICENSES" / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")',
    )
    write(path, text)


def fix_causal_test(root: Path) -> None:
    path = root / "tests/workflows/test_universal_causal_alpha_prediction_availability.py"
    text = read(path)
    if OLD_CAUSAL_TEST in text:
        text = text.replace(OLD_CAUSAL_TEST, NEW_CAUSAL_TEST, 1)
    elif "test_one_way_cost_rate_is_causal_and_ignores_future_execution_rows" not in text:
        raise RuntimeError("causal cost test shape changed unexpectedly")
    write(path, text)


def fix_checkpoint_float(root: Path) -> None:
    path = root / "tests/studio/test_checkpoint_evaluations_api.py"
    text = read(path)
    if "\nimport pytest\n" not in text:
        text = text.replace("from pathlib import Path\n", "from pathlib import Path\n\nimport pytest\n", 1)
    text = text.replace(
        'assert payload["items"][0]["totalReturn"] == 0.05',
        'assert payload["items"][0]["totalReturn"] == pytest.approx(0.05)',
    )
    write(path, text)


def remove_duplicate_docker_contract(root: Path) -> None:
    path = root / "tests/architecture/test_complete_trust_boundary_remediation.py"
    text = read(path)
    text, count = DUPLICATE_DOCKER_TEST_RE.subn("", text, count=1)
    if count == 0 and "def test_training_image_is_digest_pinned_and_generation_scoped" in text:
        raise RuntimeError("duplicate Docker contract shape changed unexpectedly")
    write(path, text)


def fix_compose_paths(root: Path) -> None:
    for relative in (
        "tests/test_training_compose_contract.py",
        "tests/architecture/test_complete_trust_boundary_remediation.py",
    ):
        path = root / relative
        text = read(path).replace('"docker" / "docker/', '"docker" / "')
        write(path, text)


def reorganize_studio_tests(root: Path) -> None:
    studio = root / "tests/studio"
    write(studio / "support.py", STUDIO_SUPPORT)
    write(studio / "conftest.py", STUDIO_CONFTEST)

    jobs = studio / "test_jobs.py"
    text = read(jobs)
    text = text.replace("from types import SimpleNamespace\n", "")
    text = text.replace(
        "from trade_rl.studio.contracts import ConfigSummary, DatasetSummary, TrainingJobRequest\n",
        "",
    )
    text = text.replace(
        "from trade_rl.studio.errors import IdentityConflict, JobOwnershipLost, ResourceNotFound\n",
        "from trade_rl.studio.errors import IdentityConflict, JobOwnershipLost\n",
    )
    text = text.replace("from trade_rl.studio.resource_ids import resource_id\n", "")
    text = text.replace("from .test_catalog import settings\n", "")
    if "from .support import FakeCatalog" not in text:
        text = text.replace(
            "from .helpers import write_run\n",
            "from .helpers import write_run\n"
            "from .support import FakeCatalog, FakeFactory, FakeProcess, request, settings\n",
            1,
        )
    text, count = re.subn(
        r"\nclass FakeProcess:.*?\n\ndef test_submit_training_persists_fixed_command_and_reconciles_success",
        "\n\ndef test_submit_training_persists_fixed_command_and_reconciles_success",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count == 0 and "class FakeProcess:" in text:
        raise RuntimeError("Studio job helper block shape changed unexpectedly")
    write(jobs, text)

    api = studio / "test_api.py"
    text = read(api)
    for line in (
        "from fastapi.testclient import TestClient\n",
        "from trade_rl.studio.api import create_app\n",
        "from trade_rl.studio.catalog import StudioCatalog\n",
        "from trade_rl.studio.jobs import JobSupervisor\n",
        "from .helpers import write_dataset, write_run\n",
        "from .test_catalog import settings\n",
        "from .test_jobs import FakeCatalog, FakeFactory, request\n",
    ):
        text = text.replace(line, "")
    if "from .support import request, studio_client as client\n" not in text:
        text = text.replace(
            "from pathlib import Path\n",
            "from pathlib import Path\n\nfrom .support import request, studio_client as client\n",
            1,
        )
    text, count = re.subn(
        r"\n\ndef client\(.*?\n\ndef test_read_endpoints_return_collision_free_validated_resources_and_no_go",
        "\n\ndef test_read_endpoints_return_collision_free_validated_resources_and_no_go",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count == 0 and "\ndef client(" in text:
        raise RuntimeError("Studio API client helper shape changed unexpectedly")
    write(api, text)

    catalog = studio / "test_catalog.py"
    text = read(catalog)
    text = text.replace("from trade_rl.studio.settings import StudioSettings\n", "")
    if "from .support import settings\n" not in text:
        text = text.replace(
            "from .helpers import write_dataset, write_run\n",
            "from .helpers import write_dataset, write_run\nfrom .support import settings\n",
            1,
        )
    text, count = re.subn(
        r"\n\ndef settings\(.*?\n\ndef write_config",
        "\n\ndef write_config",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count == 0 and "\ndef settings(" in text:
        raise RuntimeError("Studio settings helper shape changed unexpectedly")
    write(catalog, text)

    for path in sorted(studio.glob("test_*.py")):
        text = read(path)
        text = text.replace(
            "from .test_api import client",
            "from .support import studio_client as client",
        )
        text = text.replace(
            "from .test_jobs import request",
            "from .support import request",
        )
        text = text.replace(
            "from .test_catalog import settings",
            "from .support import settings",
        )
        write(path, text)


def static_contract_checks(root: Path) -> None:
    problems: list[str] = []

    for path in root.rglob("*.py"):
        try:
            text = read(path)
        except UnicodeDecodeError:
            continue
        if "docker/docker/" in text:
            problems.append(f"double Docker path: {path.relative_to(root)}")

    if "PipeConnection" in read(root / "trade_rl/integrations/nautilus/historical_streaming.py"):
        problems.append("nonportable PipeConnection import remains")

    trust = read(root / "tests/architecture/test_complete_trust_boundary_remediation.py")
    if "def test_training_image_is_digest_pinned_and_generation_scoped" in trust:
        problems.append("duplicated Docker trust-boundary test remains")

    for path in (root / "tests/studio").glob("test_*.py"):
        text = read(path)
        for old_import in (
            "from .test_api import",
            "from .test_jobs import",
            "from .test_catalog import settings",
        ):
            if old_import in text:
                problems.append(f"test-to-test Studio import remains: {path.name}: {old_import}")

    license_test = read(root / "tests/test_license_contract.py")
    if 'ROOT / "THIRD_PARTY_NOTICES.md"' in license_test:
        problems.append("obsolete root THIRD_PARTY_NOTICES contract remains")

    causal = read(root / "tests/workflows/test_universal_causal_alpha_prediction_availability.py")
    if "uses_first_executable_row_after_signal_delay" in causal:
        problems.append("future-cost leakage expectation remains")
    if "is_causal_and_ignores_future_execution_rows" not in causal:
        problems.append("causal no-lookahead regression is missing")

    for obsolete in (root / ".importlinter", root / "uv.toml"):
        if obsolete.exists():
            problems.append(f"obsolete root config still exists: {obsolete.name}")

    if problems:
        raise RuntimeError("static cleanup checks failed:\n- " + "\n- ".join(problems))


def format_changed_areas(root: Path) -> None:
    run(
        root,
        "uv",
        "run",
        "ruff",
        "format",
        "tests/studio",
        "tests/architecture/test_complete_trust_boundary_remediation.py",
        "tests/test_training_compose_contract.py",
        "tests/test_license_contract.py",
        "tests/test_nautilus_training_throughput_benchmark.py",
        "tests/workflows/test_universal_causal_alpha_prediction_availability.py",
        "scripts/nautilus_training_throughput_benchmark.py",
        "trade_rl/integrations/nautilus/historical_streaming.py",
        "trade_rl/telemetry/_indexed_storage.py",
        "trade_rl/studio/jobs.py",
        "trade_rl/studio/system_probe.py",
        "trade_rl/workflows/symbol_triplet_stage_orchestrator.py",
        "trade_rl/workflows/symbol_triplet_stage_state.py",
    )


def verify(root: Path) -> None:
    format_changed_areas(root)
    static_contract_checks(root)
    run(
        root,
        "uv",
        "run",
        "pytest",
        "-q",
        "tests/test_training_compose_contract.py",
        "tests/examples/test_docker_training_assets.py",
        "tests/studio",
        "tests/test_license_contract.py",
        "tests/test_nautilus_training_throughput_benchmark.py",
        "tests/workflows/test_universal_causal_alpha_prediction_availability.py",
    )
    run(root, "uv", "run", "ruff", "check", ".")
    run(root, "uv", "run", "ruff", "format", "--check", ".")
    run(root, "uv", "run", "mypy", ".")
    run(root, "uv", "run", "lint-imports")
    run(root, "uv", "run", "pytest", "-q")
    run(root, "git", "diff", "--check")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "trade_rl").is_dir():
        raise SystemExit(f"not a trade_rl repository root: {root}")

    branch = capture(root, "git", "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise SystemExit(
            f"expected branch {EXPECTED_BRANCH!r}, found {branch!r}; switch branches first"
        )

    if not args.allow_dirty:
        status = capture(root, "git", "status", "--porcelain")
        if status:
            raise SystemExit(
                "working tree is not clean; commit/stash unrelated changes or use --allow-dirty"
            )

    fix_historical_streaming(root)
    remove_type_ignores(root)
    restore_telemetry_comment(root)
    fix_benchmark(root)
    fix_license_contract(root)
    fix_causal_test(root)
    fix_checkpoint_float(root)
    remove_duplicate_docker_contract(root)
    fix_compose_paths(root)
    reorganize_studio_tests(root)
    static_contract_checks(root)

    if args.verify:
        verify(root)

    print("\ncontract-first test cleanup applied")
    print("review with: git status && git diff --check && git diff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
