from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.market_b4_oracle.outer_witness import (
    WitnessFilePin,
    _make_root_observed_attempt,
    implementation_source_identity,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pin(path: Path, value: dict[str, object]) -> WitnessFilePin:
    raw = _canonical(value)
    path.write_bytes(raw)
    return WitnessFilePin(len(raw), hashlib.sha256(raw).hexdigest())


def test_windows_path_witness_pair_reads_back_with_external_pins(
    tmp_path: Path,
) -> None:
    output_root = (tmp_path / "attempt").absolute()
    output_root.mkdir()
    witness_root = (tmp_path / "witness").absolute()
    witness_root.mkdir()
    start_path, finish_path = witness_root / "start.json", witness_root / "finish.json"
    attempt_uuid = "a" * 32
    argv = ["python", "-m", "tools.tier_d_nonrl", "--mode", "synthetic"]
    source = {
        "closure_sha256": "b" * 64,
        "repository_head": "c" * 40,
        "working_tree_clean": False,
    }
    start = {
        "schema": "issue810_root_owned_attempt_start_v1",
        "attempt_uuid": attempt_uuid,
        "clock": {"wall_utc_ns": 10, "monotonic_ns": 20},
        "argv": argv,
        "output_root": str(output_root),
        "implementation_source": source,
    }
    finish = {
        "schema": "issue810_root_owned_attempt_finish_v1",
        "attempt_uuid": attempt_uuid,
        "clock": {"wall_utc_ns": 11, "monotonic_ns": 21},
        "argv_sha256": hashlib.sha256(_canonical(argv)).hexdigest(),
        "output_root": str(output_root),
        "implementation_source": source,
        "child": {
            "pid": 101,
            "birth_ticks": 102,
            "exit_code": 0,
            "owned_tree_empty": True,
            "observed_process_identities": [[101, 102]],
        },
        "output_manifest": {"bytes": 1, "sha256": "d" * 64},
    }
    start_pin = _pin(start_path, start)
    finish_pin = _pin(finish_path, finish)

    observed = _make_root_observed_attempt(
        start_path,
        finish_path,
        repository_head="c" * 40,
        implementation_source_sha256="b" * 64,
    )

    assert observed.readback(
        expected_start_pin=start_pin,
        expected_finish_pin=finish_pin,
    ) == (start, finish)


def test_source_identity_uses_bounded_descriptor_reads_not_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    source_paths = (
        repository / "tools" / "market_b4_oracle" / "oracle.py",
        repository / "tools" / "market_replay_oracle" / "reader.py",
        repository / "trade_rl" / "artifacts" / "identity.py",
        repository / "tools" / "market_b4_outer_runner.py",
        repository / "tools" / "tier_d_nonrl" / "tier_d_process_tree.py",
    )
    for path in source_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# pinned synthetic source\n", encoding="utf-8")

    def forbidden_read_bytes(_: Path) -> bytes:
        raise AssertionError("source identity used an unbounded read_bytes call")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    identity = implementation_source_identity(repository)

    assert identity["files"]
    assert len(identity["closure_sha256"]) == 64


def _synthetic_source_tree(repository: Path) -> tuple[Path, ...]:
    source_paths = (
        repository / "tools" / "market_b4_oracle" / "oracle.py",
        repository / "tools" / "market_replay_oracle" / "reader.py",
        repository / "trade_rl" / "artifacts" / "identity.py",
        repository / "tools" / "market_b4_outer_runner.py",
        repository / "tools" / "tier_d_nonrl" / "tier_d_process_tree.py",
    )
    for path in source_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# pinned synthetic source\n", encoding="utf-8")
    return source_paths


def test_source_identity_rejects_file_growth_between_stat_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.market_b4_oracle import outer_witness

    repository = tmp_path / "repo"
    target = _synthetic_source_tree(repository)[0]
    original_open = outer_witness.os.open
    grew = False

    def grow_before_open(
        path: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        nonlocal grew
        if Path(path) == target and not grew:
            grew = True
            with target.open("ab") as stream:
                stream.write(b"changed after stat")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(outer_witness.os, "open", grow_before_open)

    with pytest.raises(ValueError, match="comparison source changed while opening"):
        implementation_source_identity(repository)


def test_source_identity_rejects_repository_parent_reparse_point(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    _synthetic_source_tree(repository)
    alias = tmp_path / "repo-junction"
    try:
        alias.symlink_to(repository, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="link or reparse point"):
        implementation_source_identity(alias)


def test_existing_target_rejects_reparse_ancestor(tmp_path: Path) -> None:
    from tools.market_b4_outer_runner import _safe_target

    real_parent = tmp_path / "real-parent"
    target = real_parent / "attempt"
    target.mkdir(parents=True)
    alias = tmp_path / "parent-alias"
    try:
        alias.symlink_to(real_parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="reparse point"):
        _safe_target(alias / "attempt", must_exist=True)


def test_fresh_target_rejects_missing_ancestor(tmp_path: Path) -> None:
    from tools.market_b4_outer_runner import _safe_target

    missing_parent = tmp_path / "missing-parent"

    with pytest.raises(ValueError, match="parent directory does not exist"):
        _safe_target(missing_parent / "attempt", must_exist=False)


def test_frozen_h_witness_root_stays_in_fresh_ignored_verification_scope(
    tmp_path: Path,
) -> None:
    from tools.market_b4_outer_runner import _fresh_h_witness_root

    repository = tmp_path / "observer"
    verify = repository / "output" / "verify"
    verify.mkdir(parents=True)
    candidate = verify / "attempt-1"

    assert _fresh_h_witness_root(repository, candidate) == candidate
    with pytest.raises(ValueError, match="ignored verification scope"):
        _fresh_h_witness_root(repository, tmp_path / "outside-witness")

    candidate.mkdir()
    with pytest.raises(FileExistsError, match="never overwritten"):
        _fresh_h_witness_root(repository, candidate)


def test_native_launch_rejects_arbitrary_python_before_process_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tools.market_b4_outer_runner import run_and_consume_market_arm

    def forbidden_spawn(*args: object, **kwargs: object) -> None:
        raise AssertionError("native command reached process creation")

    monkeypatch.setattr(
        "tools.market_b4_outer_runner.subprocess.Popen", forbidden_spawn
    )
    with pytest.raises(ValueError, match="exact native CLI binding"):
        run_and_consume_market_arm(
            argv=[str(Path(__import__("sys").executable).absolute()), "-c", "pass"],
            repository=Path.cwd(),
            output_root=Path.cwd() / "native-output",
            witness_root=Path.cwd() / "native-witness",
            content=object(),
            policy=object(),
            binding=SimpleNamespace(comparison_only=False),
            role="NONRL",
            reserve_plan_digest="a" * 64,
            static_inputs=object(),
        )


def test_frozen_h_plan_emits_only_the_exact_public_market_cli(tmp_path: Path) -> None:
    from tools.market_b4_outer_runner import _frozen_h_public_argv

    python = tmp_path / "h" / ".venv" / "Scripts" / "python.exe"
    attempt = tmp_path / "h" / "output" / "tier-d-nonrl" / "fresh-attempt"
    protocol = tmp_path / "protocol-v3.json"
    review = tmp_path / "review.json"

    argv = _frozen_h_public_argv(
        child_python=python,
        attempt_root=attempt,
        protocol_path=protocol,
        review_path=review,
        max_workers=4,
    )

    assert argv == (
        str(python),
        "-B",
        "-m",
        "tools.tier_d_nonrl",
        "--mode",
        "market",
        "--attempt-root",
        str(attempt),
        "--protocol",
        str(protocol),
        "--review-record",
        str(review),
        "--max-workers",
        "4",
    )
    assert "--worker" not in argv
    assert "--resume-from" not in argv


def test_frozen_h_runtime_pin_uses_exact_locked_interpreter_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from tools.market_b4_outer_runner import capture_frozen_h_runtime_pin

    source_root = tmp_path / "frozen-h"
    (source_root / ".venv" / "Scripts").mkdir(parents=True)
    python = source_root / ".venv" / "Scripts" / "python.exe"
    python.write_bytes(b"bounded synthetic interpreter pin")
    (source_root / "pyproject.toml").write_text(
        """[project]
requires-python = ">=3.12,<3.13"
dependencies = ["numpy>=1.26.0,<2.0.0", "gymnasium==0.29.1"]
[project.optional-dependencies]
forecast-gbm = ["lightgbm[scikit-learn]==4.7.0"]
""",
        encoding="utf-8",
    )
    (source_root / "uv.lock").write_text(
        """[[package]]
name = "numpy"
version = "1.26.4"
[[package]]
name = "gymnasium"
version = "0.29.1"
[[package]]
name = "lightgbm"
version = "4.7.0"
[[package]]
name = "scikit-learn"
version = "1.9.0"
[[package]]
name = "scipy"
version = "1.17.1"
""",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "candidate_run_runtime_v1",
        "python": {
            "implementation": "CPython",
            "version": "3.12.0 synthetic test runtime",
        },
        "os": {"family": "Windows", "release": "11"},
        "machine": "AMD64",
        "packages": {
            "trade-rl": "0.3.0",
            "numpy": "1.26.4",
            "gymnasium": "0.29.1",
            "lightgbm": "4.7.0",
            "stable-baselines3": None,
            "torch": None,
        },
    }
    probe = {
        "schema": "issue810_frozen_h_runtime_probe_v1",
        "source_root": str(source_root),
        "python_executable": str(python),
        "module_root": str(source_root),
        "module_resolution": {
            "working_directory": str(source_root),
            "sys_path0": str(source_root),
            "trade_rl_origin": str(source_root / "trade_rl" / "__init__.py"),
            "tier_d_nonrl_main": str(
                source_root / "tools" / "tier_d_nonrl" / "__main__.py"
            ),
        },
        "runtime_environment": manifest,
        "support_packages": {"scikit-learn": "1.9.0", "scipy": "1.17.1"},
    }
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(probe, sort_keys=True, separators=(",", ":")).encode(),
            stderr=b"",
        )

    monkeypatch.setattr("tools.market_b4_outer_runner.subprocess.run", fake_run)
    pin = capture_frozen_h_runtime_pin(source_root)

    assert pin.source_root == source_root
    assert pin.python_executable == python
    assert (
        pin.python_executable_sha256 == hashlib.sha256(python.read_bytes()).hexdigest()
    )
    assert pin.authority_payload()["runtime_environment"] == manifest
    assert pin.authority_payload()["probe_facts"]["module_root"] == str(source_root)
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == [str(python), "-B", "-c"]
    assert "-I" not in command
    assert kwargs["cwd"] == source_root
    assert kwargs["timeout"] == 30
    assert set(kwargs["env"]) <= {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUTF8",
    }
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert kwargs["env"]["PYTHONUTF8"] == "1"
    assert not {
        "PYTHONPATH",
        "PYTHONHOME",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    } & set(kwargs["env"])


def test_frozen_h_plan_cannot_be_built_from_artificial_binding(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from tools.market_b4_outer_runner import build_frozen_h_launch_plan

    with pytest.raises(ValueError, match="artificial inputs"):
        build_frozen_h_launch_plan(
            observer_repository=tmp_path,
            supervisor_attempt_root=tmp_path / "attempt",
            witness_root=tmp_path / "witness",
            content=object(),
            policy=object(),
            binding=SimpleNamespace(comparison_only=True),
            runtime_pin=object(),
            static_inputs=None,
            review_record_path=tmp_path / "review.json",
            review_record_pin=WitnessFilePin(1, "a" * 64),
        )


def test_timeout_after_popen_reaps_process_and_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.market_b4_outer_runner import _run_root_observed_attempt

    monkeypatch.setattr(
        "tools.market_b4_outer_runner._root_owned_process_tree_supported",
        lambda: True,
    )

    class FakeProcess:
        pid = 12345
        returncode: int | None = None
        terminated = False
        cleanup_started = False

        def poll(self) -> int | None:
            if self.cleanup_started:
                raise OSError("synthetic exit poll failure")
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.cleanup_started = True
            self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is None or timeout > 0
            if self.returncode is None:
                self.returncode = -15
            return self.returncode

    class FakeSample:
        def __init__(self, members: tuple[object, ...]) -> None:
            self.members = members

    class FakeTree:
        def __init__(self) -> None:
            self.root_identity = (process.pid, 99)
            self.owned = {self.root_identity: 0}
            self.last_sample = FakeSample((object(),))

        def sample(self) -> FakeSample:
            self.last_sample = FakeSample(
                () if process.returncode is not None else (object(),)
            )
            return self.last_sample

    process = FakeProcess()
    tree = FakeTree()
    repository = tmp_path / "repository"
    repository.mkdir()
    output_root = tmp_path / "output"
    witness_root = tmp_path / "witness"
    source = {
        "closure_sha256": "a" * 64,
        "files": [],
        "runtime": {"packages": {"numpy": "2", "trade-rl": "2"}},
        "repository_head": "b" * 40,
        "working_tree_clean": False,
        "git_status_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        "tools.market_b4_outer_runner.implementation_source_identity",
        lambda _: source,
    )
    monkeypatch.setattr(
        "tools.market_b4_outer_runner._git_readback",
        lambda _: ("b" * 40, False, "c" * 64),
    )
    monkeypatch.setattr(
        "tools.market_b4_outer_runner.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "tools.market_b4_outer_runner._new_process_tree", lambda _: tree
    )
    monkeypatch.setattr(
        "tools.market_b4_outer_runner._stop_owned_tree",
        lambda target, _: target.terminate(),
    )
    monkeypatch.setattr(
        "tools.market_b4_outer_runner._observe_owned_tree",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            TimeoutError("synthetic observer timeout")
        ),
    )

    with pytest.raises(TimeoutError, match="synthetic observer timeout"):
        _run_root_observed_attempt(
            argv=[str(Path(__import__("sys").executable).absolute()), "-c", "pass"],
            repository=repository,
            output_root=output_root,
            witness_root=witness_root,
            input_authority={"scope": "ARTIFICIAL_COMPARISON_ONLY"},
            timeout_seconds=0.01,
        )

    assert process.terminated
    assert process.returncode == -15
    assert not (witness_root / "finish.json").exists()
    failure = json.loads((witness_root / "failure.json").read_text(encoding="utf-8"))
    assert failure["child"]["exit_code"] == -15
    assert failure["child"]["owned_tree_empty"] is True
    assert failure["cleanup"]["status"] == "CLEANUP_INCOMPLETE"
    assert any(
        "synthetic exit poll failure" in row for row in failure["cleanup"]["errors"]
    )


@pytest.mark.skipif(
    __import__("os").name != "nt",
    reason="frozen root-owned process-tree observer is Windows-only",
)
def test_real_timeout_stops_sleeping_child_and_descendant_and_keeps_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    import sys
    import time
    from types import SimpleNamespace

    from tools.market_b4_outer_runner import _run_root_observed_attempt
    from tools.tier_d_nonrl.tier_d_process_tree import (
        ProcessTree,
        terminate_owned_tree,
        windows_process_snapshot,
    )

    repository = tmp_path / "repository"
    repository.mkdir()
    output_root = tmp_path / "output"
    witness_root = tmp_path / "witness"
    pid_file = tmp_path / "descendant.pid"
    source = {
        "closure_sha256": "a" * 64,
        "files": [],
        "runtime": {"packages": {"numpy": "2", "trade-rl": "2"}},
        "repository_head": "b" * 40,
        "working_tree_clean": False,
        "git_status_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        "tools.market_b4_outer_runner.implementation_source_identity",
        lambda _: source,
    )
    monkeypatch.setattr(
        "tools.market_b4_outer_runner._git_readback",
        lambda _: ("b" * 40, False, "c" * 64),
    )
    spawned: dict[str, object] = {}
    owned: dict[str, ProcessTree] = {}
    real_popen = subprocess.Popen
    real_new_tree = __import__(
        "tools.market_b4_outer_runner", fromlist=["_new_process_tree"]
    )._new_process_tree

    def tracked_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        spawned["process"] = process
        return process

    def tracked_tree(pid: int) -> ProcessTree:
        tree = real_new_tree(pid)
        owned["tree"] = tree
        return tree

    monkeypatch.setattr(
        "tools.market_b4_outer_runner.subprocess",
        SimpleNamespace(
            Popen=tracked_popen,
            DEVNULL=subprocess.DEVNULL,
            CREATE_NO_WINDOW=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ),
    )
    monkeypatch.setattr("tools.market_b4_outer_runner._new_process_tree", tracked_tree)
    child_code = (
        "import subprocess,sys,time; "
        "desc=subprocess.Popen([sys.executable,'-B','-c',"
        "'import time; time.sleep(30)']); "
        "open(sys.argv[1],'w',encoding='ascii').write(str(desc.pid)); "
        "time.sleep(30)"
    )
    command = [sys.executable, "-B", "-c", child_code, str(pid_file)]
    try:
        with pytest.raises(TimeoutError, match="bounded runtime"):
            _run_root_observed_attempt(
                argv=command,
                repository=repository,
                output_root=output_root,
                witness_root=witness_root,
                input_authority={"scope": "ARTIFICIAL_COMPARISON_ONLY"},
                timeout_seconds=4.0,
            )
        deadline = time.monotonic() + 2.0
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.is_file()
        descendant_pid = int(pid_file.read_text(encoding="ascii"))
        process = spawned["process"]
        tree = owned["tree"]
        assert isinstance(process, subprocess.Popen)
        assert isinstance(tree, ProcessTree)
        failure = json.loads(
            (witness_root / "failure.json").read_text(encoding="utf-8")
        )
        observed = failure["child"]
        assert observed["pid"] == process.pid
        assert type(observed["birth_ticks"]) is int
        assert observed["exit_observed"] is True
        assert observed["exit_code"] != 0
        assert observed["owned_tree_empty"] is True
        assert any(
            row[0] == descendant_pid for row in observed["observed_process_identities"]
        )
        assert "output_manifest" not in failure
        assert (witness_root / "start.json").is_file()
        assert not (witness_root / "finish.json").exists()
        assert process.wait(timeout=1) != 0
        identities = set(tree.owned)
        current_identities = {item.identity for item in windows_process_snapshot()}
        assert identities.isdisjoint(current_identities)
        assert not tree.sample().members
    finally:
        process = spawned.get("process")
        tree = owned.get("tree")
        if isinstance(process, subprocess.Popen):
            try:
                if isinstance(tree, ProcessTree):
                    terminate_owned_tree(process, tree)
                elif process.poll() is None:
                    process.terminate()
                process.wait(timeout=30)
            except Exception:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=30)
            if isinstance(tree, ProcessTree):
                assert not tree.sample().members
