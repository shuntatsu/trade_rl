"""Pure resource supervision oracles with manufactured processes and CIM rows."""

import json
import subprocess
from importlib import import_module
from types import SimpleNamespace

import pytest

from tools.tier_d_nonrl.tier_d_process_tree import ProcessTree, ProcessUsage


def api():
    try:
        return import_module("tools.tier_d_scaling.supervision")
    except ModuleNotFoundError as error:
        if error.name == "tools.tier_d_scaling.supervision":
            pytest.fail("bounded artificial scaling supervisor is missing")
        raise


def install(
    module,
    monkeypatch,
    root,
    *,
    exits=(0,),
    orphan=False,
    rss_excess=False,
    late_orphan=False,
    unobserved_root=False,
    provider_failure=False,
    cleanup_failure=False,
):
    clock, kills = [0.0], []
    outcomes = iter(exits)
    state = SimpleNamespace(
        leaf_alive=True, unrelated_alive=True, reaped=False, samples=0, cleanup_calls=0
    )

    class Process:
        pid = 10
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if self.returncode is None:
                outcome = next(outcomes)
                if isinstance(outcome, BaseException):
                    raise outcome
                if outcome is None:
                    clock[0] += timeout
                    raise subprocess.TimeoutExpired("MANUFACTURED", timeout)
                clock[0] += 2
                self.returncode = outcome
                if not orphan and not late_orphan:
                    state.leaf_alive = False
            state.reaped = True
            return self.returncode

        def terminate(self):
            self.returncode = -15

    process = Process()

    def provider():
        state.samples += 1
        if provider_failure and state.samples > 1:
            raise OSError("manufactured persistent CIM failure")
        rows = [ProcessUsage(99, 1, 99, 999_000_000)]
        if process.returncode is None and not unobserved_root:
            rows.append(ProcessUsage(10, 1, 100, 4_000_000))
        if (
            state.leaf_alive
            and not unobserved_root
            and not (late_orphan and process.returncode is None)
        ):
            rss = (
                module.LIMITS["rss_bytes"] + 1
                if rss_excess and clock[0] > 0
                else 30_000_000
            )
            rows.append(ProcessUsage(20, 10, 101, rss))
        return tuple(rows)

    tree = ProcessTree(10, provider=provider, expected_root_identity=(10, 100))

    def popen(command, **kwargs):
        assert command == ["MANUFACTURED_WORKER"]
        assert kwargs["cwd"] == module.REPO
        assert kwargs["env"]["PYTHONPATH"] == str(module.REPO)
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["stdout"].name == str(root / "worker.log")
        kwargs["stdout"].write("manufactured child output\n")
        kwargs["stdout"].flush()
        (root / "attempt").mkdir()
        (root / "attempt" / "partial.json").write_bytes(b'{"retained":"child-owned"}\n')
        return process

    def cleanup(actual, actual_tree):
        assert actual is process and actual_tree is tree
        state.cleanup_calls += 1
        try:
            sample = tree.sample()
        except OSError:
            sample = tree.last_sample
        kills.extend(member.pid for member in sample.owned_members)
        assert 99 not in kills
        if sample.ambiguous_members:
            raise RuntimeError("ambiguous RSS-only rows remain; refusing cleanup")
        state.leaf_alive = False
        if process.returncode is None:
            process.terminate()
        process.wait(timeout=30)
        if cleanup_failure:
            raise RuntimeError("manufactured cleanup verification failure")

    monkeypatch.setattr(module, "_platform_name", lambda: "nt")
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    class ManufacturedTree(ProcessTree):
        @classmethod
        def from_popen(cls, actual_process):
            assert actual_process is process
            return tree

    monkeypatch.setattr(module, "ProcessTree", ManufacturedTree)
    monkeypatch.setattr(module, "terminate_owned_tree", cleanup)
    return process, state, kills, clock


def report(root):
    return json.loads((root / "supervision.json").read_text())


def test_limits_are_fixed_and_unsupported_platform_rejects_before_any_effect(
    tmp_path, monkeypatch
):
    module, root = api(), tmp_path / "new"
    assert module.LIMITS == {
        "wall_seconds": 10_800,
        "rss_bytes": 2 * 1024**3,
        "disk_bytes": 4 * 1024**3,
        "sample_seconds": 10,
    }
    monkeypatch.setattr(module, "_platform_name", lambda: "posix")
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("spawned on unsupported platform"),
    )
    with pytest.raises(ValueError, match="Windows"):
        module.supervised(["MANUFACTURED_WORKER"], root)
    assert not root.exists()


def test_success_samples_resources_reaps_and_keeps_separate_worker_log(
    tmp_path, monkeypatch
):
    module, root = api(), tmp_path / "job"
    _, state, kills, _ = install(module, monkeypatch, root)
    assert module.supervised(["MANUFACTURED_WORKER"], root) == 0
    value = report(root)
    assert value["status"] == "COMPLETED" and value["returncode"] == 0
    assert value["failure"] is value["cleanup_error"] is None
    assert value["final_violation"] == [] and value["owned_members_remaining"] == 0
    assert value["worker_reaped"] and state.reaped and kills == []
    assert value["peak_rss_bytes"] == 34_000_000 and value["elapsed_seconds"] == 2
    assert (root / "worker.log").read_text() == "manufactured child output\n"
    events = [
        json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()
    ]
    assert {
        event["stage"] for event in events if event["event"] == "resource_sample"
    } >= {"initial", "final"}
    assert value["limits"] == module.LIMITS


def test_supervisor_fails_when_empty_samples_never_observe_worker_root(
    tmp_path, monkeypatch
):
    module, root = api(), tmp_path / "unobserved-root"
    _, state, kills, _ = install(module, monkeypatch, root, unobserved_root=True)

    assert module.supervised(["MANUFACTURED_WORKER"], root) == 1

    value = report(root)
    assert value["status"] == "FAILED"
    assert "root process birth identity was never observed" in value["failure"]
    assert value["returncode"] == 0 and state.reaped and value["worker_reaped"]
    assert value["owned_members_remaining"] == 0
    assert value["ambiguous_rss_members_remaining"] == 0
    assert kills == []


def test_sampled_aggregate_rss_kills_owned_leaf_and_never_unrelated_process(
    tmp_path, monkeypatch
):
    module, root = api(), tmp_path / "rss"
    _, state, kills, _ = install(
        module, monkeypatch, root, exits=(None,), rss_excess=True
    )
    assert module.supervised(["MANUFACTURED_WORKER"], root) == 1
    value = report(root)
    assert (
        "rss_bytes" in value["failure"]
        and value["peak_rss_bytes"] > module.LIMITS["rss_bytes"]
    )
    assert set(kills) == {10, 20} and state.unrelated_alive
    assert not state.leaf_alive and state.reaped
    assert value["owned_members_remaining"] == 0


@pytest.mark.parametrize("returncode", [0, 7])
@pytest.mark.parametrize("cap", ["disk_bytes", "wall_seconds"])
def test_final_caps_do_not_replace_nonzero_failure_or_rewrite_child_state(
    tmp_path, monkeypatch, returncode, cap
):
    module, root = api(), tmp_path / "final"
    process, state, _, _ = install(module, monkeypatch, root, exits=(returncode,))
    if cap == "disk_bytes":
        monkeypatch.setattr(
            module,
            "_disk_bytes",
            lambda _: (
                module.LIMITS["disk_bytes"] + 1 if process.returncode is not None else 0
            ),
        )
    else:
        monkeypatch.setattr(
            module,
            "monotonic",
            lambda: (
                module.LIMITS["wall_seconds"] + 1
                if process.returncode is not None
                else 0
            ),
        )
    assert module.supervised(["MANUFACTURED_WORKER"], root) == 1
    value = report(root)
    assert value["returncode"] == returncode and value["final_violation"] == [cap]
    if returncode:
        assert "7" in value["failure"] and cap not in value["failure"]
    assert value["worker_reaped"] and state.reaped
    assert (
        root / "attempt" / "partial.json"
    ).read_bytes() == b'{"retained":"child-owned"}\n'


def test_zero_exit_with_owned_orphan_is_failed_and_cleaned(tmp_path, monkeypatch):
    module, root = api(), tmp_path / "orphan"
    _, state, kills, _ = install(module, monkeypatch, root, orphan=True)
    assert module.supervised(["MANUFACTURED_WORKER"], root) == 1
    value = report(root)
    assert "descendant" in value["failure"] and value["returncode"] == 0
    assert kills == [20] and state.unrelated_alive and not state.leaf_alive
    assert value["owned_members_remaining"] == 0 and state.reaped


def test_late_ambiguous_orphan_blocks_success_and_is_never_killed(
    tmp_path, monkeypatch
):
    module, root = api(), tmp_path / "late-ambiguous"
    _, state, kills, _ = install(module, monkeypatch, root, late_orphan=True)

    assert module.supervised(["MANUFACTURED_WORKER"], root) == 1

    value = report(root)
    assert value["status"] == "FAILED"
    assert "ambiguous" in value["failure"]
    assert kills == []
    assert state.leaf_alive and state.reaped
    assert value["owned_members_remaining"] == 0
    assert value["ambiguous_rss_members_remaining"] == 1


def test_owned_leaf_found_by_final_verification_is_also_cleaned(tmp_path, monkeypatch):
    module, root = api(), tmp_path / "late-owned"
    _, state, kills, _ = install(module, monkeypatch, root, orphan=True)
    tree = ProcessTree(10, expected_root_identity=(10, 100))
    original = tree.provider

    def transiently_missing_final_snapshot():
        rows = original()
        return (
            tuple(row for row in rows if row.pid == 99) if state.samples == 2 else rows
        )

    tree.provider = transiently_missing_final_snapshot
    assert module.supervised(["MANUFACTURED_WORKER"], root) == 1
    value = report(root)
    assert not state.leaf_alive and kills == [20] and state.unrelated_alive
    assert value["owned_members_remaining"] == 0 and state.reaped


def test_persistent_snapshot_failure_still_invokes_retained_owned_cleanup(
    tmp_path, monkeypatch
):
    module, root = api(), tmp_path / "cim"
    _, state, kills, _ = install(
        module, monkeypatch, root, exits=(None,), provider_failure=True
    )
    assert module.supervised(["MANUFACTURED_WORKER"], root) == 1
    value = report(root)
    assert "persistent CIM failure" in value["failure"]
    assert state.cleanup_calls == 1 and set(kills) == {10, 20}
    assert state.reaped and not state.leaf_alive and state.unrelated_alive
    assert value["owned_members_remaining"] is None
    assert "persistent CIM failure" in value["cleanup_error"]


def test_keyboard_interrupt_and_cleanup_error_preserve_primary_separately(
    tmp_path, monkeypatch
):
    module, root = api(), tmp_path / "interrupt"
    _, state, kills, _ = install(
        module, monkeypatch, root, exits=(KeyboardInterrupt(),), cleanup_failure=True
    )
    assert module.supervised(["MANUFACTURED_WORKER"], root) == 1
    value = report(root)
    assert "KeyboardInterrupt" in value["failure"]
    assert "manufactured cleanup verification failure" in value["cleanup_error"]
    assert state.reaped and set(kills) == {10, 20} and state.unrelated_alive


def test_existing_supervisor_root_is_never_reopened(tmp_path, monkeypatch):
    module, root = api(), tmp_path / "old"
    root.mkdir()
    (root / "retained").write_bytes(b"old bytes")
    monkeypatch.setattr(module, "_platform_name", lambda: "nt")
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("reopened prior job"),
    )
    with pytest.raises(FileExistsError):
        module.supervised(["MANUFACTURED_WORKER"], root)
    assert [path.name for path in root.iterdir()] == ["retained"]


def test_summary_itself_cannot_silently_push_output_past_disk_cap(
    tmp_path, monkeypatch
):
    module, root = api(), tmp_path / "summary-cap"
    install(module, monkeypatch, root)
    monkeypatch.setattr(
        module,
        "_disk_bytes",
        lambda _: (
            module.LIMITS["disk_bytes"] + 1
            if (root / "supervision.json").exists()
            else module.LIMITS["disk_bytes"] - 1
        ),
    )
    assert module.supervised(["MANUFACTURED_WORKER"], root) == 1
    value = report(root)
    assert value["status"] == "FAILED" and value["final_violation"] == ["disk_bytes"]


def test_v3_protocol_limits_match_actual_supervision_report(tmp_path, monkeypatch):
    from tools.tier_d_scaling.fixture import write_full_artificial_dataset
    from tools.tier_d_scaling.spec import profile_protocol

    module, root = api(), tmp_path / "protocol-bound"
    dataset_root = tmp_path / "dataset"
    dataset = write_full_artificial_dataset(dataset_root)
    protocol = profile_protocol(dataset, dataset_root)
    _, state, kills, _ = install(module, monkeypatch, root)

    assert module.supervised(["MANUFACTURED_WORKER"], root) == 0
    value = report(root)
    expected_limits = {
        "wall_seconds": 10_800,
        "rss_bytes": 2 * 1024**3,
        "disk_bytes": 4 * 1024**3,
        "sample_seconds": 10,
    }
    assert protocol["schema"] == "issue888_artificial_scaling_v3"
    assert protocol["measurement"]["limits"] == expected_limits
    assert module.LIMITS == expected_limits
    assert value["limits"] == protocol["measurement"]["limits"]
    assert value["status"] == "COMPLETED" and value["returncode"] == 0
    assert state.reaped and value["worker_reaped"]
    assert value["owned_members_remaining"] == 0 and kills == []


def test_supervisor_completes_after_former_1200_second_cap(tmp_path, monkeypatch):
    module, root = api(), tmp_path / "long-prefix"
    _, state, kills, clock = install(
        module, monkeypatch, root, exits=(None,) * 130 + (0,)
    )

    assert module.supervised(["MANUFACTURED_WORKER"], root) == 0, report(root)
    value = report(root)
    assert value["status"] == "COMPLETED" and value["returncode"] == 0
    assert value["elapsed_seconds"] > 1_200 and clock[0] > 1_200
    assert value["limits"]["wall_seconds"] == 10_800
    assert value["limits"] == module.LIMITS
    assert state.reaped and value["worker_reaped"]
    assert value["owned_members_remaining"] == 0
    assert kills == [] and state.unrelated_alive
    assert (root / "attempt" / "partial.json").read_bytes() == (
        b'{"retained":"child-owned"}\n'
    )
    assert (root / "worker.log").read_text() == "manufactured child output\n"
