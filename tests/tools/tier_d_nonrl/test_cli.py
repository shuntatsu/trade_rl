"""CLI wiring only; no subprocess, Dataset, fitting, or native execution."""

import importlib
import sys
from types import SimpleNamespace

import pytest


def test_supervisor_passes_frozen_inputs_and_concurrency_to_absent_child_root(tmp_path):
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    child = tmp_path / "supervision/attempt"
    command = module.worker_command(
        "market",
        child,
        tmp_path / "review.json",
        protocol_path=tmp_path / "protocol.json",
        resume_from=tmp_path / "old-attempt",
        max_workers=3,
    )
    assert command[2:4] == ["-m", "tools.tier_d_nonrl"]
    assert command[command.index("--attempt-root") + 1] == str(child)
    assert command[command.index("--resume-from") + 1] == str(
        (tmp_path / "old-attempt").resolve()
    )
    assert command[command.index("--protocol") + 1] == str(
        (tmp_path / "protocol.json").resolve()
    )
    assert command[command.index("--max-workers") + 1] == "3"
    assert not child.exists()


def test_market_cli_refuses_missing_frozen_protocol_before_outputs(
    tmp_path, monkeypatch
):
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    root = tmp_path / "must-stay-absent"
    monkeypatch.setattr(
        sys, "argv", ["tier_d_nonrl", "--mode", "market", "--attempt-root", str(root)]
    )
    with pytest.raises(ValueError, match="frozen protocol"):
        module.main()
    assert not root.exists()


@pytest.mark.parametrize("count", [0, 7, True, 1.0])
def test_concurrency_is_a_declared_native_integer_within_one_to_six(count):
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_protocol")
    with pytest.raises(ValueError, match="1..6"):
        module.worker_count(count)


def test_non_windows_market_supervision_refuses_before_io(tmp_path, monkeypatch):
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_supervisor")
    runner = importlib.import_module("tools.tier_d_nonrl.tier_d_runner")
    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix"))

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported supervision reached protocol/source IO")

    monkeypatch.setattr(runner, "load_market_protocol", forbidden)
    root = tmp_path / "must-stay-absent"
    with pytest.raises(ValueError, match="Windows"):
        module.supervised_run("market", root, None)
    assert not root.exists()


def test_resource_proposal_bounds_pairs_by_declared_workers():
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_protocol")
    protocol = module.market_protocol(max_workers=3)
    assert (
        protocol["resource_proposal"]["maximum_inflight_pairs"]
        == protocol["concurrency"]["max_workers"]
        == 3
    )
