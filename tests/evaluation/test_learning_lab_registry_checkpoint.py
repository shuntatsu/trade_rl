"""Source-bound, synthetic trial-registry checkpoint/anchor contract."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tools import learning_lab as cli
from trade_rl.evaluation.learning_lab import registry as reg


def _record(seed: int) -> reg.TrialRecord:
    return reg.TrialRecord(
        method_id="L1-PPO",
        config_digest="a" * 64,
        seed=seed,
        split="validate",
        code_commit="b" * 40,
        metrics={"sharpe_per_period": 0.05},
        status="completed",
    )


def _registry(path: Path, count: int = 3) -> reg.TrialRegistry:
    registry = reg.TrialRegistry(path)
    for seed in range(count):
        assert registry.append(_record(seed)) == seed
    return registry


def _cli(*args: str) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(list(args), stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def test_exported_checkpoint_is_verified_last_line_digest_and_read_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trials.jsonl"
    registry = _registry(path)
    original = path.read_bytes()
    checkpoint = registry.checkpoint()
    assert checkpoint.trial_count == 3
    assert (
        checkpoint.head_sha256 == hashlib.sha256(original.splitlines()[-1]).hexdigest()
    )
    assert checkpoint.to_dict() == {
        "schema": reg.CHECKPOINT_SCHEMA,
        "trial_count": 3,
        "head_sha256": checkpoint.head_sha256,
    }
    assert registry.snapshot(expected_checkpoint=checkpoint)[1] == checkpoint
    assert path.read_bytes() == original


def test_empty_checkpoint_never_creates_a_registry(tmp_path: Path) -> None:
    path = tmp_path / "absent.jsonl"
    checkpoint = reg.TrialRegistry(path).checkpoint()
    assert checkpoint.trial_count == 0
    assert checkpoint.head_sha256 == reg.GENESIS
    assert not path.exists()
    assert reg.TrialRegistry(path).snapshot(expected_checkpoint=checkpoint) == (
        [],
        checkpoint,
    )


def test_saved_checkpoint_detects_complete_tail_deletion(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "trials.jsonl")
    pinned = registry.checkpoint()
    original = registry.path.read_bytes().splitlines(keepends=True)
    registry.path.write_bytes(b"".join(original[:-1]))
    assert registry.verify() == 2  # internally valid prefix: absence undetectable alone
    damaged = registry.path.read_bytes()
    with pytest.raises(reg.RegistryError, match="checkpoint.*count"):
        registry.snapshot(expected_checkpoint=pinned)
    assert registry.path.read_bytes() == damaged


def test_saved_checkpoint_rejects_rewritten_last_line_with_valid_chain(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "trials.jsonl")
    pinned = registry.checkpoint()
    original = registry.path.read_bytes().splitlines(keepends=True)
    registry.path.write_bytes(b"".join(original[:-1]))
    assert registry.append(_record(999)) == 2  # valid 3-line registry, different head
    assert registry.verify() == 3
    with pytest.raises(reg.RegistryError, match="checkpoint.*hash"):
        registry.snapshot(expected_checkpoint=pinned)


def test_checkpoint_requires_exact_count_unless_extensions_are_explicit(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "trials.jsonl", count=2)
    pinned = registry.checkpoint()
    registry.append(_record(2))
    with pytest.raises(reg.RegistryError, match="checkpoint.*count"):
        registry.snapshot(expected_checkpoint=pinned)
    records, latest = registry.snapshot(
        expected_checkpoint=pinned, allow_extensions=True
    )
    assert [record.seed for record in records] == [0, 1, 2]
    assert latest.trial_count == 3 and latest.head_sha256 != pinned.head_sha256


def test_checkpoint_rejects_wrong_prefix_even_with_extensions(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "trials.jsonl", count=2)
    pinned = registry.checkpoint()
    original = registry.path.read_bytes().splitlines(keepends=True)
    registry.path.write_bytes(original[0])
    registry.append(_record(97))
    registry.append(_record(98))
    with pytest.raises(reg.RegistryError, match="checkpoint.*hash"):
        registry.snapshot(expected_checkpoint=pinned, allow_extensions=True)


@pytest.mark.parametrize(
    ("count", "head"),
    (
        (True, reg.GENESIS),
        (-1, reg.GENESIS),
        (1.0, reg.GENESIS),
        (1, "ABCDEF" * 10 + "abcd"),
        (1, "no-hash"),
        (0, "a" * 64),
    ),
)
def test_checkpoint_type_and_genesis_validation_refuse(
    count: object, head: object
) -> None:
    with pytest.raises(reg.RegistryError):
        reg.RegistryCheckpoint(trial_count=count, head_sha256=head)  # type: ignore[arg-type]


def test_registry_snapshot_is_a_single_locked_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path / "trials.jsonl")
    pinned = registry.checkpoint()
    called = 0
    original = reg.TrialRegistry._read_lines

    def observe(descriptor: int) -> list[bytes]:
        nonlocal called
        called += 1
        return original(descriptor)

    monkeypatch.setattr(reg.TrialRegistry, "_read_lines", staticmethod(observe))
    records, checkpoint = registry.snapshot(expected_checkpoint=pinned)
    assert len(records) == 3 and checkpoint == pinned
    assert called == 1


def test_cli_checkpoint_export_and_anchored_registry_end_to_end(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "trials.jsonl")
    original = registry.path.read_bytes()
    code, raw, err = _cli("checkpoint", "--registry", str(registry.path))
    assert code == cli.EXIT_OK and not err
    exported = json.loads(raw)
    assert exported["schema"] == reg.CHECKPOINT_SCHEMA
    args = (
        "registry",
        "--registry",
        str(registry.path),
        "--expected-trial-count",
        str(exported["trial_count"]),
        "--expected-head-sha256",
        exported["head_sha256"],
    )
    code, raw, err = _cli(*args)
    assert code == cli.EXIT_OK and not err
    assert json.loads(raw)["checkpoint_verified"] is True
    assert json.loads(raw)["trials"] == 3
    assert registry.path.read_bytes() == original

    registry.path.write_bytes(b"".join(original.splitlines(keepends=True)[:-1]))
    damaged = registry.path.read_bytes()
    code, raw, err = _cli(*args)
    assert code == cli.EXIT_REFUSED and not raw and "checkpoint" in err
    assert registry.path.read_bytes() == damaged


def test_cli_appended_trials_require_explicit_extension_opt_in(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "trials.jsonl", count=2)
    pinned = registry.checkpoint()
    registry.append(_record(2))
    args = (
        "registry",
        "--registry",
        str(registry.path),
        "--expected-trial-count",
        str(pinned.trial_count),
        "--expected-head-sha256",
        pinned.head_sha256,
    )
    assert _cli(*args)[0] == cli.EXIT_REFUSED
    code, raw, err = _cli(*args, "--allow-checkpoint-extensions")
    assert code == cli.EXIT_OK and not err
    assert json.loads(raw)["trials"] == 3
    assert json.loads(raw)["checkpoint_verified"] is True


@pytest.mark.parametrize(
    "flags",
    (
        ("--expected-trial-count", "1"),
        ("--expected-head-sha256", "a" * 64),
        ("--allow-checkpoint-extensions",),
        ("--expected-trial-count", "-1", "--expected-head-sha256", "a" * 64),
        ("--expected-trial-count", "2", "--expected-head-sha256", "bad"),
    ),
)
def test_cli_refuses_partial_or_invalid_external_checkpoint(
    tmp_path: Path, flags: tuple[str, ...]
) -> None:
    path = tmp_path / "absent.jsonl"
    code, raw, err = _cli("registry", "--registry", str(path), *flags)
    assert code == cli.EXIT_REFUSED and not raw and "checkpoint" in err
    assert not path.exists()


def test_subprocess_cli_checkpoint_replay_requires_original_external_pin(
    tmp_path: Path,
) -> None:
    """Exercise the installed Python CLI entrypoint, not only an injected main()."""
    registry = _registry(tmp_path / "trials.jsonl")
    root = Path(__file__).resolve().parents[2]
    start = [sys.executable, "-m", "tools.learning_lab"]
    export = subprocess.run(
        [*start, "checkpoint", "--registry", str(registry.path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    checkpoint = json.loads(export.stdout)
    args = [
        *start,
        "registry",
        "--registry",
        str(registry.path),
        "--expected-trial-count",
        str(checkpoint["trial_count"]),
        "--expected-head-sha256",
        checkpoint["head_sha256"],
    ]
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == cli.EXIT_OK and not result.stderr
    assert json.loads(result.stdout)["checkpoint_verified"] is True

    lines = registry.path.read_bytes().splitlines(keepends=True)
    registry.path.write_bytes(b"".join(lines[:-1]))
    refused = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=30)
    assert refused.returncode == cli.EXIT_REFUSED
    assert not refused.stdout and "checkpoint" in refused.stderr


def test_pinned_deflated_view_includes_all_trials_from_one_verified_snapshot(
    tmp_path: Path,
) -> None:
    registry = reg.TrialRegistry(tmp_path / "trials.jsonl")
    metrics = {
        "sharpe_per_period": 0.05,
        "n_obs": 100,
        "skewness": 0.0,
        "kurtosis": 3.0,
    }
    for seed in range(2):
        assert registry.append(replace(_record(seed), metrics=metrics)) == seed
    pinned = registry.checkpoint()
    args = (
        "registry",
        "--registry",
        str(registry.path),
        "--deflated-for-last",
        "--expected-trial-count",
        str(pinned.trial_count),
        "--expected-head-sha256",
        pinned.head_sha256,
    )
    code, raw, error = _cli(*args)
    assert code == cli.EXIT_OK and not error
    result = json.loads(raw)
    assert result["checkpoint_verified"] is True
    assert result["deflated_sharpe"]["n_trials"] == 2.0
    assert result["trials"] == 2
    registry.append(replace(_record(2), metrics=metrics))
    assert _cli(*args)[0] == cli.EXIT_REFUSED
    code, raw, error = _cli(*args, "--allow-checkpoint-extensions")
    assert code == cli.EXIT_OK and not error
    extended = json.loads(raw)
    assert extended["deflated_sharpe"]["n_trials"] == 3.0
    assert extended["trials"] == 3
