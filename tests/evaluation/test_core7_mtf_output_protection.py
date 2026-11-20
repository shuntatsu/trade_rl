"""Synthetic writer boundary regressions: no outside-file publication."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.core7_mtf_study.runner import _write_json_canonical


def test_writer_is_canonical_and_write_once(tmp_path: Path) -> None:
    output = tmp_path / "new-study" / "decision.json"
    expected = {"b": 2, "a": [3, 1]}
    _write_json_canonical(output, expected)
    assert output.read_bytes() == canonical_json_bytes(expected)
    with pytest.raises(Exception, match="(already exists|File exists|occupied)"):
        _write_json_canonical(output, {"forged": True})
    assert output.read_bytes() == canonical_json_bytes(expected)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink guard")
@pytest.mark.parametrize("dangling", (False, True))
def test_writer_refuses_symlink_leaf_without_modifying_foreign_target(
    tmp_path: Path, dangling: bool
) -> None:
    output = tmp_path / "study" / "decision.json"
    output.parent.mkdir()
    foreign = tmp_path / "foreign.json"
    if not dangling:
        foreign.write_bytes(b"ORIGINAL_FOREIGN")
    output.symlink_to(foreign)

    with pytest.raises(Exception):
        _write_json_canonical(output, {"forged": True})
    assert output.is_symlink()
    if dangling:
        assert not foreign.exists()
    else:
        assert foreign.read_bytes() == b"ORIGINAL_FOREIGN"


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink guard")
def test_writer_refuses_symlink_root_without_writing_outside(
    tmp_path: Path,
) -> None:
    foreign_dir = tmp_path / "foreign-dir"
    foreign_dir.mkdir()
    (tmp_path / "study").symlink_to(foreign_dir, target_is_directory=True)

    with pytest.raises(Exception):
        _write_json_canonical(tmp_path / "study" / "decision.json", {"forged": True})
    assert not (foreign_dir / "decision.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink guard")
def test_writer_rejects_symlink_ancestor_chain_before_publication(
    tmp_path: Path,
) -> None:
    foreign_dir = tmp_path / "foreign-dir"
    foreign_dir.mkdir()
    (tmp_path / "bridge").symlink_to(foreign_dir, target_is_directory=True)
    with pytest.raises(Exception):
        _write_json_canonical(
            tmp_path / "bridge" / "study" / "decision.json", {"forged": True}
        )
    assert not (foreign_dir / "study").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink guard")
@pytest.mark.parametrize("leaf", ("study-manifest.json", "decision.json"))
def test_actual_runner_rejects_dangling_output_before_any_publication(
    tmp_path: Path, leaf: str
) -> None:
    """Exercise the real run_study gate, not only its private writer."""
    from tests.data.test_mtf6_signals import _make_alternating_dataset
    from trade_rl.evaluation.core7_mtf_study import protocol, run_study

    dataset = protocol.mark_synthetic_dataset(
        _make_alternating_dataset(n_bars=50, pairs=("BTCUSDT", "ETHUSDT"), seed=88)
    )
    output = tmp_path / "safe"
    output.mkdir()
    foreign = tmp_path / "foreign.json"
    (output / leaf).symlink_to(foreign)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_study(
            dataset,
            output_dir=output,
            expected_protocol_digest=protocol.protocol_digest(),
            allow_synthetic_unauthenticated=True,
            start_index=8,
            stop_index=48,
            fit_stop_index=8,
        )

    assert not foreign.exists()
    assert not (output / "study-manifest.json").is_file()
    assert not (output / "decision.json").is_file()
