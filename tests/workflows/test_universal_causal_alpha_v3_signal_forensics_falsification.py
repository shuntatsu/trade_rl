from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.workflows.test_universal_causal_alpha_v3_signal_forensics import (
    _api,
    _build_run,
    _digest,
    _load_metric,
    _rewrite_metric,
)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_signal_forensics_never_mutates_source_run(tmp_path: Path) -> None:
    _build_run(tmp_path)
    before = _snapshot(tmp_path)

    _api().load_causal_alpha_v3_signal_forensics(tmp_path)

    assert _snapshot(tmp_path) == before


def test_signal_forensics_report_identity_excludes_source_path(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _build_run(first_root)
    _build_run(second_root)

    first = _api().load_causal_alpha_v3_signal_forensics(first_root)
    second = _api().load_causal_alpha_v3_signal_forensics(second_root)

    assert first.digest == second.digest
    assert first.to_payload() == second.to_payload()


def test_signal_forensics_rejects_cross_fit_chronology_drift(tmp_path: Path) -> None:
    built = _build_run(tmp_path)
    fit_digest = built["fit_configs"][1]
    for symbol in ("BTCUSDT", "ETHUSDT"):
        path = built["paths_by_identity"][(fit_digest, symbol, 1)]
        metric = _load_metric(path)
        _rewrite_metric(
            path,
            replace(
                metric,
                contract_start=metric.contract_start + 7,
                contract_stop=metric.contract_stop + 7,
                contract_digest=_digest(f"drifted-contract:{symbol}:1"),
                digest="",
            ),
        )

    with pytest.raises(ValueError, match="chronological episode scope"):
        _api().load_causal_alpha_v3_signal_forensics(tmp_path)
