from __future__ import annotations

import importlib
from pathlib import Path

from tests.workflows.test_universal_causal_alpha_v3_signal_forensics import (
    _api,
    _build_run,
)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_signal_forensics_v2_historical_mode_binds_v1_without_mutating_source(
    tmp_path: Path,
) -> None:
    _build_run(tmp_path)
    before = _snapshot(tmp_path)
    base = _api().load_causal_alpha_v3_signal_forensics(tmp_path)

    v2_api = importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2"
    )
    report = v2_api.load_causal_alpha_v3_signal_forensics_v2(tmp_path)

    assert report.schema_version == "causal_alpha_v3_signal_forensics_v2"
    assert report.base_forensics_digest == base.digest
    assert report.base_forensics.to_payload() == base.to_payload()
    assert report.sidecar_mode == "historical_unavailable"
    assert report.sidecar_analysis is None
    assert report.research_only is True
    assert report.promotion_eligible is False
    assert tuple(item.to_payload() for item in report.unavailable_analyses) == tuple(
        item.to_payload() for item in base.unavailable_analyses
    )
    assert (
        report.digest
        == v2_api.load_causal_alpha_v3_signal_forensics_v2(tmp_path).digest
    )
    assert _snapshot(tmp_path) == before
