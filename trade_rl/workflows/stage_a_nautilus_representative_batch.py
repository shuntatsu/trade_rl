"""Aggregate fixed real-window Stage A Nautilus evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.stage_a_nautilus_representative_evidence import (
    RepresentativeNautilusEvidence,
    build_representative_nautilus_evidence,
    write_representative_nautilus_evidence,
)
from trade_rl.workflows.stage_a_nautilus_representative_runner import (
    run_representative_nautilus_window,
)

_REPRESENTATIVE_TIME_QUANTILES = (0.1, 0.5, 0.9)


def run_and_persist_representative_nautilus_evidence(
    *,
    markets: Mapping[float, MarketDataset],
    source_digest: str,
    store_root: str | Path,
    output_path: str | Path,
    target_exposure: float = 0.10,
) -> RepresentativeNautilusEvidence:
    """Run all maintained real windows and atomically persist aggregate evidence."""

    require_sha256(source_digest, field="source_digest")
    if len(markets) != len(_REPRESENTATIVE_TIME_QUANTILES) or set(markets) != set(
        _REPRESENTATIVE_TIME_QUANTILES
    ):
        raise ValueError(
            "representative Nautilus evidence requires exactly the 0.1, 0.5, and 0.9 windows"
        )

    windows = tuple(
        run_representative_nautilus_window(
            market=markets[time_quantile],
            time_quantile=time_quantile,
            store_root=store_root,
            target_exposure=target_exposure,
        )
        for time_quantile in _REPRESENTATIVE_TIME_QUANTILES
    )
    evidence = build_representative_nautilus_evidence(
        source_digest=source_digest,
        windows=windows,
    )
    write_representative_nautilus_evidence(output_path, evidence)
    return evidence


__all__ = ["run_and_persist_representative_nautilus_evidence"]
