from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.studio.training_metrics import StudioTrainingMetricsReader

from .support import settings


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_reader_rejects_invalid_cache_bound(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        StudioTrainingMetricsReader(
            settings(tmp_path),
            max_cached_sources=value,  # type: ignore[arg-type]
        )
