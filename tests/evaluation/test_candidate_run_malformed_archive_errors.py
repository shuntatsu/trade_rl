from __future__ import annotations

import io
import zlib

import numpy as np
import pytest

from trade_rl.evaluation.runs import artifact as candidate_artifact


# Regression for the real zlib.error observed by baseline artifact falsification run 34616635445.
# Keep the unit oracle at the Candidate artifact boundary where backend errors are normalized.
def test_load_returns_normalizes_zlib_decompression_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_decompression(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise zlib.error("corrupt compressed NPZ member")

    monkeypatch.setattr(candidate_artifact.np, "load", fail_decompression)

    payload = io.BytesIO()
    np.savez(payload, symbol_0_strategy_0=np.asarray([0.0]))
    with pytest.raises(ValueError, match="malformed candidate returns archive"):
        candidate_artifact._load_returns(
            payload.getvalue(),
            expected_keys=frozenset({"symbol_0_strategy_0"}),
        )
