"""Opt-in, externally derived pair-count envelope for TD8 failure receipts.

These are synthetic software-only receipts; no market data or research result
is read, fitted, replayed, or reclassified.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from tests.evaluation.test_decision_value_failure_receipt_robustness import (
    _as_v2,
    _persist,
    _row,
)
from trade_rl.evaluation import decision_value_dev_study as study


@pytest.mark.parametrize("limit", (0, -1, True, 1.5, "2", [], {}))
def test_external_pair_limit_must_be_a_positive_native_integer_before_io(
    tmp_path: Path, limit: object
) -> None:
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match="expected_max_ineligible_pairs"):
        study.load_incomplete_pair_attempt(
            root, expected_max_ineligible_pairs=cast(int, limit)
        )
    assert not root.exists()


def test_excess_pairs_refused_before_json_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consistent declared count must not bypass an external source ceiling."""
    pair = study.canonical_json_bytes(dict.fromkeys(study._PAIR_KEYS))
    raw = b'{"ineligible_pairs":2,"pairs":[' + pair + b"," + pair + b"]}"

    def forbidden_loads(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("over-budget pairs reached JSON decoder")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="exceeds externally verified pair limit"):
        study._decode_incomplete_receipt(raw, expected_max_ineligible_pairs=1)


@pytest.mark.parametrize("legacy_v2", (False, True))
def test_opt_in_envelope_preserves_v2_v3_receipt_bytes_and_digest(
    tmp_path: Path, legacy_v2: bool
) -> None:
    root = tmp_path / "attempt"
    _persist(root, _row(), _row(action=(0, 1)))
    if legacy_v2:
        _as_v2(root)
    original = (root / "incomplete-pairs.json").read_bytes()
    historical = study.load_incomplete_pair_attempt(root)
    digest = str(historical["receipt_digest"])
    assert historical["ineligible_pairs"] == 2

    with pytest.raises(ValueError, match="exceeds externally verified pair limit"):
        study.load_incomplete_pair_attempt(
            root,
            expected_receipt_digest=digest,
            expected_max_ineligible_pairs=1,
        )
    assert (root / "incomplete-pairs.json").read_bytes() == original

    for ceiling in (2, 3):
        assert (
            study.load_incomplete_pair_attempt(
                root,
                expected_receipt_digest=digest,
                expected_max_ineligible_pairs=ceiling,
            )
            == historical
        )
    assert study.load_incomplete_pair_attempt(root) == historical
    assert (root / "incomplete-pairs.json").read_bytes() == original
