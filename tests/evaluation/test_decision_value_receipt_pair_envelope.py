"""Opt-in, externally derived pair-count envelope for TD8 failure receipts.

These are synthetic software-only receipts; no market data or research result
is read, fitted, replayed, or reclassified.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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


def _json_value_nodes(value: object) -> int:
    """Independent decoded-tree oracle: member names are not value nodes."""
    if isinstance(value, dict):
        return 1 + sum(_json_value_nodes(child) for child in value.values())
    if isinstance(value, list):
        return 1 + sum(_json_value_nodes(child) for child in value)
    return 1


@pytest.mark.parametrize("limit", (0, -1, True, 1.5, "2", [], {}))
def test_external_node_limit_must_be_positive_native_integer_before_io(
    tmp_path: Path, limit: object
) -> None:
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match="expected_max_json_nodes"):
        study.load_incomplete_pair_attempt(
            root, expected_max_json_nodes=cast(int, limit)
        )
    assert not root.exists()


@pytest.mark.parametrize("case", ("v2", "v3", "quarantine"))
@pytest.mark.parametrize("pinned", (False, True))
def test_opt_in_node_limit_preserves_valid_receipt_at_exact_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, pinned: bool
) -> None:
    from dataclasses import replace

    from trade_rl.artifacts.hashing import content_digest

    root = tmp_path / "attempt"
    reason = 'quoted } [ , : " ' + chr(92) + chr(0x1F600)
    first = replace(_row(), reason_codes=(reason,))
    if case == "quarantine":
        first = replace(first, baseline={"terminal_cash": "broken" + reason})
    _persist(root, first, _row(action=(0, 1)))
    if case == "v2":
        _as_v2(root)
    path = root / "incomplete-pairs.json"
    value = study.load_incomplete_pair_attempt(root)
    # The reference-vintage compatibility field admits JSON-safe nested data.
    # Use a wide scalar array, empty containers, and escaped keys/values.
    pairs = cast(list[dict[str, object]], value["pairs"])
    pairs[-1]["reference_vintage"] = {
        "a": [[], {}, [1, None, False, reason], {"x": [1] * 1024}],
        reason: {"b": [True, {"c": ":"}]},
        "escaped": chr(0) + chr(92) + '"',
    }
    value["pairs_digest"] = content_digest(pairs)
    value["receipt_digest"] = content_digest(
        {k: v for k, v in value.items() if k != "receipt_digest"}
    )
    raw = study.canonical_json_bytes(value)
    path.write_bytes(raw)
    nodes = _json_value_nodes(value)
    kwargs = {
        "expected_receipt_digest": str(value["receipt_digest"]) if pinned else None,
        "expected_max_ineligible_pairs": 2,
    }
    # Missing opt-in budget must not retrospectively reject v2/v3 history.
    assert study.load_incomplete_pair_attempt(root) == value
    assert (
        study.load_incomplete_pair_attempt(
            root, expected_max_json_nodes=nodes, **kwargs
        )
        == value
    )
    assert (
        study.load_incomplete_pair_attempt(
            root, expected_max_json_nodes=nodes + 1, **kwargs
        )
        == value
    )

    def forbidden_loads(*args: object, **kwargs: object) -> object:
        raise AssertionError("over-budget nodes reached JSON decode")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="exceeds externally verified JSON node limit"):
        study.load_incomplete_pair_attempt(
            root, expected_max_json_nodes=nodes - 1, **kwargs
        )
    assert path.read_bytes() == raw


def test_node_limit_counts_array_items_and_duplicate_member_occurrences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Duplicate member names still consume input value slots, even though
    # stdlib would collapse them. Counting values is not schema validation.
    raw = b'{"pairs":[{"reference_vintage":{"a":1,"a":2,"b":[1,2,3]}}]}'

    def forbidden_loads(*args: object, **kwargs: object) -> object:
        raise AssertionError("over-budget nested nodes reached JSON decode")

    monkeypatch.setattr(study.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="exceeds externally verified JSON node limit"):
        study._decode_incomplete_receipt(raw, expected_max_json_nodes=9)


_JSON_TEXT = st.text(
    alphabet=st.sampled_from(
        [
            "a",
            "b",
            ":",
            ",",
            "[",
            "]",
            "{",
            "}",
            '"',
            chr(92),
            chr(0),
            chr(10),
            chr(0x1F600),
        ]
    ),
    max_size=24,
)
_JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers(-100, 100) | _JSON_TEXT,
    lambda children: (
        st.lists(children, max_size=5)
        | st.dictionaries(_JSON_TEXT, children, max_size=5)
    ),
    max_leaves=50,
)


@given(value=_JSON_VALUES, ascii_only=st.booleans(), pretty=st.booleans())
@settings(max_examples=150, deadline=None)
def test_lexical_value_node_count_matches_independent_tree_oracle(
    value: object, ascii_only: bool, pretty: bool
) -> None:
    import json

    raw = json.dumps(
        value,
        ensure_ascii=ascii_only,
        allow_nan=False,
        indent=2 if pretty else None,
    ).encode("utf-8")
    count = _json_value_nodes(value)
    study._preflight_receipt_json_depth(raw, expected_max_json_nodes=count)
    if count > 1:
        with pytest.raises(
            ValueError, match="exceeds externally verified JSON node limit"
        ):
            study._preflight_receipt_json_depth(raw, expected_max_json_nodes=count - 1)


def test_node_envelope_does_not_weaken_depth_or_strict_schema() -> None:
    deep = b"[" * 33 + b"0" + b"]" * 33
    with pytest.raises(ValueError, match="nesting exceeds"):
        study._preflight_receipt_json_depth(deep, expected_max_json_nodes=1000)
    # A node count within budget must not excuse malformed container syntax.
    with pytest.raises(ValueError, match="malformed"):
        study._preflight_receipt_json_depth(b'{"a":[1}', expected_max_json_nodes=10)
    with pytest.raises(ValueError, match="fields are malformed"):
        study._decode_incomplete_receipt(b"{}", expected_max_json_nodes=1)
