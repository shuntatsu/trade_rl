"""FS1 acquisition review follow-ups: refusal aggregation, robustness of the development
proof, cache strictness, checkout guard, raw checksum binding and the span-equality
oracle. Offline fixtures only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.evaluation.forward_acquisition_support import (
    AFTER_SETTLEMENT,
    CODE_SHA,
    EXCHANGE_JSON,
    Scenario,
    make_scenario,
)
from trade_rl.evaluation.forward_acquisition import acquire, plan
from trade_rl.evaluation.forward_acquisition.acquire import AcquisitionRefusal
from trade_rl.evaluation.forward_study.seal import CodeIdentity


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Scenario:
    return make_scenario(tmp_path_factory.mktemp("followups"))


def refusal(scenario: Scenario, root: Path, **kwargs: Any) -> AcquisitionRefusal:
    with pytest.raises(AcquisitionRefusal) as caught:
        scenario.run(root, **kwargs)
    return caught.value


def test_clock_and_checkout_are_still_checked_when_the_build_policy_is_refused(
    world: Scenario, tmp_path: Path
) -> None:
    bad_policy = replace(
        world.registry,
        build_policy={**world.registry.build_policy, "gap_policy": "strict"},
    )
    caught = refusal(
        world,
        tmp_path / "p",
        registry=bad_policy,
        now=AFTER_SETTLEMENT - timedelta(days=30),
        code_identity=CodeIdentity(CODE_SHA, False),
    )
    assert {
        "build_policy_digest_not_implemented",
        "build_policy_unsupported_gap_policy",
        "settlement_not_reached",
        "code_tree_dirty",
    } <= set(caught.reasons)
    # no plan exists without a parsable policy, so the identity proof is not attempted
    assert not any(reason.startswith("dev_") for reason in caught.reasons)
    missing = refusal(world, tmp_path / "q", registry=bad_policy, dev_archive_root=None)
    assert "dev_inputs_missing" in missing.reasons
    assert world.requests == []


def test_a_malformed_development_identity_payload_is_a_refusal_not_a_traceback(
    world: Scenario, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for payload, reason in (
        ("{not json", "dev_dataset_identity_payload_invalid"),
        ("[1, 2]", "dev_dataset_identity_payload_invalid"),
        (None, "dev_dataset_roster_digest_unbound"),
        ('{"metadata_evidence": 3}', "dev_dataset_roster_digest_unbound"),
    ):
        stub = SimpleNamespace(
            dataset_id=world.dev_reference.dataset_id,
            identity_verified=True,
            identity_payload_json=payload,
        )
        artifact = SimpleNamespace(artifact_digest=world.dev_reference.artifact_digest)
        monkeypatch.setattr(
            acquire,
            "load_published_market_dataset_artifact",
            lambda _root, stub=stub, artifact=artifact: (artifact, stub),
        )
        assert reason in refusal(world, tmp_path / "x").reasons
    assert world.requests == []


def test_duplicate_development_roster_urls_are_refused(
    world: Scenario, tmp_path: Path
) -> None:
    evidence = json.loads(world.dev_source_evidence.read_text(encoding="utf-8"))
    evidence["roster"].append(dict(evidence["roster"][0]))
    doubled = tmp_path / "doubled.json"
    doubled.write_text(json.dumps(evidence), encoding="utf-8")
    caught = refusal(world, tmp_path / "r", dev_source_evidence=doubled)
    assert "dev_roster_duplicate_urls" in caught.reasons


def test_leftover_temporary_cache_members_are_unexpected(
    world: Scenario, tmp_path: Path
) -> None:
    from trade_rl.integrations.binance import vision_cache_path

    cache = tmp_path / "cache"
    for item in world.item_plan.archives:
        path = vision_cache_path(cache, item.url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        path.with_suffix(".json").write_text(json.dumps({"url": item.url}))
    for item in world.item_plan.fetched:
        path = vision_cache_path(cache, item.checksum_url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        path.with_suffix(".json").write_text(json.dumps({"url": item.checksum_url}))
    assert acquire._cache_problems(world.item_plan, cache)[0] == ()
    first = vision_cache_path(cache, world.item_plan.archives[0].url)
    leftover = first.with_suffix(".bin.tmp")
    leftover.write_bytes(b"partial")
    unexpected = ("archives_unexpected",)
    assert acquire._cache_problems(world.item_plan, cache)[0] == unexpected
    leftover.unlink()
    orphan = cache / "ab" / "orphan.json"
    orphan.parent.mkdir(exist_ok=True)
    orphan.write_text("{}")
    assert acquire._cache_problems(world.item_plan, cache)[0] == unexpected
    orphan.unlink()
    (cache / "note.txt").write_text("stray")
    assert acquire._cache_problems(world.item_plan, cache)[0] == unexpected


def test_an_output_root_inside_the_checkout_is_refused(
    world: Scenario, tmp_path: Path
) -> None:
    inside = acquire._checkout_root()
    assert inside is not None
    for allow in (False, True):
        caught = refusal(world, inside / "fs1-root-never-created", allow_network=allow)
        assert "output_root_inside_checkout" in caught.reasons
    assert not (inside / "fs1-root-never-created").exists()
    assert not acquire._inside_checkout(tmp_path / "outside")
    assert acquire._inside_checkout(inside / "a" / "b")
    assert world.requests == []


@pytest.fixture(scope="module")
def success(tmp_path_factory: pytest.TempPathFactory) -> tuple[Scenario, Path]:
    base = tmp_path_factory.mktemp("followups-success")
    scenario = make_scenario(base / "world")
    root = base / "root"
    scenario.run(root)
    return scenario, root


def test_the_receipt_binds_the_raw_checksum_files_and_the_span_equality(
    success: tuple[Scenario, Path],
) -> None:
    scenario, root = success
    receipt = json.loads((root / plan.RECEIPT_NAME).read_text(encoding="utf-8"))
    rows = {row["url"]: row for row in receipt["source_archives"]["roster"]}
    for item in scenario.item_plan.fetched:
        expected = hashlib.sha256(scenario.served[item.checksum_url]).hexdigest()
        assert rows[item.url]["checksum_file_sha256"] == expected
    for item in scenario.item_plan.reused:
        assert rows[item.url]["checksum_file_sha256"] is None
    equality = receipt["development_span_equality"]
    assert equality["equal"] is True and equality["mismatched_arrays"] == []
    assert equality["development_rows"] > 1000
    assert set(equality["exchange_rule_arrays_equal"]) == set(
        acquire.EXCHANGE_RULE_ARRAYS
    )
    assert all(equality["exchange_rule_arrays_equal"].values())


def test_the_span_equality_tolerates_gaps_and_changed_exchange_rules(
    tmp_path: Path,
) -> None:
    january = datetime(2026, 1, 1, tzinfo=UTC)
    # a gap in a development month leaves a higher-clock placeholder that is visible on
    # the last development row of the development build only
    missing = {"BTCUSDT": {january + timedelta(hours=100 + n) for n in range(5)}}
    scenario = make_scenario(tmp_path / "w", missing_hours=missing)
    scenario.served[plan.EXCHANGE_INFO_URL] = EXCHANGE_JSON.replace(b'"0.01"', b'"0.1"')
    result = scenario.run(tmp_path / "out")
    assert result.receipt_path is not None
    equality = json.loads(result.receipt_path.read_text(encoding="utf-8"))[
        "development_span_equality"
    ]
    assert equality["equal"] is True
    assert equality["exchange_rule_arrays_equal"]["tick_size"] is False
    assert equality["exchange_rule_arrays_equal"]["lot_size"] is True


def test_a_rebuild_that_differs_from_the_development_dataset_is_refused(
    tmp_path: Path,
) -> None:
    scenario = make_scenario(tmp_path / "w", dev_fee_rate=0.001)
    root = tmp_path / "out"
    caught = refusal(scenario, root)
    assert caught.reasons == ("dev_span_rebuild_differs",)
    assert "fee_rate" in caught.detail
    assert not (root / plan.MANIFEST_NAME).exists()
    assert not (root / plan.DATASET_DIR).exists()


def _stub(arrays: dict[str, np.ndarray], rows: int, **overrides: Any) -> Any:
    values: dict[str, Any] = {
        "n_bars": rows,
        "symbols": ("A", "B"),
        "feature_names": ("f",),
        "identity_arrays": lambda: arrays,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dev_span_equality_reports_each_kind_of_difference() -> None:
    rows = 6
    base = {
        "close": np.arange(rows * 2, dtype=float).reshape(rows, 2),
        "features": np.arange(rows * 2, dtype=float).reshape(rows, 2),
        "tick_size": np.full((2,), 0.01),
    }
    dev = _stub(base, rows)

    def new(rows_: int, **changes: np.ndarray) -> Any:
        arrays = {
            "close": np.arange(rows_ * 2, dtype=float).reshape(rows_, 2),
            "features": np.arange(rows_ * 2, dtype=float).reshape(rows_, 2),
            "tick_size": np.full((2,), 0.01),
        }
        arrays.update(changes)
        return _stub(arrays, rows_)

    assert acquire.dev_span_equality(dev, new(9))["equal"] is True
    moved = np.arange(18, dtype=float).reshape(9, 2)
    moved[2, 1] += 1
    report = acquire.dev_span_equality(dev, new(9, close=moved))
    assert report["equal"] is False and report["mismatched_arrays"] == ["close"]
    # the last development row may differ in feature arrays only
    last_row = np.arange(18, dtype=float).reshape(9, 2)
    last_row[rows - 1] += 5
    assert acquire.dev_span_equality(dev, new(9, features=last_row))["equal"] is True
    assert acquire.dev_span_equality(dev, new(9, close=last_row))["equal"] is False
    earlier = np.arange(18, dtype=float).reshape(9, 2)
    earlier[rows - 2] += 5
    assert acquire.dev_span_equality(dev, new(9, features=earlier))["equal"] is False
    # exchange-rule arrays are recorded, not refused
    ruled = acquire.dev_span_equality(dev, new(9, tick_size=np.full((2,), 0.1)))
    assert ruled["equal"] is True
    assert ruled["exchange_rule_arrays_equal"]["tick_size"] is False
    # structural differences
    assert acquire.dev_span_equality(dev, new(rows - 1))["mismatched_arrays"] == [
        "rows"
    ]
    other = _stub(base, rows, symbols=("A", "C"), feature_names=("g",))
    assert acquire.dev_span_equality(dev, other)["mismatched_arrays"] == [
        "symbols",
        "feature_names",
    ]
