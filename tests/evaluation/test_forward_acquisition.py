"""FS1 acquisition: planning, every refusal and an end-to-end fixture build.

All offline: fixture archives, a fake transport and a miniature FS1 span. The real
registry is only planned against, never fetched.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.evaluation.forward_acquisition_support import (
    AFTER_SETTLEMENT,
    AUTH,
    CODE_SHA,
    Scenario,
    make_scenario,
)
from trade_rl.data.artifacts.publication import load_published_market_dataset_artifact
from trade_rl.evaluation.dev_realdata_study import dataset_build
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.evaluation.forward_acquisition import acquire, cli, plan
from trade_rl.evaluation.forward_acquisition.acquire import AcquisitionRefusal
from trade_rl.evaluation.forward_study import seal
from trade_rl.evaluation.forward_study.registry import (
    FROZEN_PARAMS_DIGEST,
    FS1_REGISTRY,
    payload_digest,
)
from trade_rl.evaluation.forward_study.seal import CodeIdentity, authorize_scoring
from trade_rl.integrations.binance import BinanceMarket, plan_binance_vision_cache


def real_plan() -> plan.AcquisitionPlan:
    params, problems = plan.build_params_or_problems(FS1_REGISTRY)
    assert params is not None and problems == ()
    return plan.plan_acquisition(
        params,
        development_end=datetime(2026, 8, 1, tzinfo=UTC),
        scoring_start=datetime(2026, 11, 1, tzinfo=UTC),
    )


# --- pinned identities -------------------------------------------------------------


def test_implemented_policy_digest_is_the_registry_build_policy_digest() -> None:
    assert plan.IMPLEMENTED_BUILD_POLICY_DIGEST == FS1_REGISTRY.build_policy_digest()
    assert plan.IMPLEMENTED_BUILD_POLICY_DIGEST == payload_digest(
        dict(FS1_REGISTRY.build_policy)
    )


def test_pinned_development_identity_matches_the_dev_builder_and_protocol() -> None:
    assert plan.DEV_DATASET_ID == dataset_build.STUDY_DATASET_ID == PROTOCOL.dataset_id
    assert plan.DEV_ARTIFACT_DIGEST == dataset_build.STUDY_ARTIFACT_DIGEST


# --- planning ----------------------------------------------------------------------


def test_the_pure_planner_equals_the_maintained_vision_planner() -> None:
    item = real_plan()
    maintained = plan_binance_vision_cache(
        market=BinanceMarket.USDS_M,
        symbols=item.params.symbols,
        intervals=("1h",),
        start_time=item.params.start,
        end_time=item.params.end,
    )
    assert item.archive_urls == maintained.urls
    assert len(item.archive_urls) == 760


def test_the_plan_splits_reuse_from_fetch_at_the_development_end() -> None:
    item = real_plan()
    assert len(item.reused) == 670 and len(item.fetched) == 90
    assert item.periods(item.reused)[0] == "2021-01"
    assert item.periods(item.reused)[-1] == "2026-07"
    assert item.periods(item.fetched) == tuple(
        f"{year}-{month:02d}"
        for year, month in [(2026, m) for m in range(8, 13)]
        + [(2027, m) for m in range(1, 5)]
    )
    assert len(item.fetch_urls) == 180
    assert all(url.startswith(plan.VISION_ROOT) for url in item.fetch_urls)
    assert item.fetch_urls[0].endswith("klines/BTCUSDT/1h/BTCUSDT-1h-2026-08.zip")
    assert item.fetch_urls[1].endswith("BTCUSDT-1h-2026-08.zip.CHECKSUM")
    assert any(
        url.endswith("fundingRate/ADAUSDT/ADAUSDT-fundingRate-2027-04.zip")
        for url in item.fetch_urls
    )
    # nothing from the development span or after the scoring end is requested
    assert not any("2026-07" in url or "2027-05" in url for url in item.fetch_urls)


def test_plan_only_lists_the_exact_urls_and_ranges_and_touches_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(_cache_root: Path) -> Any:
        raise AssertionError("plan-only must not create a transport")

    monkeypatch.setattr(acquire, "_real_transport", forbidden)
    root = tmp_path / "fs1-root"
    code = cli.main(["acquire", "--authorization", AUTH, "--output-root", str(root)])
    output = capsys.readouterr().out
    assert code == 0
    assert not root.exists()
    assert "PLAN ONLY: nothing was fetched" in output
    item = real_plan()
    for url in item.fetch_urls:
        assert f"  {url}\n" in output + "\n"
    assert plan.EXCHANGE_INFO_URL in output
    assert "[2021-01-01, 2027-05-01): 760 monthly archives" in output
    assert "670 archives, months 2021-01..2026-07" in output
    assert (
        "fetch: 90 archives plus 90 .CHECKSUM files, months 2026-08..2027-04" in output
    )
    assert "warm-up only, never scored: [2026-08-01, 2026-11-01)" in output
    assert "scored window: [2026-11-01, 2027-05-01)" in output
    assert "2026-07" not in "\n".join(
        line for line in output.splitlines() if line.startswith("  https://")
    )


def test_plan_only_does_not_read_development_inputs(tmp_path: Path) -> None:
    result = acquire.run_acquire(
        authorization=AUTH,
        output_root=tmp_path / "root",
        allow_network=False,
        dev_archive_root=tmp_path / "does-not-exist",
        command_line="x",
    )
    assert result.planned_only and not (tmp_path / "root").exists()


# --- refusals before anything is created or requested -------------------------------


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Scenario:
    return make_scenario(tmp_path_factory.mktemp("world"))


def refusal(scenario: Scenario, root: Path, **kwargs: Any) -> AcquisitionRefusal:
    with pytest.raises(AcquisitionRefusal) as caught:
        scenario.run(root, **kwargs)
    return caught.value


def test_a_clean_plan_only_call_on_the_miniature_world_plans(
    world: Scenario, tmp_path: Path
) -> None:
    result = world.run(tmp_path / "root", allow_network=False)
    assert result.planned_only and not (tmp_path / "root").exists()
    assert world.requests == []


def test_authorization_must_match_the_pinned_issue_comment_pattern(
    world: Scenario, tmp_path: Path
) -> None:
    for value, reason in (
        (None, "authorization_missing"),
        ("", "authorization_missing"),
        (
            "https://github.com/shuntatsu/trade_rl/issues/811#issuecomment-1",
            "authorization_malformed",
        ),
        ("https://github.com/shuntatsu/trade_rl/issues/810", "authorization_malformed"),
        (
            "https://github.com/other/trade_rl/issues/810#issuecomment-5",
            "authorization_malformed",
        ),
    ):
        for allow in (False, True):
            caught = refusal(
                world,
                tmp_path / f"r{allow}{reason}{abs(hash(value))}",
                authorization=value,
                allow_network=allow,
            )
            assert reason in caught.reasons
    assert world.requests == []


def test_an_existing_output_root_is_refused_and_left_untouched(
    world: Scenario, tmp_path: Path
) -> None:
    root = tmp_path / "taken"
    root.mkdir()
    (root / "keep.txt").write_text("existing", encoding="utf-8")
    for allow in (False, True):
        assert "output_exists" in refusal(world, root, allow_network=allow).reasons
    assert [p.name for p in root.iterdir()] == ["keep.txt"]
    assert world.requests == []


def test_a_build_policy_mismatch_is_refused(world: Scenario, tmp_path: Path) -> None:
    from dataclasses import replace

    for key, value, reason in (
        ("gap_policy", "strict", "build_policy_unsupported_gap_policy"),
        ("feature_timeframes", ["4h"], "build_policy_unsupported_feature_timeframes"),
        ("market", "spot", "build_policy_unsupported_market"),
        ("base_interval", "4h", "build_policy_unsupported_base_interval"),
        ("economics_profile", "other", "build_policy_unsupported_economics_profile"),
        (
            "baked_in_fees_and_spread",
            "0.0005",
            "build_policy_unsupported_baked_in_costs",
        ),
    ):
        changed = replace(
            world.registry, build_policy={**world.registry.build_policy, key: value}
        )
        caught = refusal(world, tmp_path / key, registry=changed, allow_network=False)
        assert "build_policy_digest_not_implemented" in caught.reasons
        assert reason in caught.reasons
    # a policy that is not the one this tool was written against is refused even if
    # every field is individually supported
    reworded = replace(
        world.registry,
        build_policy={**world.registry.build_policy, "missing_base_bar": "reworded"},
    )
    assert refusal(world, tmp_path / "w", registry=reworded).reasons == (
        "build_policy_digest_not_implemented",
    )
    # symbols and span must agree with the registry
    other_symbols = replace(
        world.registry,
        build_policy={**world.registry.build_policy, "symbols": ["BTCUSDT"]},
    )
    assert (
        "build_policy_symbols_differ_from_registry"
        in refusal(world, tmp_path / "s", registry=other_symbols).reasons
    )
    shifted = replace(
        world.registry,
        build_policy={
            **world.registry.build_policy,
            "end_time": "2026-03-31T00:00:00+00:00",
        },
    )
    assert (
        "build_policy_end_differs_from_scoring_end"
        in refusal(world, tmp_path / "e", registry=shifted).reasons
    )
    assert world.requests == []


def test_a_registry_that_differs_from_the_frozen_digest_is_refused(
    world: Scenario, tmp_path: Path
) -> None:
    env = world.environment()
    env = acquire._Environment(**{**env.__dict__, "frozen_params_digest": "0" * 64})
    with pytest.raises(AcquisitionRefusal) as caught:
        acquire._run_acquire(
            env,
            authorization=AUTH,
            output_root=tmp_path / "x",
            allow_network=False,
            dev_archive_root=None,
            dev_source_evidence=None,
            dev_dataset=None,
            command_line="x",
        )
    assert caught.value.reasons == ("registry_digest_differs_from_frozen",)


def test_the_fetch_waits_for_the_settlement_margin(
    world: Scenario, tmp_path: Path
) -> None:
    just_before = datetime(2026, 4, 1, 23, 59, 50, tzinfo=UTC)
    caught = refusal(world, tmp_path / "early", now=just_before)
    assert caught.reasons == ("settlement_not_reached",)
    # plan-only is not clock gated
    assert world.run(
        tmp_path / "plan", now=just_before, allow_network=False
    ).planned_only
    assert world.requests == [] and not (tmp_path / "early").exists()


def test_the_builder_commit_must_be_a_clean_trade_rl_checkout(
    world: Scenario, tmp_path: Path
) -> None:
    dirty = refusal(world, tmp_path / "d", code_identity=CodeIdentity(CODE_SHA, False))
    assert dirty.reasons == ("code_tree_dirty",)
    unavailable = refusal(
        world, tmp_path / "u", code_identity=CodeIdentity(None, False)
    )
    assert unavailable.reasons == ("code_identity_unavailable",)
    assert world.requests == []


def test_development_inputs_are_required_for_a_fetch(
    world: Scenario, tmp_path: Path
) -> None:
    caught = refusal(
        world,
        tmp_path / "n",
        dev_archive_root=None,
        dev_source_evidence=None,
        dev_dataset=None,
    )
    assert caught.reasons == ("dev_inputs_missing",)


def test_every_preflight_reason_is_reported_together(
    world: Scenario, tmp_path: Path
) -> None:
    taken = tmp_path / "taken"
    taken.mkdir()
    caught = refusal(
        world,
        taken,
        authorization="nope",
        dev_archive_root=None,
        code_identity=CodeIdentity(CODE_SHA, False),
        now=AFTER_SETTLEMENT - timedelta(days=30),
    )
    assert set(caught.reasons) == {
        "authorization_malformed",
        "output_exists",
        "settlement_not_reached",
        "code_tree_dirty",
        "dev_inputs_missing",
    }


def test_the_development_identity_is_proven_before_anything_is_created(
    world: Scenario, tmp_path: Path
) -> None:
    root = tmp_path / "dev-identity"
    wrong = acquire.DevReference("1" * 64, "2" * 64)
    caught = refusal(world, root, dev_reference=wrong)
    assert set(caught.reasons) == {
        "dev_dataset_id_mismatch",
        "dev_dataset_artifact_digest_mismatch",
    }
    assert not root.exists() and world.requests == []

    # a roster that does not hash to the digest bound inside the verified Dataset
    evidence = json.loads(world.dev_source_evidence.read_text(encoding="utf-8"))
    evidence["roster"][0]["sha256"] = "9" * 64
    edited = tmp_path / "edited-evidence.json"
    edited.write_text(json.dumps(evidence), encoding="utf-8")
    caught = refusal(world, root, dev_source_evidence=edited)
    assert "dev_roster_digest_mismatch" in caught.reasons
    assert "dev_archive_hash_mismatch" in caught.reasons

    # an unreadable or non-dataset directory is not the development Dataset
    caught = refusal(world, root, dev_dataset=tmp_path / "no-dataset")
    assert caught.reasons == ("dev_dataset_unreadable",)
    assert not root.exists() and world.requests == []


def test_a_tampered_or_missing_development_archive_is_refused_without_fetching(
    world: Scenario, tmp_path: Path
) -> None:
    victim = world.item_plan.reused[0]
    path = world.dev_archive_root / victim.relative_path
    original = path.read_bytes()
    root = tmp_path / "tamper"
    try:
        path.write_bytes(original + b"\0")
        assert refusal(world, root).reasons == ("dev_archive_hash_mismatch",)
        path.unlink()
        assert refusal(world, root).reasons == ("dev_archive_missing",)
    finally:
        path.write_bytes(original)
    assert not root.exists() and world.requests == []


def test_a_roster_that_lists_other_archives_than_the_plan_is_refused(
    world: Scenario, tmp_path: Path
) -> None:
    evidence = json.loads(world.dev_source_evidence.read_text(encoding="utf-8"))
    evidence["roster"].append(
        {
            "url": plan.VISION_ROOT + "/futures/um/monthly/klines/XRPUSDT/1h/x.zip",
            "sha256": "1" * 64,
            "size_bytes": 1,
        }
    )
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps(evidence), encoding="utf-8")
    caught = refusal(world, tmp_path / "r", dev_source_evidence=extra)
    assert "dev_roster_urls_differ_from_plan" in caught.reasons


# --- refusals during the run: a partial root without a manifest ---------------------


def test_a_missing_archive_on_the_server_stops_the_run_without_a_manifest(
    tmp_path: Path,
) -> None:
    probe = make_scenario(tmp_path / "w1")
    victim = probe.item_plan.fetched[3].url
    scenario = make_scenario(tmp_path / "w2", omit_from_server=[victim])
    root = tmp_path / "out"
    caught = refusal(scenario, root)
    assert caught.reasons == ("fetch_failed",)
    assert victim in caught.detail
    assert not (root / plan.MANIFEST_NAME).exists()
    assert not (root / plan.DATASET_DIR).exists()


def test_an_archive_that_does_not_match_its_official_checksum_is_refused(
    tmp_path: Path,
) -> None:
    probe = make_scenario(tmp_path / "w1")
    victim = probe.item_plan.fetched[1].url
    scenario = make_scenario(tmp_path / "w2", wrong_checksum_for=victim)
    root = tmp_path / "out"
    assert refusal(scenario, root).reasons == ("archive_checksum_mismatch",)
    assert not (root / plan.MANIFEST_NAME).exists()


def test_a_malformed_checksum_file_is_refused(tmp_path: Path) -> None:
    probe = make_scenario(tmp_path / "w1")
    victim = probe.item_plan.fetched[0]
    scenario = make_scenario(tmp_path / "w2")
    scenario.served[victim.checksum_url] = b"not a checksum file\n"
    assert refusal(scenario, tmp_path / "out").reasons == (
        "archive_checksum_file_malformed",
    )


def test_unexpected_and_missing_cache_members_are_detected(
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

    stray_url = plan.VISION_ROOT + "/futures/um/monthly/klines/BTCUSDT/1h/stray.zip"
    stray = vision_cache_path(cache, stray_url)
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"x")
    stray.with_suffix(".json").write_text(json.dumps({"url": stray_url}))
    assert acquire._cache_problems(world.item_plan, cache)[0] == (
        "archives_unexpected",
    )

    stray.unlink()
    vision_cache_path(cache, world.item_plan.archives[0].url).unlink()
    assert acquire._cache_problems(world.item_plan, cache)[0] == ("archives_missing",)


def test_a_gap_beyond_the_policy_is_refused_and_leaves_no_manifest(
    tmp_path: Path,
) -> None:
    march = datetime(2026, 3, 1, tzinfo=UTC)
    # 40 of 744 hours missing for one symbol: 94.6% tradable, below the 99% guard
    missing = {"BTCUSDT": {march + timedelta(hours=100 + n) for n in range(40)}}
    scenario = make_scenario(tmp_path / "w", missing_hours=missing)
    root = tmp_path / "out"
    caught = refusal(scenario, root)
    assert caught.reasons == ("gap_tradable_fraction_low_BTCUSDT",)
    assert not (root / plan.MANIFEST_NAME).exists()
    assert not (root / plan.DATASET_DIR).exists()
    assert not (root / plan.RECEIPT_NAME).exists()


def test_an_hour_missing_for_every_symbol_is_a_non_tradable_row_not_a_hole(
    tmp_path: Path,
) -> None:
    hole = datetime(2026, 3, 10, 5, tzinfo=UTC)
    scenario = make_scenario(tmp_path / "w", missing_everywhere=[hole])
    result = scenario.run(tmp_path / "out")
    _, dataset = load_published_market_dataset_artifact(tmp_path / "out" / "dataset")
    stamps = dataset.timestamps.astype("datetime64[h]")
    assert np.all(np.diff(stamps.astype(np.int64)) == 1)
    row = int(np.flatnonzero(stamps == np.datetime64("2026-03-10T06"))[0])
    assert not np.asarray(dataset.tradable)[row].any()
    assert result.manifest_path is not None


def test_the_gap_guard_refuses_an_irregular_timeline_wrong_span_and_symbols() -> None:
    registry = FS1_REGISTRY
    hours = np.arange(
        np.datetime64("2021-01-01T01"),
        np.datetime64("2021-01-03T01"),
        dtype="datetime64[h]",
    ).astype("datetime64[ns]")
    irregular = np.delete(hours, 5)
    stub = SimpleNamespace(
        symbols=registry.symbols,
        timestamps=irregular,
        tradable=np.ones((irregular.size, 5), dtype=bool),
    )
    assert acquire.dataset_gap_problems(stub, registry) == (  # type: ignore[arg-type]
        "gap_timeline_not_regular_hourly",
    )
    regular = SimpleNamespace(
        symbols=("BTCUSDT",),
        timestamps=hours,
        tradable=np.ones((hours.size, 1), dtype=bool),
    )
    problems = acquire.dataset_gap_problems(regular, registry)  # type: ignore[arg-type]
    assert "dataset_symbols_mismatch" in problems
    assert "dataset_last_row_not_scoring_end" in problems
    assert "dataset_window_empty" in problems


def test_a_few_missing_bars_inside_the_policy_are_admitted(tmp_path: Path) -> None:
    march = datetime(2026, 3, 1, tzinfo=UTC)
    # 3 of 744 hours (99.6% tradable): a non-tradable row, not a refusal
    missing = {"ETHUSDT": {march + timedelta(hours=200 + n) for n in range(3)}}
    scenario = make_scenario(tmp_path / "w", missing_hours=missing)
    result = scenario.run(tmp_path / "out")
    assert result.manifest_path is not None and result.manifest_path.is_file()
    receipt = json.loads((tmp_path / "out" / plan.RECEIPT_NAME).read_text("utf-8"))
    assert receipt["dataset"]["gap_report"]["ETHUSDT:1h"]["missing_bars"] == 3


# --- the successful fixture build ---------------------------------------------------


@pytest.fixture(scope="module")
def success(tmp_path_factory: pytest.TempPathFactory) -> tuple[Scenario, Path, Any]:
    base = tmp_path_factory.mktemp("success")
    scenario = make_scenario(base / "world")
    root = base / "fs1-root"
    result = scenario.run(root)
    return scenario, root, result


def test_the_run_fetches_exactly_the_planned_requests_and_nothing_else(
    success: tuple[Scenario, Path, Any],
) -> None:
    scenario, _, _ = success
    expected = [*scenario.item_plan.fetch_urls, plan.EXCHANGE_INFO_URL]
    assert scenario.requests == expected
    # no development archive (and nothing from 2026-01/02) was requested
    assert not any(
        item.url in set(scenario.requests) for item in scenario.item_plan.reused
    )


def test_the_output_root_has_the_documented_layout(
    success: tuple[Scenario, Path, Any],
) -> None:
    _, root, result = success
    assert sorted(path.name for path in root.iterdir()) == sorted(
        [
            plan.CACHE_DIR,
            plan.EXCHANGE_INFO_DIR,
            plan.DATASET_DIR,
            plan.MANIFEST_NAME,
            plan.RECEIPT_NAME,
        ]
    )
    assert (
        result.manifest_sha256
        == hashlib.sha256((root / plan.MANIFEST_NAME).read_bytes()).hexdigest()
    )


def test_the_manifest_is_accepted_by_the_scoring_seal_end_to_end(
    success: tuple[Scenario, Path, Any],
) -> None:
    scenario, root, result = success
    registry = scenario.registry
    manifest_bytes = (root / plan.MANIFEST_NAME).read_bytes()
    permit = seal._authorize(
        registry=registry,
        frozen_params_digest=registry.digest(),
        now=AFTER_SETTLEMENT + timedelta(days=1),
        authorization="https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-777",
        expected_params_digest=registry.digest(),
        sealed_code_sha=CODE_SHA,
        code_identity=CodeIdentity(CODE_SHA, True),
        manifest_bytes=manifest_bytes,
        expected_manifest_sha256=result.manifest_sha256,
    )
    assert permit.manifest.dataset_id == result.dataset_id
    assert permit.manifest.authorization_reference == AUTH
    assert permit.manifest.build_policy_digest == registry.build_policy_digest()
    _, dataset = load_published_market_dataset_artifact(root / plan.DATASET_DIR)
    # the loaded dataset satisfies every dataset check of the scoring evaluator
    assert seal.dataset_problems(permit, dataset) == ()
    assert tuple(dataset.symbols) == registry.symbols
    assert str(dataset.timestamps[-1]).startswith("2026-04-01T00:00:00")


def test_the_manifest_binds_every_archive_the_receipt_and_the_exchange_snapshot(
    success: tuple[Scenario, Path, Any],
) -> None:
    scenario, root, result = success
    manifest = seal.parse_manifest((root / plan.MANIFEST_NAME).read_bytes())
    receipt_bytes = (root / plan.RECEIPT_NAME).read_bytes()
    receipt = json.loads(receipt_bytes)
    sources = manifest.source_manifest
    for item in scenario.item_plan.archives:
        assert (
            sources[item.url] == hashlib.sha256(scenario.archives[item.url]).hexdigest()
        )
    assert sources[plan.SOURCE_KEY_RECEIPT] == hashlib.sha256(receipt_bytes).hexdigest()
    assert (
        sources[plan.SOURCE_KEY_EXCHANGE_INFO]
        == (receipt["exchange_information"]["raw_payload_sha256"])
    )
    assert (
        sources[plan.SOURCE_KEY_DATASET_ARTIFACT]
        == receipt["dataset"]["artifact_digest"]
    )
    assert (
        sources[plan.SOURCE_KEY_DEV_ROSTER]
        == receipt["development_reuse"]["roster_digest"]
    )
    assert len(sources) == len(scenario.item_plan.archives) + 4
    assert manifest.fetched_utc == result.fetched_utc
    assert datetime.fromisoformat(manifest.sealed_utc) >= datetime.fromisoformat(
        manifest.fetched_utc
    )
    assert datetime.fromisoformat(manifest.fetched_utc) >= acquire.settlement_instant(
        scenario.registry
    )


def test_the_receipt_records_commit_command_authorization_and_reuse(
    success: tuple[Scenario, Path, Any],
) -> None:
    scenario, root, result = success
    receipt = json.loads((root / plan.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["schema_version"] == plan.RECEIPT_SCHEMA
    assert receipt["builder"]["git_head"] == CODE_SHA
    assert receipt["builder"]["tracked_tree_clean"] is True
    assert receipt["command_line"].startswith(
        "python -m trade_rl.evaluation.forward_acq"
    )
    assert receipt["authorization_reference"] == AUTH
    assert receipt["build_policy_digest"] == scenario.registry.build_policy_digest()
    assert receipt["registry_params_digest"] == scenario.registry.digest()
    assert (
        receipt["development_reuse"]["dataset_id"] == scenario.dev_reference.dataset_id
    )
    assert receipt["development_reuse"]["archive_count"] == len(
        scenario.item_plan.reused
    )
    origins = [row["origin"] for row in receipt["source_archives"]["roster"]]
    assert origins.count("development_reuse") == len(scenario.item_plan.reused)
    assert origins.count("fetched") == len(scenario.item_plan.fetched)
    assert receipt["dataset"]["dataset_id"] == result.dataset_id
    assert receipt["economic_replay_executed"] is False
    fractions = receipt["dataset"]["tradable_fraction_development_end_to_scoring_end"]
    assert set(fractions) == set(scenario.registry.symbols)
    assert receipt["exchange_information"]["source_uri"] == plan.EXCHANGE_INFO_URL


def test_the_dataset_was_built_under_the_pinned_policy(
    success: tuple[Scenario, Path, Any],
) -> None:
    scenario, root, _ = success
    _, dataset = load_published_market_dataset_artifact(root / plan.DATASET_DIR)
    identity = json.loads(dataset.identity_payload_json or "{}")
    evidence = identity["metadata_evidence"]
    assert evidence["schema_version"] == "fs1_source_evidence_v1"
    assert evidence["build_policy_digest"] == scenario.registry.build_policy_digest()
    assert identity["execution_economics"]["fee_rate"] == 0.0
    assert identity["execution_economics"]["spread_rate"] == 0.0
    assert identity["execution_economics"]["name"] == "bot_overlay_only_assumptions_v1"
    assert set(dataset.feature_names) and any("4h" in n for n in dataset.feature_names)
    assert any("1d" in n for n in dataset.feature_names)


def test_a_second_run_into_the_same_root_is_refused(
    success: tuple[Scenario, Path, Any],
) -> None:
    scenario, root, _ = success
    before = (root / plan.MANIFEST_NAME).read_bytes()
    caught = refusal(scenario, root)
    assert caught.reasons == ("output_exists",)
    assert (root / plan.MANIFEST_NAME).read_bytes() == before


# --- FS1 authorization on a synthetic manifest (real registry, faked clock) ---------


def synthetic_manifest(**overrides: Any) -> bytes:
    stamps = np.array(
        ["2021-01-01T01:00:00", "2027-05-01T00:00:00"], dtype="datetime64[s]"
    )
    dataset = SimpleNamespace(
        dataset_id="a" * 64,
        symbols=FS1_REGISTRY.symbols,
        timestamps=stamps,
        n_bars=2,
    )
    arguments: dict[str, Any] = {
        "source_manifest": {plan.SOURCE_KEY_RECEIPT: "b" * 64},
        "build_policy_digest": FS1_REGISTRY.build_policy_digest(),
        "authorization_reference": AUTH,
        "fetched_utc": "2027-05-02T01:00:00Z",
        "sealed_utc": "2027-05-02T02:00:00Z",
    }
    arguments.update(overrides)
    return seal.manifest_bytes_from_dataset(dataset, **arguments)  # type: ignore[arg-type]


def test_scoring_authorization_accepts_a_manifest_shaped_like_the_acquisition_output() -> (
    None
):
    raw = synthetic_manifest()
    permit = authorize_scoring(
        now=datetime(2027, 5, 2, 0, 0, tzinfo=UTC),
        authorization="https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-99",
        expected_params_digest=FROZEN_PARAMS_DIGEST,
        sealed_code_sha=CODE_SHA,
        code_identity=CodeIdentity(CODE_SHA, True),
        manifest_bytes=raw,
        expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert permit.params_digest == FROZEN_PARAMS_DIGEST
    assert permit.manifest.fetched_utc == "2027-05-02T01:00:00Z"


def test_a_manifest_fetched_before_settlement_would_be_refused_by_scoring() -> None:
    raw = synthetic_manifest(
        fetched_utc="2027-05-01T12:00:00Z", sealed_utc="2027-05-02T02:00:00Z"
    )
    with pytest.raises(seal.Fs1Refusal) as caught:
        authorize_scoring(
            now=datetime(2027, 5, 2, 3, 0, tzinfo=UTC),
            authorization=AUTH,
            expected_params_digest=FROZEN_PARAMS_DIGEST,
            sealed_code_sha=CODE_SHA,
            code_identity=CodeIdentity(CODE_SHA, True),
            manifest_bytes=raw,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        )
    assert caught.value.reasons == ("manifest_fetched_before_settlement",)


def test_the_acquisition_settlement_instant_is_the_scoring_instant() -> None:
    assert acquire.settlement_instant(FS1_REGISTRY) == datetime(
        2027, 5, 2, 0, 0, tzinfo=UTC
    )


def test_checksum_files_parse_strictly() -> None:
    digest = "ab" * 32
    assert acquire.parse_checksum(f"{digest}  f.zip\n", filename="f.zip") == digest
    assert acquire.parse_checksum(f"{digest} *f.zip", filename="f.zip") == digest
    assert acquire.parse_checksum(f"{digest}  other.zip", filename="f.zip") is None
    assert acquire.parse_checksum(digest, filename="f.zip") is None
    assert acquire.parse_checksum("zz  f.zip", filename="f.zip") is None
    assert acquire.parse_checksum(f"{digest.upper()}  f.zip", filename="f.zip") is None


def test_the_public_cli_reports_a_refusal_with_exit_code_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    taken = tmp_path / "taken"
    taken.mkdir()
    code = cli.main(["acquire", "--authorization", "bad", "--output-root", str(taken)])
    assert code == 2
    error = capsys.readouterr().err
    assert "authorization_malformed" in error and "output_exists" in error
