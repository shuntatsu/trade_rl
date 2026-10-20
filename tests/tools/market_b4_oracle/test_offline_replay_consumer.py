from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from tests.tools.market_b4_oracle._native_replay_fixture import _public_native_fixture
from tools.market_replay_oracle.artifact import VerifiedDatasetContent


def test_frozen_native_roster_is_derived_as_exact_48_without_row_replay(
    tmp_path: Path,
) -> None:
    from tests.tools.market_replay_oracle.test_protocol_pair import _load_pair
    from tools.market_b4_oracle.offline_replay_consumer import (
        derive_frozen_native_market_roster,
    )

    pair, *_ = _load_pair(tmp_path)
    protocol = json.loads(pair.native_canonical)
    timestamps = np.datetime64("2024-12-01T01:00:00", "ns") + np.arange(
        9504
    ) * np.timedelta64(1, "h")
    timestamps.setflags(write=False)
    features = np.zeros((9504, 2, 6), dtype=np.float64)
    features.setflags(write=False)
    content = VerifiedDatasetContent(
        protocol["dataset"]["dataset_id"],
        MappingProxyType({"timestamps": timestamps, "features": features}),
        MappingProxyType(
            {
                "symbols": ("BTCUSDT", "ETHUSDT"),
                "feature_names": tuple(protocol["dataset"]["feature_names"]),
                "calendar_kind": "continuous_24_7",
                "nominal_bar_hours": 1.0,
            }
        ),
    )

    roster = derive_frozen_native_market_roster(
        protocol_pair=pair,
        content=content,
    )

    assert len(roster) == 48
    assert (
        len({(arm.account_id, arm.scenario, arm.native_role) for arm in roster}) == 48
    )
    assert {arm.scenario for arm in roster} == {"BASEL0", "COST2L0", "BASEL2"}
    assert {arm.role for arm in roster} == {"NONRL", "CASH"}
    assert {arm.configuration_id for arm in roster} == {
        "H24_RIDGE_A1",
        "H24_RIDGE_A10",
        "H72_RIDGE_A1",
        "H72_RIDGE_A10",
    }
    assert {arm.binding.window.actor_advances for arm in roster} == {6527}
    assert {arm.binding.window.reserve_advances for arm in roster} == {72}
    assert {arm.binding.window.terminal_stop for arm in roster} == {9502}
    assert all(arm.binding.comparison_only is False for arm in roster)


def test_native_matrix_consumer_requires_full_static_input_pins(
    tmp_path: Path,
) -> None:
    from tools.market_b4_oracle.offline_replay_consumer import (
        consume_indexed_market_matrix,
    )
    from tools.market_b4_oracle.output_index import (
        ArtifactPin,
        load_pinned_output_index,
    )

    root = tmp_path / "indexed-output"
    root.mkdir()
    placeholder = b"{}"
    (root / "placeholder.json").write_bytes(placeholder)
    manifest = json.dumps(
        {
            "placeholder.json": {
                "bytes": len(placeholder),
                "sha256": hashlib.sha256(placeholder).hexdigest(),
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    (root / "output-manifest.json").write_bytes(manifest)
    manifest_pin = ArtifactPin(len(manifest), hashlib.sha256(manifest).hexdigest())
    index = load_pinned_output_index(root, manifest_pin)

    with pytest.raises(ValueError, match="all exact H/v3, V4, source and Dataset pins"):
        consume_indexed_market_matrix(
            output_index=index,
            protocol_pair=object(),
            content=object(),
            static_inputs=None,
        )


@pytest.mark.parametrize("role", ["nonrl", "cash"])
def test_indexed_row_consumer_compares_full_window_and_rejects_missing_actor_row(
    tmp_path, role: str
) -> None:
    from tools.market_b4_oracle.offline_replay_consumer import (
        _metadata_arm,
        consume_indexed_market_arm,
    )
    from tools.market_b4_oracle.output_index import (
        ArtifactPin,
        load_pinned_output_index,
    )
    from trade_rl.artifacts import content_digest

    owned, _, native, _, _ = _public_native_fixture(
        tmp_path,
        0,
        scenario="BASEL0",
        role=role,
        horizon=72,
    )
    forecast_bytes = (tmp_path / f"stream-0-BASEL0-{role}-h72.json").read_bytes()

    def indexed_output(
        receipt: dict[str, object],
        directory: str,
        *,
        corrupt_tail: bool = False,
        forecast_content: bytes | None = None,
    ):
        receipt_bytes = gzip.compress(
            json.dumps(
                receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8"),
            mtime=0,
        )
        if corrupt_tail:
            receipt_bytes += b"!"
        files = {
            "receipts/arm.json.gz": receipt_bytes,
            "forecasts/arm.json": (
                forecast_bytes if forecast_content is None else forecast_content
            ),
        }
        output_root = tmp_path / directory
        for name, raw in files.items():
            path = output_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        manifest = json.dumps(
            {
                name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                for name, raw in files.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        (output_root / "output-manifest.json").write_bytes(manifest)
        return load_pinned_output_index(
            output_root,
            ArtifactPin(len(manifest), hashlib.sha256(manifest).hexdigest()),
        )

    kwargs = {
        "receipt_name": "receipts/arm.json.gz",
        "forecast_name": "forecasts/arm.json",
        "content": owned.content,
        "policy": owned.policy,
        "binding": owned.binding,
        "role": role.upper(),
        "reserve_plan_digest": owned.reserve_plan_digest,
    }
    complete = consume_indexed_market_arm(
        output_index=indexed_output(native, "indexed-complete"), **kwargs
    )
    assert complete.status == "ARTIFICIAL_COMPARISON_COMPLETE"
    assert (complete.actor_rows, complete.reserve_rows) == (32, 8)
    assert complete.output_index_rechecked
    assert not complete.outer_attempt_authenticated
    assert not complete.full_b4_authorized

    from tools.market_b4_oracle.offline_replay_consumer import (
        consume_root_observed_market_arm,
    )

    with pytest.raises(
        ValueError, match="root-observed start and finish pins are required"
    ):
        consume_root_observed_market_arm(
            output_index=indexed_output(native, "indexed-no-root-witness"),
            expected_start_pin=None,
            expected_finish_pin=None,
            **kwargs,
        )

    missing_actor = json.loads(json.dumps(native))
    missing_actor["native_receipt"]["rows"].pop()
    with pytest.raises(ValueError, match="actor receipt rows differ from bound window"):
        consume_indexed_market_arm(
            output_index=indexed_output(missing_actor, "indexed-missing-row"), **kwargs
        )

    changed_identity = json.loads(json.dumps(native))
    changed_identity["native_receipt"]["rows"][0]["native"]["proposal"]["decision"][
        "baseline"
    ]["context"]["state_digest"] = "0" * 64
    with pytest.raises(
        ValueError, match="native actor context state digest differs from owned replay"
    ):
        consume_indexed_market_arm(
            output_index=indexed_output(
                changed_identity, "indexed-mutated-actor-identity"
            ),
            **kwargs,
        )

    stale_context_content = json.loads(json.dumps(native))
    stale_actor = stale_context_content["native_receipt"]["rows"][0]["native"]
    stale_actor["proposal"]["decision"]["baseline"]["context"]["cash"] += 1.0
    with pytest.raises(
        ValueError, match="native actor context content differs from owned replay"
    ):
        consume_indexed_market_arm(
            output_index=indexed_output(
                stale_context_content, "indexed-stale-context-content"
            ),
            **kwargs,
        )

    stale_input_content = json.loads(json.dumps(native))
    stale_actor = stale_input_content["native_receipt"]["rows"][0]["native"]
    stale_actor["proposal"]["decision"]["baseline"]["inputs"][
        "expected_simple_return"
    ] += 0.01
    with pytest.raises(
        ValueError, match="native actor proposal content differs from owned replay"
    ):
        consume_indexed_market_arm(
            output_index=indexed_output(
                stale_input_content, "indexed-stale-input-content"
            ),
            **kwargs,
        )

    changed_bar_count = json.loads(json.dumps(native))
    changed_bar_count["native_receipt"]["rows"][0]["native"]["execution"][
        "bars_advanced"
    ] = True
    with pytest.raises(ValueError, match="native execution row differs"):
        consume_indexed_market_arm(
            output_index=indexed_output(changed_bar_count, "indexed-bool-bar-count"),
            **kwargs,
        )

    if role == "cash":
        changed_fill_count = json.loads(json.dumps(native))
        assert (
            changed_fill_count["native_receipt"]["rows"][0]["native"]["book"][
                "fill_count"
            ]
            == 0
        )
        changed_fill_count["native_receipt"]["rows"][0]["native"]["book"][
            "fill_count"
        ] = False
        with pytest.raises(ValueError, match="native book counters or termination"):
            consume_indexed_market_arm(
                output_index=indexed_output(
                    changed_fill_count, "indexed-bool-fill-count"
                ),
                **kwargs,
            )

    changed_profile = json.loads(json.dumps(native))
    plan = changed_profile["plan"]
    plan["native_plan"]["cell"]["runtime"]["cost_payload"]["multiplier"] += 1.0
    plan_digest = hashlib.sha256(
        json.dumps(
            plan,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    changed_profile["plan_digest"] = plan_digest
    changed_profile["native_receipt"]["plan"] = plan["native_plan"]
    changed_profile["native_receipt"]["cell_plan_digest"] = hashlib.sha256(
        json.dumps(
            plan["native_plan"]["cell"],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(
        ValueError,
        match="receipt runtime/cost/risk profile differs from frozen binding",
    ):
        _metadata_arm(changed_profile, owned, plan_digest)

    changed_forecast = json.loads(forecast_bytes)
    digest_replacements: dict[str, str] = {}
    for vintage in changed_forecast["vintages"]:
        old_digest = vintage["digest"]
        vintage["model"]["alpha"] = 10.0
        vintage_content = {
            key: value for key, value in vintage.items() if key != "digest"
        }
        vintage["digest"] = content_digest(vintage_content)
        digest_replacements[old_digest] = vintage["digest"]
    for packet in changed_forecast["packets"]:
        packet["vintage_digest"] = digest_replacements[packet["vintage_digest"]]
        packet_content = {
            key: value for key, value in packet.items() if key != "digest"
        }
        packet["digest"] = content_digest(packet_content)
    changed_forecast["causal_scope_digest"] = content_digest(
        {
            "vintages": changed_forecast["vintages"],
            "packets": changed_forecast["packets"],
        }
    )
    changed_forecast_bytes = json.dumps(
        changed_forecast,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with pytest.raises(
        ValueError, match="forecast vintage differs from the bound configuration"
    ):
        consume_indexed_market_arm(
            output_index=indexed_output(
                native,
                "indexed-mutated-alpha-configuration",
                forecast_content=changed_forecast_bytes,
            ),
            **kwargs,
        )

    with pytest.raises(ValueError, match="trailing bytes or another member"):
        consume_indexed_market_arm(
            output_index=indexed_output(
                native, "indexed-corrupt-tail", corrupt_tail=True
            ),
            **kwargs,
        )
