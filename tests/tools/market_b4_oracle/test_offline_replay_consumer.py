from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

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


def test_static_readback_uses_paired_v4_binding_for_native_v3_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route a synthetic pair through the strict binding, stubbing external IO."""
    from tests.tools.market_replay_oracle.test_exact_fill_binding import _binding
    from tests.tools.market_replay_oracle.test_protocol_pair import _load_pair
    from tools.market_b4_oracle import offline_replay_consumer as consumer
    from tools.market_b4_oracle.frozen_market_replay import (
        FrozenMarketBinding,
        FrozenMarketPolicy,
    )
    from tools.market_replay_oracle import (
        DatasetContentPin,
        ExactFillAttemptBinding,
        NativeSourcePin,
    )
    from tools.market_replay_oracle.profiles import effective_profile

    pair, _, _, v3_pin, v4_pin = _load_pair(tmp_path)
    v4_document = json.loads(pair.comparison_canonical)
    content = _binding().content
    native_document = json.loads(pair.native_canonical)
    risk = native_document["policy"]["risk"]
    binding = FrozenMarketBinding.from_declared_protocol_pair(
        protocol_pair=pair,
        content=content,
        configuration_id="H24_RIDGE_A1",
        scenario="BASEL0",
        native_role="nonrl",
        selected_symbol_index=0,
    )
    policy = FrozenMarketPolicy.from_effective_profile(
        dataset_id=content.dataset_id,
        profile=effective_profile("BASEL0", binding.window.actor_advances, risk),
        risk=risk,
    )
    dataset = v4_document["dataset"]
    pins = consumer.StaticInputReadback(
        protocol_v3_path=tmp_path / "native-h-v3.json",
        protocol_v3_pin=v3_pin,
        protocol_v4_path=tmp_path / "comparison-h-v4.json",
        protocol_v4_pin=v4_pin,
        source_root=tmp_path,
        source_pin=NativeSourcePin(pair.native_pin.source_head, "2" * 64),
        dataset_root=Path(dataset["root"]),
        dataset_pin=DatasetContentPin(
            dataset["dataset_id"],
            dataset["input_sha256"]["manifest.json"],
            dataset["input_sha256"]["arrays.npz"],
            7453,
            5864784,
        ),
    )
    from tools.market_replay_oracle import exact_fill_binding

    class DeclaredAbsolutePath:
        """Adapt the frozen Windows path's absolute check for this unit fixture."""

        def __init__(self, value: str | Path) -> None:
            self.value = str(value)

        def is_absolute(self) -> bool:
            return True

        def __str__(self) -> str:
            return self.value

    monkeypatch.setattr(
        exact_fill_binding,
        "verify_native_source",
        lambda _root, pin: {
            "head_sha": pin.head_sha,
            "snapshot_sha256": pin.snapshot_sha256,
        },
    )
    monkeypatch.setattr(exact_fill_binding, "load_pinned_v4", lambda *_args: content)
    monkeypatch.setattr(
        exact_fill_binding, "_dataset_still_pinned", lambda *_args: None
    )
    monkeypatch.setattr(exact_fill_binding, "Path", DeclaredAbsolutePath)
    actual_exact_fill_loader = exact_fill_binding.load_pinned_exact_fill_attempt
    observed: list[tuple[object, ...]] = []

    def exact_fill_readback(*args: Any, **kwargs: Any) -> Any:
        observed.append(args)
        return actual_exact_fill_loader(*args, **kwargs)

    monkeypatch.setattr(
        consumer,
        "load_pinned_exact_fill_attempt",
        exact_fill_readback,
        raising=False,
    )
    monkeypatch.setattr(consumer, "load_pinned_v4", lambda *_args: content)

    readback = consumer._readback_static_inputs(
        pins=pins,
        binding=binding,
        content=content,
        policy=policy,
    )

    assert readback[0] == pair
    assert isinstance(readback[1], ExactFillAttemptBinding)
    assert readback[1].payload()["clock"] == {
        "actor_start": 2903,
        "actor_stop": 9430,
        "settlement_stop": 9502,
    }
    assert readback[2] is content
    assert observed == [
        (
            pins.protocol_v4_path,
            pins.protocol_v4_pin,
            pins.source_root,
            pins.source_pin,
            pins.dataset_root,
            pins.dataset_pin,
        )
    ]

    real_pair_loader = consumer.load_declared_protocol_pair
    pair_checks = 0

    def change_native_pin_on_final_check(*args: Any, **kwargs: Any) -> Any:
        nonlocal pair_checks
        pair_checks += 1
        if pair_checks == 2:
            pins.protocol_v3_path.write_bytes(pins.protocol_v3_path.read_bytes() + b" ")
        return real_pair_loader(*args, **kwargs)

    monkeypatch.setattr(
        consumer, "load_declared_protocol_pair", change_native_pin_on_final_check
    )
    with pytest.raises(ValueError, match="native protocol raw byte count mismatch"):
        consumer._readback_static_inputs(
            pins=pins,
            binding=binding,
            content=content,
            policy=policy,
        )
    assert pair_checks == 2


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

    if role == "nonrl":
        changed_fill_event = json.loads(json.dumps(native))
        fill_event = next(
            event
            for row in changed_fill_event["native_receipt"]["rows"]
            for event in row["native"]["order_events"]
            if event["event_type"] in {"filled", "partial_fill"}
        )
        fill_event["execution_price"] += 0.01
        with pytest.raises(
            ValueError, match="observed fill batch differs|native fill events differ"
        ):
            consume_indexed_market_arm(
                output_index=indexed_output(
                    changed_fill_event, "indexed-mutated-fill-event"
                ),
                **kwargs,
            )

    changed_terminal_cash = json.loads(json.dumps(native))
    changed_terminal_cash["final_book"]["cash"] += 1.0
    with pytest.raises(ValueError, match="native cash differs from the owned replay"):
        consume_indexed_market_arm(
            output_index=indexed_output(
                changed_terminal_cash, "indexed-mutated-terminal-cash"
            ),
            **kwargs,
        )

    changed_terminal_quantity = json.loads(json.dumps(native))
    changed_terminal_quantity["final_book"]["exact_quantities"] = ["1", "0"]
    with pytest.raises(
        ValueError, match="native exact quantities differ from the owned replay"
    ):
        consume_indexed_market_arm(
            output_index=indexed_output(
                changed_terminal_quantity, "indexed-mutated-terminal-quantity"
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


def _indexed_original_native_receipt(
    tmp_path: Path, receipt: dict[str, object], name: str
):
    from tools.market_b4_oracle.output_index import (
        ArtifactPin,
        load_pinned_output_index,
    )

    raw = gzip.compress(
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8"),
        mtime=0,
    )
    root = tmp_path / name
    path = root / "receipts/arm.json.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    manifest = json.dumps(
        {
            "receipts/arm.json.gz": {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    (root / "output-manifest.json").write_bytes(manifest)
    return load_pinned_output_index(
        root, ArtifactPin(len(manifest), hashlib.sha256(manifest).hexdigest())
    )


def _native_event_consumer_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.tools.market_replay_oracle.test_exact_fill_binding import (
        _binding as _exact_fill_binding,
    )
    from tests.tools.market_replay_oracle.test_protocol_pair import _load_pair
    from tools.market_b4_oracle import offline_replay_consumer as consumer
    from tools.market_b4_oracle.frozen_market_replay import (
        FrozenMarketBinding,
        FrozenMarketPolicy,
    )
    from tools.market_replay_oracle.profiles import effective_profile

    pair, *_ = _load_pair(tmp_path)
    exact_fill = _exact_fill_binding()
    content = exact_fill.content
    protocol = json.loads(pair.native_canonical)
    risk = protocol["policy"]["risk"]
    binding = FrozenMarketBinding.from_declared_protocol_pair(
        protocol_pair=pair,
        content=content,
        configuration_id="H24_RIDGE_A1",
        scenario="BASEL0",
        native_role="nonrl",
        selected_symbol_index=0,
    )
    policy = FrozenMarketPolicy.from_effective_profile(
        dataset_id=content.dataset_id,
        profile=effective_profile("BASEL0", 6527, risk),
        risk=risk,
    )
    pins = consumer.StaticInputReadback(
        protocol_v3_path=tmp_path / "native-h-v3.json",
        protocol_v3_pin=pair.native_pin,
        protocol_v4_path=tmp_path / "comparison-h-v4.json",
        protocol_v4_pin=pair.comparison_pin,
        source_root=tmp_path,
        source_pin=consumer.NativeSourcePin(pair.native_pin.source_head, "2" * 64),
        dataset_root=Path("C:/synthetic-pinned-dataset"),
        dataset_pin=consumer.DatasetContentPin(
            content.dataset_id,
            "3" * 64,
            "4" * 64,
            7453,
            5864784,
        ),
    )
    monkeypatch.setattr(
        consumer,
        "_readback_static_inputs",
        lambda **_kwargs: (pair, exact_fill, content),
    )
    return consumer, exact_fill, binding, policy, pins


def test_native_event_precondition_uses_full_original_pinned_receipt_before_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.tools.market_replay_oracle.test_event_chain import (
        ARM,
        _actor_row,
        _actor_stream,
        _for_context,
        _reserve_stream,
    )
    from tests.tools.market_replay_oracle.test_events import _legacy

    consumer, exact_fill, binding, policy, pins = _native_event_consumer_fixture(
        tmp_path, monkeypatch
    )
    account, scenario, role = ARM
    rows = list(_actor_stream(exact_fill))
    rejection = _for_context(
        _legacy(
            fill=True,
        ),
        exact_fill.event_context(account, scenario, role, 1),
    )
    rows[0] = _actor_row(exact_fill, 1, events=[rejection])
    receipt = {
        "native_receipt": {"rows": rows},
        "closing_rows": list(_reserve_stream(exact_fill)),
    }
    index = _indexed_original_native_receipt(tmp_path, receipt, "native-v1-fill")

    event_validation_calls: list[object] = []
    with monkeypatch.context() as context:
        context.setattr(
            consumer,
            "_validate_native_event_rows",
            lambda **_kwargs: event_validation_calls.append(None),
            raising=False,
        )
        with pytest.raises(
            ValueError, match="root-observed start and finish pins are required"
        ):
            consumer.consume_root_observed_market_arm(
                output_index=index,
                receipt_name="receipts/arm.json.gz",
                forecast_name="forecasts/arm.json",
                content=exact_fill.content,
                policy=policy,
                binding=binding,
                role="NONRL",
                reserve_plan_digest="5" * 64,
                static_inputs=pins,
                root_witness=None,
                expected_start_pin=None,
                expected_finish_pin=None,
            )
    assert event_validation_calls == []

    # The missing forecast is intentional: v4 validation must reject the
    # original receipt before forecast or Q replay is reached.
    with pytest.raises(ValueError, match="positive lot fill requires v2 witness"):
        consumer.consume_indexed_market_arm(
            output_index=index,
            receipt_name="receipts/arm.json.gz",
            forecast_name="forecasts/arm.json",
            content=exact_fill.content,
            policy=policy,
            binding=binding,
            role="NONRL",
            reserve_plan_digest="5" * 64,
            static_inputs=pins,
        )


@pytest.mark.parametrize(
    ("receipt_variant", "expected_error"),
    [
        ("complete", "artifact absent from pinned output index"),
        ("valid_v2_fill", "artifact absent from pinned output index"),
        ("wrong_v2_step", "event lot differs from expected effective lot"),
        ("wrong_clock", "event processing_index differs from context"),
        ("wrong_context", "event dataset_id differs from context"),
        ("rejected", "native event receipt contains rejected native facts"),
        ("missing_tail", "missing native facts are only valid at an actor tail"),
        (
            "short_reserve",
            "native receipt lacks a complete, nonrejected strict-v4 event chain",
        ),
    ],
)
def test_native_event_precondition_requires_complete_nonrejected_original_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_variant: str,
    expected_error: str,
) -> None:
    from tests.tools.market_replay_oracle.test_event_chain import (
        ARM,
        _actor_row,
        _actor_stream,
        _for_context,
        _reserve_stream,
    )

    consumer, exact_fill, binding, policy, pins = _native_event_consumer_fixture(
        tmp_path, monkeypatch
    )
    rows = list(_actor_stream(exact_fill))
    reserve_rows = list(_reserve_stream(exact_fill))
    if receipt_variant == "short_reserve":
        reserve_rows = reserve_rows[:-1]
    if receipt_variant in {"complete", "valid_v2_fill"}:
        from tests.tools.market_replay_oracle.test_events import _event, _legacy

        if receipt_variant == "complete":
            event = _legacy(
                fill=False,
                event_type="rejected",
                new_status="rejected",
                reason="risk_control",
            )
        else:
            event = _event(
                filled_lot_count=1,
                lot_size=0.125,
                remaining_quantity=0.875,
                filled_quantity=0.125,
                filled_notional=12.5,
                capacity_after=87.5,
                participation_rate=0.125,
            )
        rows[0] = _actor_row(
            exact_fill,
            1,
            events=[
                _for_context(
                    event,
                    exact_fill.event_context(*ARM, 1),
                )
            ],
        )
    elif receipt_variant in {"wrong_v2_step", "wrong_clock", "wrong_context"}:
        from tests.tools.market_replay_oracle.test_events import _event

        event = _for_context(
            _event(
                filled_lot_count=1,
                lot_size=0.125,
                remaining_quantity=0.875,
                filled_quantity=0.125,
                filled_notional=12.5,
                capacity_after=87.5,
                participation_rate=0.125,
            ),
            exact_fill.event_context(*ARM, 1),
        )
        if receipt_variant == "wrong_v2_step":
            event["lot_size"] = 0.25
        elif receipt_variant == "wrong_clock":
            event["processing_index"] = 2905
        else:
            event["dataset_id"] = "f" * 64
        rows[0] = _actor_row(exact_fill, 1, events=[event])
    elif receipt_variant == "rejected":
        rows[0] = _actor_row(
            exact_fill,
            1,
            rejected={"error_type": "ValueError", "message": "native rejection"},
        )
    elif receipt_variant == "missing_tail":
        rows[-1] = _actor_row(exact_fill, len(rows), events=None)
        rows[-1]["native"] = None
    index = _indexed_original_native_receipt(
        tmp_path,
        {"native_receipt": {"rows": rows}, "closing_rows": reserve_rows},
        f"native-{receipt_variant}",
    )

    with pytest.raises(ValueError, match=expected_error):
        consumer.consume_indexed_market_arm(
            output_index=index,
            receipt_name="receipts/arm.json.gz",
            forecast_name="forecasts/arm.json",
            content=exact_fill.content,
            policy=policy,
            binding=binding,
            role="NONRL",
            reserve_plan_digest="5" * 64,
            static_inputs=pins,
        )


def test_native_event_precondition_rejects_replaced_indexed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.tools.market_replay_oracle.test_event_chain import (
        _actor_stream,
        _reserve_stream,
    )

    consumer, exact_fill, binding, policy, pins = _native_event_consumer_fixture(
        tmp_path, monkeypatch
    )
    index = _indexed_original_native_receipt(
        tmp_path,
        {
            "native_receipt": {"rows": list(_actor_stream(exact_fill))},
            "closing_rows": list(_reserve_stream(exact_fill)),
        },
        "native-replaced-receipt",
    )
    receipt_path = index.path("receipts/arm.json.gz")
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="output file roster or identity changed"):
        consumer.consume_indexed_market_arm(
            output_index=index,
            receipt_name="receipts/arm.json.gz",
            forecast_name="forecasts/arm.json",
            content=exact_fill.content,
            policy=policy,
            binding=binding,
            role="NONRL",
            reserve_plan_digest="5" * 64,
            static_inputs=pins,
        )
