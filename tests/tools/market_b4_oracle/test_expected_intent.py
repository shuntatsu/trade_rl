from __future__ import annotations

import hashlib
import math
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

import tools.market_b4_oracle.expected_intent as _module
from trade_rl.simulation.accounting import BookState as NativeBookState
from trade_rl.simulation.orders.model import (
    OrderBookState as NativeOrderBookState,
)
from trade_rl.simulation.orders.model import (
    OrderIntent as NativeOrderIntent,
)
from trade_rl.simulation.orders.model import (
    OrderType as NativeOrderType,
)
from trade_rl.simulation.orders.model import (
    PendingOrder as NativePendingOrder,
)
from trade_rl.simulation.orders.model import (
    TimeInForce as NativeTimeInForce,
)
from trade_rl.simulation.orders.reconciliation import reconcile_target

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def api():
    return _module


POLICY = "a" * 64
DATASET = "synthetic-dataset"
TARGET = "synthetic-target"


def intent(**changes):
    values = {
        "dataset_id": DATASET,
        "target_identity": TARGET,
        "execution_policy_digest": POLICY,
        "symbol_index": 0,
        "requested_quantity": 0.25,
        "order_type": "market",
        "time_in_force": "gtc",
        "limit_price": None,
        "stop_price": None,
        "submit_index": 3,
        "eligible_index": 4,
        "expiry_index": None,
        "submission_reference_price": 123.456789,
        "decision_equity": 100000.00000000003,
    }
    values.update(changes)
    return api().ExpectedIntent.create(**values)


def reconcile(**changes):
    values = {
        "exact_quantities": (Fraction(0), Fraction(0)),
        "current_weights": (0.0, 0.0),
        "final_risk_weights": (0.5, 0.0),
        "decision_equity": 100000.0,
        "valuation_prices": (100.0, 200.0),
        "multipliers": (1.0, 1.0),
        "reference_prices": (101.0, 201.0),
        "dataset_id": DATASET,
        "target_identity": TARGET,
        "execution_policy_digest": POLICY,
        "submit_index": 3,
        "latency_bars": 1,
        "pending_orders": (),
    }
    values.update(changes)
    return api().reconcile_expected(**values)


def native_intent(**changes):
    values = {
        "dataset_id": DATASET,
        "target_identity": TARGET,
        "execution_policy_digest": POLICY,
        "symbol_index": 0,
        "requested_quantity": 0.25,
        "order_type": NativeOrderType.MARKET,
        "time_in_force": NativeTimeInForce.GTC,
        "limit_price": None,
        "stop_price": None,
        "submit_index": 3,
        "eligible_index": 4,
        "expiry_index": None,
        "submission_reference_price": 123.456789,
        "decision_equity": 100000.00000000003,
    }
    values.update(changes)
    values["order_type"] = NativeOrderType(values["order_type"])
    values["time_in_force"] = NativeTimeInForce(values["time_in_force"])
    return NativeOrderIntent.create(**values)


def native_reconcile(**changes):
    values = {
        "exact_quantities": (Fraction(0), Fraction(0)),
        "final_risk_weights": (0.5, 0.0),
        "decision_equity": 100000.0,
        "valuation_prices": (100.0, 200.0),
        "multipliers": (1.0, 1.0),
        "reference_prices": (101.0, 201.0),
        "dataset_id": DATASET,
        "target_identity": TARGET,
        "execution_policy_digest": POLICY,
        "submit_index": 3,
        "latency_bars": 1,
        "pending_orders": (),
    }
    values.update(changes)
    quantities = np.asarray(
        [api().project_quantity(value) for value in values["exact_quantities"]],
        dtype=np.float64,
    )
    prices = np.asarray(values["valuation_prices"], dtype=np.float64)
    multipliers = np.asarray(values["multipliers"], dtype=np.float64)
    position_value = float(np.sum(quantities * prices * multipliers))
    equity = float(values["decision_equity"])
    native_book = NativeBookState(
        quantities=quantities,
        cash=equity - position_value,
        mark_prices=prices,
        peak_value=max(equity, 1.0),
        contract_multipliers=multipliers,
        _exact_quantities=tuple(str(value) for value in values["exact_quantities"]),
    )
    native_order_book = NativeOrderBookState(
        active_orders=values["pending_orders"], terminal_orders=()
    )
    return reconcile_target(
        dataset_id=values["dataset_id"],
        target_identity=values["target_identity"],
        execution_policy_digest=values["execution_policy_digest"],
        target_weights=np.asarray(values["final_risk_weights"], dtype=np.float64),
        book=native_book,
        order_book=native_order_book,
        reference_prices=np.asarray(values["reference_prices"], dtype=np.float64),
        decision_equity=values["decision_equity"],
        submit_index=values["submit_index"],
        latency_bars=values["latency_bars"],
        order_type=NativeOrderType.MARKET,
        time_in_force=NativeTimeInForce.GTC,
        expiry_index=None,
        limit_offset_rate=0.0,
        maximum_gross=1.0,
        valuation_prices=prices,
    )


def native_pending(**changes):
    return NativePendingOrder.from_intent(native_intent(**changes))


class ExpectedIntentTests(unittest.TestCase):
    def test_public_differential_imports_are_from_q_checkout(self):
        self.assertEqual(
            Path(reconcile_target.__code__.co_filename).resolve(),
            REPOSITORY_ROOT
            / "trade_rl"
            / "simulation"
            / "orders"
            / "reconciliation.py",
        )

    def test_expected_identity_matches_native_public_fixture(self):
        order = intent(
            dataset_id="synthetic-manufactured-dataset",
            target_identity="synthetic-manufactured-target",
            execution_policy_digest="0" * 64,
        )
        self.assertEqual(
            order.order_id,
            "7c0897802073abf7195660a111941c42612618ca0806ba61725b149c618ee8dd",
        )
        payload = order.identity_payload()
        self.assertEqual(payload["schema_version"], "order_intent_v1")
        self.assertNotIn("reduce_only", payload)
        self.assertEqual(
            set(payload),
            {
                "dataset_id",
                "decision_equity",
                "eligible_index",
                "execution_policy_digest",
                "expiry_index",
                "limit_price",
                "order_type",
                "replaced_order_id",
                "requested_quantity",
                "schema_version",
                "stop_price",
                "submission_reference_price",
                "submit_index",
                "symbol_index",
                "target_identity",
                "time_in_force",
            },
        )
        self.assertEqual(
            hashlib.sha256(api().canonical_json_bytes(payload)).hexdigest(),
            order.order_id,
        )
        payload["target_identity"] = "mutated-copy"
        self.assertEqual(
            order.identity_payload()["target_identity"], "synthetic-manufactured-target"
        )
        with self.assertRaises(ValueError):
            replace(order, order_id="0" * 64)

    def test_canonical_payload_matches_native_public_order_intent(self):
        own = intent()
        native = native_intent()
        self.assertEqual(own.canonical_payload(), native.canonical_payload())
        self.assertNotIn("schema_version", own.canonical_payload())
        self.assertIn("order_id", own.canonical_payload())
        self.assertNotIn("reduce_only", own.canonical_payload())

        own_reduce = intent(requested_quantity=-1.0, reduce_only=True)
        native_reduce = native_intent(requested_quantity=-1.0, reduce_only=True)
        self.assertEqual(
            own_reduce.canonical_payload(), native_reduce.canonical_payload()
        )
        self.assertTrue(own_reduce.canonical_payload()["reduce_only"])

    def test_native_identity_numeric_fields_require_original_float_types(self):
        for field in (
            "requested_quantity",
            "submission_reference_price",
            "decision_equity",
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                intent(**{field: 1})

    def test_one_ulp_equity_change_changes_identity(self):
        fixture = {
            "dataset_id": "synthetic-manufactured-dataset",
            "target_identity": "synthetic-manufactured-target",
            "execution_policy_digest": "0" * 64,
        }
        high = intent(decision_equity=100000.00000000003, **fixture)
        low = intent(decision_equity=100000.00000000001, **fixture)
        self.assertEqual(
            high.order_id,
            "7c0897802073abf7195660a111941c42612618ca0806ba61725b149c618ee8dd",
        )
        self.assertEqual(
            low.order_id,
            "5f296159d1ef08af6dd9518b4694265ac6695dfd714b220605929731cc28fcf8",
        )
        self.assertNotEqual(high.order_id, low.order_id)

    def test_canonical_json_preserves_signed_zero_and_invalid_order_is_rejected(self):
        self.assertNotEqual(
            api().canonical_json_bytes({"value": -0.0}),
            api().canonical_json_bytes({"value": 0.0}),
        )
        with self.assertRaises(ValueError):
            intent(requested_quantity=0.0)
        reducing = intent(requested_quantity=-1.0, reduce_only=True)
        self.assertTrue(reducing.identity_payload()["reduce_only"])
        self.assertEqual(reducing.recompute_order_id(), reducing.order_id)

    def test_fractional_two_symbol_market_sizing_matches_native_without_lot_rounding(
        self,
    ):
        inputs = {"final_risk_weights": (0.333333333, 0.0)}
        result = reconcile(**inputs)
        native = native_reconcile(**inputs)
        self.assertEqual(result.desired_quantities, tuple(native.desired_quantities))
        self.assertEqual(result.residual_quantities, tuple(native.residual_quantities))
        self.assertEqual(len(result.new_intents), 1)
        self.assertEqual(result.new_intents[0].symbol_index, 0)
        self.assertEqual(result.new_intents[0].requested_quantity, 333.333333)
        self.assertNotEqual(result.new_intents[0].requested_quantity % 1.0, 0.0)
        self.assertEqual(result.new_intents[0].order_type, "market")
        self.assertEqual(result.new_intents[0].time_in_force, "gtc")
        self.assertEqual(result.new_intents[0].submission_reference_price, 101.0)
        self.assertEqual(
            result.new_intents[0].canonical_payload(),
            native.new_intents[0].canonical_payload(),
        )

    def test_compatible_partial_pending_is_kept_by_exact_remaining_projection(self):
        pending = api().ExpectedPending.from_intent(
            intent(requested_quantity=500.0, decision_equity=100000.0),
            exact_filled_quantity=Fraction(100),
            status="partially_filled",
        )
        result = reconcile(
            exact_quantities=(Fraction(100), Fraction(0)),
            current_weights=(0.1, 0.0),
            pending_orders=(pending,),
        )
        self.assertEqual(pending.remaining_exact_quantity, Fraction(400))
        self.assertEqual(pending.remaining_quantity, 400.0)
        self.assertEqual(result.kept_orders, (pending,))
        self.assertEqual(result.cancelled_orders, ())
        self.assertEqual(result.new_intents, ())

    def test_compatible_pending_aggregate_is_kept(self):
        first = api().ExpectedPending.from_intent(
            intent(requested_quantity=200.0, eligible_index=4),
            exact_filled_quantity=Fraction(0),
        )
        second = api().ExpectedPending.from_intent(
            intent(requested_quantity=300.0, eligible_index=5),
            exact_filled_quantity=Fraction(0),
        )
        result = reconcile(pending_orders=(second, first))
        self.assertEqual(result.kept_orders, (second, first))
        self.assertEqual(result.cancelled_orders, ())
        self.assertEqual(result.new_intents, ())

    def test_compatible_kept_orders_preserve_global_native_order(self):
        eth = api().ExpectedPending.from_intent(
            intent(symbol_index=1, requested_quantity=300.0),
            exact_filled_quantity=Fraction(0),
        )
        btc = api().ExpectedPending.from_intent(
            intent(symbol_index=0, requested_quantity=100.0),
            exact_filled_quantity=Fraction(0),
        )
        native_eth = native_pending(symbol_index=1, requested_quantity=300.0)
        native_btc = native_pending(symbol_index=0, requested_quantity=100.0)
        own = reconcile(
            exact_quantities=(Fraction(100), Fraction(100)),
            current_weights=(0.1, 0.1),
            final_risk_weights=(0.2, 0.4),
            valuation_prices=(100.0, 100.0),
            pending_orders=(eth, btc),
        )
        native = native_reconcile(
            exact_quantities=(Fraction(100), Fraction(100)),
            final_risk_weights=(0.2, 0.4),
            valuation_prices=(100.0, 100.0),
            pending_orders=(native_eth, native_btc),
        )
        self.assertEqual(own.kept_orders, (eth, btc))
        self.assertEqual(
            tuple(row.order_id for row in own.kept_orders),
            tuple(row.intent.order_id for row in native.order_book.active_orders),
        )
        self.assertEqual(own.new_intents, ())
        self.assertEqual(native.new_intents, ())

    def test_incompatible_orders_cancel_in_eligibility_then_id_order(self):
        later = api().ExpectedPending.from_intent(
            intent(
                requested_quantity=500.0, eligible_index=8, dataset_id="other-dataset"
            ),
            exact_filled_quantity=Fraction(0),
        )
        earlier = api().ExpectedPending.from_intent(
            intent(
                requested_quantity=500.0, eligible_index=4, dataset_id="other-dataset"
            ),
            exact_filled_quantity=Fraction(0),
        )
        result = reconcile(pending_orders=(later, earlier))
        self.assertEqual(
            tuple(row.order.order_id for row in result.cancelled_orders),
            (earlier.order_id, later.order_id),
        )
        self.assertTrue(
            all(row.reason == "superseded" for row in result.cancelled_orders)
        )
        self.assertEqual(result.new_intents[0].replaced_order_id, earlier.order_id)

    def test_incompatible_orders_with_equal_eligibility_cancel_by_id(self):
        first = api().ExpectedPending.from_intent(
            intent(requested_quantity=500.0, eligible_index=4, dataset_id="other"),
            exact_filled_quantity=Fraction(0),
        )
        second = api().ExpectedPending.from_intent(
            intent(
                requested_quantity=500.0,
                eligible_index=4,
                dataset_id="other",
                target_identity="other-target",
            ),
            exact_filled_quantity=Fraction(0),
        )
        ordered = tuple(sorted((first, second), key=lambda row: row.order_id))
        own = reconcile(pending_orders=(ordered[1], ordered[0]))
        native_pending_rows = (
            native_pending(
                requested_quantity=500.0, eligible_index=4, dataset_id="other"
            ),
            native_pending(
                requested_quantity=500.0,
                eligible_index=4,
                dataset_id="other",
                target_identity="other-target",
            ),
        )
        native = native_reconcile(pending_orders=native_pending_rows)
        self.assertEqual(
            tuple(row.order.order_id for row in own.cancelled_orders),
            tuple(row.order_id for row in native.cancelled_orders),
        )
        self.assertEqual(
            tuple(row.order.order_id for row in own.cancelled_orders),
            tuple(row.order_id for row in ordered),
        )
        self.assertEqual(
            own.new_intents[0].canonical_payload(),
            native.new_intents[0].canonical_payload(),
        )

    def test_policy_and_reduce_only_mismatches_replace_in_native_order(self):
        cases = (
            ("policy", intent(execution_policy_digest="b" * 64)),
            ("reduce", intent(reduce_only=True)),
        )
        for label, order in cases:
            with self.subTest(label=label):
                pending = api().ExpectedPending.from_intent(
                    order, exact_filled_quantity=Fraction(0)
                )
                native_order = native_intent(
                    execution_policy_digest="b" * 64 if label == "policy" else POLICY,
                    reduce_only=label == "reduce",
                )
                own = reconcile(pending_orders=(pending,))
                native = native_reconcile(
                    pending_orders=(NativePendingOrder.from_intent(native_order),)
                )
                self.assertEqual(
                    tuple(row.order.order_id for row in own.cancelled_orders),
                    tuple(row.order_id for row in native.cancelled_orders),
                )
                self.assertEqual(
                    own.new_intents[0].canonical_payload(),
                    native.new_intents[0].canonical_payload(),
                )

    def test_hold_cancels_every_active_order_in_original_order_without_new_ids(self):
        first = api().ExpectedPending.from_intent(
            intent(requested_quantity=1.0, symbol_index=0),
            exact_filled_quantity=Fraction(0),
        )
        second = api().ExpectedPending.from_intent(
            intent(requested_quantity=-1.0, symbol_index=1),
            exact_filled_quantity=Fraction(0),
        )
        result = reconcile(
            exact_quantities=(Fraction(0), Fraction(0)),
            current_weights=(0.25, -0.25),
            final_risk_weights=(0.25, -0.25),
            target_identity=None,
            pending_orders=(second, first),
        )
        self.assertEqual(result.desired_quantities, None)
        self.assertEqual(result.new_intents, ())
        self.assertEqual(
            tuple(row.order for row in result.cancelled_orders), (second, first)
        )
        self.assertTrue(
            all(row.reason == "quantity_hold" for row in result.cancelled_orders)
        )

    def test_legacy_flatten_uses_inward_exact_delta_and_reduce_only(self):
        huge = Fraction(2**53 + 1)
        inputs = {
            "exact_quantities": (huge, Fraction(0)),
            "current_weights": (0.25, 0.0),
            "final_risk_weights": (0.0, 0.0),
            "decision_equity": 1.0,
            "valuation_prices": (1.0, 1.0),
            "multipliers": (1.0, 1.0),
            "reference_prices": (1.0, 1.0),
        }
        result = reconcile(**inputs)
        native = native_reconcile(
            exact_quantities=inputs["exact_quantities"],
            final_risk_weights=inputs["final_risk_weights"],
            decision_equity=inputs["decision_equity"],
            valuation_prices=inputs["valuation_prices"],
            multipliers=inputs["multipliers"],
            reference_prices=inputs["reference_prices"],
        )
        self.assertEqual(len(result.new_intents), 1)
        order = result.new_intents[0]
        self.assertEqual(order.requested_quantity, -float(2**53))
        self.assertTrue(order.reduce_only)
        self.assertEqual(order.replaced_order_id, None)
        self.assertEqual(
            order.canonical_payload(), native.new_intents[0].canonical_payload()
        )

    def test_negative_huge_legacy_flatten_matches_native_inward_projection(self):
        huge = Fraction(-(2**53 + 1))
        inputs = {
            "exact_quantities": (huge, Fraction(0)),
            "current_weights": (-0.25, 0.0),
            "final_risk_weights": (0.0, 0.0),
            "decision_equity": 1.0,
            "valuation_prices": (1.0, 1.0),
            "multipliers": (1.0, 1.0),
            "reference_prices": (1.0, 1.0),
        }
        own = reconcile(**inputs)
        native = native_reconcile(
            exact_quantities=inputs["exact_quantities"],
            final_risk_weights=inputs["final_risk_weights"],
            decision_equity=inputs["decision_equity"],
            valuation_prices=inputs["valuation_prices"],
            multipliers=inputs["multipliers"],
            reference_prices=inputs["reference_prices"],
        )
        self.assertEqual(own.new_intents[0].requested_quantity, float(2**53))
        self.assertTrue(own.new_intents[0].reduce_only)
        self.assertEqual(
            own.new_intents[0].canonical_payload(),
            native.new_intents[0].canonical_payload(),
        )

    def test_changed_target_cannot_exceed_frozen_maximum_gross(self):
        near_limit = {"final_risk_weights": (1.0 + 0.5e-12, 0.0)}
        accepted = reconcile(**near_limit)
        native_accepted = native_reconcile(**near_limit)
        self.assertEqual(len(accepted.new_intents), 1)
        self.assertEqual(
            accepted.new_intents[0].canonical_payload(),
            native_accepted.new_intents[0].canonical_payload(),
        )
        with self.assertRaisesRegex(ValueError, "gross"):
            reconcile(final_risk_weights=(1.0 + 2e-12, 0.0))
        with self.assertRaisesRegex(ValueError, "gross"):
            native_reconcile(final_risk_weights=(1.0, 1.0))

    def test_tiny_residual_is_suppressed(self):
        result = reconcile(
            final_risk_weights=(5e-15, 0.0),
            decision_equity=100.0,
            valuation_prices=(1.0, 1.0),
            multipliers=(1.0, 1.0),
        )
        self.assertEqual(result.new_intents, ())
        self.assertEqual(result.residual_quantities, (0.0, 0.0))

    def test_target_identity_is_required_outside_hold(self):
        with self.assertRaises(ValueError):
            reconcile(target_identity=None)

    def test_pending_remaining_quantity_projects_exact_fraction(self):
        pending = api().ExpectedPending.from_intent(
            intent(requested_quantity=0.3),
            exact_filled_quantity=Fraction(1, 10),
        )
        self.assertEqual(pending.remaining_exact_quantity, Fraction(1, 5))
        self.assertLessEqual(Fraction(str(pending.remaining_quantity)), Fraction(1, 5))

    def test_only_market_gtc_is_supported(self):
        with self.assertRaises(ValueError):
            intent(order_type="limit")
        with self.assertRaises(ValueError):
            intent(time_in_force="day")

    def test_duplicate_pending_ids_and_malformed_pairs_reject(self):
        one = api().ExpectedPending.from_intent(
            intent(), exact_filled_quantity=Fraction(0)
        )
        duplicate = api().ExpectedPending.from_intent(
            intent(), exact_filled_quantity=Fraction(0)
        )
        with self.assertRaises(ValueError):
            reconcile(pending_orders=(one, duplicate))
        with self.assertRaises(ValueError):
            reconcile(current_weights=(0.0,))
        with self.assertRaises(ValueError):
            reconcile(current_weights=(math.nan, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
