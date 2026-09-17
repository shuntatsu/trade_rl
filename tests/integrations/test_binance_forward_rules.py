import copy
import hashlib
import json
from datetime import timedelta

import pytest

from tests.integrations.test_binance_forward import NOW
from trade_rl.integrations.binance.forward_rules import (
    capture_forward_rules,
    read_forward_rules,
)


def symbol(venue, name):
    result = dict(
        symbol=name,
        status="TRADING",
        baseAsset=name[:-4],
        quoteAsset="USDT",
        orderTypes=["LIMIT", "MARKET"],
        filters=[
            dict(
                filterType="PRICE_FILTER",
                minPrice="0.01",
                maxPrice="1000000",
                tickSize="0.01",
            ),
            dict(
                filterType="LOT_SIZE", minQty="0.001", maxQty="1000", stepSize="0.001"
            ),
            dict(filterType="MARKET_LOT_SIZE", minQty="0", maxQty="100", stepSize="0"),
        ],
    )
    if venue == "spot":
        result.update(isSpotTradingAllowed=True)
        result["filters"].append(
            dict(
                filterType="NOTIONAL",
                minNotional="5",
                maxNotional="100000",
                applyMinToMarket=True,
                applyMaxToMarket=False,
                avgPriceMins=5,
            )
        )
    else:
        result.update(contractType="PERPETUAL", marginAsset="USDT")
        result["filters"].append(dict(filterType="MIN_NOTIONAL", notional="50"))
    return result


class RuleFeed:
    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate

    def _request_bytes(self, url):
        self.calls.append(url)
        venue = "perpetual" if "fapi" in url else "spot"
        payload = dict(
            serverTime=1,
            symbols=[symbol(venue, name) for name in ("BTCUSDT", "ETHUSDT")],
        )
        if self.mutate:
            self.mutate(venue, payload)
        return json.dumps(payload).encode()


def capture(root, feed=None, **kwargs):
    return capture_forward_rules(
        root,
        transport=feed or RuleFeed(),
        clock=lambda: NOW,
        monotonic=lambda: 1.0,
        **kwargs,
    )


def test_rules_preserve_raw_bytes_flags_and_disabled_market_step(tmp_path):
    root = tmp_path / "rules"
    result = capture(root)
    spot = result["rules"]["spot"]["BTCUSDT"]
    perp = result["rules"]["perpetual"]["BTCUSDT"]
    assert spot["lot_size"] == 0.001 and spot["minimum_quantity"] == 0.001
    assert spot["maximum_quantity"] == 100
    assert spot["minimum_notional"] == 5 and spot["maximum_notional"] is None
    assert spot["notional_average_minutes"] == 5
    assert perp["minimum_notional"] == 50 and perp["notional_average_minutes"] is None
    assert result["eligible"] is True and result["production_eligible"] is False
    digest = hashlib.sha256((root / "rules.json").read_bytes()).hexdigest()
    assert (
        read_forward_rules(root, expected_sha256=digest, as_of=NOW + timedelta(hours=1))
        == result
    )
    with pytest.raises(ValueError, match="digest"):
        read_forward_rules(root, expected_sha256="0" * 64)


def test_lot_intersection_uses_common_multiple_and_tightest_bounds(tmp_path):
    def mutate(venue, payload):
        for item in payload["symbols"]:
            item["filters"][1].update(stepSize="0.002", minQty="0.004")
            item["filters"][2].update(stepSize="0.003", minQty="0.009", maxQty="10")

    result = capture(tmp_path / "rules", RuleFeed(mutate))
    rule = result["rules"]["spot"]["BTCUSDT"]
    assert (
        rule["lot_size"] == 0.006
        and rule["minimum_quantity"] == 0.009
        and rule["maximum_quantity"] == 10
    )


@pytest.mark.parametrize(
    "defect",
    [
        "halt",
        "base",
        "margin",
        "contract",
        "market",
        "duplicate_symbol",
        "missing_symbol",
        "duplicate_filter",
        "missing_filter",
        "negative",
        "nan",
        "flag",
        "spot_disabled",
        "risk_control",
    ],
)
def test_unsupported_metadata_stays_failed_and_preserves_raw(tmp_path, defect):
    def mutate(venue, payload):
        item = payload["symbols"][0]
        if defect == "halt":
            item["status"] = "BREAK"
        elif defect == "base":
            item["baseAsset"] = "WRONG"
        elif defect == "margin" and venue == "perpetual":
            item["marginAsset"] = "USDC"
        elif defect == "contract" and venue == "perpetual":
            item["contractType"] = "CURRENT_QUARTER"
        elif defect == "market":
            item["orderTypes"] = ["LIMIT"]
        elif defect == "duplicate_symbol":
            payload["symbols"].append(copy.deepcopy(item))
        elif defect == "missing_symbol":
            payload["symbols"].pop()
        elif defect == "duplicate_filter":
            item["filters"].append(copy.deepcopy(item["filters"][0]))
        elif defect == "missing_filter":
            item["filters"].pop(1)
        elif defect == "negative":
            item["filters"][2]["maxQty"] = "-1"
        elif defect == "nan":
            item["filters"][0]["tickSize"] = "nan"
        elif defect == "flag" and venue == "spot":
            item["filters"][-1]["applyMinToMarket"] = "true"
        elif defect == "spot_disabled" and venue == "spot":
            item["isSpotTradingAllowed"] = False
        elif defect == "risk_control" and venue == "perpetual":
            item["filters"].append(
                dict(
                    filterType="POSITION_RISK_CONTROL", positionControlSide="LONG_ONLY"
                )
            )

    root = tmp_path / "rules"
    with pytest.raises(ValueError):
        capture(root, RuleFeed(mutate))
    assert (root / "failure.json").exists() and not (root / "rules.json").exists()
    assert list(root.glob("*.raw"))


@pytest.mark.parametrize("seconds", [-1, 3600.001])
def test_rule_consumption_rejects_future_and_expired_metadata(tmp_path, seconds):
    root = tmp_path / "rules"
    capture(root)
    with pytest.raises(ValueError):
        read_forward_rules(root, as_of=NOW + timedelta(seconds=seconds))


def test_rules_are_write_once_before_any_network(tmp_path):
    root = tmp_path / "rules"
    capture(root)
    feed = RuleFeed()
    with pytest.raises(FileExistsError):
        capture(root, feed)
    assert not feed.calls


def test_modified_rule_summary_is_not_accepted(tmp_path):
    root = tmp_path / "rules"
    result = capture(root)
    result["rules"]["spot"]["BTCUSDT"]["minimum_notional"] = 0
    (root / "rules.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="summary"):
        read_forward_rules(root)


def test_slow_rule_capture_preserves_failure(tmp_path):
    root = tmp_path / "rules"
    ticks = iter([0.0, 0.0, 6.0, 6.0, 6.0])
    with pytest.raises(ValueError):
        capture_forward_rules(
            root, transport=RuleFeed(), clock=lambda: NOW, monotonic=lambda: next(ticks)
        )
    assert (root / "failure.json").exists() and not (root / "rules.json").exists()


def test_rule_precision_cannot_silently_loosen_quantity_grid(tmp_path):
    def mutate(venue, payload):
        payload["symbols"][0]["filters"][1]["stepSize"] = "0.00100000000000000001"

    with pytest.raises(ValueError, match="precision"):
        capture(tmp_path / "rules", RuleFeed(mutate))


def test_unquoted_decimal_token_cannot_bypass_rule_precision_validation(tmp_path):
    class NumericFeed(RuleFeed):
        def _request_bytes(self, url):
            return (
                super()
                ._request_bytes(url)
                .replace(b'"stepSize": "0.001"', b'"stepSize": 0.00100000000000000001')
            )

    with pytest.raises(ValueError, match="precision"):
        capture(tmp_path / "rules", NumericFeed())


def test_inactive_notional_and_active_maximum_keep_market_flags(tmp_path):
    def mutate(venue, payload):
        if venue == "spot":
            for item in payload["symbols"]:
                item["filters"][-1].update(
                    applyMinToMarket=False, applyMaxToMarket=True
                )

    result = capture(tmp_path / "rules", RuleFeed(mutate))
    rule = result["rules"]["spot"]["BTCUSDT"]
    assert rule["minimum_notional"] == 0 and rule["maximum_notional"] == 100000
    assert rule["notional_average_minutes"] == 5


def test_nonintersecting_quantity_rules_are_ineligible(tmp_path):
    def mutate(venue, payload):
        payload["symbols"][0]["filters"][1].update(stepSize="3", minQty="2", maxQty="2")

    with pytest.raises(ValueError, match="intersection"):
        capture(tmp_path / "rules", RuleFeed(mutate))


def test_network_failure_preserves_received_metadata_without_eligibility(tmp_path):
    class FailingFeed(RuleFeed):
        def _request_bytes(self, url):
            if "fapi" in url:
                raise RuntimeError("network unavailable")
            return super()._request_bytes(url)

    root = tmp_path / "rules"
    with pytest.raises(RuntimeError, match="unavailable"):
        capture(root, FailingFeed())
    assert (root / "spot_rules.raw").exists() and (root / "failure.json").exists()
    assert not (root / "rules.json").exists()
