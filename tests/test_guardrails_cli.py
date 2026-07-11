import json

from mars_lite.trading.guardrails import PositionSnapshot, main


class FakeExecutor:
    def __init__(self, *, fail_cancel: bool = False, partial: bool = False):
        self.blocked = False
        self.open_orders = ["open-1"]
        self.positions = {"BTCUSDT": 2.0, "ETHUSDT": -1.0}
        self.fail_cancel = fail_cancel
        self.partial = partial
        self.submitted_client_ids = []

    def block_new_risk(self, reason, idempotency_key):
        self.blocked = True

    def cancel_all_orders(self, reason, idempotency_key):
        if self.fail_cancel:
            return []
        cancelled = list(self.open_orders)
        self.open_orders = []
        return cancelled

    def list_open_order_ids(self):
        return list(self.open_orders)

    def list_positions(self):
        return [PositionSnapshot(symbol, quantity) for symbol, quantity in self.positions.items()]

    def submit_reduce_only_market_order(self, symbol, side, quantity, client_order_id):
        self.submitted_client_ids.append(client_order_id)
        if not self.partial:
            self.positions[symbol] = 0.0
        else:
            self.positions[symbol] *= 0.5
        return f"reduce-{symbol}"

    def reconcile(self):
        return None


def test_flatten_without_executor_fails_closed(capsys):
    ret = main(
        [
            "--action",
            "flatten",
            "--idempotency-key",
            "incident-1234",
            "--output-format",
            "json",
        ]
    )
    assert ret == 2
    data = json.loads(capsys.readouterr().out)
    assert data["success"] is False
    assert data["action"] == "flatten_not_executed"


def test_flatten_executes_and_reconciles(capsys):
    executor = FakeExecutor()
    ret = main(
        [
            "--action",
            "flatten",
            "--reason",
            "emergency stop",
            "--idempotency-key",
            "incident-1234",
            "--output-format",
            "json",
        ],
        executor=executor,
    )
    assert ret == 0
    data = json.loads(capsys.readouterr().out)
    assert data["success"] is True
    assert data["blocked_new_risk"] is True
    assert data["remaining_open_order_ids"] == []
    assert all(abs(value) == 0.0 for value in data["remaining_positions"].values())
    assert len(executor.submitted_client_ids) == 2


def test_flatten_never_claims_success_when_orders_remain(capsys):
    executor = FakeExecutor(fail_cancel=True)
    ret = main(
        [
            "--action",
            "flatten",
            "--idempotency-key",
            "incident-1234",
        ],
        executor=executor,
    )
    assert ret == 1
    data = json.loads(capsys.readouterr().out)
    assert data["success"] is False
    assert data["remaining_open_order_ids"] == ["open-1"]


def test_flatten_retries_partial_residuals(capsys):
    executor = FakeExecutor(partial=True)
    ret = main(
        [
            "--action",
            "flatten",
            "--idempotency-key",
            "incident-1234",
            "--max-reconcile-rounds",
            "3",
            "--position-tolerance",
            "0.3",
        ],
        executor=executor,
    )
    assert ret == 0
    data = json.loads(capsys.readouterr().out)
    assert data["success"] is True


def test_scale_is_explicitly_advisory(capsys):
    ret = main(["--action", "scale", "--scale", "0.5"])
    assert ret == 0
    data = json.loads(capsys.readouterr().out)
    assert data["action"] == "scale"
    assert "advisory" in data["warnings"][0]
