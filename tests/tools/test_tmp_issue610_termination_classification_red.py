from tools.tmp_issue610_exact_once_orchestrator import partition_termination_violations


def test_new_termination_is_gate_failure_but_malformed_is_invalid() -> None:
    new, invalid = partition_termination_violations(
        ("new PPO termination: seed=0 BTCUSDT reason=margin_call",)
    )
    assert new == ("new PPO termination: seed=0 BTCUSDT reason=margin_call",)
    assert invalid == ()

    new, invalid = partition_termination_violations(
        ("candidate PPO termination evidence malformed: seed=0 BTCUSDT",)
    )
    assert new == ()
    assert invalid == (
        "candidate PPO termination evidence malformed: seed=0 BTCUSDT",
    )


def test_unknown_termination_violation_fails_closed_as_invalid() -> None:
    new, invalid = partition_termination_violations(("unexpected termination error",))
    assert new == ()
    assert invalid == ("unexpected termination error",)
