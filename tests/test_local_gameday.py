from mars_lite.serving.local_gameday import (
    exit_code_for_summary,
    run_local_gameday,
)

_EXPECTED_SCENARIOS = {
    "healthy_activation",
    "content_mutation_identity",
    "timeframe_freshness",
    "stale_data_fail_closed",
    "replay_rejection",
    "bundle_rejection_preserves_healthy_runtime",
    "rollback",
}


def test_exit_code_is_nonzero_when_any_scenario_fails() -> None:
    assert exit_code_for_summary({"passed": True, "scenarios": []}) == 0
    assert (
        exit_code_for_summary(
            {
                "passed": False,
                "scenarios": [
                    {"name": "healthy_activation", "passed": False, "details": {}}
                ],
            }
        )
        == 1
    )


def test_local_gameday_reports_all_required_scenarios(tmp_path) -> None:
    summary = run_local_gameday(tmp_path)

    assert summary["passed"] is True
    assert {item["name"] for item in summary["scenarios"]} == _EXPECTED_SCENARIOS
    assert all(item["passed"] is True for item in summary["scenarios"])


def test_local_gameday_is_deterministic_for_fixed_inputs(tmp_path) -> None:
    first = run_local_gameday(tmp_path / "first")
    second = run_local_gameday(tmp_path / "second")

    assert first == second
