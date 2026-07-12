from mars_lite.pipeline.release_eligibility import derive_release_eligibility


def _derive(**overrides: object):
    values: dict[str, object] = {
        "forced": False,
        "skip_p0": False,
        "skip_pbt": False,
        "skip_wf": False,
        "skip_gate": False,
        "sealed_holdout_used": True,
        "p0_passed": True,
        "signal_gate_passed": True,
        "walk_forward_passed": True,
        "gate2_passed": True,
        "significance_passed": None,
    }
    values.update(overrides)
    return derive_release_eligibility(**values)  # type: ignore[arg-type]


def test_normal_run_is_release_eligible() -> None:
    result = _derive()

    assert result.eligible is True
    assert result.skipped_gates == ()
    assert result.optimization_steps_skipped == ()
    assert result.required_gates["significance"] == "not_required"


def test_forced_run_is_not_release_eligible() -> None:
    result = _derive(forced=True, significance_passed=True)

    assert result.eligible is False
    assert result.forced is True


def test_skipped_pbt_is_recorded_but_not_disqualifying() -> None:
    result = _derive(skip_pbt=True, significance_passed=True)

    assert result.eligible is True
    assert result.optimization_steps_skipped == ("pbt",)


def test_missing_holdout_is_not_release_eligible() -> None:
    result = _derive(sealed_holdout_used=False, significance_passed=True)

    assert result.eligible is False


def test_skipped_mandatory_gate_is_recorded_and_disqualifying() -> None:
    result = _derive(skip_wf=True, significance_passed=True)

    assert result.eligible is False
    assert result.skipped_gates == ("walk_forward",)
    assert result.required_gates["walk_forward"] == "skipped"


def test_skipped_signal_gate_is_not_misreported_as_gate2() -> None:
    result = _derive(skip_gate=True, significance_passed=True)

    assert result.eligible is False
    assert result.skipped_gates == ("signal_gate",)
    assert result.required_gates["signal_gate"] == "skipped"
    assert result.required_gates["gate2"] == "passed"


def test_failed_mandatory_gate_is_disqualifying() -> None:
    result = _derive(gate2_passed=False, significance_passed=True)

    assert result.eligible is False
    assert result.required_gates["gate2"] == "failed"


def test_serialization_preserves_immutable_contract_fields() -> None:
    result = _derive(skip_pbt=True)

    assert result.to_dict() == {
        "eligible": True,
        "forced": False,
        "skipped_gates": (),
        "optimization_steps_skipped": ("pbt",),
        "sealed_holdout_used": True,
        "required_gates": {
            "p0": "passed",
            "signal_gate": "passed",
            "walk_forward": "passed",
            "gate2": "passed",
            "significance": "not_required",
        },
    }
