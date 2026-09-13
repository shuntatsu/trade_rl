from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import pytest

MODULE = "research.issue522_portable_exp003_prereg"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "Experiment 0003 preregistration helper is not implemented"
    return import_module(MODULE)


def _payload():
    module = _module()
    return module.expected_prereg_index(
        definition_digest="d" * 64,
        requested_candidate_config_digest="c" * 64,
        structural_report_sha256="a" * 64,
    )


def test_formal_rule_is_exact_and_result_blind() -> None:
    module = _module()
    assert module.formal_rule_payload() == {
        "accept": {
            "positive_trend_factor_effect_symbols_min": 4,
            "median_trend_excess_total_return_strictly_positive": True,
            "candidate_trend_positive_total_return_symbols": 5,
            "candidate_trend_turnover_strictly_below_baseline": True,
        },
        "keep_baseline": {
            "positive_trend_factor_effect_symbols_max": 2,
            "median_trend_excess_total_return_non_positive": True,
            "candidate_trend_positive_total_return_symbols_max": 3,
        },
        "otherwise": "INCONCLUSIVE",
    }


def test_validate_prereg_index_accepts_exact_payload() -> None:
    module = _module()
    payload = _payload()
    module.validate_prereg_index_payload(
        payload,
        definition_digest="d" * 64,
        requested_candidate_config_digest="c" * 64,
        structural_report_sha256="a" * 64,
    )


def test_validate_prereg_index_rejects_extra_key() -> None:
    module = _module()
    payload = _payload()
    payload["post_result_override"] = True
    with pytest.raises(RuntimeError, match="key set"):
        module.validate_prereg_index_payload(
            payload,
            definition_digest="d" * 64,
            requested_candidate_config_digest="c" * 64,
            structural_report_sha256="a" * 64,
        )


def test_validate_prereg_index_rejects_rule_tampering() -> None:
    module = _module()
    payload = _payload()
    payload["formal_decision_rule"]["accept"][  # type: ignore[index]
        "candidate_trend_positive_total_return_symbols"
    ] = 4
    with pytest.raises(RuntimeError, match="formal_decision_rule"):
        module.validate_prereg_index_payload(
            payload,
            definition_digest="d" * 64,
            requested_candidate_config_digest="c" * 64,
            structural_report_sha256="a" * 64,
        )


def test_validate_prereg_index_rejects_candidate_executed_flag() -> None:
    module = _module()
    payload = _payload()
    payload["candidate_executed"] = True
    with pytest.raises(RuntimeError, match="candidate_executed"):
        module.validate_prereg_index_payload(
            payload,
            definition_digest="d" * 64,
            requested_candidate_config_digest="c" * 64,
            structural_report_sha256="a" * 64,
        )


def test_validate_prereg_index_rejects_structural_digest_drift() -> None:
    module = _module()
    payload = _payload()
    with pytest.raises(RuntimeError, match="structural_report_sha256"):
        module.validate_prereg_index_payload(
            payload,
            definition_digest="d" * 64,
            requested_candidate_config_digest="c" * 64,
            structural_report_sha256="b" * 64,
        )
