"""Regression oracles for exact pytest count gates on optional platforms."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tools.ci_exact_pytest_counts import require_exact_junit_tests


def _report(
    tmp_path: Path,
    *,
    tests: int = 2,
    skipped: int = 0,
    failures: int = 0,
    errors: int = 0,
    cases: int | None = None,
    hidden_skip: bool = False,
) -> Path:
    root = ET.Element("testsuites")
    suite = ET.SubElement(
        root,
        "testsuite",
        tests=str(tests),
        skipped=str(skipped),
        failures=str(failures),
        errors=str(errors),
    )
    for index in range(tests if cases is None else cases):
        case = ET.SubElement(suite, "testcase", name=f"case_{index}")
        if index < skipped or (index == 0 and hidden_skip):
            ET.SubElement(case, "skipped")
    path = tmp_path / "cases.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_exact_two_passed_cases_are_accepted(tmp_path: Path) -> None:
    require_exact_junit_tests(_report(tmp_path), expected_tests=2)


@pytest.mark.parametrize(
    ("tests", "skipped", "failures", "errors", "cases", "hidden_skip"),
    [
        (2, 2, 0, 0, None, False),  # pytest returns success for all-skipped
        (2, 1, 0, 0, None, False),
        (2, 0, 1, 0, None, False),
        (2, 0, 0, 1, None, False),
        (0, 0, 0, 0, None, False),
        (1, 0, 0, 0, None, False),
        (2, 0, 0, 0, 1, False),  # forged/incomplete count metadata
        (2, 0, 0, 0, None, True),  # inconsistent testcase contents
    ],
)
def test_missing_skipped_or_failed_cases_are_rejected(
    tmp_path: Path,
    tests: int,
    skipped: int,
    failures: int,
    errors: int,
    cases: int | None,
    hidden_skip: bool,
) -> None:
    report = _report(
        tmp_path,
        tests=tests,
        skipped=skipped,
        failures=failures,
        errors=errors,
        cases=cases,
        hidden_skip=hidden_skip,
    )
    with pytest.raises(ValueError):
        require_exact_junit_tests(report, expected_tests=2)


def test_missing_and_malformed_reports_are_rejected(tmp_path: Path) -> None:
    absent = tmp_path / "absent.xml"
    with pytest.raises(ValueError):
        require_exact_junit_tests(absent, expected_tests=2)
    malformed = tmp_path / "malformed.xml"
    malformed.write_text("<testsuites>", encoding="utf-8")
    with pytest.raises(ValueError):
        require_exact_junit_tests(malformed, expected_tests=2)


def test_invalid_expected_count_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        require_exact_junit_tests(_report(tmp_path), expected_tests=0)


def test_full_history_count_accepts_four_real_cases(tmp_path: Path) -> None:
    require_exact_junit_tests(_report(tmp_path, tests=4, cases=4), expected_tests=4)


def test_full_history_count_rejects_declaration_without_cases(tmp_path: Path) -> None:
    # A four-count suite with zero testcase records fooled the former inline
    # check; the shared validator requires actual executed testcase elements.
    with pytest.raises(ValueError, match="testcase count"):
        require_exact_junit_tests(_report(tmp_path, tests=4, cases=0), expected_tests=4)
