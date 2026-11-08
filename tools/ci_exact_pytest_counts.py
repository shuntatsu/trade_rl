"""Fail closed when optional-platform pytest runs skip or omit required tests."""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def require_exact_junit_tests(report: Path, *, expected_tests: int) -> None:
    """Check JUnit testcase outcomes, not just pytest's zero exit code."""
    if isinstance(expected_tests, bool) or not isinstance(expected_tests, int):
        raise ValueError("expected_tests must be a positive integer")
    if expected_tests <= 0:
        raise ValueError("expected_tests must be a positive integer")
    try:
        root = ET.parse(report).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError("JUnit report is missing or malformed") from error

    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    else:
        suites = []
    if len(suites) != 1:
        raise ValueError("JUnit report must contain exactly one testsuite")

    suite = suites[0]
    names = ("tests", "skipped", "failures", "errors")
    try:
        counts = tuple(int(suite.attrib[name]) for name in names)
    except (KeyError, ValueError) as error:
        raise ValueError("JUnit counters are missing or invalid") from error
    if counts != (expected_tests, 0, 0, 0):
        raise ValueError(f"Unexpected JUnit test or skipped/failed counts: {counts}")

    cases = suite.findall("testcase")
    if len(cases) != expected_tests:
        raise ValueError("JUnit testcase count does not match the declared count")
    for case in cases:
        for outcome in ("skipped", "failure", "error"):
            if case.find(outcome) is not None:
                raise ValueError("JUnit testcase contains a skipped or failed outcome")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assert exact passed pytest count")
    parser.add_argument("report_name", help="JUnit filename below RUNNER_TEMP")
    parser.add_argument("expected_tests", type=int)
    args = parser.parse_args()
    if args.report_name != Path(args.report_name).name:
        parser.error("report_name must be a basename")
    runner_temp = os.environ.get("RUNNER_TEMP")
    if not runner_temp:
        parser.error("RUNNER_TEMP is required")

    try:
        require_exact_junit_tests(
            Path(runner_temp) / args.report_name,
            expected_tests=args.expected_tests,
        )
    except ValueError as error:
        parser.exit(1, f"JUnit validation failed: {error}\n")
    print(f"JUnit: exactly {args.expected_tests} passed, none skipped or failed")


if __name__ == "__main__":
    main()
