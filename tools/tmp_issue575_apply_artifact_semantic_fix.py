from __future__ import annotations

from pathlib import Path


PATH = Path("trade_rl/evaluation/experiments/bootstrap/perp_index_basis_calibration.py")


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one replacement site, found {text.count(old)}")
    return text.replace(old, new, 1)


text = PATH.read_text(encoding="utf-8")

marker = '''\n\n@dataclass(frozen=True, slots=True)\nclass PerpIndexBasisCalibrationResult:\n'''
helper = '''\n\ndef _symbol_semantic_failures(\n    item: PerpIndexBasisSymbolCalibration,\n    protocol: PerpIndexBasisProtocol,\n) -> tuple[str, ...]:\n    """Re-derive the frozen coverage/numeric gate from one symbol result."""\n\n    failures: list[str] = []\n    minimum = protocol.minimum_eligible_observations_per_symbol\n    if item.eligible_observations < minimum:\n        failures.append(f"{item.symbol}:eligible_observations<{minimum}")\n    if item.numerator is None:\n        failures.append(f"{item.symbol}:numerator_not_finite")\n    if item.denominator is None or item.denominator <= 0.0:\n        failures.append(f"{item.symbol}:denominator_not_finite_positive")\n\n    if failures:\n        if item.beta is not None:\n            raise ValueError(\n                "calibration beta must be null when coverage or numeric gate fails"\n            )\n        return tuple(failures)\n\n    assert item.numerator is not None\n    assert item.denominator is not None\n    expected_beta = item.numerator / item.denominator\n    if not math.isfinite(expected_beta):\n        if item.beta is not None:\n            raise ValueError("calibration beta must be null when regression is non-finite")\n        return (f"{item.symbol}:beta_not_finite",)\n    if item.beta is None:\n        raise ValueError("calibration beta is missing despite finite regression inputs")\n    if item.beta != expected_beta:\n        raise ValueError(\n            "calibration beta does not match canonical numerator / denominator"\n        )\n    return ()\n\n\n@dataclass(frozen=True, slots=True)\nclass PerpIndexBasisCalibrationResult:\n'''
text = replace_once(text, marker, helper)

old = '''        if self.status not in {\n            protocol.valid_status,\n            protocol.reject_status,\n            protocol.invalid_coverage_status,\n        }:\n            raise ValueError("calibration status is not canonical")\n        if any(not isinstance(item, str) or not item for item in self.failures):\n            raise ValueError("calibration failures must be non-empty strings")\n        if self.failures and self.status != protocol.invalid_coverage_status:\n            raise ValueError("calibration failures require INVALID_BASIS_COVERAGE")\n        if not self.failures:\n            expected_status = (\n                protocol.valid_status\n                if self.negative_slope_count >= protocol.required_negative_symbol_slopes\n                else protocol.reject_status\n            )\n            if self.status != expected_status:\n                raise ValueError(\n                    "calibration decision does not match frozen slope gate"\n                )\n'''
new = '''        if self.status not in {\n            protocol.valid_status,\n            protocol.reject_status,\n            protocol.invalid_coverage_status,\n        }:\n            raise ValueError("calibration status is not canonical")\n        if any(not isinstance(item, str) or not item for item in self.failures):\n            raise ValueError("calibration failures must be non-empty strings")\n\n        expected_failures = tuple(\n            failure\n            for item in self.symbol_results\n            for failure in _symbol_semantic_failures(item, protocol)\n        )\n        if self.failures != expected_failures:\n            raise ValueError(\n                "calibration failures do not match frozen coverage/numeric semantics"\n            )\n        if expected_failures:\n            if self.status != protocol.invalid_coverage_status:\n                raise ValueError(\n                    "coverage/numeric failures require INVALID_BASIS_COVERAGE"\n                )\n        else:\n            expected_status = (\n                protocol.valid_status\n                if self.negative_slope_count >= protocol.required_negative_symbol_slopes\n                else protocol.reject_status\n            )\n            if self.status != expected_status:\n                raise ValueError(\n                    "calibration decision does not match frozen slope gate"\n                )\n'''
text = replace_once(text, old, new)
PATH.write_text(text, encoding="utf-8")
