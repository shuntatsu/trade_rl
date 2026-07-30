from __future__ import annotations

import base64
import gzip
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _replace_exact(text: str, old: str, new: str, *, field: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{field}: expected one replacement target, observed {count}")
    return text.replace(old, new)


def _emit(name: str, content: str) -> None:
    encoded = base64.b64encode(gzip.compress(content.encode("utf-8"), mtime=0)).decode(
        "ascii"
    )
    print(f"A4_PATCH_BEGIN:{name}")
    print(encoded)
    print(f"A4_PATCH_END:{name}")


def test_emit_exact_a4_runner_patch_material() -> None:
    pipeline_path = ROOT / "examples" / "binance-multitimeframe" / "full_research_pipeline.py"
    pipeline = pipeline_path.read_text(encoding="utf-8")
    pipeline = _replace_exact(
        pipeline,
        "_EXPECTED_POLICY_OBSERVATIONS = 231_026",
        "_EXPECTED_POLICY_OBSERVATIONS = 217_886",
        field="runner observation count",
    )
    pipeline = _replace_exact(
        pipeline,
        '''    expected_dataset_features = (
        (*expected_features, *(f"15m__symbol_id_{symbol}" for symbol in _SYMBOL_POOL))
        if use_postgres
        else expected_features
    )''',
        "    expected_dataset_features = expected_features",
        field="runner feature contract",
    )

    tests_path = ROOT / "tests" / "examples" / "test_binance_multitimeframe_full_assets.py"
    tests = tests_path.read_text(encoding="utf-8")
    tests = _replace_exact(
        tests,
        '    assert "231_026" in content',
        '    assert "217_886" in content',
        field="runner source assertion",
    )
    tests = _replace_exact(
        tests,
        '''    assert namespace["_EXPECTED_POLICY_OBSERVATIONS"] == 231_026
    assert 231_026 - 231_005 == 3 * ORDER_OBSERVATION_WIDTH''',
        '''    assert namespace["_EXPECTED_POLICY_OBSERVATIONS"] == 217_886
    assert 217_886 - 217_865 == 3 * ORDER_OBSERVATION_WIDTH''',
        field="runner observation assertions",
    )

    _emit("full_research_pipeline.py", pipeline)
    _emit("test_binance_multitimeframe_full_assets.py", tests)
    pytest.fail("A4 patch material emitted for Git Data commit assembly")
