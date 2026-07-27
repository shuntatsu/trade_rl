from __future__ import annotations

import ast
from pathlib import Path

_SEQUENCE_POLICY_TEST = Path("tests/rl/test_sequence_policy_core.py")
_REQUIRED_TESTS = {
    "test_projection_after_selection_matches_legacy_in_float64",
    "test_projection_after_selection_preserves_float32_gradient_semantics",
}
_FORBIDDEN_TEST = "test_projection_after_selection_matches_legacy_outputs_and_gradients"


def test_sequence_projection_equivalence_uses_stable_contracts() -> None:
    tree = ast.parse(_SEQUENCE_POLICY_TEST.read_text(encoding="utf-8"))
    test_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert _REQUIRED_TESTS <= test_names
    assert _FORBIDDEN_TEST not in test_names
