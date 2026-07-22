from __future__ import annotations

from pathlib import Path

SOURCE = Path("tests/rl/test_sequence_policy_core.py")
REQUIRED = (
    "def test_projection_after_selection_matches_legacy_in_float64()",
    "def test_projection_after_selection_preserves_float32_gradient_semantics()",
)
FORBIDDEN = "def test_projection_after_selection_matches_legacy_outputs_and_gradients()"


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    missing = [name for name in REQUIRED if name not in source]
    if missing:
        raise SystemExit(f"missing stable projection contracts: {missing}")
    if FORBIDDEN in source:
        raise SystemExit("backend-sensitive legacy test still exists")
    print("sequence projection test contract is stable")


if __name__ == "__main__":
    main()
