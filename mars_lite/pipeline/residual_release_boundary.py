from __future__ import annotations


def validate_residual_invocation(*, action_mode: str, no_register: bool) -> None:
    """Keep residual runs research-only until sealed multi-fold release wiring exists."""

    if action_mode not in {"direct", "baseline-residual"}:
        raise ValueError(f"unknown action_mode: {action_mode}")
    if action_mode == "baseline-residual" and not no_register:
        raise RuntimeError(
            "baseline-residual registration requires the sealed multi-fold residual "
            "release workflow; run with --no-register for research until that gate is "
            "implemented and evidenced"
        )
