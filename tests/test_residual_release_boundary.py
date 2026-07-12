import pytest

from mars_lite.pipeline.residual_release_boundary import validate_residual_invocation


def test_direct_mode_is_unchanged() -> None:
    validate_residual_invocation(action_mode="direct", no_register=False)


def test_residual_research_requires_no_register() -> None:
    validate_residual_invocation(action_mode="baseline-residual", no_register=True)


def test_residual_registration_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="sealed multi-fold"):
        validate_residual_invocation(action_mode="baseline-residual", no_register=False)
