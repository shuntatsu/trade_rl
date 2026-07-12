import numpy as np

from mars_lite.trading.htf_constraint import HTFProposalConstraint


def test_directional_constraint_zeroes_only_opposing_positions() -> None:
    constraint = HTFProposalConstraint(threshold=0.3, neutral_scale=0.5)
    proposal = np.array([0.4, -0.4, 0.2, -0.2])
    htf = np.array([0.8, 0.8, -0.8, -0.8])

    result = constraint.apply(proposal, htf)

    np.testing.assert_allclose(result.weights, np.array([0.4, 0.0, 0.0, -0.2]))
    assert result.zeroed_fraction == 0.5


def test_neutral_constraint_is_proposal_idempotent() -> None:
    constraint = HTFProposalConstraint(threshold=0.3, neutral_scale=0.5)
    desired = np.array([0.10, -0.10])
    htf = np.array([0.0, 0.0])

    first = constraint.apply(desired, htf)
    second = constraint.apply(desired, htf)

    np.testing.assert_allclose(first.weights, np.array([0.05, -0.05]))
    np.testing.assert_allclose(second.weights, first.weights)
    assert first.neutral_scaled_fraction == 1.0


def test_constraint_never_increases_gross() -> None:
    constraint = HTFProposalConstraint()
    proposal = np.array([0.6, -0.4])

    result = constraint.apply(proposal, np.array([0.0, 1.0]))

    assert np.abs(result.weights).sum() <= np.abs(proposal).sum() + 1e-12
