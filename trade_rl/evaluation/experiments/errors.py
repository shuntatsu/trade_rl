"""Fail-closed error taxonomy for controlled development experiments."""

from __future__ import annotations


class ControlledExperimentError(Exception):
    """Base error for controlled experiment lifecycle failures."""


class ContractViolationError(ControlledExperimentError, ValueError):
    """An immutable experiment contract is malformed or self-inconsistent."""


class ArtifactIntegrityError(ControlledExperimentError, ValueError):
    """Persisted evidence cannot be trusted as an exact immutable snapshot."""


class InvalidExperimentStateError(ControlledExperimentError, RuntimeError):
    """A lifecycle operation is not valid for the reconstructed Study state."""


class UncontrolledDeltaError(ControlledExperimentError, ValueError):
    """Resolved evidence contains an undeclared or otherwise invalid delta."""


class ExperimentBudgetExceededError(InvalidExperimentStateError):
    """A Study cannot define another Experiment within its frozen budget."""


class StudyFrozenError(InvalidExperimentStateError):
    """A mutation was attempted after the Study freeze artifact exists."""


__all__ = [
    "ArtifactIntegrityError",
    "ContractViolationError",
    "ControlledExperimentError",
    "ExperimentBudgetExceededError",
    "InvalidExperimentStateError",
    "StudyFrozenError",
    "UncontrolledDeltaError",
]
