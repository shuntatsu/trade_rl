"""Typed failure modes at the Studio API boundary."""

from __future__ import annotations


class StudioError(Exception):
    """Base class for expected Studio failures."""

    code = "studio_error"


class ResourceNotFound(StudioError):
    code = "resource_not_found"


class InvalidStudioRequest(StudioError):
    code = "invalid_request"


class IdentityConflict(StudioError):
    code = "identity_conflict"


class ArtifactInvalid(StudioError):
    code = "artifact_invalid"


class JobOwnershipLost(StudioError):
    code = "job_ownership_lost"


__all__ = [
    "ArtifactInvalid",
    "IdentityConflict",
    "InvalidStudioRequest",
    "JobOwnershipLost",
    "ResourceNotFound",
    "StudioError",
]
