"""Verified observation transformation for serving bundles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.serving.bundle import ServingBundle

NORMALIZER_FILE = "normalizer.json"


@dataclass(frozen=True, slots=True)
class ServingObservationPipeline:
    observation_size: int
    normalizer: ObservationNormalizer | None

    @classmethod
    def load(cls, bundle: ServingBundle) -> ServingObservationPipeline:
        manifest = bundle.manifest
        raw_normalizer = bundle.normalizer
        if raw_normalizer is not None and not isinstance(
            raw_normalizer, ObservationNormalizer
        ):
            raise ValueError("serving bundle normalizer type is invalid")
        normalizer = raw_normalizer
        if manifest.normalizer_digest is None:
            if normalizer is not None:
                raise ValueError("serving bundle contains an unbound normalizer")
            return cls(observation_size=manifest.observation_size, normalizer=None)
        if normalizer is None or normalizer.digest != manifest.normalizer_digest:
            raise ValueError("serving normalizer digest mismatch")
        if normalizer.size != manifest.observation_size:
            raise ValueError("serving normalizer observation size mismatch")
        if normalizer.observation_schema != manifest.observation_schema:
            raise ValueError("serving normalizer observation schema mismatch")
        if (
            normalizer.dataset_id is not None
            and normalizer.dataset_id != manifest.dataset_id
        ):
            raise ValueError("serving normalizer dataset identity mismatch")
        return cls(observation_size=manifest.observation_size, normalizer=normalizer)

    def transform(self, observation: np.ndarray) -> np.ndarray:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        if vector.shape != (self.observation_size,) or not np.isfinite(vector).all():
            raise ValueError("observation violates the active observation schema")
        if self.normalizer is None:
            return vector.copy()
        return self.normalizer.transform(vector)


__all__ = ["NORMALIZER_FILE", "ServingObservationPipeline"]
