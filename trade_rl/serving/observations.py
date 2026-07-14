"""Verified observation transformation for serving bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from trade_rl.rl.normalization import ObservationNormalizer, normalizer_from_payload
from trade_rl.serving.bundle import ServingBundle

NORMALIZER_FILE = "normalizer.json"


@dataclass(frozen=True, slots=True)
class ServingObservationPipeline:
    observation_size: int
    normalizer: ObservationNormalizer | None

    @classmethod
    def load(cls, bundle: ServingBundle) -> ServingObservationPipeline:
        manifest = bundle.manifest
        if manifest.normalizer_digest is None:
            return cls(observation_size=manifest.observation_size, normalizer=None)
        path = bundle.root / NORMALIZER_FILE
        if not path.is_file() or path.is_symlink():
            raise ValueError("serving bundle normalizer artifact is missing")
        normalizer = normalizer_from_payload(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if normalizer.digest != manifest.normalizer_digest:
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
