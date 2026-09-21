from __future__ import annotations

import numpy as np
import pytest

from trade_rl.strategies.rl.ppo import PPOIntentStrategy


@pytest.mark.parametrize("feature_name", [[[]], [{"nested": "name"}]])
def test_ppo_adapter_rejects_unhashable_feature_names_as_value_error(
    feature_name: object,
) -> None:
    class Policy:
        def predict(
            self,
            observation: np.ndarray,
            *,
            deterministic: bool = True,
        ) -> tuple[np.ndarray, None]:
            return np.asarray(1), None

    with pytest.raises(ValueError, match="feature_names"):
        PPOIntentStrategy(
            Policy(),
            feature_indices=(0,),
            feature_names=(feature_name,),  # type: ignore[arg-type]
        )
