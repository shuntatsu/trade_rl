from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"{label} no longer matches: {source.count(old)} occurrences")
    updated = source.replace(old, new, 1)
    compile(updated, str(path), "exec")
    path.write_text(updated, encoding="utf-8")


serving = Path("trade_rl/integrations/sb3_serving.py")
replace_once(
    serving,
    "from trade_rl.data.market import MarketDataset\n",
    """from trade_rl.data.market import MarketDataset
from trade_rl.integrations.sb3_ensemble import predict_deterministic_mean_action
""",
    "serving helper import",
)
replace_once(
    serving,
    """def _validated_action(
    raw: object, *, action_size: int, member_index: int
) -> np.ndarray:
    action = np.asarray(raw, dtype=np.float32).reshape(-1)
    if action.shape != (action_size,):
        raise ValueError(f"SB3 ensemble member {member_index} action shape mismatch")
    if not np.isfinite(action).all():
        raise ValueError(f"SB3 ensemble member {member_index} action must be finite")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError(f"SB3 ensemble member {member_index} action violates bounds")
    return action


def _mean_action(actions: list[np.ndarray]) -> np.ndarray:
    averaged = np.mean(np.stack(actions, axis=0), axis=0, dtype=np.float64)
    if not np.isfinite(averaged).all():
        raise ValueError("SB3 ensemble mean action must be finite")
    return np.asarray(averaged, dtype=np.float32)


""",
    "",
    "serving duplicate aggregation helpers",
)
replace_once(
    serving,
    """        actions: list[np.ndarray] = []
        for member_index, model in enumerate(self.models):
            try:
                raw, _ = model.predict(vector, deterministic=True)
            except Exception as error:
                raise ValueError(
                    f"SB3 ensemble member {member_index} prediction failed"
                ) from error
            actions.append(
                _validated_action(
                    raw, action_size=self.action_size, member_index=member_index
                )
            )
        return _mean_action(actions)
""",
    """        return predict_deterministic_mean_action(
            self.models,
            vector,
            action_size=self.action_size,
            context="SB3 ensemble",
        )
""",
    "flat Serving aggregation",
)
replace_once(
    serving,
    """        actions: list[np.ndarray] = []
        for member_index, model in enumerate(self.models):
            try:
                raw, _ = model.predict(structured, deterministic=True)
            except Exception as error:
                raise ValueError(
                    f"SB3 ensemble member {member_index} prediction failed"
                ) from error
            actions.append(
                _validated_action(
                    raw, action_size=self.action_size, member_index=member_index
                )
            )
        return _mean_action(actions)
""",
    """        return predict_deterministic_mean_action(
            self.models,
            structured,
            action_size=self.action_size,
            context="SB3 ensemble",
        )
""",
    "structured Serving aggregation",
)

walk_forward = Path("trade_rl/workflows/_market_walk_forward_core.py")
replace_once(
    walk_forward,
    "from trade_rl.integrations.checkpoints import StableBaselines3CheckpointLoader\n",
    """from trade_rl.integrations.checkpoints import StableBaselines3CheckpointLoader
from trade_rl.integrations.sb3_ensemble import predict_deterministic_mean_action
""",
    "walk-forward helper import",
)
replace_once(
    walk_forward,
    """        actions: list[np.ndarray] = []
        for index, model in enumerate(self.models):
            raw, _ = model.predict(observation, deterministic=True)
            action = np.asarray(raw, dtype=np.float32).reshape(-1)
            if not np.isfinite(action).all():
                raise ValueError(
                    f"ensemble member {index} returned a non-finite action"
                )
            if np.any(action < -1.0) or np.any(action > 1.0):
                raise ValueError(
                    f"ensemble member {index} returned an out-of-range action"
                )
            actions.append(action)
        shapes = {item.shape for item in actions}
        if len(shapes) != 1:
            raise ValueError("ensemble member action shapes disagree")
        mean = np.mean(np.stack(actions, axis=0), axis=0, dtype=np.float64)
        return np.asarray(mean, dtype=np.float32), None
""",
    """        action = predict_deterministic_mean_action(
            self.models,
            observation,
            context="deployable ensemble",
        )
        return action, None
""",
    "walk-forward aggregation",
)
