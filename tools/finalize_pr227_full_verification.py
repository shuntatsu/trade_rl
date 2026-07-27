from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}")
    updated = text.replace(old, new, 1)
    ast.parse(updated, filename=path)
    target.write_text(updated, encoding="utf-8")


def _normalize_gpu_smoke() -> None:
    _replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''            "observation_encoder": "invalid_legacy_combination"
            if (True) and (False)
            else "hierarchical_sequence_v2"
            if (True)
            else "asset_set"
            if (False)
            else "flat_mlp",
''',
        '''            "observation_encoder": "hierarchical_sequence_v2",
''',
    )


def _enable_sequence_bc_to_ppo_audit() -> None:
    path = "tools/run_training_capability_audit.py"
    _replace_once(
        path,
        '''        sequence_dropout=0.0,
        max_policy_parameters=2_000_000,
        device="cpu",
''',
        '''        sequence_dropout=0.0,
        max_policy_parameters=2_000_000,
        behavior_cloning_epochs=1,
        behavior_cloning_batch_size=16,
        behavior_cloning_validation_fraction=0.1,
        device="cpu",
''',
    )
    _replace_once(
        path,
        '''    if architecture["architecture"].get("encoder") != "MultiTimeframeTCNEncoder":
        raise RuntimeError("structured sequence encoder was not instantiated")
    return {
''',
        '''    if architecture["architecture"].get("encoder") != "MultiTimeframeTCNEncoder":
        raise RuntimeError("structured sequence encoder was not instantiated")
    behavior_cloning_path = output.parent / "behavior-cloning.json"
    if not behavior_cloning_path.is_file():
        raise RuntimeError("structured sequence behavior cloning evidence is missing")
    behavior_cloning = json.loads(behavior_cloning_path.read_text(encoding="utf-8"))
    for field in ("initial_mse", "final_mse"):
        if not np.isfinite(float(behavior_cloning[field])):
            raise RuntimeError(
                f"structured sequence behavior cloning {field} is invalid"
            )
    return {
''',
    )
    _replace_once(
        path,
        '''        "actual_timesteps": result.actual_timesteps,
        "observation_schema": result.observation_schema,
''',
        '''        "actual_timesteps": result.actual_timesteps,
        "behavior_cloning": {
            "final_mse": behavior_cloning["final_mse"],
            "initial_mse": behavior_cloning["initial_mse"],
            "sample_count": behavior_cloning["sample_count"],
            "status": "pass",
        },
        "observation_schema": result.observation_schema,
''',
    )


def main() -> None:
    _normalize_gpu_smoke()
    _enable_sequence_bc_to_ppo_audit()


if __name__ == "__main__":
    main()
