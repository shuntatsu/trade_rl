from __future__ import annotations

from pathlib import Path


path = Path(".github/scripts/apply_h4_parallel_sequence_env.py")
source = path.read_text(encoding="utf-8")
old = '''def _build_training_environment(
    factory: Callable[[], gym.Env[Any, Any]],
    n_envs: int,
    *,
    subprocesses: bool = True,
) -> Any:
    if n_envs == 1:
        return factory()
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    factories = [factory for _ in range(n_envs)]
    if subprocesses:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


'''
new = '''def _build_training_environment(
    factory: Callable[[], Any],
    n_envs: int,
    *,
    subprocesses: bool = True,
) -> Any:
    if n_envs == 1:
        return factory()

    if subprocesses:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        return SubprocVecEnv(
            [factory for _ in range(n_envs)],
            start_method="spawn",
        )
    from stable_baselines3.common.vec_env import DummyVecEnv

    return DummyVecEnv([factory for _ in range(n_envs)])


'''
count = source.count(old)
if count != 1:
    raise SystemExit(f"H4 patch helper seam changed: expected one match, got {count}")
path.write_text(source.replace(old, new), encoding="utf-8")
