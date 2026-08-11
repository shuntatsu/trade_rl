from pathlib import Path

path = Path("trade_rl/integrations/sb3_runtime.py")
text = path.read_text()
if "def build_episode_oracle_batch_for_environment(" in text:
    raise SystemExit(0)
text = text.replace(
    "from trade_rl.learning.episode_oracle_bc import oracle_episode_sampling_config\n",
    "from trade_rl.learning.episode_oracle_bc import (\n"
    "    oracle_episode_sampling_config,\n"
    "    resolve_episode_initial_weights,\n"
    ")\n",
    1,
)
text = text.replace(
    "from trade_rl.learning.episode_oracle_teacher import OracleEpisodeSamplingConfig\n",
    "from trade_rl.learning.episode_oracle_teacher import (\n"
    "    EpisodeOracleBatch,\n"
    "    OracleEpisodeSamplingConfig,\n"
    "    build_episode_oracle_batch,\n"
    ")\n",
    1,
)
marker = "\n\ndef _oracle_solver_config() -> OracleSolverConfig:\n"
if marker not in text:
    raise SystemExit("sb3_runtime solver marker not found")
block = r'''


def build_episode_oracle_batch_for_environment(
    environment: Any,
    *,
    train_range: tuple[int, int],
    seed: int,
    n_envs: int,
) -> EpisodeOracleBatch:
    """Build Oracle episode evidence inside one explicit train-only index range."""

    start, stop = train_range
    dataset = getattr(environment, "dataset", None)
    n_bars = getattr(dataset, "n_bars", None)
    minimum_start_index = getattr(environment, "minimum_start_index", None)
    if (
        isinstance(n_bars, bool)
        or not isinstance(n_bars, int)
        or n_bars <= 0
        or isinstance(minimum_start_index, bool)
        or not isinstance(minimum_start_index, int)
        or minimum_start_index < 0
    ):
        raise ValueError("Oracle environment does not expose a valid trainable dataset")
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < minimum_start_index
        or stop <= start
        or stop > n_bars
    ):
        raise ValueError("Oracle train_range is outside the environment trainable range")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("Oracle seed must fit in an unsigned 32-bit integer")
    if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError("Oracle n_envs must be a positive integer")

    teacher_config = oracle_teacher_config_for_environment(environment)
    sampling_config = _oracle_episode_sampling_config(
        environment,
        train_range=(start, stop),
        seed=seed,
    )
    solver_config = _oracle_solver_config()
    workers = _teacher_worker_count(n_envs, solver_config=solver_config)
    return build_episode_oracle_batch(
        dataset,
        minimum_start_index=start,
        maximum_stop_index=stop,
        sampling_config=sampling_config,
        teacher_config=teacher_config,
        initial_weight_provider=lambda mode, index: resolve_episode_initial_weights(
            environment,
            mode,
            index,
        ),
        max_workers=workers,
        solver_config=solver_config,
        accelerator_backend=_oracle_accelerator_backend(solver_config),
    )
'''
text = text.replace(marker, block + marker, 1)
path.write_text(text)
compile(text, str(path), "exec")
