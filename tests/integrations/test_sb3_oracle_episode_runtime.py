from __future__ import annotations

from types import SimpleNamespace


def test_build_episode_oracle_batch_for_environment_binds_explicit_train_range(
    monkeypatch,
) -> None:
    import trade_rl.integrations.sb3_runtime as module
    from trade_rl.integrations.sb3_runtime import (
        build_episode_oracle_batch_for_environment,
    )

    environment = SimpleNamespace(
        dataset=SimpleNamespace(n_bars=100),
        minimum_start_index=5,
    )
    teacher_config = object()
    sampling_config = object()
    solver_config = SimpleNamespace(selection="numpy")
    sentinel = object()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "oracle_teacher_config_for_environment",
        lambda value: teacher_config if value is environment else None,
    )

    def sampling(value, *, train_range, seed):
        assert value is environment
        observed["train_range"] = train_range
        observed["seed"] = seed
        return sampling_config

    monkeypatch.setattr(module, "_oracle_episode_sampling_config", sampling)
    monkeypatch.setattr(module, "_oracle_solver_config", lambda: solver_config)
    monkeypatch.setattr(
        module,
        "_teacher_worker_count",
        lambda n_envs, *, solver_config: n_envs,
    )
    monkeypatch.setattr(module, "_oracle_accelerator_backend", lambda _config: None)
    monkeypatch.setattr(
        module,
        "resolve_episode_initial_weights",
        lambda _environment, _mode, _index: None,
    )

    def build(dataset, **kwargs):
        assert dataset is environment.dataset
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "build_episode_oracle_batch", build)

    result = build_episode_oracle_batch_for_environment(
        environment,
        train_range=(7, 40),
        seed=31,
        n_envs=4,
    )

    assert result is sentinel
    assert observed["train_range"] == (7, 40)
    assert observed["seed"] == 31
    assert observed["minimum_start_index"] == 7
    assert observed["maximum_stop_index"] == 40
    assert observed["sampling_config"] is sampling_config
    assert observed["teacher_config"] is teacher_config
    assert observed["max_workers"] == 4
    assert observed["solver_config"] is solver_config
