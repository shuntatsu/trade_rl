from pathlib import Path

path = Path("tests/learning/test_oracle_bellman_torch.py")
text = path.read_text(encoding="utf-8")
old = '''def test_cuda_solver_contains_real_reduce_overhead_compile_path() -> None:
    import inspect

    source = inspect.getsource(solve_torch_cuda_oracle_batch)
    assert "torch.compile(" in source
    assert 'mode="reduce-overhead"' in source
'''
new = '''def test_cuda_solver_contains_real_reduce_overhead_compile_path() -> None:
    import inspect

    from trade_rl.learning.oracle_bellman_torch import _prepare_compiled_core

    source = inspect.getsource(_prepare_compiled_core)
    assert "torch.compile(" in source
    assert 'mode="reduce-overhead"' in source
'''
if text.count(old) != 1:
    raise SystemExit("Task 11 compile-path test anchor changed")
text = text.replace(old, new)
compile_test = '''


def test_compile_setup_failure_uses_eager_mode(monkeypatch) -> None:
    from trade_rl.learning.oracle_bellman_torch import _prepare_compiled_core

    def fail_compile(*args, **kwargs):
        raise RuntimeError("torch.compile is unavailable")

    monkeypatch.setattr("torch.compile", fail_compile)
    compiled, reason = _prepare_compiled_core(
        OracleSolverConfig(
            selection="cuda",
            compile_mode="reduce_overhead",
            compile_chunk_size=8,
        )
    )

    assert compiled is None
    assert reason == "compile_setup_failed:RuntimeError"
'''
if "def test_compile_setup_failure_uses_eager_mode" not in text:
    text += compile_test
path.write_text(text, encoding="utf-8")

path = Path("tests/learning/test_oracle_solver.py")
text = path.read_text(encoding="utf-8")
aggregation_test = '''


def test_orchestrator_aggregates_variable_cuda_runtime_provenance(monkeypatch) -> None:
    from dataclasses import replace

    import trade_rl.learning.oracle_solver as oracle_solver_module
    from trade_rl.learning.oracle_bellman_contracts import (
        OracleSolveResult,
        OracleSolverProvenance,
    )

    market = _market()
    teacher = OracleTeacherConfig(execution_cost=ExecutionCostConfig.zero())
    calls = 0

    def fake_torch_backend(*, tape, states, episode_inputs, parameters, solver_config):
        nonlocal calls
        calls += 1
        numpy_result = solve_numpy_oracle_batch(
            tape=tape,
            states=states,
            episode_inputs=episode_inputs,
            parameters=parameters,
            solver_config=replace(
                solver_config,
                selection="numpy",
                compile_mode="disabled",
            ),
        )
        fallback = "compile_failed:Unsupported" if calls == 2 else None
        return OracleSolveResult(
            targets=numpy_result.targets,
            final_scores=numpy_result.final_scores,
            provenance=OracleSolverProvenance(
                backend="torch_cuda",
                solver_config_digest=solver_config.digest,
                market_tape_digest=tape.digest,
                numeric_dtype="float64",
                tie_tolerance=solver_config.tie_tolerance,
                episode_batch_size=solver_config.episode_batch_size,
                target_state_block_size=2 if calls == 1 else 1,
                compile_mode="reduce_overhead" if calls == 1 else "disabled",
                compile_chunk_size=solver_config.compile_chunk_size,
                fallback_reason=fallback,
                oom_retry_performed=calls == 3,
                solver_wall_time_seconds=float(calls),
                peak_device_memory_bytes=100 * calls,
                torch_version="test",
                cuda_version="test",
                device_name="test",
                compute_capability="0.0",
            ),
        )

    monkeypatch.setattr(
        oracle_solver_module,
        "solve_torch_cuda_oracle_batch",
        fake_torch_backend,
    )

    result = solve_oracle_episodes(
        market,
        states=_portfolio_states(market, teacher),
        episode_inputs=_inputs(),
        parameters=teacher.bellman_parameters,
        solver_config=OracleSolverConfig(
            selection="cuda",
            episode_batch_size=1,
            target_state_block_size=2,
            compile_mode="reduce_overhead",
            compile_chunk_size=8,
        ),
    )

    assert calls == 3
    assert result.provenance.target_state_block_size == 1
    assert result.provenance.compile_mode == "disabled"
    assert result.provenance.fallback_reason == "compile_failed:Unsupported"
    assert result.provenance.oom_retry_performed is True
    assert result.provenance.solver_wall_time_seconds == 6.0
    assert result.provenance.peak_device_memory_bytes == 300
'''
if "def test_orchestrator_aggregates_variable_cuda_runtime_provenance" not in text:
    text += aggregation_test
path.write_text(text, encoding="utf-8")
