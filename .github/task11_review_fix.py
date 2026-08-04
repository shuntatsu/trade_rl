from pathlib import Path

path = Path("trade_rl/learning/oracle_bellman_torch.py")
text = path.read_text(encoding="utf-8")
anchor = '''def _run_compiled_or_eager(
    *,
    compiled: Callable[[], _ResultT],
    eager: Callable[[], _ResultT],
) -> tuple[_ResultT, CompileMode, str | None]:
'''
helper = '''def _prepare_compiled_core(
    solver_config: OracleSolverConfig,
) -> tuple[Callable[..., TorchBellmanResult] | None, str | None]:
    if solver_config.compile_mode != "reduce_overhead":
        return None, None
    try:
        return (
            torch.compile(
                _solve_torch_oracle_batch_core,
                mode="reduce-overhead",
                fullgraph=False,
            ),
            None,
        )
    except Exception as error:
        if not _is_compile_failure(error):
            raise
        return None, f"compile_setup_failed:{type(error).__name__}"


'''
if "def _prepare_compiled_core(" not in text:
    if text.count(anchor) != 1:
        raise SystemExit("Task 11 compile helper anchor changed")
    text = text.replace(anchor, helper + anchor)
old = '''    compiled_core: Callable[..., TorchBellmanResult] | None = None
    if solver_config.compile_mode == "reduce_overhead":
        compiled_core = torch.compile(
            _solve_torch_oracle_batch_core,
            mode="reduce-overhead",
            fullgraph=False,
        )
'''
new = '''    compiled_core, compile_setup_reason = _prepare_compiled_core(solver_config)
'''
if text.count(old) != 1:
    raise SystemExit("Task 11 direct compile anchor changed")
text = text.replace(old, new)
old = '''        if compiled_core is None:
            return eager(), "disabled", None
'''
new = '''        if compiled_core is None:
            return eager(), "disabled", compile_setup_reason
'''
if text.count(old) != 1:
    raise SystemExit("Task 11 eager setup fallback anchor changed")
path.write_text(text, encoding="utf-8")

path = Path("trade_rl/learning/oracle_solver.py")
text = path.read_text(encoding="utf-8")
old = '''    stable_fields = (
        "backend",
        "solver_config_digest",
        "market_tape_digest",
        "numeric_dtype",
        "tie_tolerance",
        "episode_batch_size",
        "target_state_block_size",
        "compile_mode",
        "compile_chunk_size",
        "fallback_reason",
        "solver_contract",
        "tie_break_contract",
        "torch_version",
        "cuda_version",
        "device_name",
        "compute_capability",
    )
'''
new = '''    stable_fields = (
        "backend",
        "solver_config_digest",
        "market_tape_digest",
        "numeric_dtype",
        "tie_tolerance",
        "episode_batch_size",
        "compile_chunk_size",
        "solver_contract",
        "tie_break_contract",
        "torch_version",
        "cuda_version",
        "device_name",
        "compute_capability",
    )
'''
if text.count(old) != 1:
    raise SystemExit("Task 11 provenance stable-fields anchor changed")
text = text.replace(old, new)
old = '''    device_peaks = [
        value.peak_device_memory_bytes
        for value in provenances
        if value.peak_device_memory_bytes is not None
    ]
    aggregate = replace(
        first,
        oom_retry_performed=any(value.oom_retry_performed for value in provenances),
        solver_wall_time_seconds=sum(wall_times) if wall_times else None,
        peak_host_memory_bytes=max(host_peaks) if host_peaks else None,
        peak_device_memory_bytes=max(device_peaks) if device_peaks else None,
        digest="",
    )
'''
new = '''    device_peaks = [
        value.peak_device_memory_bytes
        for value in provenances
        if value.peak_device_memory_bytes is not None
    ]
    effective_blocks = [
        value.target_state_block_size
        for value in provenances
        if value.target_state_block_size is not None
    ]
    compile_modes = {value.compile_mode for value in provenances}
    fallback_reasons = sorted(
        {
            value.fallback_reason
            for value in provenances
            if value.fallback_reason is not None
        }
    )
    aggregate = replace(
        first,
        target_state_block_size=(
            min(effective_blocks) if effective_blocks else None
        ),
        compile_mode=(
            "reduce_overhead"
            if compile_modes == {"reduce_overhead"}
            else "disabled"
        ),
        fallback_reason=(
            ";".join(fallback_reasons) if fallback_reasons else None
        ),
        oom_retry_performed=any(
            value.oom_retry_performed for value in provenances
        ),
        solver_wall_time_seconds=sum(wall_times) if wall_times else None,
        peak_host_memory_bytes=max(host_peaks) if host_peaks else None,
        peak_device_memory_bytes=max(device_peaks) if device_peaks else None,
        digest="",
    )
'''
if text.count(old) != 1:
    raise SystemExit("Task 11 provenance aggregation anchor changed")
path.write_text(text.replace(old, new), encoding="utf-8")
