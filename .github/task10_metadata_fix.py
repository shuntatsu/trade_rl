from pathlib import Path

path = Path("trade_rl/operations/oracle_teacher_benchmark.py")
text = path.read_text(encoding="utf-8")
old = '''            "target_state_block_size": config.target_state_block_size,
            "compile_mode": config.compile_mode,
            "compile_chunk_size": config.compile_chunk_size,
'''
new = '''            "requested_target_state_block_size": config.target_state_block_size,
            "actual_target_state_block_size": (
                None if provenance is None else provenance.target_state_block_size
            ),
            "requested_compile_mode": config.compile_mode,
            "actual_compile_mode": (
                None if provenance is None else provenance.compile_mode
            ),
            "compile_chunk_size": config.compile_chunk_size,
            "fallback_reason": (
                None if provenance is None else provenance.fallback_reason
            ),
            "oom_retry_performed": (
                False if provenance is None else provenance.oom_retry_performed
            ),
'''
if text.count(old) != 1:
    raise SystemExit("Task 10 benchmark metadata anchor changed")
path.write_text(text.replace(old, new), encoding="utf-8")

path = Path("tests/operations/test_oracle_teacher_benchmark.py")
text = path.read_text(encoding="utf-8")
old = '''    assert batched.metadata["actual_backend"] == "numpy"
    assert "market-tape construction" in str(batched.metadata["total_wall_scope"])
'''
new = '''    assert batched.metadata["actual_backend"] == "numpy"
    assert batched.metadata["requested_compile_mode"] == "disabled"
    assert batched.metadata["actual_compile_mode"] == "disabled"
    assert batched.metadata["fallback_reason"] is None
    assert batched.metadata["oom_retry_performed"] is False
    assert "market-tape construction" in str(batched.metadata["total_wall_scope"])
'''
if text.count(old) != 1:
    raise SystemExit("Task 10 benchmark metadata test anchor changed")
path.write_text(text.replace(old, new), encoding="utf-8")
