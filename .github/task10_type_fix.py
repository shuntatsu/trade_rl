from pathlib import Path

path = Path("trade_rl/operations/oracle_teacher_benchmark.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "        for contract in contracts:\n            result = solve_oracle_episodes(\n",
    "        for contract in contracts:\n            serial_result = solve_oracle_episodes(\n",
)
text = text.replace(
    "            targets.extend(result.targets)\n"
    "            scores.extend(result.final_scores.tolist())\n"
    "            provenances.append(result.provenance)\n",
    "            targets.extend(serial_result.targets)\n"
    "            scores.extend(serial_result.final_scores.tolist())\n"
    "            provenances.append(serial_result.provenance)\n",
)
text = text.replace(
    "    result: OracleSolveResult = solve_oracle_episodes(\n",
    "    batched_result: OracleSolveResult = solve_oracle_episodes(\n",
)
text = text.replace(
    "        output_digest=_target_payload_digest(result.targets, result.final_scores),\n"
    "        solver_seconds=result.provenance.solver_wall_time_seconds,\n"
    "        peak_device_allocated_bytes=result.provenance.peak_device_memory_bytes,\n"
    "        peak_device_reserved_bytes=None,\n"
    "        provenance=result.provenance,\n",
    "        output_digest=_target_payload_digest(\n"
    "            batched_result.targets, batched_result.final_scores\n"
    "        ),\n"
    "        solver_seconds=batched_result.provenance.solver_wall_time_seconds,\n"
    "        peak_device_allocated_bytes=(\n"
    "            batched_result.provenance.peak_device_memory_bytes\n"
    "        ),\n"
    "        peak_device_reserved_bytes=None,\n"
    "        provenance=batched_result.provenance,\n",
)
path.write_text(text, encoding="utf-8")
