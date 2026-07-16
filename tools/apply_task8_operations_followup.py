from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 8 operations follow-up anchor in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        "tests/examples/test_docker_training_assets.py",
        '    combined = "\n".join((architecture, research, runbook))\n',
        '    combined = "\\n".join((architecture, research, runbook))\n',
    )
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''    resolved = dict(result)
    actual_timesteps = int(resolved.get("actual_timesteps", 0))
    return resolved, {
''',
        '''    resolved = dict(result)
    artifact_path = Path(str(resolved["artifact_path"]))
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    ensemble_payload = json.loads(
        (artifact_path / "ensemble.json").read_text(encoding="utf-8")
    )
    actual_timesteps = int(ensemble_payload["actual_timesteps"])
    resolved["actual_timesteps"] = actual_timesteps
    return resolved, {
''',
    )


if __name__ == "__main__":
    main()
