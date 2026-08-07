from __future__ import annotations

from pathlib import Path


RUN_FILE_OLD = '''    walk_forward = json.loads(
        (root / "examples/binance-multitimeframe/walk-forward-full.json").read_text(
            encoding="utf-8"
        )
    )["candidates"][0]["run"]["training"]
'''
RUN_FILE_NEW = '''    walk_forward_config = json.loads(
        (root / "examples/binance-multitimeframe/walk-forward-full.json").read_text(
            encoding="utf-8"
        )
    )
    run_file = walk_forward_config["candidates"][0]["run_file"]
    walk_forward = json.loads(
        (root / "examples/binance-multitimeframe" / run_file).read_text(
            encoding="utf-8"
        )
    )["training"]
'''

CHECKPOINT_OLD = "        checkpoint_interval_steps=8,\n"
CHECKPOINT_NEW = "        checkpoint_interval_steps=4,\n"

DOC_OLD = (
    "Action schemaは`portfolio_action_v3`です。Maintained Target-weight modeでは"
    "Action shapeは`(1,)`、Action nameは`target_weight:BTCUSDT`です。\n"
)
DOC_NEW = (
    "Action schemaは`portfolio_action_v3`です。Maintained Target-weight modeでは"
    "Action shapeは`(1,)`、Action nameは`target_weight:BTCUSDT`です。"
    "階層Actorはchange intensityとtarget exposureを分離し、最終Actionを合成します。\n"
)


def replace_exact(path: Path, old: str, new: str, *, expected: int = 1) -> None:
    content = path.read_text(encoding="utf-8")
    observed = content.count(old)
    if observed != expected:
        raise RuntimeError(
            f"{path}: expected {expected} exact replacements, observed {observed}"
        )
    path.write_text(content.replace(old, new), encoding="utf-8")


def main() -> None:
    replace_exact(
        Path("tests/integrations/test_parallel_sequence_environments.py"),
        RUN_FILE_OLD,
        RUN_FILE_NEW,
    )
    replace_exact(
        Path("tests/integrations/test_sequence_runtime_acceleration.py"),
        RUN_FILE_OLD,
        RUN_FILE_NEW,
    )
    replace_exact(
        Path("tests/integrations/test_single_symbol_training_smoke.py"),
        CHECKPOINT_OLD,
        CHECKPOINT_NEW,
    )
    replace_exact(Path("docs/ARCHITECTURE.md"), DOC_OLD, DOC_NEW)


if __name__ == "__main__":
    main()
