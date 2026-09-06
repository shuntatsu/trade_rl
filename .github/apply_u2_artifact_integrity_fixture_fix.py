from pathlib import Path


path = Path("tests/workflows/test_universal_trade_rl_u2_selection_final.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, *, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, got {count}")
    text = text.replace(old, new, 1)


replace_once(
    '''from trade_rl.artifacts.hashing import content_digest
''',
    '''from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_trade_rl_u2_time_partition import U2_DECISION_STEP_NS
''',
    label="U2 decision step import",
)

replace_once(
    '''    checkpoints = dict(checkpoint_closure.checkpoint_digests)
    symbols = ("DEV_A", "DEV_B")
    pairs = tuple(
''',
    '''    checkpoints = dict(checkpoint_closure.checkpoint_digests)
    symbols = ("DEV_A", "DEV_B")
    d1_timestamps = tuple(
        1_000_000_000 + index * U2_DECISION_STEP_NS for index in range(4)
    )
    d2_timestamps = tuple(
        1_000_000_000 + (10 + index) * U2_DECISION_STEP_NS for index in range(4)
    )
    pairs = tuple(
''',
    label="paired timestamp definitions",
)

replace_once(
    '''        for window, cell, timestamps in (
            ("development_future_1", "D1", (100, 200, 300, 400)),
            ("development_future_2", "D2", (500, 600, 700, 800)),
        )
''',
    '''        for window, cell, timestamps in (
            ("development_future_1", "D1", d1_timestamps),
            ("development_future_2", "D2", d2_timestamps),
        )
''',
    label="paired timestamp fixture",
)

path.write_text(text, encoding="utf-8")
