from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "trade_rl/serving/normalizer.py"
text = path.read_text(encoding="utf-8")
text = text.replace(
    "from pathlib import Path\n",
    "from pathlib import Path\nfrom typing import cast\n",
    1,
)
text = text.replace(
    '''        mean = np.asarray(raw["mean"], dtype=np.float64)
        scale = np.asarray(raw["scale"], dtype=np.float64)
        passthrough = tuple(int(value) for value in raw["passthrough_indices"])
''',
    '''        mean = np.asarray(cast(list[float], raw["mean"]), dtype=np.float64)
        scale = np.asarray(cast(list[float], raw["scale"]), dtype=np.float64)
        passthrough = tuple(
            int(value) for value in cast(list[int], raw["passthrough_indices"])
        )
''',
    1,
)
text = text.replace(
    '''            train_start=int(raw["train_start"]),
            train_end=int(raw["train_end"]),
            clip=float(raw["clip"]),
            epsilon=float(raw["epsilon"]),
''',
    '''            train_start=cast(int, raw["train_start"]),
            train_end=cast(int, raw["train_end"]),
            clip=float(cast(int | float, raw["clip"])),
            epsilon=float(cast(int | float, raw["epsilon"])),
''',
    1,
)
path.write_text(text, encoding="utf-8")
