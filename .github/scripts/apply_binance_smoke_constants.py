from __future__ import annotations

from pathlib import Path


path = Path("scripts/run_binance_e2e_smoke.py")
text = path.read_text(encoding="utf-8")
marker = '_DATA_BINANCE_COMMAND = ("data", "binance")'
if marker not in text:
    text = text.replace(
        "from typing import Any\n\n\n",
        "from typing import Any\n\n\n"
        '_DATA_BINANCE_COMMAND = ("data", "binance")\n'
        '_TRAIN_RUN_COMMAND = ("train", "run")\n'
        '_WALK_FORWARD_RUN_COMMAND = ("walk-forward", "run")\n\n\n',
        1,
    )
    text = text.replace(
        "    return [\n        \"data\", \"binance\",\n",
        "    return [\n        *_DATA_BINANCE_COMMAND,\n",
        1,
    )
    text = text.replace(
        "        [\n            \"train\", \"run\",\n",
        "        [\n            *_TRAIN_RUN_COMMAND,\n",
        1,
    )
    text = text.replace(
        "        [\n            \"walk-forward\", \"run\",\n",
        "        [\n            *_WALK_FORWARD_RUN_COMMAND,\n",
        1,
    )
path.write_text(text, encoding="utf-8")
