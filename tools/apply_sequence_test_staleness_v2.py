"""Update direct encoder tests to pass the explicit quality-staleness plane."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pattern = re.compile(
        r"(?P<indent>[ \t]+)sequences=sequences,\n"
        r"(?P=indent)available=available,\n"
        r"(?P=indent)snapshot="
    )
    for path in (ROOT / "tests").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        replacement = (
            r"\g<indent>sequences=sequences,\n"
            r"\g<indent>available=available,\n"
            r"\g<indent>staleness={\n"
            r"\g<indent>    key: __import__(\"torch\").zeros_like(value, dtype=__import__(\"torch\").float32)\n"
            r"\g<indent>    for key, value in available.items()\n"
            r"\g<indent>},\n"
            r"\g<indent>snapshot="
        )
        updated = pattern.sub(replacement, text)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
