from __future__ import annotations

import sys
from pathlib import Path

import apply_task3_shared_actor as base

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 3 type anchor in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def add_tests() -> None:
    base.add_tests()


def add_implementation() -> None:
    base.add_implementation()
    replace_once(
        "trade_rl/rl/policies.py",
        "        self.mlp_extractor = SharedAssetActorCriticExtractor(\n",
        "        self.mlp_extractor = SharedAssetActorCriticExtractor(  # type: ignore[assignment]\n",
    )
    replace_once(
        "trade_rl/rl/policies.py",
        "        self.optimizer = self.optimizer_class(\n",
        "        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task3_shared_actor_fix.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
