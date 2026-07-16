from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "trade_rl/integrations/compact_rollout_buffer.py"
    text = path.read_text(encoding="utf-8")
    old = '''    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> DictRolloutBufferSamples:
'''
    new = '''    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> Any:
'''
    if old not in text:
        raise RuntimeError("missing Task 5 return annotation anchor")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
