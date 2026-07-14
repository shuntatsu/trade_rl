from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

normalization_path = ROOT / "trade_rl/rl/normalization.py"
normalization = normalization_path.read_text(encoding="utf-8")
old_range = '''        if (self.absolute_train_start is None) != (self.absolute_train_end is None):
            raise ValueError("absolute normalizer range must be specified as a pair")
        if self.absolute_train_start is not None:
            if (
                isinstance(self.absolute_train_start, bool)
                or isinstance(self.absolute_train_end, bool)
                or not isinstance(self.absolute_train_start, int)
                or not isinstance(self.absolute_train_end, int)
                or self.absolute_train_start < 0
                or self.absolute_train_end <= self.absolute_train_start
            ):
                raise ValueError("absolute normalizer training range is invalid")
'''
new_range = '''        absolute_start = self.absolute_train_start
        absolute_end = self.absolute_train_end
        if (absolute_start is None) != (absolute_end is None):
            raise ValueError("absolute normalizer range must be specified as a pair")
        if absolute_start is not None and absolute_end is not None:
            if (
                isinstance(absolute_start, bool)
                or isinstance(absolute_end, bool)
                or not isinstance(absolute_start, int)
                or not isinstance(absolute_end, int)
                or absolute_start < 0
                or absolute_end <= absolute_start
            ):
                raise ValueError("absolute normalizer training range is invalid")
'''
if normalization.count(old_range) != 1:
    raise RuntimeError("normalizer absolute range validation block changed")
normalization_path.write_text(
    normalization.replace(old_range, new_range, 1),
    encoding="utf-8",
)

execution_path = ROOT / "trade_rl/simulation/execution.py"
execution = execution_path.read_text(encoding="utf-8")
old_multiplier = '''        if not np.allclose(dataset.contract_multipliers, 1.0, rtol=0.0, atol=1e-12):
'''
new_multiplier = '''        if not np.allclose(
            dataset.resolved_array("contract_multipliers"),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
'''
if execution.count(old_multiplier) != 1:
    raise RuntimeError("execution multiplier validation block changed")
execution_path.write_text(
    execution.replace(old_multiplier, new_multiplier, 1),
    encoding="utf-8",
)
