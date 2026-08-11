from __future__ import annotations

import copy
import json
from pathlib import Path

# Allow Discounted Lagrangian to encode the gamma difference through a real-time half-life.
module_path = Path("trade_rl/workflows/universal_full_research_training.py")
module = module_path.read_text(encoding="utf-8")
old = '''    if remove_gamma:\n        payload.pop("gamma", None)\n    for key in tuple(payload):\n'''
new = '''    if remove_gamma:\n        payload.pop("gamma", None)\n        payload.pop("discount_half_life_hours", None)\n    for key in tuple(payload):\n'''
if old not in module:
    raise SystemExit("U6 fixed-condition gamma strip anchor not found")
module = module.replace(old, new, 1)
old = '''def _strip_gamma(config: ResidualTrainingConfig) -> dict[str, object]:\n    payload = _training_payload(config)\n    payload.pop("gamma", None)\n    return payload\n'''
new = '''def _strip_gamma(config: ResidualTrainingConfig) -> dict[str, object]:\n    payload = _training_payload(config)\n    payload.pop("gamma", None)\n    payload.pop("discount_half_life_hours", None)\n    return payload\n'''
if old not in module:
    raise SystemExit("U6 discounted gamma strip anchor not found")
module = module.replace(old, new, 1)
module_path.write_text(module, encoding="utf-8")
compile(module, str(module_path), "exec")

root = Path("examples/binance-multitimeframe")
source_paths = {
    "universal-u6-ppo.json": root / "training-target-weight-growth-ppo.json",
    "universal-u6-lagrangian.json": root / "training-target-weight-constrained-growth.json",
    "universal-u6-discounted.json": root / "training-target-weight-constrained-growth-discounted.json",
}
common = json.loads((root / "training-target-weight-growth-ppo.json").read_text(encoding="utf-8"))
for output_name, source_path in source_paths.items():
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(common)
    training = copy.deepcopy(source["training"])
    training["behavior_cloning_seed"] = 17
    training["behavior_cloning_critic_warm_start_steps"] = 512
    training["behavior_cloning_joint_warm_start_steps"] = 128
    training["behavior_cloning_critic_warm_start_learning_rate"] = 0.0003
    training["behavior_cloning_joint_warm_start_actor_lr_scale"] = 0.1
    config["training"] = training
    (root / output_name).write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

start_path = Path("START.md")
start = start_path.read_text(encoding="utf-8")
start = start.replace(
    "examples/binance-multitimeframe/training-target-weight-constrained-growth-discounted.json",
    "examples/binance-multitimeframe/universal-u6-discounted.json",
)
start = start.replace(
    "examples/binance-multitimeframe/training-target-weight-constrained-growth.json",
    "examples/binance-multitimeframe/universal-u6-lagrangian.json",
)
start = start.replace(
    "examples/binance-multitimeframe/training-target-weight-growth-ppo.json",
    "examples/binance-multitimeframe/universal-u6-ppo.json",
)
needle = "3つのauthored `TrainingRunConfig`は、training algorithm/cost-family/gammaの比較上必要な差分を除き、Environment / Risk / Reward / Trend / Action / Executionを同一にしてください。"
replacement = needle + " 上のcanonical U6 configはBC seedを17に固定し、critic-only warm startを512 step、conservative joint warm startを128 step有効化した初期maintained設定です。これらのstep数はソフトウェア契約用の初期値であり、実データ上の最適性を主張しません。"
if needle not in start:
    raise SystemExit("START U6 comparison paragraph anchor not found")
start = start.replace(needle, replacement, 1)
start_path.write_text(start, encoding="utf-8")
