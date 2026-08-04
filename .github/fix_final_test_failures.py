from pathlib import Path


path = Path("trade_rl/rl/lagrangian_probe.py")
text = path.read_text(encoding="utf-8")
old = '''    worker_count = min(max_workers, episode_count)
    if "fork" not in mp.get_all_start_methods():
        return tuple(
'''
new = '''    worker_count = min(max_workers, episode_count)
    if worker_count == 1 or "fork" not in mp.get_all_start_methods():
        return tuple(
'''
if text.count(old) != 1:
    raise SystemExit("Lagrangian probe serial-path anchor changed")
text = text.replace(old, new)
old = '''        # A full-market environment dirties enough inherited Python/NumPy pages
        # that running episodes in the parent or reusing a child steadily grows
        # RSS. Even with one worker, isolate and recycle every episode so its
        # allocator/COW state is returned to the OS.
        with context.Pool(processes=worker_count, maxtasksperchild=1) as pool:
'''
new = '''        # Parallel full-market probes use one task per child so inherited
        # Python/NumPy allocator and COW state is returned to the OS after each
        # episode. A single worker stays in-process to preserve deterministic
        # environment lifecycle and instrumentation semantics.
        with context.Pool(processes=worker_count, maxtasksperchild=1) as pool:
'''
if text.count(old) != 1:
    raise SystemExit("Lagrangian probe isolation comment anchor changed")
path.write_text(text.replace(old, new), encoding="utf-8")


path = Path("trade_rl/learning/oracle_teacher.py")
text = path.read_text(encoding="utf-8")
old = '''    return project_portfolio_targets_numpy(
        weights[None, :, :, :],
        portfolio_value=values[None, :],
        market_notional=market_notional,
        config=config,
    )[0]
'''
new = '''    liquidity = np.asarray(market_notional, dtype=np.float64)
    if liquidity.ndim == 1:
        liquidity = liquidity[None, :]
    return project_portfolio_targets_numpy(
        weights[None, :, :, :],
        portfolio_value=values[None, :],
        market_notional=liquidity,
        config=config,
    )[0]
'''
if text.count(old) != 1:
    raise SystemExit("Oracle portfolio projection wrapper anchor changed")
path.write_text(text.replace(old, new), encoding="utf-8")


path = Path("trade_rl/integrations/sb3_training.py")
text = path.read_text(encoding="utf-8")
old = '''    raw = os.environ.get("TRADE_RL_TEACHER_WORKERS", "1").strip()
    try:
        configured = int(raw)
    except ValueError as error:
        raise ValueError("TRADE_RL_TEACHER_WORKERS must be an integer") from error
'''
new = '''    raw = os.environ.get("TRADE_RL_TEACHER_WORKERS", "").strip()
    try:
        configured = n_envs if not raw else int(raw)
    except ValueError as error:
        raise ValueError("TRADE_RL_TEACHER_WORKERS must be an integer") from error
'''
if text.count(old) != 1:
    raise SystemExit("teacher worker default anchor changed")
path.write_text(text.replace(old, new), encoding="utf-8")


path = Path(
    "examples/binance-multitimeframe/training-target-weight-constrained-growth.json"
)
text = path.read_text(encoding="utf-8")
replacements = {
    '"behavior_cloning_gate_prediction_threshold": 0.55': (
        '"behavior_cloning_gate_prediction_threshold": 0.49'
    ),
    '"behavior_cloning_gate_loss_weight": 1.0': (
        '"behavior_cloning_gate_loss_weight": 1.25'
    ),
    '"behavior_cloning_max_positive_class_weight": 1.2': (
        '"behavior_cloning_max_positive_class_weight": 1.4'
    ),
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"constrained growth profile anchor changed: {old}")
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
