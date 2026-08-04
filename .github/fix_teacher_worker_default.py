from pathlib import Path

path = Path("trade_rl/integrations/sb3_training.py")
text = path.read_text(encoding="utf-8")
old = '''    raw = os.environ.get("TRADE_RL_TEACHER_WORKERS", "").strip()
    try:
        configured = n_envs if not raw else int(raw)
    except ValueError as error:
'''
new = '''    raw = os.environ.get("TRADE_RL_TEACHER_WORKERS", "").strip()
    try:
        if raw:
            configured = int(raw)
        else:
            configured = n_envs if solver_config is None else 1
    except ValueError as error:
'''
if text.count(old) != 1:
    raise SystemExit("teacher worker default anchor changed")
path.write_text(text.replace(old, new), encoding="utf-8")
