from pathlib import Path

path = Path("trade_rl/workflows/universal_full_research_training.py")
text = path.read_text()
old = '''    payload.pop("algorithm", None)\n    if remove_gamma:\n        payload.pop("gamma", None)\n    for key in tuple(payload):\n        if key.startswith("cost_") or key.startswith("lagrangian_"):\n            payload.pop(key)\n    return payload\n'''
new = '''    payload.pop("algorithm", None)\n    payload.pop("cost_critic", None)\n    payload.pop("lagrangian", None)\n    if remove_gamma:\n        payload.pop("gamma", None)\n    for key in tuple(payload):\n        if key.startswith("cost_") or key.startswith("lagrangian_"):\n            payload.pop(key)\n    return payload\n'''
if old not in text:
    raise SystemExit("U6 comparison strip anchor not found")
text = text.replace(old, new, 1)
path.write_text(text)
compile(text, str(path), "exec")
