from pathlib import Path

path = Path("trade_rl/workflows/universal_causal_alpha_fitting.py")
text = path.read_text(encoding="utf-8")
old = '''    if predicted.shape != realized.shape or predicted.size == 0:\n        raise ValueError("causal alpha prediction diagnostics require aligned samples")\n    if not np.isfinite(predicted).all() or not np.isfinite(realized).all():\n        raise ValueError("causal alpha prediction diagnostics require finite samples")\n    predicted_std = float(predicted.std())\n'''
new = '''    if predicted.shape != realized.shape or predicted.size == 0:\n        raise ValueError("causal alpha prediction diagnostics require aligned samples")\n    if not np.isfinite(predicted).all():\n        raise ValueError("causal alpha predictions must be finite")\n    realized_mask = np.isfinite(realized)\n    if not np.any(realized_mask):\n        raise ValueError("causal alpha prediction diagnostics have no realized labels")\n    predicted = predicted[realized_mask]\n    realized = realized[realized_mask]\n    predicted_std = float(predicted.std())\n'''
if text.count(old) != 1:
    raise SystemExit("realized diagnostics patch target drifted")
path.write_text(text.replace(old, new), encoding="utf-8")
