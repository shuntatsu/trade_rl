from pathlib import Path

config_paths = (
    Path("examples/binance-multitimeframe/universal-u6-ppo.json"),
    Path("examples/binance-multitimeframe/universal-u6-lagrangian.json"),
    Path("examples/binance-multitimeframe/universal-u6-discounted.json"),
)
old_teacher = '"behavior_cloning_teacher": "oracle"'
new_teacher = '"behavior_cloning_teacher": "causal_alpha_ridge"'
for path in config_paths:
    text = path.read_text(encoding="utf-8")
    if text.count(old_teacher) != 1:
        raise SystemExit(f"canonical teacher token drifted: {path}")
    path.write_text(text.replace(old_teacher, new_teacher), encoding="utf-8")

docs = Path("docs/CONFIGURATION.md")
text = docs.read_text(encoding="utf-8")
old = "`behavior_cloning_epochs > 0`はPPO-familyでだけ有効です。Teacherは`oracle`または`trend_baseline`です。"
new = (
    "`behavior_cloning_epochs > 0`はPPO-familyでだけ有効です。Teacherは`oracle`、"
    "`trend_baseline`、またはtrain-only fitted teacherの`causal_alpha_ridge`です。"
    "Canonical Universal U6は`causal_alpha_ridge`を使用し、`oracle`/`trend_baseline`は"
    "診断・互換経路としてのみ保持します。"
)
if text.count(old) != 1:
    raise SystemExit("configuration teacher documentation target drifted")
text = text.replace(old, new)
old_example = '"behavior_cloning_teacher": "oracle"'
if text.count(old_example) != 1:
    raise SystemExit("configuration BC teacher example target drifted")
text = text.replace(old_example, new_teacher)
docs.write_text(text, encoding="utf-8")
