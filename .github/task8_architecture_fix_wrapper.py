from pathlib import Path
import runpy

path = Path(".github/task8_architecture_fix.py")
text = path.read_text(encoding="utf-8")
old = '''path = Path(".importlinter")
text = path.read_text(encoding="utf-8")
anchor = \'\'\'forbidden_modules =
    stable_baselines3
    sb3_contrib
    torch
\'\'\'
replacement = \'\'\'forbidden_modules =
    stable_baselines3
    sb3_contrib
    torch
allow_indirect_imports = True
ignore_imports =
    trade_rl.learning.oracle_bellman_torch -> torch
    trade_rl.learning.oracle_transition_torch -> torch
\'\'\'
if text.count(anchor) != 1:
    raise SystemExit("learning framework contract changed unexpectedly")
path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")
'''
new = '''path = Path(".importlinter")
text = path.read_text(encoding="utf-8")
section = "[importlinter:contract:learning-frameworks]"
next_section = "[importlinter:contract:workflow-frameworks]"
prefix, remainder = text.split(section, maxsplit=1)
block, suffix = remainder.split(next_section, maxsplit=1)
anchor = \'\'\'forbidden_modules =
    stable_baselines3
    sb3_contrib
    torch
\'\'\'
replacement = \'\'\'forbidden_modules =
    stable_baselines3
    sb3_contrib
    torch
allow_indirect_imports = True
ignore_imports =
    trade_rl.learning.oracle_bellman_torch -> torch
    trade_rl.learning.oracle_transition_torch -> torch
\'\'\'
if block.count(anchor) != 1:
    raise SystemExit("learning framework contract changed unexpectedly")
resolved = prefix + section + block.replace(anchor, replacement, 1) + next_section + suffix
path.write_text(resolved, encoding="utf-8")
'''
if text.count(old) != 1:
    raise SystemExit("Task 8 architecture fix source changed unexpectedly")
path.write_text(text.replace(old, new), encoding="utf-8")
runpy.run_path(str(path), run_name="__main__")
