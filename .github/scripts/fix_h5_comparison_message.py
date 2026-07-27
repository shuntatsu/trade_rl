from pathlib import Path

path = Path("examples/binance-multitimeframe/compare_gpu_training_smoke.py")
source = path.read_text(encoding="utf-8")
old = '        raise ValueError("training artifact and smoke actual timesteps differ")\n'
new = '        raise ValueError("GPU comparison workload differs between training artifact and smoke")\n'
if source.count(old) != 1:
    raise SystemExit("H5 workload diagnostic seam changed")
path.write_text(source.replace(old, new), encoding="utf-8")
