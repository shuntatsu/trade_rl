from pathlib import Path

path = Path("tests/examples/test_docker_training_assets.py")
source = path.read_text(encoding="utf-8")
old = '    assert "gpu_sequence_target_oracle_bc_training_smoke_v6" in smoke\n'
new = '    assert "gpu_sequence_target_oracle_bc_training_smoke_v7" in smoke\n'
if source.count(old) != 1:
    raise SystemExit("H5 Docker asset schema seam changed")
path.write_text(source.replace(old, new), encoding="utf-8")
