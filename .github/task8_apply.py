from pathlib import Path
import base64
import zlib

PAYLOADS = {
    ".github/task8_payload_bellman.txt": "trade_rl/learning/oracle_bellman_torch.py",
    ".github/task8_payload_solver.txt": "trade_rl/learning/oracle_solver.py",
    ".github/task8_payload_test_bellman.txt": "tests/learning/test_oracle_bellman_torch.py",
    ".github/task8_payload_test_solver.txt": "tests/learning/test_oracle_solver.py",
}

for payload_path, target_path in PAYLOADS.items():
    encoded = Path(payload_path).read_text(encoding="utf-8").strip()
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(zlib.decompress(base64.b85decode(encoded)))
