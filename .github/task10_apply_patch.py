from __future__ import annotations

import argparse
import base64
import hashlib
import subprocess
import zlib
from pathlib import Path

PATCHES = {
    "tests": (
        Path(".github/task10_tests.patch.b85"),
        "c038826988a6374cdf2096cf6540f5594218d7c02ed2093bbe2ebd1b6072ba6d",
    ),
    "implementation": (
        Path(".github/task10_impl.patch.b85"),
        "047a2e70cf4ca377f8a8e326c664e282c8aa6e5eabe079fb8ddfb740c80a884d",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=PATCHES)
    arguments = parser.parse_args()
    payload_path, expected_digest = PATCHES[arguments.kind]
    payload = zlib.decompress(base64.b85decode(payload_path.read_text().strip()))
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != expected_digest:
        raise SystemExit(
            f"Task 10 {arguments.kind} patch digest mismatch: {actual_digest}"
        )
    patch_path = Path(f"/tmp/task10-{arguments.kind}.patch")
    patch_path.write_bytes(payload)
    subprocess.run(["git", "apply", "--check", str(patch_path)], check=True)
    subprocess.run(["git", "apply", str(patch_path)], check=True)


if __name__ == "__main__":
    main()
