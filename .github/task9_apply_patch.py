from __future__ import annotations

import argparse
import base64
import hashlib
import subprocess
import zlib
from pathlib import Path

PATCHES = {
    "tests": (
        Path(".github/task9_tests.patch.b85"),
        "bbf2366dc21b4b4937f8447b1cd249f425e4e87a7f9bfddc76158fe0d9351940",
    ),
    "implementation": (
        Path(".github/task9_implementation.patch.b85"),
        "3c142d26f1d5e513041b13d07bece371c77100429cbb7cee2190fbb9610d6d20",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=PATCHES)
    args = parser.parse_args()
    payload_path, expected = PATCHES[args.kind]
    payload = zlib.decompress(base64.b85decode(payload_path.read_text().strip()))
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise SystemExit(f"Task 9 {args.kind} patch digest mismatch: {actual}")
    patch_path = Path(f"/tmp/task9-{args.kind}.patch")
    patch_path.write_bytes(payload)
    subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        check=True,
    )
    subprocess.run(["git", "apply", str(patch_path)], check=True)


if __name__ == "__main__":
    main()
