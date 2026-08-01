#!/usr/bin/env python3
"""Fail closed unless PyTorch can see CUDA, and record the visible device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def write_cuda_preflight_evidence(
    output: Path,
    torch_probe: Any,
) -> dict[str, object]:
    """Validate CUDA visibility and atomically publish device evidence."""

    if not torch_probe.cuda.is_available():
        raise RuntimeError("CUDA is required for maintained full training")

    device_index = 0
    capability = torch_probe.cuda.get_device_capability(device_index)
    properties = torch_probe.cuda.get_device_properties(device_index)
    evidence: dict[str, object] = {
        "capability": [int(capability[0]), int(capability[1])],
        "device_index": device_index,
        "device_name": str(torch_probe.cuda.get_device_name(device_index)),
        "resolved_device": f"cuda:{device_index}",
        "total_memory_bytes": int(properties.total_memory),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/cuda-preflight.json"),
    )
    return parser


def main() -> int:
    import torch

    args = build_parser().parse_args()
    evidence = write_cuda_preflight_evidence(args.output, torch)
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
