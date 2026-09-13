from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MAX_BYTES = 500_000


class BundleBudgetError(ValueError):
    """Raised when the built Guide exceeds its JavaScript bundle budget."""


def check_javascript_bundle_budget(
    dist: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> None:
    if max_bytes <= 0:
        raise BundleBudgetError("max_bytes must be positive")
    if not dist.is_dir():
        raise BundleBudgetError(f"missing build directory: {dist}")

    chunks = sorted(path for path in dist.rglob("*.js") if path.is_file())
    if not chunks:
        raise BundleBudgetError(f"no JavaScript chunks found under: {dist}")

    oversized = [
        (path.relative_to(dist).as_posix(), path.stat().st_size)
        for path in chunks
        if path.stat().st_size > max_bytes
    ]
    if oversized:
        details = ", ".join(f"{path}={size} bytes" for path, size in oversized)
        raise BundleBudgetError(
            f"JavaScript bundle budget exceeded ({max_bytes} bytes): {details}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Human Guide JavaScript bundle size.")
    parser.add_argument("dist", type=Path)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        check_javascript_bundle_budget(args.dist, max_bytes=args.max_bytes)
    except BundleBudgetError as exc:
        print(f"guide bundle budget: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
