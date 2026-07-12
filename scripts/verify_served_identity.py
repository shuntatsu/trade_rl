"""Verify that the live Serving Plane exposes one exact approved identity."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from typing import Any, Callable

_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def identity_matches(
    payload: dict[str, Any],
    *,
    expected_version: str,
    expected_digest: str,
    expected_release_git_sha: str,
) -> bool:
    """Return true only when readiness reports the approved active identity."""

    return (
        payload.get("status") in {"ready", "degraded"}
        and payload.get("active_version") == expected_version
        and payload.get("bundle_digest") == expected_digest
        and payload.get("release_git_sha") == expected_release_git_sha
    )


def fetch_json(url: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Fetch and decode a readiness JSON object."""

    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "trade-rl-deployer"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = response.getcode()
        if status != 200:
            raise RuntimeError(f"readiness returned HTTP {status}")
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("readiness payload must be a JSON object")
    return payload


def verify_with_retries(
    *,
    url: str,
    expected_version: str,
    expected_digest: str,
    expected_release_git_sha: str,
    attempts: int,
    interval_seconds: float,
    fetch: Callable[[str], dict[str, Any]] = fetch_json,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll readiness until the approved identity is observed or attempts expire."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if interval_seconds < 0:
        raise ValueError("interval_seconds must be non-negative")

    for attempt in range(1, attempts + 1):
        try:
            payload = fetch(url)
            if identity_matches(
                payload,
                expected_version=expected_version,
                expected_digest=expected_digest,
                expected_release_git_sha=expected_release_git_sha,
            ):
                print(
                    "served identity verified: "
                    f"version={expected_version} digest={expected_digest} "
                    f"release_git_sha={expected_release_git_sha}"
                )
                return True
            print(f"attempt {attempt}: served identity mismatch: {payload}")
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"attempt {attempt}: readiness check failed: {exc}")
        if attempt < attempts:
            sleep(interval_seconds)
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the model identity currently exposed by /ready"
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--release-git-sha", required=True)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.url.startswith(("http://", "https://")):
        raise SystemExit("--url must use http or https")
    if not args.version.strip():
        raise SystemExit("--version must be non-empty")
    if _DIGEST_RE.fullmatch(args.digest) is None:
        raise SystemExit("--digest must be a 64-character hexadecimal SHA-256")
    if _GIT_SHA_RE.fullmatch(args.release_git_sha) is None:
        raise SystemExit("--release-git-sha must be a 40-character hexadecimal SHA")

    matched = verify_with_retries(
        url=args.url,
        expected_version=args.version,
        expected_digest=args.digest.lower(),
        expected_release_git_sha=args.release_git_sha.lower(),
        attempts=args.attempts,
        interval_seconds=args.interval_seconds,
    )
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
