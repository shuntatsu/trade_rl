from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from trade_rl.evaluation.experiments.bootstrap.spot_aggtrades_source_prereg import (
    canonical_spot_aggtrades_source_protocol,
)
from trade_rl.evaluation.experiments.bootstrap.spot_aggtrades_structural_validator import (
    build_structural_report,
    canonical_structural_report_bytes,
    validate_archive_bytes,
)

PROTOCOL_HEAD = "9932775127c79a85d81fb304850c3349460e4a2a"
PROTOCOL_DIGEST = "808f999915815259762067aaff381a88589a7c28ceb0383d8aa0626d5c67e8d4"
PROTOCOL_SEAL_RUN_ID = 34928427543
PROTOCOL_SEAL_ARTIFACT_ID = 10381040623
PROTOCOL_SEAL_ARTIFACT_API_DIGEST = (
    "84b3637ab61dcc6ba9942a2b41907db326bc5fd48643e437cc4505ff17f9f62f"
)
PROTOCOL_FRESH_ARTIFACT_ID = 10381305016
PROTOCOL_FRESH_ARTIFACT_API_DIGEST = (
    "d15223aa90a1feaeb8cd85baad9d374422f55d4cfb7d262e52ecc9e7978440a0"
)
VALIDATOR_HEAD = "a8234444cb4306c37d80c47832200f8303f8928c"
VALIDATOR_VERIFICATION_RUN_ID = 34930082702
_USER_AGENT = "trade-rl-issue578-structural-probe/1.0"


def _fetch(url: str) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45.0) as response:  # noqa: S310
                payload = response.read()
            if not payload:
                raise RuntimeError(f"empty response for {url}")
            return payload
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            last_error = error
            if error.code not in {408, 418, 429, 500, 502, 503, 504}:
                raise
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
        if attempt < 2:
            time.sleep(1.0 * (2**attempt))
    raise RuntimeError(f"failed to fetch exact source after retries: {url}: {last_error}")


def _build_report(fetcher=_fetch) -> dict[str, object]:
    protocol = canonical_spot_aggtrades_source_protocol()
    if protocol.digest != PROTOCOL_DIGEST:
        raise RuntimeError("protocol digest differs from sealed Issue 578 authority")

    entries: list[dict[str, object]] = []
    for symbol in protocol.symbols:
        for date in protocol.dates:
            url = protocol.url_template.format(symbol=symbol, date=date)
            archive = fetcher(url)
            checksum = fetcher(url + protocol.checksum_suffix)
            entries.append(
                validate_archive_bytes(
                    protocol,
                    symbol=symbol,
                    date=date,
                    archive_bytes=archive,
                    checksum_bytes=checksum,
                )
            )

    return build_structural_report(
        protocol,
        entries,
        protocol_head=PROTOCOL_HEAD,
        protocol_seal_run_id=PROTOCOL_SEAL_RUN_ID,
        protocol_seal_artifact_id=PROTOCOL_SEAL_ARTIFACT_ID,
        protocol_seal_artifact_api_digest=PROTOCOL_SEAL_ARTIFACT_API_DIGEST,
        protocol_fresh_artifact_id=PROTOCOL_FRESH_ARTIFACT_ID,
        protocol_fresh_artifact_api_digest=PROTOCOL_FRESH_ARTIFACT_API_DIGEST,
        validator_head=VALIDATOR_HEAD,
        validator_verification_run_id=VALIDATOR_VERIFICATION_RUN_ID,
    )


def _synthetic_sources() -> dict[str, bytes]:
    protocol = canonical_spot_aggtrades_source_protocol()
    sources: dict[str, bytes] = {}
    for symbol in protocol.symbols:
        for date in protocol.dates:
            url = protocol.url_template.format(symbol=symbol, date=date)
            archive_name = url.rsplit("/", 1)[-1]
            member_name = archive_name.removesuffix(".zip") + ".csv"
            timestamp = int(
                datetime.strptime(date, "%Y-%m-%d")
                .replace(tzinfo=UTC)
                .timestamp()
                * 1_000
            ) + 1_000
            row = f"1,100,1,1,1,{timestamp},False,True\n".encode()
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr(member_name, row)
            payload = stream.getvalue()
            digest = hashlib.sha256(payload).hexdigest()
            sources[url] = payload
            sources[url + protocol.checksum_suffix] = (
                f"{digest}  {archive_name}\n".encode("ascii")
            )
    return sources


def _self_test() -> None:
    synthetic = _synthetic_sources()

    def fetcher(url: str) -> bytes | None:
        return synthetic.get(url)

    report = _build_report(fetcher)
    if report["status"] != "PASS_SPOT_AGGTRADES_SOURCE":
        raise AssertionError("synthetic all-good report did not pass")
    encoded = canonical_structural_report_bytes(report)
    decoded = json.loads(encoded)
    if decoded["validator_head"] != VALIDATOR_HEAD:
        raise AssertionError("validator authority missing from canonical bytes")
    if decoded["validator_verification_run_id"] != VALIDATOR_VERIFICATION_RUN_ID:
        raise AssertionError("validator verification authority missing from canonical bytes")
    if decoded["economic_values_inspected"] is not False:
        raise AssertionError("self-test crossed economic-value boundary")

    tampered = dict(report)
    archive_reports = [dict(item) for item in tampered["archive_reports"]]
    archive_reports[0]["price_mean"] = 123.0
    tampered["archive_reports"] = archive_reports
    payload = dict(tampered)
    payload.pop("content_digest")
    from trade_rl.artifacts.hashing import content_digest

    tampered["content_digest"] = content_digest(payload)
    try:
        canonical_structural_report_bytes(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("semantic-closure tampering was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.output is None:
        raise SystemExit("--output is required outside --self-test")

    report = _build_report()
    encoded = canonical_structural_report_bytes(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)


if __name__ == "__main__":
    main()
