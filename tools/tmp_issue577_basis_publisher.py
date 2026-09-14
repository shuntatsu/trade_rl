from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from tools.tmp_issue577_basis_common import (
    CALIBRATION_HEAD,
    ISSUE_NUMBER,
    PROTOCOL_DIGEST,
    SOURCE_IMPLEMENTATION_HEAD,
    authority_summary,
    build_dataset,
    build_result,
    build_source_manifest,
    download_all_sources,
    load_index_preflight,
    manifest_bytes,
    result_bytes,
)
from trade_rl.artifacts.canonical import canonical_json_bytes


def _digest_body(payload: dict[str, object]) -> dict[str, object]:
    body = dict(payload)
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return {**body, "content_digest": digest}


def execute(
    *,
    output_dir: Path,
    preflight_path: Path,
    expected_source_manifest: Path,
    workflow_run_id: int,
) -> None:
    if output_dir.exists():
        raise RuntimeError("publisher output directory already exists")

    # Freeze and compare all raw-source identities before constructing any Dataset or
    # computing the real training relation. A changed historical archive fails here.
    preflight = load_index_preflight(preflight_path)
    entries, contract_by_symbol, index_by_symbol = download_all_sources(preflight)
    manifest = build_source_manifest(entries)
    source_bytes = manifest_bytes(manifest)
    frozen_source_bytes = expected_source_manifest.read_bytes()
    if source_bytes != frozen_source_bytes:
        raise RuntimeError(
            "combined raw source manifest differs from frozen pre-training authority"
        )
    frozen_source_sha = hashlib.sha256(frozen_source_bytes).hexdigest()
    manifest_digest = str(manifest["content_digest"])

    # This is the first point at which the real basis Dataset / training relation may
    # be constructed. The raw source bytes have already passed the frozen gate above.
    dataset = build_dataset(
        contract_by_symbol,
        index_by_symbol,
        source_manifest_digest=manifest_digest,
    )
    result = build_result(dataset, source_manifest_digest=manifest_digest)
    published_result_bytes = result_bytes(result)
    if result.dataset_id != dataset.dataset_id:
        raise RuntimeError("result Dataset ID differs from reconstructed Dataset")
    if result.source_manifest_digest != manifest_digest:
        raise RuntimeError("result source-manifest digest differs")

    metadata = _digest_body(
        {
            "schema_version": "issue577_basis_training_publisher_metadata_v1",
            "issue_number": ISSUE_NUMBER,
            "workflow_run_id": workflow_run_id,
            "protocol_digest": PROTOCOL_DIGEST,
            "source_implementation_head": SOURCE_IMPLEMENTATION_HEAD,
            "calibration_head": CALIBRATION_HEAD,
            "dataset_id": dataset.dataset_id,
            "source_manifest_digest": manifest_digest,
            "source_manifest_json_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "frozen_source_manifest_json_sha256": frozen_source_sha,
            "source_manifest_matches_frozen_authority": True,
            "result_content_digest": result.digest,
            "result_json_sha256": hashlib.sha256(published_result_bytes).hexdigest(),
            "training_relation_executed": True,
            "interpretation_deferred_until_fresh_reconstruction": True,
            "economic_values_logged": False,
            "evaluation_pnl_inspected": False,
            "evaluation_execution_authorized": False,
            "final_test_authorized": False,
            "shared_cash_profitability_established": False,
            "production_eligible": False,
            "live_trading_authorized": False,
            "authority": authority_summary(),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source-manifest.json").write_bytes(source_bytes)
    (output_dir / "result.json").write_bytes(published_result_bytes)
    (output_dir / "publisher-metadata.json").write_bytes(canonical_json_bytes(metadata))

    if sorted(path.name for path in output_dir.iterdir()) != [
        "publisher-metadata.json",
        "result.json",
        "source-manifest.json",
    ]:
        raise RuntimeError("publisher artifact member roster differs")

    # Deliberately do not print status, beta, numerator, denominator or feature values.
    print("PUBLISHER_RESULT_CREATED=true")
    print("PUBLISHER_INTERPRETATION_DEFERRED=true")
    print("SOURCE_MANIFEST_MATCHES_FROZEN_AUTHORITY=true")
    print(f"DATASET_ID={dataset.dataset_id}")
    print(f"SOURCE_MANIFEST_DIGEST={manifest_digest}")
    print(f"RESULT_CONTENT_DIGEST={result.digest}")
    print("ECONOMIC_VALUES_LOGGED=false")
    print("EVALUATION_PNL_INSPECTED=false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-path", type=Path, required=True)
    parser.add_argument("--expected-source-manifest", type=Path, required=True)
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        default=int(os.environ.get("GITHUB_RUN_ID", "0")),
    )
    args = parser.parse_args()
    if args.workflow_run_id <= 0:
        raise SystemExit("workflow run ID must be positive")
    execute(
        output_dir=args.output_dir,
        preflight_path=args.preflight_path,
        expected_source_manifest=args.expected_source_manifest,
        workflow_run_id=args.workflow_run_id,
    )


if __name__ == "__main__":
    main()
