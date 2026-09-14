from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_calibration import (
    load_perp_index_basis_calibration_result,
)

from tmp_issue577_basis_common import (
    ISSUE_NUMBER,
    manifest_bytes,
    reconstruct_from_raw,
    result_bytes,
)


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def execute(
    *,
    published_dir: Path,
    verified_dir: Path,
    preflight_path: Path,
) -> None:
    if verified_dir.exists():
        raise RuntimeError("fresh verification output directory already exists")
    expected_names = {
        "publisher-metadata.json",
        "result.json",
        "source-manifest.json",
    }
    actual_names = {path.name for path in published_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError("published artifact member roster differs")

    publisher_metadata = _load_json_object(published_dir / "publisher-metadata.json")
    if publisher_metadata.get("issue_number") != ISSUE_NUMBER:
        raise RuntimeError("publisher metadata issue authority differs")
    if publisher_metadata.get("interpretation_deferred_until_fresh_reconstruction") is not True:
        raise RuntimeError("publisher did not defer interpretation")
    for key in (
        "economic_values_logged",
        "evaluation_pnl_inspected",
        "evaluation_execution_authorized",
        "final_test_authorized",
        "shared_cash_profitability_established",
        "production_eligible",
        "live_trading_authorized",
    ):
        if publisher_metadata.get(key) is not False:
            raise RuntimeError(f"publisher crossed forbidden boundary: {key}")

    published_source = (published_dir / "source-manifest.json").read_bytes()
    published_result = (published_dir / "result.json").read_bytes()
    loaded_published = load_perp_index_basis_calibration_result(
        published_dir / "result.json"
    )

    manifest, dataset, result = reconstruct_from_raw(preflight_path)
    fresh_source = manifest_bytes(manifest)
    fresh_result = result_bytes(result)
    if fresh_source != published_source:
        raise RuntimeError("fresh source manifest bytes differ from publisher")
    if fresh_result != published_result:
        raise RuntimeError("fresh calibration result bytes differ from publisher")
    if result.dataset_id != dataset.dataset_id:
        raise RuntimeError("fresh result Dataset ID differs")
    if loaded_published.dataset_id != dataset.dataset_id:
        raise RuntimeError("publisher Dataset ID differs from fresh reconstruction")
    if loaded_published.digest != result.digest:
        raise RuntimeError("publisher result digest differs from fresh reconstruction")

    source_sha = hashlib.sha256(fresh_source).hexdigest()
    result_sha = hashlib.sha256(fresh_result).hexdigest()
    if publisher_metadata.get("source_manifest_json_sha256") != source_sha:
        raise RuntimeError("publisher metadata source SHA differs")
    if publisher_metadata.get("result_json_sha256") != result_sha:
        raise RuntimeError("publisher metadata result SHA differs")
    if publisher_metadata.get("dataset_id") != dataset.dataset_id:
        raise RuntimeError("publisher metadata Dataset ID differs")
    if publisher_metadata.get("result_content_digest") != result.digest:
        raise RuntimeError("publisher metadata result digest differs")

    evidence = {
        "schema_version": "issue577_basis_training_fresh_verification_v1",
        "issue_number": ISSUE_NUMBER,
        "dataset_id": dataset.dataset_id,
        "source_manifest_digest": str(manifest["content_digest"]),
        "source_manifest_json_sha256": source_sha,
        "result_content_digest": result.digest,
        "result_json_sha256": result_sha,
        "source_manifest_byte_equal": True,
        "result_byte_equal": True,
        "dataset_id_equal": True,
        "fresh_raw_refetch_completed": True,
        "fresh_reconstruction_passed": True,
        "economic_values_logged": False,
        "evaluation_pnl_inspected": False,
        "evaluation_execution_authorized": False,
        "final_test_authorized": False,
        "shared_cash_profitability_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    verified_dir.mkdir(parents=True, exist_ok=False)
    (verified_dir / "fresh-source-manifest.json").write_bytes(fresh_source)
    (verified_dir / "fresh-result.json").write_bytes(fresh_result)
    (verified_dir / "verification.json").write_bytes(canonical_json_bytes(evidence))

    # Deliberately do not print status, beta or per-symbol regression statistics.
    print("FRESH_RAW_REFETCH_COMPLETED=true")
    print("SOURCE_MANIFEST_BYTE_EQUAL=true")
    print("RESULT_BYTE_EQUAL=true")
    print("DATASET_ID_EQUAL=true")
    print("FRESH_RECONSTRUCTION_PASSED=true")
    print("ECONOMIC_VALUES_LOGGED=false")
    print("EVALUATION_PNL_INSPECTED=false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument("--verified-dir", type=Path, required=True)
    parser.add_argument("--preflight-path", type=Path, required=True)
    args = parser.parse_args()
    execute(
        published_dir=args.published_dir,
        verified_dir=args.verified_dir,
        preflight_path=args.preflight_path,
    )


if __name__ == "__main__":
    main()
