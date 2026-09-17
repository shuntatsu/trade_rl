from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

SOURCE_RUN_ID = 34803217815
SOURCE_ARTIFACT_ID = 10331899302
SOURCE_ARTIFACT_NAME = "issue539-calibrated-causal-successor-v3-34803217815"
SOURCE_API_DIGEST = "sha256:89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce"
EXPECTED_DATASET_ID = "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
EXPECTED_DATASET_ARTIFACT_DIGEST = "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7"
IMPLEMENTATION_HEAD = "3d3bbc416b2ac2b9051496e3ead6e8ff96170b45"
PROTOCOL_DIGEST = "167af235eeba44b9ade780c95ef7bab11c7b211d81018b82ece5d7044f7c4eb3"
IMPLEMENTATION_SEAL_SHA256 = "d152072586b6b5a10cb4deb9e5280cc3711e1ac93c4d673a6ef99c7491b2758a"
IMPLEMENTATION_PRIMARY_ARTIFACT_ID = 10487276312
IMPLEMENTATION_PRIMARY_API_DIGEST = "sha256:708593b15300f07ee616157b33c36807b3b1e25814c9139b9668793ab9bc4431"
IMPLEMENTATION_FRESH_ARTIFACT_ID = 10486378100
IMPLEMENTATION_FRESH_API_DIGEST = "sha256:c6c31df2b7ff97f3fe9066baa87f6c482c946ed16d403d5b1deaece3294a8c50"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_source_metadata(path: Path) -> None:
    data = json.loads(path.read_text())
    if data.get("id") != SOURCE_ARTIFACT_ID:
        raise SystemExit("source Artifact ID drift")
    if data.get("name") != SOURCE_ARTIFACT_NAME:
        raise SystemExit("source Artifact name drift")
    if data.get("digest") != SOURCE_API_DIGEST:
        raise SystemExit("source Artifact API digest drift")
    if data.get("expired") is not False:
        raise SystemExit("source Artifact is expired")
    if (data.get("workflow_run") or {}).get("id") != SOURCE_RUN_ID:
        raise SystemExit("source workflow run drift")


def extract_dataset_only(zip_path: Path, dataset_root: Path) -> str:
    raw = zip_path.read_bytes()
    zip_digest = "sha256:" + sha256(raw)
    if zip_digest != SOURCE_API_DIGEST:
        raise SystemExit(f"source ZIP digest mismatch: {zip_digest}")

    if dataset_root.exists():
        raise SystemExit("Dataset extraction root already exists")
    dataset_root.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        manifests = [
            info
            for info in infos
            if not info.is_dir() and info.filename.endswith("dataset/manifest.json")
        ]
        if len(manifests) != 1:
            raise SystemExit(
                f"expected exactly one dataset/manifest.json, got {len(manifests)}"
            )
        manifest = PurePosixPath(manifests[0].filename)
        dataset_prefix = PurePosixPath(*manifest.parts[:-2], "dataset")
        arrays_name = str(dataset_prefix / "arrays.npz")
        if sum(1 for info in infos if info.filename == arrays_name) != 1:
            raise SystemExit("matching dataset/arrays.npz is missing")

        extracted: list[str] = []
        for info in infos:
            name = PurePosixPath(info.filename)
            try:
                relative = name.relative_to(dataset_prefix)
            except ValueError:
                continue
            if info.is_dir():
                continue
            if not relative.parts or any(
                part in {"", ".", ".."} for part in relative.parts
            ):
                raise SystemExit(f"unsafe Dataset member: {info.filename!r}")
            mode = info.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise SystemExit(f"Dataset member must not be a symlink: {info.filename!r}")
            target = dataset_root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.append(str(relative))

    if not {"manifest.json", "arrays.npz"}.issubset(extracted):
        raise SystemExit(f"Dataset extraction incomplete: {sorted(extracted)!r}")
    if any("study" in PurePosixPath(name).parts for name in extracted):
        raise SystemExit("study content unexpectedly extracted")
    return sha256(raw)


def build_seal(
    *,
    metadata_path: Path,
    zip_path: Path,
    dataset_root: Path,
    authority_run_id: int,
) -> tuple[bytes, str]:
    verify_source_metadata(metadata_path)
    source_zip_sha256 = extract_dataset_only(zip_path, dataset_root)

    from trade_rl.data import (
        inspect_published_market_dataset_artifact,
        load_market_dataset_artifact,
    )
    from trade_rl.evaluation.experiments.bootstrap import (
        ridge_economic_gate_execution_authority as authority_module,
    )
    from trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation import (
        canonical_ridge_economic_gate_evaluation_spec,
    )

    published = inspect_published_market_dataset_artifact(dataset_root)
    if published.artifact_digest != EXPECTED_DATASET_ARTIFACT_DIGEST:
        raise SystemExit("Dataset artifact digest drift")
    dataset = load_market_dataset_artifact(dataset_root)
    if dataset.dataset_id != EXPECTED_DATASET_ID:
        raise SystemExit("Dataset ID drift")

    evaluator_called = False

    def forbidden_evaluator(*args: object, **kwargs: object) -> object:
        nonlocal evaluator_called
        evaluator_called = True
        raise AssertionError("PRE-P&L authority must not fit or replay")

    authority_module.evaluate_ridge_economic_gate = forbidden_evaluator
    spec = canonical_ridge_economic_gate_evaluation_spec()
    if spec.protocol_digest != PROTOCOL_DIGEST:
        raise SystemExit("sealed protocol digest drift")
    authority = authority_module.build_ridge_economic_gate_pre_pnl_cost_authority(
        dataset,
        authority_implementation_head=IMPLEMENTATION_HEAD,
        authority_run_id=authority_run_id,
        spec=spec,
    )
    if evaluator_called:
        raise SystemExit("economic evaluator was called during authority construction")
    if authority.result_blind is not True or authority.evaluation_pnl_inspected is not False:
        raise SystemExit("authority result-blind flags are invalid")

    payload: dict[str, object] = {
        "schema_version": "issue616_real_pre_pnl_cost_authority_seal_v1",
        "preregistration_issue": 616,
        "implementation_pr": 625,
        "protocol_digest": PROTOCOL_DIGEST,
        "implementation_head": IMPLEMENTATION_HEAD,
        "implementation_seal_sha256": IMPLEMENTATION_SEAL_SHA256,
        "implementation_primary_artifact_id": IMPLEMENTATION_PRIMARY_ARTIFACT_ID,
        "implementation_primary_api_digest": IMPLEMENTATION_PRIMARY_API_DIGEST,
        "implementation_fresh_artifact_id": IMPLEMENTATION_FRESH_ARTIFACT_ID,
        "implementation_fresh_api_digest": IMPLEMENTATION_FRESH_API_DIGEST,
        "source_bundle_run_id": SOURCE_RUN_ID,
        "source_bundle_artifact_id": SOURCE_ARTIFACT_ID,
        "source_bundle_artifact_name": SOURCE_ARTIFACT_NAME,
        "source_bundle_api_digest": SOURCE_API_DIGEST,
        "source_bundle_zip_sha256": source_zip_sha256,
        "dataset_id": dataset.dataset_id,
        "dataset_artifact_digest": published.artifact_digest,
        "authority_run_id": authority_run_id,
        "authority_digest": authority.digest,
        "authority": authority.to_payload(),
        "result_blind": True,
        "source_bundle_downloaded": True,
        "dataset_files_extracted_only": True,
        "study_files_extracted": False,
        "study_content_read": False,
        "candidate_result_content_read": False,
        "model_fit_performed": False,
        "evaluation_execution_performed": False,
        "evaluation_pnl_inspected": False,
        "final_test_accessed": False,
        "shared_cash_profitability_established": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    canonical_payload = json.dumps(
        payload, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    document = {
        "seal_payload_sha256": sha256(canonical_payload),
        "payload": payload,
    }
    raw = (
        json.dumps(document, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    return raw, authority.digest


def append_output(path: Path, name: str, value: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def command_build(args: argparse.Namespace) -> None:
    raw, authority_digest = build_seal(
        metadata_path=args.metadata,
        zip_path=args.zip,
        dataset_root=args.dataset_root,
        authority_run_id=args.run_id,
    )
    args.output.write_bytes(raw)
    seal_sha256 = sha256(raw)
    if args.primary_b64 is not None:
        primary = base64.b64decode(args.primary_b64, validate=True)
        if raw != primary:
            raise SystemExit("fresh PRE-P&L authority is not byte-identical")
        if args.primary_sha256 != seal_sha256:
            raise SystemExit("fresh PRE-P&L authority seal SHA mismatch")
        if args.primary_authority_digest != authority_digest:
            raise SystemExit("fresh PRE-P&L authority digest mismatch")
    if args.github_output is not None:
        append_output(args.github_output, "seal_sha256", seal_sha256)
        append_output(args.github_output, "authority_digest", authority_digest)
        if args.primary_b64 is None:
            append_output(args.github_output, "seal_b64", base64.b64encode(raw).decode())
    document = json.loads(raw)
    authority = document["payload"]["authority"]
    print(f"PRE_PNL_AUTHORITY_DIGEST={authority_digest}")
    print(f"PRE_PNL_AUTHORITY_SEAL_SHA256={seal_sha256}")
    print(f"N_EVALUATION_ROWS={authority['n_evaluation_rows']}")
    print("DATASET_ONLY_EXTRACTION=true")
    print("STUDY_CONTENT_READ=false")
    print("MODEL_FIT_PERFORMED=false")
    print("EVALUATION_EXECUTION_PERFORMED=false")
    print("EVALUATION_PNL_INSPECTED=false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--metadata", type=Path, required=True)
    build.add_argument("--zip", type=Path, required=True)
    build.add_argument("--dataset-root", type=Path, required=True)
    build.add_argument("--run-id", type=int, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--github-output", type=Path)
    build.add_argument("--primary-b64")
    build.add_argument("--primary-sha256")
    build.add_argument("--primary-authority-digest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        command_build(args)


if __name__ == "__main__":
    main()
