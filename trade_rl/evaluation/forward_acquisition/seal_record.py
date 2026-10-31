"""The exact Markdown body of the future Issue #810 FS1 seal comment.

The generator reads an acquisition root's seal manifest and receipt, checks that they are
the FS1 ones and agree with each other, and returns text. It only prints: it has no
network client, posts nothing, and treats the sealed code commit as an operator-supplied
value (it must be a commit on ``main`` that contains the FS1 contract).

The text states the procedure notes that belong to the seal:

* LOW-A: pre-seal Revision 2 of the FS1 design record changed no digest, falsifier,
  threshold or arm and is covered by the independent-review receipt on PR #1049;
* LOW-B: a hard kill of the scoring process can leave a zero-byte claimed output file; it
  is reported on Issue #810 and removed by hand, never silently.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from trade_rl.evaluation.forward_acquisition.acquire import AcquisitionRefusal
from trade_rl.evaluation.forward_acquisition.plan import (
    MANIFEST_NAME,
    RECEIPT_NAME,
    RECEIPT_SCHEMA,
    SOURCE_KEY_DATASET_ARTIFACT,
    SOURCE_KEY_DEV_ROSTER,
    SOURCE_KEY_EXCHANGE_INFO,
    SOURCE_KEY_RECEIPT,
)
from trade_rl.evaluation.forward_study.registry import (
    FROZEN_PARAMS_DIGEST,
    FS1_REGISTRY,
    Fs1Registry,
)
from trade_rl.evaluation.forward_study.seal import parse_manifest

_GIT_SHA: Final = re.compile(r"^[0-9a-f]{40}$")
LOW_A_NOTE: Final = (
    "Pre-seal Revision 2 of the FS1 design record (2026-10-30; no change to the params "
    "digest, the falsifiers, the thresholds or the arms) is covered by the "
    "independent-review receipt on PR #1049."
)
LOW_B_PROCEDURE: Final = (
    "The scoring entrypoint claims its output path atomically before it reads the "
    "dataset and removes the claim on any failure. A hard kill (for example SIGKILL or "
    "power loss) can leave a zero-byte claimed output file, and the next run then "
    "refuses with `output_exists`. Such a file is reported on Issue #810 and removed "
    "by hand by the operator, never silently, and never by a script. A non-empty "
    "output is a completed result: it is never removed, overwritten or replaced, and "
    "a second result file would be a protocol violation to be reported."
)


@dataclass(frozen=True)
class SealRecordInputs:
    manifest_bytes: bytes
    receipt_bytes: bytes
    sealed_code_sha: str


def _utc(text: str) -> datetime:
    parsed = datetime.fromisoformat(text)
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def _row(label: str, value: str) -> str:
    return f"| {label} | {value} |"


def _render(
    inputs: SealRecordInputs,
    *,
    registry: Fs1Registry,
    frozen_params_digest: str,
) -> str:
    reasons: list[str] = []
    if not _GIT_SHA.match(inputs.sealed_code_sha):
        reasons.append("sealed_code_sha_malformed")
    manifest = None
    try:
        manifest = parse_manifest(inputs.manifest_bytes)
    except ValueError:
        reasons.append("manifest_invalid")
    receipt: dict[str, Any] = {}
    try:
        loaded = json.loads(inputs.receipt_bytes.decode("utf-8"))
        if (
            not isinstance(loaded, dict)
            or loaded.get("schema_version") != RECEIPT_SCHEMA
        ):
            raise ValueError("not an FS1 acquisition receipt")
        receipt = loaded
    except (UnicodeDecodeError, ValueError):
        reasons.append("receipt_invalid")
    if registry.digest() != frozen_params_digest:
        reasons.append("registry_digest_differs_from_frozen")
    if reasons or manifest is None:
        raise AcquisitionRefusal(tuple(dict.fromkeys(reasons)))

    manifest_sha = hashlib.sha256(inputs.manifest_bytes).hexdigest()
    receipt_sha = hashlib.sha256(inputs.receipt_bytes).hexdigest()
    sources = manifest.source_manifest
    earliest = datetime.fromisoformat(registry.scoring_end).replace(
        tzinfo=UTC
    ) + timedelta(hours=registry.settlement_margin_hours)
    dataset = receipt.get("dataset", {})
    builder = receipt.get("builder", {})
    exchange = receipt.get("exchange_information", {})
    reuse = receipt.get("development_reuse", {})
    archives = receipt.get("source_archives", {})
    if manifest.build_policy_digest != registry.build_policy_digest():
        reasons.append("manifest_build_policy_mismatch")
    if manifest.symbols != registry.symbols:
        reasons.append("manifest_symbols_mismatch")
    if sources.get(SOURCE_KEY_RECEIPT) != receipt_sha:
        reasons.append("receipt_sha256_not_bound_by_manifest")
    if dataset.get("dataset_id") != manifest.dataset_id:
        reasons.append("receipt_dataset_id_differs_from_manifest")
    if receipt.get("authorization_reference") != manifest.authorization_reference:
        reasons.append("receipt_authorization_differs_from_manifest")
    if receipt.get("fetched_utc") != manifest.fetched_utc:
        reasons.append("receipt_fetched_utc_differs_from_manifest")
    if receipt.get("registry_params_digest") != registry.digest():
        reasons.append("receipt_params_digest_differs_from_registry")
    if exchange.get("raw_payload_sha256") != sources.get(SOURCE_KEY_EXCHANGE_INFO):
        reasons.append("exchange_information_not_bound_by_manifest")
    if dataset.get("artifact_digest") != sources.get(SOURCE_KEY_DATASET_ARTIFACT):
        reasons.append("dataset_artifact_not_bound_by_manifest")
    if reuse.get("roster_digest") != sources.get(SOURCE_KEY_DEV_ROSTER):
        reasons.append("development_roster_not_bound_by_manifest")
    if _utc(manifest.fetched_utc) < earliest:
        reasons.append("manifest_fetched_before_settlement")
    if _utc(manifest.sealed_utc) < _utc(manifest.fetched_utc):
        reasons.append("manifest_sealed_before_fetched")
    if reasons:
        raise AcquisitionRefusal(tuple(dict.fromkeys(reasons)))

    root = str(receipt.get("output_root", "<acquisition root>"))
    earliest_text = earliest.strftime("%Y-%m-%dT%H:%MZ")
    sealed = inputs.sealed_code_sha
    builder_head = builder.get("git_head")
    same_commit = builder_head == sealed
    archive_urls = [key for key in sources if not key.startswith("fs1:")]
    lines = [
        "## FS1 seal record (Issue #810)",
        "",
        "Printed by `python -m trade_rl.evaluation.forward_acquisition seal-record`. No "
        "tool posts it: the user decides whether to post it as an Issue #810 comment. "
        "It records software and data identity only. It is not the G4 scoring "
        "authorization, it does not authorize sealed PPO, unused/final evaluation or "
        "any live or paper order, and it contains no economic result.",
        "",
        "| Field | Value |",
        "|---|---|",
        _row("Sealed code commit", f"`{sealed}`"),
        _row("FROZEN_PARAMS_DIGEST", f"`{registry.digest()}`"),
        _row("Build-policy digest", f"`{registry.build_policy_digest()}`"),
        _row("Dataset seal manifest SHA-256", f"`{manifest_sha}`"),
        _row("Dataset id", f"`{manifest.dataset_id}`"),
        _row("UTC fetch time (`fetched_utc`)", manifest.fetched_utc),
        _row("UTC seal time (`sealed_utc`)", manifest.sealed_utc),
        _row("Data-fetch authorization", manifest.authorization_reference),
        _row("Scoring allowed at or after", earliest_text),
        _row(
            "Dataset builder commit",
            f"`{builder_head}` (tracked tree clean: {builder.get('tracked_tree_clean')})"
            + ("" if same_commit else " - differs from the sealed code commit"),
        ),
        _row("Acquisition receipt SHA-256", f"`{receipt_sha}`"),
        _row(
            "Source archives",
            f"{len(archive_urls)} archives, roster digest "
            f"`{archives.get('roster_digest')}`; {archives.get('fetched_count')} fetched, "
            f"{reuse.get('archive_count')} reused read-only from development dataset "
            f"`{reuse.get('dataset_id')}` (artifact digest "
            f"`{reuse.get('artifact_digest')}`, roster digest `{reuse.get('roster_digest')}`)",
        ),
        _row(
            "Exchange-rule snapshot",
            f"{exchange.get('source_uri')} retrieved {exchange.get('retrieved_at')}, "
            f"raw payload SHA-256 `{exchange.get('raw_payload_sha256')}`",
        ),
        _row("Dataset artifact digest", f"`{dataset.get('artifact_digest')}`"),
        "",
        "### Dataset builder command",
        "",
        "```",
        str(receipt.get("command_line")),
        "```",
        "",
        "### Scoring command (only after a separate G4 scoring authorization comment "
        f"exists and the clock is at or after {earliest_text})",
        "",
        "```",
        "python -m trade_rl.evaluation.forward_study \\",
        f"  --dataset {root}/dataset \\",
        f"  --manifest {root}/{MANIFEST_NAME} \\",
        f"  --manifest-sha256 {manifest_sha} \\",
        "  --authorization <ISSUE-810-G4-SCORING-AUTHORIZATION-COMMENT-URL> \\",
        f"  --params-digest {registry.digest()} \\",
        f"  --sealed-code-sha {sealed} \\",
        "  --output <NEW-RESULT-PATH>",
        "```",
        "",
        "### Procedure notes",
        "",
        f"- LOW-A: {LOW_A_NOTE}",
        f"- LOW-B: {LOW_B_PROCEDURE}",
        f"- The receipt (`{RECEIPT_NAME}`, SHA-256 above) is bound by the manifest's "
        f"`{SOURCE_KEY_RECEIPT}` entry; the builder commit and the command above are "
        "read from it, and compliance of the run with the pinned build policy is "
        "procedural, as the FS1 design record states.",
        "- Provenance of the archives: fetched archives were checked against their "
        "official `.CHECKSUM` files (the raw checksum bytes are hashed in the receipt); "
        "the reused development archives have no official checksum check and are "
        "verified only against the development roster SHA-256 bound in the verified "
        "development Dataset. The rebuilt development span was compared with that "
        "Dataset (`development_span_equality` in the receipt).",
        "- Scoring remains refused until the wall clock, this seal, both user "
        "authorizations and every parameter, code and dataset check hold.",
    ]
    return "\n".join(lines) + "\n"


def render_seal_record(output_root: Path, sealed_code_sha: str) -> str:
    """Seal-record Markdown for an acquisition root, using the frozen FS1 registry."""

    manifest_path = output_root / MANIFEST_NAME
    receipt_path = output_root / RECEIPT_NAME
    missing = [
        reason
        for reason, path in (
            ("manifest_missing", manifest_path),
            ("receipt_missing", receipt_path),
        )
        if not path.is_file()
    ]
    if missing:
        raise AcquisitionRefusal(tuple(missing))
    return _render(
        SealRecordInputs(
            manifest_bytes=manifest_path.read_bytes(),
            receipt_bytes=receipt_path.read_bytes(),
            sealed_code_sha=sealed_code_sha,
        ),
        registry=FS1_REGISTRY,
        frozen_params_digest=FROZEN_PARAMS_DIGEST,
    )


__all__ = [
    "LOW_A_NOTE",
    "LOW_B_PROCEDURE",
    "SealRecordInputs",
    "render_seal_record",
]
