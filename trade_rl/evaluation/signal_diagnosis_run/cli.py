"""Command line entry of the signal diagnosis (writes new files only).

Nothing is tunable here: the frozen protocol fixes every parameter. The command refuses
to start without a well-formed owner G4 authorization comment URL, refuses a non-empty
output directory, refuses a dirty Git working tree and a missing LightGBM (both before
any data is read), and never touches the network. The published Dataset is admitted from its
manifest and the streamed digest of its payload before the loader parses any array.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from trade_rl.data.artifacts.publication import load_published_market_dataset_artifact
from trade_rl.evaluation.signal_diagnosis.protocol import (
    SCOPE_REAL,
    SignalDiagnosisProtocol,
    SignalDiagnosisProtocolError,
    default_signal_diagnosis_protocol,
)
from trade_rl.evaluation.signal_diagnosis_run.panel import (
    admit_artifact_manifest,
    panel_from_dataset,
)
from trade_rl.evaluation.signal_diagnosis_run.runner import (
    capture_code_provenance,
    run_signal_diagnosis,
    validate_authorization_url,
    write_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trade_rl.evaluation.signal_diagnosis_run",
        description=(
            "Run the preregistered signal diagnosis (Issue #1146) once on the consumed "
            "development Dataset. Parameters are fixed by the protocol."
        ),
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--g4-authorization-url",
        required=True,
        help="owner's issue / pull-request comment URL authorizing this G4 run",
    )
    return parser


_MANIFEST = "manifest.json"
_ARRAYS = "arrays.npz"
_CHUNK = 1 << 20


def admit_artifact_directory(root: Path, protocol: SignalDiagnosisProtocol) -> None:
    """Gate a published Dataset directory before the loader parses any array.

    Reads the small manifest and checks it (:func:`admit_artifact_manifest`), then
    streams the arrays file through SHA-256 in chunks that are discarded unparsed and
    requires the pinned arrays digest. Only an artifact whose bytes are the pinned,
    already consumed development Dataset reaches the loader.
    """

    manifest_path, arrays_path = root / _MANIFEST, root / _ARRAYS
    for path in (manifest_path, arrays_path):
        if path.is_symlink() or not path.is_file():
            raise SignalDiagnosisProtocolError(f"{path.name} is not a regular file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SignalDiagnosisProtocolError("artifact manifest must be a JSON object")
    admit_artifact_manifest(manifest, protocol)
    digest = hashlib.sha256()
    with arrays_path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    if digest.hexdigest() != protocol.data.arrays_digest:
        raise SignalDiagnosisProtocolError(
            "the arrays file is not the pinned development Dataset payload"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        url = validate_authorization_url(args.g4_authorization_url)
    except SignalDiagnosisProtocolError as error:
        print(error, file=sys.stderr)
        return 2
    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        print("refusing to overwrite an existing output", file=sys.stderr)
        return 2
    try:
        provenance = capture_code_provenance()
        provenance.require_clean()
    except SignalDiagnosisProtocolError as error:
        print(f"halted before any data was read: {error}", file=sys.stderr)
        return 2
    if importlib.util.find_spec("lightgbm") is None:
        print(
            "halted before any data was read: LightGBM is missing "
            "(uv sync --extra forecast-gbm)",
            file=sys.stderr,
        )
        return 2
    protocol = default_signal_diagnosis_protocol()
    try:
        admit_artifact_directory(args.dataset, protocol)
    except (SignalDiagnosisProtocolError, OSError, ValueError) as error:
        print(f"refused before any array was read: {error}", file=sys.stderr)
        return 2
    artifact, dataset = load_published_market_dataset_artifact(args.dataset)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    panel = panel_from_dataset(
        dataset,
        artifact_digest=artifact.artifact_digest,
        arrays_digest=str(manifest["arrays_digest"]),
    )
    del dataset
    try:
        run = run_signal_diagnosis(
            panel,
            protocol,
            scope=SCOPE_REAL,
            code_provenance=provenance,
            g4_authorization_url=url,
            progress=lambda message: print(message, flush=True),
        )
    except SignalDiagnosisProtocolError as error:
        print(f"refused, no result written: {error}", file=sys.stderr)
        return 2
    write_outputs(output_dir, run)
    print(f"verdict={run.label}")
    return 0 if run.label not in (None, "INVALID") else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
