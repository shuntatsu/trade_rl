"""Command line entry of the signal diagnosis (writes new files only).

Nothing is tunable here: the frozen protocol fixes every parameter. The command refuses
to start without a well-formed owner G4 authorization comment URL, refuses a non-empty
output directory, refuses a dirty Git working tree and a missing LightGBM (both before
any data is read), and never touches the network. The published Dataset is admitted from its
manifest and the digest of its payload, both held in memory, and decoded from those same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from trade_rl.data.artifacts import load_dataset_from_verified_bytes
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


def read_artifact_files(root: Path) -> tuple[bytes, bytes]:
    """Read the manifest and the arrays file once each, unparsed, into memory.

    These bytes are the only copy the run ever uses: they are verified and decoded
    from memory, and no path is opened again, so replacing the directory afterwards
    changes nothing.
    """

    out: list[bytes] = []
    for name in (_MANIFEST, _ARRAYS):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise SignalDiagnosisProtocolError(f"{name} is not a regular file")
        with path.open("rb") as stream:
            out.append(stream.read())
    return out[0], out[1]


def admit_artifact_bytes(
    manifest_bytes: bytes, arrays_bytes: bytes, protocol: SignalDiagnosisProtocol
) -> str:
    """Gate artifact bytes before anything is decoded; return the arrays SHA-256.

    The manifest is parsed as small JSON and checked (:func:`admit_artifact_manifest`);
    the arrays bytes, held in memory and not parsed, must hash to the pinned arrays
    digest. Only bytes that are the pinned, already consumed development Dataset pass.
    """

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SignalDiagnosisProtocolError("artifact manifest is not JSON") from error
    if not isinstance(manifest, dict):
        raise SignalDiagnosisProtocolError("artifact manifest must be a JSON object")
    admit_artifact_manifest(manifest, protocol)
    digest = hashlib.sha256(arrays_bytes).hexdigest()
    if digest != protocol.data.arrays_digest:
        raise SignalDiagnosisProtocolError(
            "the arrays bytes are not the pinned development Dataset payload"
        )
    return digest


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
        manifest_bytes, arrays_bytes = read_artifact_files(args.dataset)
        arrays_digest = admit_artifact_bytes(manifest_bytes, arrays_bytes, protocol)
    except (SignalDiagnosisProtocolError, OSError, ValueError) as error:
        print(f"refused before any array was decoded: {error}", file=sys.stderr)
        return 2
    # Decode from the very bytes that were verified; no path is opened again.
    dataset, artifact_digest = load_dataset_from_verified_bytes(
        manifest_bytes, arrays_bytes
    )
    del manifest_bytes, arrays_bytes
    panel = panel_from_dataset(
        dataset, artifact_digest=artifact_digest, arrays_digest=arrays_digest
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
