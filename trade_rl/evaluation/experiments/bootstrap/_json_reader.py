"""Single-snapshot reads for bootstrap JSON configuration and evidence."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import cast

from trade_rl.artifacts.verified_file import open_regular_binary


class _SourceChangedError(ValueError):
    """Distinguish a detected mutation from malformed input bytes."""


def _identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _read_json_object(
    path: Path, *, label: str, invalid_json_message: str | None = None
) -> dict[str, object]:
    """Reject observed replacement/mutation before decoding a JSON snapshot.

    The descriptor remains the no-follow read authority. The pathname
    recheck detects disappearance or replacement; it is not a filesystem
    transaction, a byte budget, or authentication of historical contents.
    """
    regular_error = f"{label} must be a regular file"
    invalid_error = invalid_json_message or f"{label} is invalid JSON"
    changed_error = f"{label} changed during read"
    try:
        inspected = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError(regular_error) from error
    if not stat.S_ISREG(inspected.st_mode) or path.is_symlink() or not path.is_file():
        raise ValueError(regular_error)
    expected = _identity(inspected)
    try:
        with open_regular_binary(path, field=label) as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != expected:
                raise _SourceChangedError(changed_error)
            encoded = handle.read()
            finished = os.fstat(handle.fileno())
            try:
                visible = os.stat(path, follow_symlinks=False)
            except OSError as error:
                raise _SourceChangedError(changed_error) from error
            if len(encoded) != inspected.st_size or any(
                not stat.S_ISREG(current.st_mode) or _identity(current) != expected
                for current in (finished, visible)
            ):
                raise _SourceChangedError(changed_error)
    except _SourceChangedError:
        raise
    except ValueError as error:
        raise ValueError(regular_error) from error
    except FileNotFoundError as error:
        raise _SourceChangedError(changed_error) from error
    except OSError as error:
        raise ValueError(invalid_error) from error

    try:
        decoded = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(invalid_error) from error
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)
