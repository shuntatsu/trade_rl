"""Closed little-endian tensor bytes used by the optional transition sidecar."""

from __future__ import annotations

from typing import Any

import numpy as np

from trade_rl.artifacts import content_digest


def array_hex(values: Any, dtype: str, shape: tuple[int, ...]) -> str:
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype(dtype):
        raise ValueError("transition tensor dtype differs from its declaration")
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("transition tensor shape/finite values differ")
    return np.ascontiguousarray(values, dtype=dtype).tobytes().hex()


def read_array(value: object, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if type(value) is not str:
        raise ValueError("transition tensor must be native hexadecimal bytes")
    try:
        raw = bytes.fromhex(value)
        result = np.frombuffer(raw, dtype=dtype).reshape(shape)
    except ValueError as error:
        raise ValueError("transition tensor byte length/encoding differs") from error
    if raw.hex() != value or not np.isfinite(result).all():
        raise ValueError("transition tensor must have canonical finite bytes")
    return result


def buffer_payload(rows: list[dict[str, Any]], width: int) -> dict[str, str]:
    """The exact chronological arrays before SB3's RNG-consuming get()."""
    result = {}
    for name, key, shape in (
        ("observations", "observation", (1, width)),
        ("actions", "action", (1,)),
        ("values", "value", (1, 1)),
        ("log_probs", "log_prob", (1,)),
        ("rewards", "reward", (1,)),
    ):
        dtype = "<i8" if key == "action" else "<f4"
        values = np.stack([read_array(row["actor"][key], dtype, shape) for row in rows])
        if name == "actions":
            values = values.reshape(len(rows), 1, 1).astype("<f4")
        else:
            values = values.reshape(
                len(rows), 1, width if name == "observations" else 1
            )
            if name != "observations":
                values = values.reshape(len(rows), 1)
        result[name] = values.astype("<f4").tobytes().hex()
    result["episode_starts"] = (
        np.asarray([[row["episode_start"]] for row in rows], dtype="<f4")
        .tobytes()
        .hex()
    )
    return result


def buffer_digest(rows: list[dict[str, Any]], width: int) -> str:
    return content_digest(buffer_payload(rows, width))
