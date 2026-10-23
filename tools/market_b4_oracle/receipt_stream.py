"""Bounded structural scans of original gzip receipts.

Rows are provisional until the iterator is exhausted and ``proof`` is available.
The future composition driver must discard its private replay state on failure.
This module checks bytes and JSON structure, not native financial schemas,
process provenance, order admission, account arithmetic or B4 completeness.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import re
import stat
import zlib
from collections.abc import Generator, Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import BinaryIO, Literal, cast

_CHUNK = 65536
_ACTOR = ("native_receipt", "rows")
_RESERVE = ("closing_rows",)
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
_PLAIN_STRING = re.compile(r'[^"\\\x00-\x1f]+')
_HEX = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ReceiptPin:
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.bytes) is not int or self.bytes <= 0:
            raise ValueError("receipt pin requires a positive exact byte count")
        if type(self.sha256) is not str or not _HEX.fullmatch(self.sha256):
            raise ValueError("receipt pin requires a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ReceiptLimits:
    compressed_bytes: int = 256 * 1024**2
    decompressed_bytes: int = 512 * 1024**2
    row_bytes: int = 1024**2
    scalar_bytes: int = 1024**2
    metadata_bytes: int = 16 * 1024**2
    depth: int = 64
    object_members: int = 4096
    array_items: int = 20000
    total_values: int = 16000000

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            ceiling = field.default
            if (
                type(ceiling) is not int
                or type(value) is not int
                or not 0 < value <= ceiling
            ):
                raise ValueError("receipt limits must be positive bounded integers")


@dataclass(frozen=True, slots=True)
class ReceiptScan:
    actor_rows: int
    reserve_rows: int
    selected_rows: int
    compressed_bytes: int
    decompressed_bytes: int
    compressed_sha256: str

    @property
    def scope(self) -> str:
        return "STRUCTURAL_ORIGINAL_RECEIPT_SCAN_ONLY"


def _regular_path(path: Path) -> Path:
    path = path.absolute()
    for part in (*reversed(path.parents), path):
        information = part.lstat()
        if stat.S_ISLNK(information.st_mode) or getattr(
            information, "st_file_attributes", 0
        ) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError("receipt paths must not contain links or reparse points")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("receipt must be a regular file")
    return path


class _Compressed:
    def __init__(self, path: Path, pin: ReceiptPin, limits: ReceiptLimits) -> None:
        self.path, self.pin, self.limits = path, pin, limits
        self.compressed = self.decompressed = 0
        self.hasher = hashlib.sha256()

    def _finish(self, stream: BinaryIO, opening: os.stat_result) -> None:
        if (
            self.compressed != self.pin.bytes
            or self.hasher.hexdigest() != self.pin.sha256
        ):
            raise ValueError("compressed receipt does not match its exact pin")
        current = _regular_path(self.path).stat()
        if not os.path.samestat(
            opening, os.fstat(stream.fileno())
        ) or not os.path.samestat(opening, current):
            raise ValueError("receipt file identity changed while scanning")
        # Recheck the current path using bounded reads, rather than retaining a
        # second compressed copy or trusting an unchanged size/mtime.
        count = 0
        digest = hashlib.sha256()
        with self.path.open("rb") as second:
            if not os.path.samestat(opening, os.fstat(second.fileno())):
                raise ValueError("receipt was replaced before final readback")
            while raw := second.read(_CHUNK):
                count += len(raw)
                if count > self.pin.bytes:
                    raise ValueError("receipt grew during final readback")
                digest.update(raw)
        if count != self.pin.bytes or digest.hexdigest() != self.pin.sha256:
            raise ValueError("receipt changed during final readback")

    def chunks(self) -> Generator[bytes, None, None]:
        path = _regular_path(self.path)
        with path.open("rb") as stream:
            opening = os.fstat(stream.fileno())
            if not stat.S_ISREG(opening.st_mode) or opening.st_size != self.pin.bytes:
                raise ValueError("receipt file length differs from the pin")
            if opening.st_size > self.limits.compressed_bytes:
                raise ValueError("compressed receipt exceeds its limit")
            inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
            pending = b""
            drain = False
            while True:
                if not pending and not drain:
                    pending = stream.read(_CHUNK)
                    self.compressed += len(pending)
                    if self.compressed > min(
                        self.pin.bytes, self.limits.compressed_bytes
                    ):
                        raise ValueError("compressed receipt grew beyond its limit")
                    self.hasher.update(pending)
                    if not pending:
                        raise ValueError("gzip receipt is truncated")
                maximum = min(
                    _CHUNK, self.limits.decompressed_bytes - self.decompressed + 1
                )
                decoded = inflater.decompress(pending, maximum)
                pending = inflater.unconsumed_tail
                self.decompressed += len(decoded)
                if self.decompressed > self.limits.decompressed_bytes:
                    raise ValueError("decompressed receipt exceeds its limit")
                if inflater.unused_data:
                    raise ValueError(
                        "gzip receipt has trailing bytes or another member"
                    )
                if decoded:
                    yield decoded
                if inflater.eof:
                    if pending or stream.read(1):
                        raise ValueError("gzip receipt has trailing compressed bytes")
                    self._finish(stream, opening)
                    return
                drain = bool(pending) or len(decoded) == maximum


class _Text:
    def __init__(self, source: _Compressed, limits: ReceiptLimits) -> None:
        self.chunks = source.chunks()
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer = ""
        self.position = 0
        self.ended = False
        self.offset = self.metadata_bytes = 0
        self.in_rows = False
        self.row_start: int | None = None
        self.limits = limits

    def peek(self) -> str:
        while self.position == len(self.buffer) and not self.ended:
            try:
                self.buffer = self.decoder.decode(next(self.chunks), final=False)
            except StopIteration:
                self.buffer = self.decoder.decode(b"", final=True)
                self.ended = True
            self.position = 0
        return self.buffer[self.position : self.position + 1]

    def take(self, count: int = 1) -> str:
        if not self.peek() or count > len(self.buffer) - self.position:
            raise ValueError("incomplete JSON token")
        result = self.buffer[self.position : self.position + count]
        self.position += count
        self.offset += len(result.encode("utf-8"))
        if not self.in_rows:
            self.metadata_bytes += len(result.encode("utf-8"))
            if self.metadata_bytes > self.limits.metadata_bytes:
                raise ValueError("receipt metadata exceeds its limit")
        if (
            self.row_start is not None
            and self.offset - self.row_start > self.limits.row_bytes
        ):
            raise ValueError("receipt row exceeds its limit")
        return result

    def space(self) -> None:
        while self.peek() in (" ", "\t", "\r", "\n"):
            self.take()

    def expect(self, expected: str) -> None:
        if self.take() != expected:
            raise ValueError("invalid JSON punctuation")

    def close(self) -> None:
        self.chunks.close()


class _Parser:
    def __init__(self, text: _Text, phase: str, limits: ReceiptLimits) -> None:
        self.text, self.phase, self.limits = text, phase, limits
        self.values = 0
        self.rows = {"actor": 0, "reserve": 0}
        self.seen: set[tuple[str | int, ...]] = set()

    def string(self) -> str:
        text = self.text
        start = text.offset
        parts = [text.take()]
        while True:
            character = text.peek()
            if not character or ord(character) < 32:
                raise ValueError("unterminated string or unescaped control character")
            if character == '"':
                parts.append(text.take())
                break
            if character == "\\":
                parts.append(text.take())
                escaped = text.take()
                parts.append(escaped)
                if escaped == "u":
                    parts.extend(text.take() for _ in range(4))
            else:
                match = _PLAIN_STRING.match(text.buffer, text.position)
                if match is None:
                    raise ValueError("invalid JSON string")
                parts.append(text.take(match.end() - text.position))
            if text.offset - start > self.limits.scalar_bytes:
                raise ValueError("JSON scalar exceeds its limit")
        if text.offset - start > self.limits.scalar_bytes:
            raise ValueError("JSON scalar exceeds its limit")
        value = json.loads("".join(parts))
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("unpaired Unicode surrogate")
        return cast(str, value)

    def scalar(self) -> object:
        text = self.text
        character = text.peek()
        if character == '"':
            return self.string()
        for literal, value in (("true", True), ("false", False), ("null", None)):
            if character == literal[0]:
                for expected in literal:
                    text.expect(expected)
                return value
        pieces = []
        while text.peek() and text.peek() in "0123456789eE+-.":
            pieces.append(text.take())
            if len(pieces) > min(4096, self.limits.scalar_bytes):
                raise ValueError("JSON number exceeds its limit")
        token = "".join(pieces)
        if not _NUMBER.fullmatch(token):
            raise ValueError("invalid JSON scalar")
        result = json.loads(token)
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return cast(object, result)

    def value(
        self, path: tuple[str | int, ...], depth: int, keep: bool
    ) -> Generator[dict[str, object], None, object]:
        self.values += 1
        if self.values > self.limits.total_values:
            raise ValueError("receipt has too many JSON values")
        text = self.text
        text.space()
        character = text.peek()
        if path in (_ACTOR, _RESERVE):
            if character != "[":
                raise ValueError("both receipt row sections must be arrays")
            return (yield from self.row_array(path, depth))
        if character in ("{", "["):
            if depth >= self.limits.depth:
                raise ValueError("JSON nesting exceeds its limit")
            if character == "{":
                return (yield from self.object_value(path, depth, keep))
            return (yield from self.array(path, depth, keep))
        result = self.scalar()
        return result if keep else None

    def object_value(
        self, path: tuple[str | int, ...], depth: int, keep: bool
    ) -> Generator[dict[str, object], None, object]:
        text = self.text
        text.expect("{")
        text.space()
        keys: set[str] = set()
        result: dict[str, object] = {}
        if text.peek() != "}":
            while True:
                if text.peek() != '"':
                    raise ValueError("JSON object requires a string key")
                key = self.string()
                if key in keys:
                    raise ValueError("duplicate JSON object key")
                keys.add(key)
                if len(keys) > self.limits.object_members:
                    raise ValueError("JSON object exceeds its member limit")
                text.space()
                text.expect(":")
                child = yield from self.value((*path, key), depth + 1, keep)
                if keep:
                    result[key] = child
                text.space()
                if text.peek() != ",":
                    break
                text.take()
                text.space()
        text.expect("}")
        return result if keep else None

    def array(
        self, path: tuple[str | int, ...], depth: int, keep: bool
    ) -> Generator[dict[str, object], None, object]:
        text = self.text
        text.expect("[")
        text.space()
        result: list[object] = []
        count = 0
        if text.peek() != "]":
            while True:
                if count >= self.limits.array_items:
                    raise ValueError("JSON array exceeds its item limit")
                child = yield from self.value((*path, count), depth + 1, keep)
                if keep:
                    result.append(child)
                count += 1
                text.space()
                if text.peek() != ",":
                    break
                text.take()
        text.expect("]")
        return result if keep else None

    def row_array(
        self, path: tuple[str | int, ...], depth: int
    ) -> Generator[dict[str, object], None, object]:
        if depth >= self.limits.depth:
            raise ValueError("JSON nesting exceeds its limit")
        self.seen.add(path)
        phase = "actor" if path == _ACTOR else "reserve"
        text = self.text
        text.expect("[")
        text.in_rows = True
        try:
            text.space()
            if text.peek() != "]":
                while True:
                    count = self.rows[phase]
                    if count >= self.limits.array_items:
                        raise ValueError("receipt has too many rows")
                    text.space()
                    if text.peek() != "{":
                        raise ValueError("receipt rows must be objects")
                    text.row_start = text.offset
                    row = yield from self.value(
                        (*path, count), depth + 1, phase == self.phase
                    )
                    text.row_start = None
                    self.rows[phase] += 1
                    if phase == self.phase:
                        yield cast(dict[str, object], row)
                    text.space()
                    if text.peek() != ",":
                        break
                    text.take()
            text.in_rows = False
            text.expect("]")
            return []
        finally:
            text.in_rows = False
            text.row_start = None


class PinnedReceiptRows:
    """One-use structural scan; yielded rows do not establish a valid receipt."""

    def __init__(
        self,
        path: Path,
        pin: ReceiptPin,
        *,
        phase: Literal["actor", "reserve"],
        limits: ReceiptLimits = ReceiptLimits(),
    ) -> None:
        if type(pin) is not ReceiptPin or type(limits) is not ReceiptLimits:
            raise ValueError("requires exact receipt pin and limit declarations")
        pin.__post_init__()
        limits.__post_init__()
        if phase not in ("actor", "reserve"):
            raise ValueError("unknown receipt section")
        self._path, self._pin, self._phase, self._limits = (
            Path(path),
            pin,
            phase,
            limits,
        )
        self._started = False
        self._proof: ReceiptScan | None = None
        self._metadata: bytes | None = None

    @property
    def proof(self) -> ReceiptScan:
        if self._proof is None:
            raise ValueError("receipt has not passed complete structural validation")
        return self._proof

    @property
    def metadata(self) -> dict[str, object]:
        self.proof
        return cast(dict[str, object], json.loads(cast(bytes, self._metadata)))

    def __iter__(self) -> Iterator[dict[str, object]]:
        if self._started:
            raise ValueError("receipt scan is one-use")
        self._started = True
        return self._iterate()

    def _iterate(self) -> Iterator[dict[str, object]]:
        source = _Compressed(self._path, self._pin, self._limits)
        text = _Text(source, self._limits)
        parser = _Parser(text, self._phase, self._limits)
        try:
            metadata = yield from parser.value((), 0, True)
            text.space()
            if text.peek() or not isinstance(metadata, dict):
                raise ValueError("receipt requires one complete JSON object")
            if parser.seen != {_ACTOR, _RESERVE}:
                raise ValueError("receipt is missing an actor or reserve row section")
            encoded = json.dumps(
                metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            if len(encoded) > self._limits.metadata_bytes:
                raise ValueError("retained receipt metadata exceeds its limit")
            self._metadata = encoded
            self._proof = ReceiptScan(
                actor_rows=parser.rows["actor"],
                reserve_rows=parser.rows["reserve"],
                selected_rows=parser.rows[self._phase],
                compressed_bytes=source.compressed,
                decompressed_bytes=source.decompressed,
                compressed_sha256=source.hasher.hexdigest(),
            )
        except (OSError, UnicodeError, zlib.error) as error:
            raise ValueError(
                "invalid or changed original compressed receipt"
            ) from error
        finally:
            text.close()
