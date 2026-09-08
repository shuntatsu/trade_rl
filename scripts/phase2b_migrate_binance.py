from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS = ROOT / "trade_rl/integrations"
OLD_BINANCE = INTEGRATIONS / "binance.py"
OLD_CACHE = INTEGRATIONS / "binance_cache.py"
OLD_FROZEN = INTEGRATIONS / "frozen_binance_metadata.py"
NEW = INTEGRATIONS / "binance"

EXPECTED_BINANCE_DEFS = {
    "BinanceMarket",
    "BinanceTransportMode",
    "BinanceTransportError",
    "validate_cached_vision_payload",
    "BinanceUnsupportedContractError",
    "_freeze_json",
    "_freeze_json_object",
    "_mutable_json",
    "_mutable_json_object",
    "BinanceInstrumentMetadata",
    "BinanceExchangeInfoSnapshot",
    "BinanceDatasetBuildResult",
    "_market",
    "_mode",
    "_aware_utc",
    "_epoch_ms",
    "_normalize_epoch_ms",
    "_interval_ms",
    "_day_floor_ms",
    "_iter_days",
    "_iter_months",
    "vision_kline_url",
    "vision_monthly_kline_url",
    "_next_month",
    "plan_vision_kline_urls",
    "vision_funding_url",
    "_csv_rows_from_zip",
    "_looks_like_header",
    "_finite_float",
    "BinancePublicTransport",
    "_parse_kline_rows",
    "_align_funding",
    "BinanceMarketDataSource",
    "_filter_value",
    "_metadata_from_exchange_info",
    "_optional_values",
    "_optional_datetimes",
    "_extended_timeframe_feature_definitions",
    "binance_multitimeframe_feature_specs",
    "_default_features",
    "build_binance_market_dataset",
}
EXPECTED_CACHE_DEFS = {
    "_VisionArchiveTransport",
    "BinanceVisionCachePlan",
    "BinanceVisionCacheReport",
    "_aware_utc",
    "_ordered_nonempty",
    "_official_urls",
    "_next_month",
    "plan_binance_vision_cache",
    "vision_cache_path",
    "inspect_binance_vision_urls",
    "inspect_binance_vision_cache",
    "require_complete_binance_vision_cache",
    "sync_binance_vision_urls",
    "sync_binance_vision_cache",
}
EXPECTED_FROZEN_DEFS = {
    "_ExchangeInfoTransport",
    "_require_mapping",
    "_require_non_empty",
    "_parse_utc",
    "FrozenBinanceExchangeInfoTransport",
}


def source_and_defs(path: Path) -> tuple[str, dict[str, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise RuntimeError(f"could not extract {node.name} from {path}")
            result[node.name] = segment
    return source, result


def assignment(source: str, name: str) -> str:
    tree = ast.parse(source)
    matches: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets: list[ast.expr]
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise RuntimeError(f"could not extract assignment {name}")
            matches.append(segment)
    if len(matches) != 1:
        raise RuntimeError(f"expected one assignment for {name}, found {len(matches)}")
    return matches[0]


def compose(preamble: str, parts: list[str], trailer: str = "") -> str:
    body = "\n\n\n".join(part.rstrip() for part in parts if part.strip())
    result = preamble.rstrip() + "\n\n\n" + body + "\n"
    if trailer:
        result += "\n\n" + trailer.strip() + "\n"
    return result


def replace_method(class_source: str, name: str, replacement: str) -> str:
    pattern = re.compile(
        rf"^    def {re.escape(name)}\(.*?(?=^    (?:def |@staticmethod|@classmethod))",
        re.MULTILINE | re.DOTALL,
    )
    updated, count = pattern.subn(replacement.rstrip() + "\n\n", class_source)
    if count != 1:
        raise RuntimeError(f"expected one method replacement for {name}, found {count}")
    return updated


def rewrite_old_helper_imports() -> None:
    replacements = {
        "trade_rl.integrations.binance_cache": "trade_rl.integrations.binance.cache",
        "trade_rl.integrations.frozen_binance_metadata": "trade_rl.integrations.binance.metadata",
    }
    for base in (ROOT / "trade_rl", ROOT / "tests"):
        for path in sorted(base.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            updated = text
            for old, new in replacements.items():
                updated = updated.replace(old, new)
            if updated != text:
                path.write_text(updated, encoding="utf-8")


def main() -> None:
    binance_source, binance_defs = source_and_defs(OLD_BINANCE)
    _, cache_defs = source_and_defs(OLD_CACHE)
    _, frozen_defs = source_and_defs(OLD_FROZEN)
    if set(binance_defs) != EXPECTED_BINANCE_DEFS:
        raise RuntimeError(
            f"binance definition inventory drift: {sorted(set(binance_defs) ^ EXPECTED_BINANCE_DEFS)}"
        )
    if set(cache_defs) != EXPECTED_CACHE_DEFS:
        raise RuntimeError(
            f"cache definition inventory drift: {sorted(set(cache_defs) ^ EXPECTED_CACHE_DEFS)}"
        )
    if set(frozen_defs) != EXPECTED_FROZEN_DEFS:
        raise RuntimeError(
            f"frozen definition inventory drift: {sorted(set(frozen_defs) ^ EXPECTED_FROZEN_DEFS)}"
        )

    if NEW.exists():
        raise RuntimeError("new Binance package already exists")
    NEW.mkdir()

    types_preamble = '''"""Dependency-neutral Binance adapter types and validation helpers."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
'''
    types_parts = [
        'BINANCE_USER_AGENT = "trade-rl/0.3 public-market-data"',
        *[
            binance_defs[name]
            for name in (
                "BinanceMarket",
                "BinanceTransportMode",
                "BinanceTransportError",
                "BinanceUnsupportedContractError",
                "_market",
                "_mode",
                "_aware_utc",
                "_finite_float",
            )
        ],
    ]
    types_trailer = '''__all__ = [
    "BinanceMarket",
    "BinanceTransportError",
    "BinanceTransportMode",
    "BinanceUnsupportedContractError",
]'''
    (NEW / "types.py").write_text(
        compose(types_preamble, types_parts, types_trailer), encoding="utf-8"
    )

    vision_preamble = '''"""Binance Vision URL planning and archive parsing."""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta

from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportError,
    _aware_utc,
    _market,
)
'''
    vision_parts = [
        assignment(binance_source, "_INTERVAL_MILLISECONDS"),
        assignment(binance_source, "_VISION_ROOT"),
        *[
            binance_defs[name]
            for name in (
                "_epoch_ms",
                "_normalize_epoch_ms",
                "_interval_ms",
                "_day_floor_ms",
                "_iter_days",
                "_iter_months",
                "vision_kline_url",
                "vision_monthly_kline_url",
                "_next_month",
                "plan_vision_kline_urls",
                "vision_funding_url",
                "_csv_rows_from_zip",
                "_looks_like_header",
            )
        ],
    ]
    vision_trailer = '''__all__ = [
    "plan_vision_kline_urls",
    "vision_funding_url",
    "vision_kline_url",
    "vision_monthly_kline_url",
]'''
    (NEW / "vision.py").write_text(
        compose(vision_preamble, vision_parts, vision_trailer), encoding="utf-8"
    )

    cache_preamble = '''"""Deterministic Binance Vision cache planning, integrity and synchronization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from trade_rl.integrations.binance.types import (
    BINANCE_USER_AGENT,
    BinanceMarket,
    BinanceTransportError,
    _aware_utc,
)
from trade_rl.integrations.binance.vision import (
    _VISION_ROOT,
    plan_vision_kline_urls,
    vision_funding_url,
)

_VISION_PREFIX = f"{_VISION_ROOT}/"
'''
    cache_names = (
        "_VisionArchiveTransport",
        "BinanceVisionCachePlan",
        "BinanceVisionCacheReport",
        "_ordered_nonempty",
        "_official_urls",
        "_next_month",
        "plan_binance_vision_cache",
        "vision_cache_path",
        "inspect_binance_vision_urls",
        "inspect_binance_vision_cache",
        "require_complete_binance_vision_cache",
        "sync_binance_vision_urls",
        "sync_binance_vision_cache",
    )
    cache_parts = [binance_defs["validate_cached_vision_payload"]]
    cache_parts.extend(cache_defs[name] for name in cache_names)
    cache_text = compose(cache_preamble, cache_parts)
    cache_text = cache_text.replace(
        "transport: _VisionArchiveTransport | BinancePublicTransport,",
        "transport: _VisionArchiveTransport,",
    )
    cache_write = '''def write_cached_vision_payload(
    *,
    url: str,
    cache_path: Path,
    payload: bytes,
    etag: str | None,
    last_modified: str | None,
) -> None:
    """Atomically publish raw Vision bytes and immutable acquisition evidence."""

    evidence = {
        "acquired_at": datetime.now(UTC).isoformat(),
        "downloader": BINANCE_USER_AGENT,
        "etag": etag,
        "last_modified": last_modified,
        "schema_version": "binance_vision_raw_cache_v1",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "url": url,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path = cache_path.with_suffix(".json")
    binary_temporary = cache_path.with_suffix(".bin.tmp")
    evidence_temporary = evidence_path.with_suffix(".json.tmp")
    binary_temporary.write_bytes(payload)
    evidence_temporary.write_text(
        json.dumps(evidence, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\\n",
        encoding="utf-8",
    )
    binary_temporary.replace(cache_path)
    evidence_temporary.replace(evidence_path)
'''
    cache_all = '''__all__ = [
    "BinanceVisionCachePlan",
    "BinanceVisionCacheReport",
    "inspect_binance_vision_cache",
    "inspect_binance_vision_urls",
    "plan_binance_vision_cache",
    "require_complete_binance_vision_cache",
    "sync_binance_vision_cache",
    "sync_binance_vision_urls",
    "vision_cache_path",
]'''
    (NEW / "cache.py").write_text(
        cache_text.rstrip() + "\n\n\n" + cache_write + "\n\n" + cache_all + "\n",
        encoding="utf-8",
    )

    metadata_preamble = '''"""Binance exchange metadata, instrument contracts and frozen evidence."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from trade_rl.data.contracts import (
    InstrumentContract,
    InstrumentExecutionRule,
    VolumeUnit,
)
from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportMode,
    _finite_float,
)
from trade_rl.integrations.binance.vision import _normalize_epoch_ms
'''
    metadata_parts = [
        binance_defs[name]
        for name in (
            "_freeze_json",
            "_freeze_json_object",
            "_mutable_json",
            "_mutable_json_object",
            "BinanceInstrumentMetadata",
            "BinanceExchangeInfoSnapshot",
            "_filter_value",
            "_metadata_from_exchange_info",
        )
    ]
    metadata_parts.extend(
        frozen_defs[name]
        for name in (
            "_ExchangeInfoTransport",
            "_require_mapping",
            "_require_non_empty",
            "_parse_utc",
            "FrozenBinanceExchangeInfoTransport",
        )
    )
    metadata_trailer = '''__all__ = [
    "BinanceExchangeInfoSnapshot",
    "BinanceInstrumentMetadata",
    "FrozenBinanceExchangeInfoTransport",
]'''
    (NEW / "metadata.py").write_text(
        compose(metadata_preamble, metadata_parts, metadata_trailer), encoding="utf-8"
    )

    transport_preamble = '''"""Bounded public Binance REST and Vision transport."""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from trade_rl.integrations.binance.cache import (
    validate_cached_vision_payload,
    vision_cache_path,
    write_cached_vision_payload,
)
from trade_rl.integrations.binance.metadata import (
    BinanceExchangeInfoSnapshot,
    _mutable_json_object,
)
from trade_rl.integrations.binance.types import (
    BINANCE_USER_AGENT,
    BinanceMarket,
    BinanceTransportError,
    BinanceTransportMode,
    _aware_utc,
    _finite_float,
    _market,
    _mode,
)
from trade_rl.integrations.binance.vision import (
    _VISION_ROOT,
    _csv_rows_from_zip,
    _interval_ms,
    _iter_months,
    _looks_like_header,
    _normalize_epoch_ms,
    plan_vision_kline_urls,
    vision_funding_url,
)
'''
    transport_parts = [
        assignment(binance_source, name)
        for name in ("_REST_BASE", "_REST_KLINES", "_REST_EXCHANGE_INFO", "_REST_FUNDING")
    ]
    transport_parts.append("_USER_AGENT = BINANCE_USER_AGENT")
    transport_class = binance_defs["BinancePublicTransport"]
    transport_class = replace_method(
        transport_class,
        "_vision_cache_path",
        '''    def _vision_cache_path(self, url: str) -> Path | None:
        if self.cache_root is None or not url.startswith(f"{_VISION_ROOT}/"):
            return None
        return vision_cache_path(self.cache_root, url)''',
    )
    transport_class = replace_method(
        transport_class,
        "_validated_cached_vision_payload",
        '''    def _validated_cached_vision_payload(self, url: str, cache_path: Path) -> bytes:
        return validate_cached_vision_payload(url, cache_path)''',
    )
    transport_class = replace_method(
        transport_class,
        "_write_vision_cache",
        '''    def _write_vision_cache(
        self,
        *,
        url: str,
        cache_path: Path,
        payload: bytes,
        etag: str | None,
        last_modified: str | None,
    ) -> None:
        write_cached_vision_payload(
            url=url,
            cache_path=cache_path,
            payload=payload,
            etag=etag,
            last_modified=last_modified,
        )''',
    )
    transport_parts.append(transport_class)
    (NEW / "transport.py").write_text(
        compose(
            transport_preamble,
            transport_parts,
            '__all__ = ["BinancePublicTransport"]',
        ),
        encoding="utf-8",
    )

    dataset_preamble = '''"""Binance market-series conversion, source adapter and dataset assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureAlignment,
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    InstrumentExecutionRule,
    MarketBuildConfig,
    VolumeUnit,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.source import MarketDataSource, RawMarketSeries
from trade_rl.integrations.binance.metadata import (
    BinanceExchangeInfoSnapshot,
    BinanceInstrumentMetadata,
    _metadata_from_exchange_info,
)
from trade_rl.integrations.binance.transport import BinancePublicTransport
from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportError,
    BinanceTransportMode,
    BinanceUnsupportedContractError,
    _aware_utc,
    _finite_float,
    _market,
)
from trade_rl.integrations.binance.vision import (
    _epoch_ms,
    _interval_ms,
    _normalize_epoch_ms,
)
'''
    dataset_parts = [
        binance_defs[name]
        for name in (
            "BinanceDatasetBuildResult",
            "_parse_kline_rows",
            "_align_funding",
            "BinanceMarketDataSource",
            "_optional_values",
            "_optional_datetimes",
            "_extended_timeframe_feature_definitions",
            "binance_multitimeframe_feature_specs",
            "_default_features",
            "build_binance_market_dataset",
        )
    ]
    dataset_trailer = '''__all__ = [
    "BinanceDatasetBuildResult",
    "BinanceMarketDataSource",
    "binance_multitimeframe_feature_specs",
    "build_binance_market_dataset",
]'''
    (NEW / "dataset.py").write_text(
        compose(dataset_preamble, dataset_parts, dataset_trailer), encoding="utf-8"
    )

    facade = '''"""Public Binance market-data adapter facade."""

from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.integrations.binance.dataset import (
    BinanceDatasetBuildResult,
    BinanceMarketDataSource,
    binance_multitimeframe_feature_specs,
    build_binance_market_dataset,
)
from trade_rl.integrations.binance.metadata import (
    BinanceExchangeInfoSnapshot,
    BinanceInstrumentMetadata,
)
from trade_rl.integrations.binance.transport import BinancePublicTransport
from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportError,
    BinanceTransportMode,
    BinanceUnsupportedContractError,
)
from trade_rl.integrations.binance.vision import (
    plan_vision_kline_urls,
    vision_funding_url,
    vision_kline_url,
    vision_monthly_kline_url,
)

__all__ = [
    "BinanceDatasetBuildResult",
    "BinanceExchangeInfoSnapshot",
    "BinanceInstrumentMetadata",
    "InstrumentExecutionRule",
    "BinanceMarket",
    "BinanceMarketDataSource",
    "BinancePublicTransport",
    "BinanceTransportError",
    "BinanceTransportMode",
    "BinanceUnsupportedContractError",
    "binance_multitimeframe_feature_specs",
    "build_binance_market_dataset",
    "plan_vision_kline_urls",
    "vision_funding_url",
    "vision_kline_url",
    "vision_monthly_kline_url",
]
'''
    (NEW / "__init__.py").write_text(facade, encoding="utf-8")

    integrations_init = INTEGRATIONS / "__init__.py"
    text = integrations_init.read_text(encoding="utf-8")
    old = "from trade_rl.integrations.frozen_binance_metadata import (\n    FrozenBinanceExchangeInfoTransport,\n)"
    new = "from trade_rl.integrations.binance.metadata import FrozenBinanceExchangeInfoTransport"
    if text.count(old) != 1:
        raise RuntimeError("integrations frozen metadata import did not match exactly once")
    integrations_init.write_text(text.replace(old, new), encoding="utf-8")

    rewrite_old_helper_imports()

    OLD_BINANCE.unlink()
    OLD_CACHE.unlink()
    OLD_FROZEN.unlink()

    # No executable old helper-module import may survive.
    retired = (
        "trade_rl.integrations.binance_cache",
        "trade_rl.integrations.frozen_binance_metadata",
    )
    leftovers: list[str] = []
    for base in (ROOT / "trade_rl", ROOT / "tests"):
        for path in sorted(base.rglob("*.py")):
            content = path.read_text(encoding="utf-8")
            if any(name in content for name in retired):
                leftovers.append(str(path.relative_to(ROOT)))
    if leftovers:
        raise RuntimeError(f"retired Binance helper imports remain: {leftovers}")

    # Remove the one-shot migration machinery from the produced tree.
    (ROOT / "scripts/phase2b_migrate_binance.py").unlink()
    (ROOT / ".github/workflows/phase2b-binance-migration.yml").unlink()


if __name__ == "__main__":
    main()
