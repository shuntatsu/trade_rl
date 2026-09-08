from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS = ROOT / "trade_rl" / "integrations"
OLD_BINANCE = INTEGRATIONS / "binance.py"
OLD_CACHE = INTEGRATIONS / "binance_cache.py"
OLD_FROZEN = INTEGRATIONS / "frozen_binance_metadata.py"
NEW = INTEGRATIONS / "binance"

BINANCE_TYPES = (
    "BinanceMarket",
    "BinanceTransportMode",
    "BinanceTransportError",
    "BinanceUnsupportedContractError",
    "_market",
    "_mode",
    "_aware_utc",
    "_finite_float",
)
BINANCE_CACHE = ("validate_cached_vision_payload",)
BINANCE_METADATA = (
    "_freeze_json",
    "_freeze_json_object",
    "_mutable_json",
    "_mutable_json_object",
    "BinanceInstrumentMetadata",
    "BinanceExchangeInfoSnapshot",
    "_filter_value",
    "_metadata_from_exchange_info",
)
BINANCE_VISION = (
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
BINANCE_TRANSPORT = ("BinancePublicTransport",)
BINANCE_DATASET = (
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
CACHE_DEFS = (
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
)
FROZEN_DEFS = (
    "_ExchangeInfoTransport",
    "_require_mapping",
    "_require_non_empty",
    "_parse_utc",
    "FrozenBinanceExchangeInfoTransport",
)
EXPECTED_BINANCE_PUBLIC = (
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
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _definitions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    result: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = {}
    for node in _tree(path).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in result:
                raise RuntimeError(f"duplicate top-level definition {node.name} in {path}")
            result[node.name] = node
    return result


def _node_text(path: Path, node: ast.AST) -> str:
    source = _source(path)
    lines = source.splitlines(keepends=True)
    lineno = getattr(node, "lineno")
    decorators = getattr(node, "decorator_list", ())
    if decorators:
        lineno = min(lineno, *(decorator.lineno for decorator in decorators))
    end_lineno = getattr(node, "end_lineno")
    return "".join(lines[lineno - 1 : end_lineno]).rstrip()


def _definition_text(path: Path, names: tuple[str, ...]) -> str:
    definitions = _definitions(path)
    missing = set(names) - set(definitions)
    if missing:
        raise RuntimeError(f"missing definitions in {path}: {sorted(missing)}")
    return "\n\n\n".join(_node_text(path, definitions[name]) for name in names)


def _assignment_text(path: Path, name: str) -> str:
    for node in _tree(path).body:
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if name in targets:
                return _node_text(path, node)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return _node_text(path, node)
    raise RuntimeError(f"missing assignment {name} in {path}")


def _all_tuple(path: Path) -> tuple[str, ...]:
    for node in _tree(path).body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                value = ast.literal_eval(node.value)
                return tuple(str(item) for item in value)
    raise RuntimeError(f"missing __all__ in {path}")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _render(header: str, *parts: str, footer: str = "") -> str:
    sections = [dedent(header).strip(), *(part.strip() for part in parts if part.strip())]
    if footer.strip():
        sections.append(dedent(footer).strip())
    return "\n\n\n".join(sections) + "\n"


def _verify_inventory() -> None:
    expected_binance = set(
        BINANCE_TYPES
        + BINANCE_CACHE
        + BINANCE_METADATA
        + BINANCE_VISION
        + BINANCE_TRANSPORT
        + BINANCE_DATASET
    )
    observed_binance = set(_definitions(OLD_BINANCE))
    if observed_binance != expected_binance:
        raise RuntimeError(
            "Binance inventory mismatch: "
            f"missing={sorted(observed_binance - expected_binance)} "
            f"stale={sorted(expected_binance - observed_binance)}"
        )
    observed_cache = set(_definitions(OLD_CACHE))
    if observed_cache != set(CACHE_DEFS):
        raise RuntimeError(
            "Binance cache inventory mismatch: "
            f"observed={sorted(observed_cache)} expected={sorted(CACHE_DEFS)}"
        )
    observed_frozen = set(_definitions(OLD_FROZEN))
    if observed_frozen != set(FROZEN_DEFS):
        raise RuntimeError(
            "Frozen metadata inventory mismatch: "
            f"observed={sorted(observed_frozen)} expected={sorted(FROZEN_DEFS)}"
        )
    if _all_tuple(OLD_BINANCE) != EXPECTED_BINANCE_PUBLIC:
        raise RuntimeError("old Binance __all__ no longer matches the recorded public contract")
    total = len(observed_binance) + len(observed_cache) + len(observed_frozen)
    if total != 60:
        raise RuntimeError(f"expected 60 classified Binance definitions, observed {total}")


def _update_inventory_document() -> None:
    path = ROOT / "docs" / "plans" / "2026-09-08-binance-definition-inventory.md"
    text = path.read_text(encoding="utf-8")
    replacements = {
        "Every top-level class/function in the three pre-refactor Binance modules is classified below.":
            "Every top-level class/function in the three pre-refactor Binance modules is classified below.",
        "binance_cache.py: 13 top-level classes/functions":
            "binance_cache.py: 14 top-level classes/functions",
        "All 59 observed definitions above have exactly one MOVE or DELETE disposition.":
            "All 60 observed definitions above have exactly one MOVE or DELETE disposition.",
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"inventory document marker missing: {old}")
        text = text.replace(old, new, 1)
    text = text.replace(
        "Inventory workflow observed:\n\n```text\nbinance.py: 41 top-level classes/functions",
        "Inventory workflow observed (the original summary undercounted `binance_cache.py` by one; the AST log itself lists all 14 definitions):\n\n```text\nbinance.py: 41 top-level classes/functions",
        1,
    )
    path.write_text(text, encoding="utf-8")


def _build_types() -> None:
    header = '''
    """Dependency-neutral Binance adapter contracts and validation helpers."""

    from __future__ import annotations

    import math
    from datetime import UTC, datetime
    from enum import StrEnum
    '''
    _write(NEW / "types.py", _render(header, _definition_text(OLD_BINANCE, BINANCE_TYPES)))


def _build_vision() -> None:
    header = '''
    """Pure Binance Vision URL planning and archive parsing."""

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
    constants = "\n\n".join(
        (
            _assignment_text(OLD_BINANCE, "_INTERVAL_MILLISECONDS"),
            _assignment_text(OLD_BINANCE, "_VISION_ROOT"),
        )
    )
    _write(
        NEW / "vision.py",
        _render(header, constants, _definition_text(OLD_BINANCE, BINANCE_VISION)),
    )


def _build_cache() -> None:
    header = '''
    """Deterministic Binance Vision cache planning, validation, and synchronization."""

    from __future__ import annotations

    import hashlib
    import json
    from collections.abc import Sequence
    from dataclasses import dataclass
    from datetime import datetime
    from pathlib import Path
    from typing import Protocol

    from trade_rl.integrations.binance.types import (
        BinanceMarket,
        BinanceTransportError,
        _aware_utc,
    )
    from trade_rl.integrations.binance.vision import (
        plan_vision_kline_urls,
        vision_funding_url,
    )
    '''
    cache_names = tuple(name for name in CACHE_DEFS if name != "_aware_utc")
    cache_body = _definition_text(OLD_CACHE, cache_names)
    cache_body = cache_body.replace(
        "transport: _VisionArchiveTransport | BinancePublicTransport,",
        "transport: _VisionArchiveTransport,",
    )
    footer = '''
    __all__ = [
        "BinanceVisionCachePlan",
        "BinanceVisionCacheReport",
        "inspect_binance_vision_cache",
        "inspect_binance_vision_urls",
        "plan_binance_vision_cache",
        "require_complete_binance_vision_cache",
        "sync_binance_vision_cache",
        "sync_binance_vision_urls",
        "vision_cache_path",
    ]
    '''
    _write(
        NEW / "cache.py",
        _render(
            header,
            _assignment_text(OLD_CACHE, "_VISION_PREFIX"),
            _definition_text(OLD_BINANCE, BINANCE_CACHE),
            cache_body,
            footer=footer,
        ),
    )


def _build_metadata() -> None:
    header = '''
    """Binance exchange metadata contracts and verified frozen snapshots."""

    from __future__ import annotations

    import hashlib
    import json
    import math
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
    footer = '''
    __all__ = [
        "BinanceExchangeInfoSnapshot",
        "BinanceInstrumentMetadata",
        "FrozenBinanceExchangeInfoTransport",
    ]
    '''
    _write(
        NEW / "metadata.py",
        _render(
            header,
            _definition_text(OLD_BINANCE, BINANCE_METADATA),
            _definition_text(OLD_FROZEN, FROZEN_DEFS),
            footer=footer,
        ),
    )


def _build_transport() -> None:
    header = '''
    """Bounded public Binance REST and Vision transport."""

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

    from trade_rl.integrations.binance.cache import validate_cached_vision_payload
    from trade_rl.integrations.binance.metadata import (
        BinanceExchangeInfoSnapshot,
        _mutable_json_object,
    )
    from trade_rl.integrations.binance.types import (
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
    constants = "\n\n".join(
        _assignment_text(OLD_BINANCE, name)
        for name in (
            "_REST_BASE",
            "_REST_KLINES",
            "_REST_EXCHANGE_INFO",
            "_REST_FUNDING",
            "_USER_AGENT",
        )
    )
    _write(
        NEW / "transport.py",
        _render(
            header,
            constants,
            _definition_text(OLD_BINANCE, BINANCE_TRANSPORT),
            footer='__all__ = ["BinancePublicTransport"]',
        ),
    )


def _build_dataset() -> None:
    header = '''
    """Binance row conversion, market-data source, and dataset assembly."""

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
        InstrumentExecutionRule,
        MarketBuildConfig,
        VolumeUnit,
    )
    from trade_rl.data.market import MarketDataset
    from trade_rl.data.source import MarketDataSource, RawMarketSeries
    from trade_rl.integrations.binance.metadata import (
        BinanceInstrumentMetadata,
        _metadata_from_exchange_info,
    )
    from trade_rl.integrations.binance.transport import BinancePublicTransport
    from trade_rl.integrations.binance.types import (
        BinanceMarket,
        BinanceTransportMode,
        BinanceUnsupportedContractError,
        _aware_utc,
        _finite_float,
        _market,
        _mode,
    )
    from trade_rl.integrations.binance.vision import (
        _epoch_ms,
        _interval_ms,
        _normalize_epoch_ms,
    )
    '''
    footer = '''
    __all__ = [
        "BinanceDatasetBuildResult",
        "BinanceMarketDataSource",
        "binance_multitimeframe_feature_specs",
        "build_binance_market_dataset",
    ]
    '''
    _write(
        NEW / "dataset.py",
        _render(
            header,
            _definition_text(OLD_BINANCE, BINANCE_DATASET),
            footer=footer,
        ),
    )


def _build_facade() -> None:
    facade = '''
    """Maintained Binance integration facade."""

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
    _write(NEW / "__init__.py", dedent(facade))


def _rewrite_python_imports() -> None:
    replacements = {
        "trade_rl.integrations.binance_cache": "trade_rl.integrations.binance.cache",
        "trade_rl.integrations.frozen_binance_metadata": "trade_rl.integrations.binance.metadata",
    }
    for root_name in ("trade_rl", "tests", "scripts"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            updated = text
            for old, new in replacements.items():
                updated = updated.replace(old, new)
            if updated != text:
                path.write_text(updated, encoding="utf-8")


def _assert_no_stale_imports() -> None:
    stale = (
        "trade_rl.integrations.binance_cache",
        "trade_rl.integrations.frozen_binance_metadata",
    )
    offenders: list[str] = []
    for root_name in ("trade_rl", "tests"):
        for path in (ROOT / root_name).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(value in text for value in stale):
                offenders.append(str(path.relative_to(ROOT)))
    if offenders:
        raise RuntimeError(f"stale Binance private imports remain: {offenders}")


def main() -> None:
    if NEW.exists():
        raise RuntimeError(f"new Binance package already exists: {NEW}")
    for path in (OLD_BINANCE, OLD_CACHE, OLD_FROZEN):
        if not path.is_file():
            raise RuntimeError(f"expected old Binance source is absent: {path}")

    _verify_inventory()
    _update_inventory_document()
    _build_types()
    _build_vision()
    _build_cache()
    _build_metadata()
    _build_transport()
    _build_dataset()
    _build_facade()
    _rewrite_python_imports()

    OLD_BINANCE.unlink()
    OLD_CACHE.unlink()
    OLD_FROZEN.unlink()
    _assert_no_stale_imports()

    for path in NEW.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    print("Phase 2B Binance split created successfully")


if __name__ == "__main__":
    main()
