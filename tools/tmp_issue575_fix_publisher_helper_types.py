from __future__ import annotations

from pathlib import Path


PUBLISHER = Path("tools/tmp_issue575_basis_publisher.py")
VERIFIER = Path("tools/tmp_issue575_basis_verifier.py")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement site, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


helper_anchor = '''def _canonical_digest(payload: dict[str, object]) -> str:\n    raw = json.dumps(\n        payload, sort_keys=True, separators=(",", ":"), allow_nan=False\n    ).encode("utf-8")\n    return hashlib.sha256(raw).hexdigest()\n'''
helper_block = helper_anchor + '''\n\ndef _nonnegative_int(value: object, *, field: str) -> int:\n    if isinstance(value, bool) or not isinstance(value, int) or value < 0:\n        raise RuntimeError(f"{field} must be a non-negative integer")\n    return value\n'''
replace_once(PUBLISHER, helper_anchor, helper_block)
replace_once(VERIFIER, helper_anchor, helper_block)

replace_once(
    PUBLISHER,
    '''    for key, expected in required.items():\n        if payload.get(key) != expected:\n            raise RuntimeError(f"index preflight field mismatch: {key}")\n''',
    '''    for field_name, expected in required.items():\n        if payload.get(field_name) != expected:\n            raise RuntimeError(f"index preflight field mismatch: {field_name}")\n''',
)
replace_once(
    PUBLISHER,
    '''        key = (symbol, month)\n        if key in result:\n            raise RuntimeError("index preflight contains duplicate entry")\n''',
    '''        entry_key = (symbol, month)\n        if entry_key in result:\n            raise RuntimeError("index preflight contains duplicate entry")\n''',
)
replace_once(
    PUBLISHER,
    '''        result[key] = dict(raw_entry)\n        observed_order.append(key)\n''',
    '''        result[entry_key] = dict(raw_entry)\n        observed_order.append(entry_key)\n''',
)
replace_once(PUBLISHER, '''            close = float(row[4])\n''', '''            close = float(str(row[4]))\n''')
replace_once(
    PUBLISHER,
    '''        symbol: sum(\n            int(entry["missing_grid_rows"])\n            for entry in entries\n            if entry["symbol"] == symbol\n        )\n''',
    '''        symbol: sum(\n            _nonnegative_int(\n                entry["missing_grid_rows"], field="perp missing_grid_rows"\n            )\n            for entry in entries\n            if entry["symbol"] == symbol\n        )\n''',
)

replace_once(
    VERIFIER,
    '''    fresh_entries: list[dict[str, object]] = []\n    rows = {symbol: [] for symbol in SYMBOLS}\n''',
    '''    fresh_entries: list[dict[str, object]] = []\n    rows: dict[str, list[tuple[int, float, float, float, float, float]]] = {\n        symbol: [] for symbol in SYMBOLS\n    }\n''',
)
replace_once(VERIFIER, '''            close = float(row[4])\n''', '''            close = float(str(row[4]))\n''')
replace_once(
    VERIFIER,
    '''        symbol: sum(\n            int(item["missing_grid_rows"])\n            for item in entries\n            if item["symbol"] == symbol\n        )\n''',
    '''        symbol: sum(\n            _nonnegative_int(\n                item["missing_grid_rows"], field="perp missing_grid_rows"\n            )\n            for item in entries\n            if item["symbol"] == symbol\n        )\n''',
)
