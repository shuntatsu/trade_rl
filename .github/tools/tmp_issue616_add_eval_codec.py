from pathlib import Path

path = Path("trade_rl/evaluation/experiments/bootstrap/ridge_economic_gate_evaluation.py")
text = path.read_text()

old = "from dataclasses import dataclass, fields\nfrom statistics import median\n"
new = "from dataclasses import dataclass, fields\nfrom pathlib import Path\nfrom statistics import median\n"
if text.count(old) != 1:
    raise SystemExit("import insertion point drift")
text = text.replace(old, new)

old = '''        _require_hex(self.baseline_return_sha256, field="baseline_return_sha256")
        _require_hex(self.candidate_return_sha256, field="candidate_return_sha256")


def _return_sha256'''
new = '''        _require_hex(self.baseline_return_sha256, field="baseline_return_sha256")
        _require_hex(self.candidate_return_sha256, field="candidate_return_sha256")

    def to_payload(self) -> dict[str, object]:
        return {
            item.name: _as_json_value(getattr(self, item.name)) for item in fields(self)
        }


def _return_sha256'''
if text.count(old) != 1:
    raise SystemExit("symbol payload insertion point drift")
text = text.replace(old, new)

old = '''        for field_name, expected in fixed_flags.items():
            value = getattr(self, field_name)
            if type(value) is not bool or value is not expected:
                raise ValueError(f"{field_name} violates the sealed research boundary")


def _exact_timestamp_index'''
new = '''        for field_name, expected in fixed_flags.items():
            value = getattr(self, field_name)
            if type(value) is not bool or value is not expected:
                raise ValueError(f"{field_name} violates the sealed research boundary")

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "by_symbol":
                payload[item.name] = [row.to_payload() for row in self.by_symbol]
            else:
                payload[item.name] = _as_json_value(value)
        return payload


def _exact_timestamp_index'''
if text.count(old) != 1:
    raise SystemExit("evaluation payload insertion point drift")
text = text.replace(old, new)

marker = "\n\n__all__ = [\n"
if text.count(marker) != 1:
    raise SystemExit("__all__ insertion point drift")
codec = '''

_RESULT_SCHEMA_VERSION = "ridge_economic_gate_evaluation_result_v1"


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\\n"
    ).encode("utf-8")


def canonical_ridge_economic_gate_evaluation_bytes(
    result: RidgeEconomicGateEvaluation,
) -> bytes:
    """Encode one strict immutable result document in canonical JSON bytes."""

    if not isinstance(result, RidgeEconomicGateEvaluation):
        raise TypeError("result must be a RidgeEconomicGateEvaluation")
    payload = result.to_payload()
    document = {
        "schema_version": _RESULT_SCHEMA_VERSION,
        "content_digest": content_digest(payload),
        "result": payload,
    }
    return _canonical_json_bytes(document)


def _require_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return value


def _require_exact_keys(
    payload: dict[str, object],
    *,
    expected: set[str],
    field: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{field} keys differ from sealed schema")


def _tuple_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string array")
    return tuple(value)


def _decode_symbol_result(value: object) -> RidgeEconomicGateSymbolResult:
    raw = _require_object(value, field="by_symbol item")
    expected = {item.name for item in fields(RidgeEconomicGateSymbolResult)}
    _require_exact_keys(raw, expected=expected, field="by_symbol item")
    converted = dict(raw)
    converted["baseline_termination_reasons"] = _tuple_strings(
        raw["baseline_termination_reasons"], field="baseline_termination_reasons"
    )
    converted["candidate_termination_reasons"] = _tuple_strings(
        raw["candidate_termination_reasons"], field="candidate_termination_reasons"
    )
    return RidgeEconomicGateSymbolResult(**converted)  # type: ignore[arg-type]


def load_ridge_economic_gate_evaluation(
    path: str | Path,
) -> RidgeEconomicGateEvaluation:
    """Load canonical result bytes and reject schema, digest, or semantic forgery."""

    raw_bytes = Path(path).read_bytes()
    try:
        document_value = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("evaluation result must contain valid JSON") from error
    document = _require_object(document_value, field="evaluation result")
    _require_exact_keys(
        document,
        expected={"schema_version", "content_digest", "result"},
        field="evaluation result",
    )
    if document["schema_version"] != _RESULT_SCHEMA_VERSION:
        raise ValueError("evaluation result schema_version is not supported")
    digest = _require_hex(document["content_digest"], field="content_digest")
    result_raw = _require_object(document["result"], field="result")
    expected_result_keys = {item.name for item in fields(RidgeEconomicGateEvaluation)}
    _require_exact_keys(result_raw, expected=expected_result_keys, field="result")

    converted = dict(result_raw)
    converted["symbols"] = _tuple_strings(result_raw["symbols"], field="symbols")
    by_symbol_raw = result_raw["by_symbol"]
    if not isinstance(by_symbol_raw, list):
        raise ValueError("by_symbol must be an array")
    converted["by_symbol"] = tuple(_decode_symbol_result(item) for item in by_symbol_raw)
    result = RidgeEconomicGateEvaluation(**converted)  # type: ignore[arg-type]
    if content_digest(result.to_payload()) != digest:
        raise ValueError("evaluation result content_digest mismatch")
    canonical = canonical_ridge_economic_gate_evaluation_bytes(result)
    if raw_bytes != canonical:
        raise ValueError("evaluation result bytes are not canonical JSON")
    return result
'''
text = text.replace(marker, codec + marker)

old = '''    "RidgeEconomicGateSymbolResult",
    "canonical_ridge_economic_gate_evaluation_spec",
    "evaluate_ridge_economic_gate",'''
new = '''    "RidgeEconomicGateSymbolResult",
    "canonical_ridge_economic_gate_evaluation_bytes",
    "canonical_ridge_economic_gate_evaluation_spec",
    "evaluate_ridge_economic_gate",
    "load_ridge_economic_gate_evaluation",'''
if text.count(old) != 1:
    raise SystemExit("__all__ content insertion point drift")
path.write_text(text.replace(old, new))
