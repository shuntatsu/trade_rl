from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected exactly one patch site")
    source.write_text(text.replace(old, new, 1), encoding="utf-8")


module = "trade_rl/evaluation/experiments/bootstrap/signed_taker_flow_calibration.py"
test = "tests/evaluation/experiments/bootstrap/test_signed_taker_flow_calibration.py"

replace_once(
    module,
    "from trade_rl.artifacts.hashing import content_digest\n",
    "from trade_rl.artifacts.canonical import canonical_json_bytes\n"
    "from trade_rl.artifacts.hashing import content_digest\n",
)
replace_once(
    module,
    '''    def to_artifact_payload(self) -> dict[str, object]:
        return {**self.to_payload(), "content_digest": self.digest}
''',
    '''    def _require_publication_authority(self) -> None:
        if self.calibration_head is None:
            raise ValueError("calibration_head is required for artifact publication")
        if self.source_manifest_digest is None:
            raise ValueError("source_manifest_digest is required for artifact publication")

    def to_artifact_payload(self) -> dict[str, object]:
        self._require_publication_authority()
        return {**self.to_payload(), "content_digest": self.digest}
''',
)
replace_once(
    module,
    '''    try:
        raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("signed taker-flow calibration result is malformed") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("signed taker-flow calibration result must be a JSON object")
''',
    '''    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        raw_text = raw_bytes.decode("utf-8")
        raw: Any = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("signed taker-flow calibration result is malformed") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError("signed taker-flow calibration result must be a JSON object")
    try:
        canonical_bytes = canonical_json_bytes(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("signed taker-flow calibration result is malformed") from error
    if raw_bytes != canonical_bytes:
        raise ValueError("signed taker-flow calibration result must use canonical JSON bytes")
''',
)
replace_once(
    module,
    '''    calibration_head_raw = raw["calibration_head"]
    calibration_head = (
        None
        if calibration_head_raw is None
        else _require_hex(calibration_head_raw, length=40, field="calibration_head")
    )
    source_manifest_raw = raw["source_manifest_digest"]
    source_manifest_digest = (
        None
        if source_manifest_raw is None
        else _require_hex(
            source_manifest_raw,
            length=64,
            field="source_manifest_digest",
        )
    )
''',
    '''    calibration_head = _require_hex(
        raw["calibration_head"], length=40, field="calibration_head"
    )
    source_manifest_digest = _require_hex(
        raw["source_manifest_digest"],
        length=64,
        field="source_manifest_digest",
    )
''',
)
replace_once(
    test,
    '''    result = module.calibrate_signed_taker_flow(_dataset(), protocol)
    artifact = result.to_artifact_payload()

    assert artifact["protocol_digest"] == protocol.digest
''',
    '''    result = module.calibrate_signed_taker_flow(
        _dataset(),
        protocol,
        calibration_head="4" * 40,
        source_manifest_digest="5" * 64,
    )
    artifact = result.to_artifact_payload()

    assert artifact["protocol_digest"] == protocol.digest
''',
)

path = Path(test)
text = path.read_text(encoding="utf-8")
marker = "def test_artifact_publication_requires_bound_authority_and_canonical_bytes"
if marker in text:
    raise RuntimeError("artifact publication regression test already exists")
text += '''


def test_artifact_publication_requires_bound_authority_and_canonical_bytes(
    tmp_path: Path,
) -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    unbound = module.calibrate_signed_taker_flow(_dataset(), protocol)

    with pytest.raises(ValueError, match="calibration_head|source_manifest|publication"):
        unbound.to_artifact_payload()

    unbound_payload = {**unbound.to_payload(), "content_digest": unbound.digest}
    path = tmp_path / "unbound.json"
    path.write_text(
        json.dumps(
            unbound_payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="calibration_head|source_manifest"):
        module.load_signed_taker_flow_calibration_result(path)

    bound = module.calibrate_signed_taker_flow(
        _dataset(),
        protocol,
        calibration_head="6" * 40,
        source_manifest_digest="7" * 64,
    )
    artifact = bound.to_artifact_payload()
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical|bytes"):
        module.load_signed_taker_flow_calibration_result(path)
'''
path.write_text(text, encoding="utf-8")
