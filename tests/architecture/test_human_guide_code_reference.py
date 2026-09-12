ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
json = __import__("json")
contract = __import__("guide.tools.content_contract", fromlist=["*"])


def _assert_contract_error(callable_object, message: str) -> None:
    try:
        callable_object()
    except contract.GuideContractError as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"expected GuideContractError containing {message!r}")


def _code_symbol() -> dict[str, object]:
    return {
        "qualified_name": "trade_rl.demo.run",
        "kind": "function",
        "path": "trade_rl/demo.py",
        "start_line": 1,
        "end_line": 3,
        "signature": "def run(value: int) -> int",
        "source_sha256": "a" * 64,
        "local_names": ["result", "value"],
    }


def _code_reference() -> dict[str, object]:
    return {
        "id": "demo-run",
        "symbol": "trade_rl.demo.run",
        "kind": "function",
        "source_sha256": "a" * 64,
        "label_ja": "デモ処理",
        "description_ja": "値を処理します。",
        "variables": [
            {
                "name": "result",
                "label_ja": "処理結果",
                "description_ja": "関数が返す値です。",
            }
        ],
        "tests": ["tests/evaluation/test_single_symbol_replay.py"],
    }


def test_validate_code_reference_rejects_missing_symbol_and_wrong_kind() -> None:
    reference = _code_reference()
    _assert_contract_error(
        lambda: contract.validate_code_reference(ROOT, reference, {}),
        "missing code symbol",
    )

    symbol = _code_symbol()
    reference["kind"] = "method"
    _assert_contract_error(
        lambda: contract.validate_code_reference(
            ROOT,
            reference,
            {str(symbol["qualified_name"]): symbol},
        ),
        "code symbol kind mismatch",
    )


def test_validate_code_reference_rejects_stale_digest_and_unknown_variable() -> None:
    symbol = _code_symbol()
    symbols = {str(symbol["qualified_name"]): symbol}

    stale = _code_reference()
    stale["source_sha256"] = "0" * 64
    _assert_contract_error(
        lambda: contract.validate_code_reference(ROOT, stale, symbols),
        "stale code symbol fingerprint",
    )

    unknown_variable = _code_reference()
    unknown_variable["variables"] = [
        {
            "name": "does_not_exist",
            "label_ja": "存在しない変数",
            "description_ja": "検証用です。",
        }
    ]
    _assert_contract_error(
        lambda: contract.validate_code_reference(ROOT, unknown_variable, symbols),
        "unknown code variable",
    )


def test_validate_code_reference_rejects_invalid_test_path() -> None:
    symbol = _code_symbol()
    reference = _code_reference()
    reference["tests"] = ["../outside.py"]
    _assert_contract_error(
        lambda: contract.validate_code_reference(
            ROOT,
            reference,
            {str(symbol["qualified_name"]): symbol},
        ),
        "invalid code reference test path",
    )


def test_refresh_code_updates_only_source_digest(tmp_path) -> None:
    source = tmp_path / "trade_rl" / "demo.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def run(value: int) -> int:\n"
        "    result = value + 1\n"
        "    return result\n",
        encoding="utf-8",
    )
    test_path = tmp_path / "tests" / "demo" / "test_demo.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("def test_demo() -> None:\n    pass\n", encoding="utf-8")
    topic_path = tmp_path / "guide" / "content" / "topics" / "demo.json"
    topic_path.parent.mkdir(parents=True)
    reference = {
        "id": "demo-run",
        "symbol": "trade_rl.demo.run",
        "kind": "function",
        "source_sha256": "0" * 64,
        "label_ja": "デモ処理",
        "description_ja": "値を処理します。",
        "variables": [
            {
                "name": "result",
                "label_ja": "処理結果",
                "description_ja": "関数が返す値です。",
            }
        ],
        "tests": ["tests/demo/test_demo.py"],
    }
    topic_path.write_text(
        json.dumps(
            {"id": "demo", "code_references": [reference]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    before = json.loads(json.dumps(reference, ensure_ascii=False))
    contract.refresh_code_references(["demo"], root=tmp_path)
    after = json.loads(topic_path.read_text(encoding="utf-8"))["code_references"][0]

    assert after["source_sha256"] != "0" * 64
    assert {key: value for key, value in after.items() if key != "source_sha256"} == {
        key: value for key, value in before.items() if key != "source_sha256"
    }
