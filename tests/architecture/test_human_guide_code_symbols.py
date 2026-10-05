import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def _code_symbols():
    return __import__("guide.tools.code_symbols", fromlist=["*"])


def _symbols(index: dict[str, object]) -> dict[str, dict[str, object]]:
    raw = index["symbols"]
    assert isinstance(raw, list)
    return {
        str(entry["qualified_name"]): entry for entry in raw if isinstance(entry, dict)
    }


def test_index_resolves_replay_risk_execution_and_local_names() -> None:
    code_symbols = _code_symbols()
    index = code_symbols.build_symbol_index(ROOT / "trade_rl", revision="a" * 40)
    symbols = _symbols(index)

    replay = symbols["trade_rl.evaluation.replay.run_single_symbol_replay"]
    assert replay["kind"] == "function"
    assert replay["path"] == "trade_rl/evaluation/replay.py"
    assert "desired_quantity" in replay["local_names"]
    assert "target_weight" in replay["local_names"]

    risk = symbols["trade_rl.risk.pretrade.PreTradeRisk.constrain"]
    assert risk["kind"] == "method"
    assert risk["path"] == "trade_rl/risk/pretrade.py"

    execution = symbols["trade_rl.simulation.execution.MarketExecutor.execute_interval"]
    assert execution["kind"] == "method"
    assert execution["path"] == "trade_rl/simulation/execution.py"


def test_index_parses_module_without_executing_it(tmp_path) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    (package / "danger.py").write_text(
        "raise RuntimeError('must not execute')\n\n"
        "def safe(value: int) -> int:\n"
        "    result = value + 1\n"
        "    return result\n",
        encoding="utf-8",
    )

    index = code_symbols.build_symbol_index(package, revision="b" * 40)
    symbols = _symbols(index)

    assert "trade_rl.danger.safe" in symbols
    assert symbols["trade_rl.danger.safe"]["local_names"] == ["result", "value"]


def test_nested_scope_names_do_not_leak_into_parent_function(tmp_path) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    (package / "nested.py").write_text(
        "def outer(value: int) -> int:\n"
        "    result = value\n"
        "    def inner() -> int:\n"
        "        nested_only = 1\n"
        "        return nested_only\n"
        "    class Local:\n"
        "        class_only = 1\n"
        "    return result\n",
        encoding="utf-8",
    )

    symbols = _symbols(code_symbols.build_symbol_index(package, revision="c" * 40))
    outer = symbols["trade_rl.nested.outer"]

    assert "result" in outer["local_names"]
    assert "value" in outer["local_names"]
    assert "nested_only" not in outer["local_names"]
    assert "class_only" not in outer["local_names"]


def test_decorator_only_change_updates_source_digest_and_start_line(tmp_path) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    source = package / "decorated.py"
    source.write_text(
        "def decorator_one(func):\n"
        "    return func\n\n"
        "@decorator_one\n"
        "def demo(value: int) -> int:\n"
        "    return value\n",
        encoding="utf-8",
    )

    first = _symbols(code_symbols.build_symbol_index(package, revision="d" * 40))[
        "trade_rl.decorated.demo"
    ]
    source.write_text(
        "def decorator_one(func):\n"
        "    return func\n\n"
        "@decorator_two\n"
        "def demo(value: int) -> int:\n"
        "    return value\n",
        encoding="utf-8",
    )
    second = _symbols(code_symbols.build_symbol_index(package, revision="d" * 40))[
        "trade_rl.decorated.demo"
    ]

    assert first["start_line"] == 4
    assert second["start_line"] == 4
    assert first["source_sha256"] != second["source_sha256"]


def test_comprehension_target_does_not_leak_but_walrus_binding_does(tmp_path) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    (package / "comprehension.py").write_text(
        "def outer(xs: list[int]) -> list[int]:\n"
        "    ys = [item for item in xs if (seen := item)]\n"
        "    return ys\n",
        encoding="utf-8",
    )

    outer = _symbols(code_symbols.build_symbol_index(package, revision="e" * 40))[
        "trade_rl.comprehension.outer"
    ]

    assert "xs" in outer["local_names"]
    assert "ys" in outer["local_names"]
    assert "seen" in outer["local_names"]
    assert "item" not in outer["local_names"]


def test_index_rejects_source_file_symlink_that_escapes_source_root(tmp_path) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def escaped() -> None:\n    pass\n", encoding="utf-8")
    try:
        (package / "escape.py").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    try:
        code_symbols.build_symbol_index(package, revision="f" * 40)
    except code_symbols.GuideCodeSymbolError as exc:
        assert "escapes source root" in str(exc)
    else:
        raise AssertionError(
            "expected GuideCodeSymbolError for escaping source symlink"
        )


def test_index_rejects_non_commit_revision(tmp_path) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    (package / "demo.py").write_text(
        "def demo() -> None:\n    pass\n", encoding="utf-8"
    )

    try:
        code_symbols.build_symbol_index(package, revision="main")
    except code_symbols.GuideCodeSymbolError as exc:
        assert "40-character hexadecimal" in str(exc)
    else:
        raise AssertionError("expected GuideCodeSymbolError for non-commit revision")


@pytest.mark.parametrize("accessors", ["setter", "deleter", "both"])
def test_property_accessors_share_one_complete_symbol(tmp_path, accessors) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    source = (
        "class Account:\n"
        "    @property\n"
        "    def value(self) -> int:\n"
        "        return self._value\n"
    )
    if accessors in ("setter", "both"):
        source += (
            "\n"
            "    @value.setter\n"
            "    def value(self, amount: int) -> None:\n"
            "        normalized = abs(amount)\n"
            "        self._value = normalized\n"
        )
    if accessors in ("deleter", "both"):
        source += (
            "\n"
            "    @value.deleter\n"
            "    def value(self) -> None:\n"
            "        previous = self._value\n"
            "        self._value = previous - previous\n"
        )
    (package / "account.py").write_text(source, encoding="utf-8")

    index = code_symbols.build_symbol_index(package, revision="a" * 40)
    symbols = _symbols(index)
    symbol = symbols["trade_rl.account.Account.value"]

    assert len(symbols) == 2
    assert symbol["kind"] == "method"
    assert symbol["start_line"] == 2
    assert symbol["end_line"] == len(source.splitlines())
    expected_source = "".join(source.splitlines(keepends=True)[1:])
    assert (
        symbol["source_sha256"]
        == __import__("hashlib").sha256(expected_source.encode("utf-8")).hexdigest()
    )
    expected_locals = {"self"}
    if accessors in ("setter", "both"):
        expected_locals.update(("amount", "normalized"))
    if accessors in ("deleter", "both"):
        expected_locals.add("previous")
    assert symbol["local_names"] == sorted(expected_locals)


def test_setter_only_change_updates_complete_property_fingerprint(tmp_path) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    source = package / "account.py"
    body = (
        "class Account:\n"
        "    @property\n"
        "    def value(self) -> int:\n"
        "        return self._value\n"
        "    @value.setter\n"
        "    def value(self, amount: int) -> None:\n"
        "        self._value = abs(amount)\n"
    )
    source.write_text(body, encoding="utf-8")
    first = _symbols(code_symbols.build_symbol_index(package, revision="b" * 40))[
        "trade_rl.account.Account.value"
    ]
    source.write_text(body.replace("abs(amount)", "max(amount, 0)"), encoding="utf-8")
    second = _symbols(code_symbols.build_symbol_index(package, revision="b" * 40))[
        "trade_rl.account.Account.value"
    ]

    assert first["source_sha256"] != second["source_sha256"]


@pytest.mark.parametrize(
    "declarations",
    [
        "    def value(self):\n        pass\n" * 2,
        "    @property\n    def value(self):\n        pass\n" * 2,
        "    @property\n    def value(self):\n        pass\n"
        + "    @value.setter\n    def value(self, amount):\n        pass\n" * 2,
        "    @property\n    def value(self):\n        pass\n"
        "    @other.setter\n    def value(self, amount):\n        pass\n",
        "    def value(self):\n        pass\n"
        "    @value.setter\n    def value(self, amount):\n        pass\n",
        "    @property\n    def value(self):\n        pass\n"
        "    @wrap\n    @value.setter\n    def value(self, amount):\n        pass\n",
        "    @property\n    def value(self):\n        pass\n"
        "    @value.setter()\n    def value(self, amount):\n        pass\n",
        "    @property\n    def value(self):\n        pass\n"
        "    def intervening(self):\n        pass\n"
        "    @value.setter\n    def value(self, amount):\n        pass\n",
    ],
)
def test_property_support_preserves_ambiguous_duplicate_rejection(
    tmp_path, declarations
) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    (package / "account.py").write_text(
        "class Account:\n" + declarations, encoding="utf-8"
    )

    with pytest.raises(
        code_symbols.GuideCodeSymbolError,
        match=r"duplicate code symbol: trade_rl\.account\.Account\.value",
    ):
        code_symbols.build_symbol_index(package, revision="c" * 40)
