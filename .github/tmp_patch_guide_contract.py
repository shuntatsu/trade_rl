from pathlib import Path

path = Path("guide/tools/content_contract.py")
text = path.read_text(encoding="utf-8")

constants = '_ALLOWED_CODE_KINDS = {"function", "class", "method"}\n_ALLOWED_ROLES ='
replacement = '_ALLOWED_CODE_KINDS = {"function", "class", "method"}\n_ALLOWED_CODE_MAP_RELATIONS = {"data-flow", "calls"}\n_ALLOWED_ROLES ='
if constants not in text:
    raise SystemExit("compat patch: constants marker missing")
text = text.replace(constants, replacement, 1)

marker = "\ndef _validate_topic_metadata(topic: dict[str, Any], *, topic_id: str) -> None:\n"
if text.count(marker) != 1:
    raise SystemExit("compat patch: metadata marker mismatch")

compat = r'''

def validate_sequence_visualization(
    topic: dict[str, object],
    *,
    topic_id: str,
) -> None:
    """Validate the retired sequence payload shape for compatibility tests/tools."""

    visualization = topic.get("visualization")
    if not isinstance(visualization, dict) or visualization.get("kind") != "sequence":
        raise GuideContractError(f"topic {topic_id} has no sequence visualization")

    raw_actors = visualization.get("actors")
    raw_messages = visualization.get("messages")
    if not isinstance(raw_actors, list) or not raw_actors:
        raise GuideContractError(f"topic {topic_id} sequence actors must be non-empty")
    if not isinstance(raw_messages, list):
        raise GuideContractError(f"topic {topic_id} sequence messages must be a list")

    code_reference_ids = {
        reference["id"]
        for reference in _topic_code_references(topic, topic_id=topic_id)
    }
    actor_ids: list[str] = []
    for index, actor in enumerate(raw_actors):
        if not isinstance(actor, dict):
            raise GuideContractError(f"topic {topic_id} has malformed sequence actor")
        actor_id = _required_string(
            actor,
            "id",
            label=f"topic {topic_id} sequence actor {index}",
        )
        _required_string(
            actor,
            "label_ja",
            label=f"topic {topic_id} sequence actor {actor_id}",
        )
        actor_ids.append(actor_id)
        code_ref = actor.get("code_ref")
        if code_ref is not None:
            if not isinstance(code_ref, str) or not code_ref:
                raise GuideContractError(
                    f"topic {topic_id} sequence actor {actor_id} has malformed code reference"
                )
            if code_ref not in code_reference_ids:
                raise GuideContractError(
                    f"unknown code reference in topic {topic_id} sequence actor: {code_ref}"
                )
    if len(set(actor_ids)) != len(actor_ids):
        raise GuideContractError(f"topic {topic_id} sequence has duplicate actor ids")
    actor_id_set = set(actor_ids)

    message_ids: list[str] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise GuideContractError(f"topic {topic_id} has malformed sequence message")
        message_id = _required_string(
            message,
            "id",
            label=f"topic {topic_id} sequence message {index}",
        )
        message_ids.append(message_id)
    if len(set(message_ids)) != len(message_ids):
        raise GuideContractError(f"topic {topic_id} sequence has duplicate message ids")

    for index, message in enumerate(raw_messages):
        assert isinstance(message, dict)
        message_id = message_ids[index]
        source = _required_string(
            message,
            "from",
            label=f"topic {topic_id} sequence message {message_id}",
        )
        target = _required_string(
            message,
            "to",
            label=f"topic {topic_id} sequence message {message_id}",
        )
        _required_string(
            message,
            "label_ja",
            label=f"topic {topic_id} sequence message {message_id}",
        )
        if source not in actor_id_set or target not in actor_id_set:
            raise GuideContractError(
                f"unknown actor reference in topic {topic_id} sequence: {source} -> {target}"
            )
        code_ref = message.get("code_ref")
        if code_ref is not None:
            if not isinstance(code_ref, str) or not code_ref:
                raise GuideContractError(
                    f"topic {topic_id} sequence message {message_id} has malformed code reference"
                )
            if code_ref not in code_reference_ids:
                raise GuideContractError(
                    f"unknown code reference in topic {topic_id} sequence message: {code_ref}"
                )
        state_changes = message.get("state_changes_ja")
        if state_changes is not None and (
            not isinstance(state_changes, list)
            or not all(isinstance(item, str) and item for item in state_changes)
        ):
            raise GuideContractError(
                f"topic {topic_id} sequence message {message_id} has malformed state changes"
            )


def validate_code_map_visualization(
    topic: dict[str, object],
    *,
    topic_id: str,
) -> None:
    """Validate the retired code-map payload shape for compatibility tests/tools."""

    visualization = topic.get("visualization")
    if not isinstance(visualization, dict) or visualization.get("kind") != "code-map":
        raise GuideContractError(f"topic {topic_id} has no code-map visualization")

    raw_nodes = visualization.get("nodes")
    raw_edges = visualization.get("edges")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise GuideContractError(f"topic {topic_id} code-map nodes must be non-empty")
    if not isinstance(raw_edges, list):
        raise GuideContractError(f"topic {topic_id} code-map edges must be a list")

    code_reference_ids = {
        reference["id"]
        for reference in _topic_code_references(topic, topic_id=topic_id)
    }
    node_ids: list[str] = []
    for index, node in enumerate(raw_nodes):
        if not isinstance(node, dict):
            raise GuideContractError(f"topic {topic_id} has malformed code-map node")
        node_id = _required_string(
            node,
            "id",
            label=f"topic {topic_id} code-map node {index}",
        )
        _required_string(
            node,
            "label_ja",
            label=f"topic {topic_id} code-map node {node_id}",
        )
        _required_string(
            node,
            "description_ja",
            label=f"topic {topic_id} code-map node {node_id}",
        )
        node_ids.append(node_id)
        code_ref = node.get("code_ref")
        if code_ref is not None:
            if not isinstance(code_ref, str) or not code_ref:
                raise GuideContractError(
                    f"topic {topic_id} code-map node {node_id} has malformed code reference"
                )
            if code_ref not in code_reference_ids:
                raise GuideContractError(
                    f"unknown code reference in topic {topic_id} code-map node: {code_ref}"
                )
    if len(set(node_ids)) != len(node_ids):
        raise GuideContractError(f"topic {topic_id} code-map has duplicate node ids")
    node_id_set = set(node_ids)

    indegree = {node_id: 0 for node_id in node_ids}
    outgoing = {node_id: [] for node_id in node_ids}
    for index, edge in enumerate(raw_edges):
        if not isinstance(edge, dict):
            raise GuideContractError(f"topic {topic_id} has malformed code-map edge")
        source = _required_string(
            edge,
            "from",
            label=f"topic {topic_id} code-map edge {index}",
        )
        target = _required_string(
            edge,
            "to",
            label=f"topic {topic_id} code-map edge {index}",
        )
        relation = _required_string(
            edge,
            "relation",
            label=f"topic {topic_id} code-map edge {index}",
        )
        _required_string(
            edge,
            "label_ja",
            label=f"topic {topic_id} code-map edge {index}",
        )
        if source not in node_id_set or target not in node_id_set:
            raise GuideContractError(
                f"unknown node reference in topic {topic_id} code-map: {source} -> {target}"
            )
        if relation not in _ALLOWED_CODE_MAP_RELATIONS:
            raise GuideContractError(
                f"unknown code-map relation in topic {topic_id}: {relation}"
            )
        outgoing[source].append(target)
        indegree[target] += 1

    ready = [node_id for node_id in node_ids if indegree[node_id] == 0]
    visited = 0
    while ready:
        node_id = ready.pop(0)
        visited += 1
        for target in outgoing[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(node_ids):
        raise GuideContractError(f"topic {topic_id} code-map contains a cycle")


def _code_reference_test_path(root: Path, path_text: str) -> Path:
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != "tests"
        or path.suffix != ".py"
    ):
        raise GuideContractError(f"invalid code reference test path: {path_text}")
    resolved_root = root.resolve()
    resolved = (root / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise GuideContractError(f"invalid code reference test path: {path_text}") from exc
    if not resolved.is_file():
        raise GuideContractError(f"invalid code reference test path: {path_text}")
    return resolved


def validate_code_reference(
    root: Path,
    reference: dict[str, object],
    symbols: dict[str, dict[str, object]],
    *,
    check_digest: bool = True,
) -> None:
    """Validate one code reference independently of the active Guide content layout."""

    raw = dict(reference)
    reference_id = _required_string(raw, "id", label="code reference")
    symbol_name = _required_string(raw, "symbol", label=f"code reference {reference_id}")
    expected_kind = _required_string(raw, "kind", label=f"code reference {reference_id}")
    if expected_kind not in _ALLOWED_CODE_KINDS:
        raise GuideContractError(f"invalid code symbol kind: {expected_kind}")
    _required_string(raw, "label_ja", label=f"code reference {reference_id}")
    _required_string(raw, "description_ja", label=f"code reference {reference_id}")

    symbol = symbols.get(symbol_name)
    if symbol is None:
        raise GuideContractError(f"missing code symbol: {symbol_name}")
    actual_kind = symbol.get("kind")
    if actual_kind != expected_kind:
        raise GuideContractError(
            f"code symbol kind mismatch: {symbol_name}: expected {expected_kind}, actual {actual_kind}"
        )

    expected_digest = raw.get("source_sha256")
    if not isinstance(expected_digest, str) or not _HEX_64_RE.fullmatch(expected_digest):
        raise GuideContractError(f"invalid code symbol fingerprint: {symbol_name}")
    actual_digest = symbol.get("source_sha256")
    if check_digest and actual_digest != expected_digest:
        raise GuideContractError(
            f"stale code symbol fingerprint: {symbol_name}: "
            f"expected {expected_digest}, actual {actual_digest}"
        )

    local_names = symbol.get("local_names")
    if not isinstance(local_names, list) or not all(
        isinstance(name, str) for name in local_names
    ):
        raise GuideContractError(f"malformed code symbol local names: {symbol_name}")
    allowed_names = set(local_names)
    raw_variables = raw.get("variables", [])
    if not isinstance(raw_variables, list):
        raise GuideContractError(f"code reference {reference_id} variables must be a list")
    variable_names: list[str] = []
    for variable in raw_variables:
        if not isinstance(variable, dict):
            raise GuideContractError(f"code reference {reference_id} has malformed variable")
        name = _required_string(variable, "name", label=f"code reference {reference_id} variable")
        _required_string(variable, "label_ja", label=f"code reference {reference_id} variable")
        _required_string(
            variable,
            "description_ja",
            label=f"code reference {reference_id} variable",
        )
        variable_names.append(name)
        if name not in allowed_names:
            raise GuideContractError(f"unknown code variable: {symbol_name}.{name}")
    if len(set(variable_names)) != len(variable_names):
        raise GuideContractError(f"code reference {reference_id} has duplicate variables")

    raw_tests = raw.get("tests", [])
    if not isinstance(raw_tests, list) or not all(
        isinstance(test_path, str) and test_path for test_path in raw_tests
    ):
        raise GuideContractError(f"code reference {reference_id} tests must be string paths")
    for test_path in raw_tests:
        _code_reference_test_path(root, test_path)


def _symbol_map_for_root(root: Path) -> dict[str, dict[str, object]]:
    code_symbols = _code_symbols_module()
    index = code_symbols.build_symbol_index(root / "trade_rl", revision="0" * 40)
    raw_symbols = index.get("symbols")
    if not isinstance(raw_symbols, list):
        raise GuideContractError("code symbol index has malformed symbols")
    symbols: dict[str, dict[str, object]] = {}
    for raw in raw_symbols:
        if not isinstance(raw, dict):
            raise GuideContractError("code symbol index has malformed entry")
        name = raw.get("qualified_name")
        if not isinstance(name, str) or not name:
            raise GuideContractError("code symbol index has malformed qualified name")
        symbols[name] = raw
    return symbols


def refresh_code_references(topic_ids: list[str], root: Path = ROOT) -> None:
    """Refresh only source digests, supporting v2 metadata and legacy temp fixtures."""

    if not topic_ids:
        raise GuideContractError("refresh-code requires at least one explicit topic id")

    meta_dir = root / "guide" / "content" / "meta"
    legacy_dir = root / "guide" / "content" / "topics"
    records_dir = meta_dir if meta_dir.is_dir() else legacy_dir
    symbols = _symbol_map_for_root(root)
    for topic_id in topic_ids:
        topic_path = records_dir / f"{topic_id}.json"
        if not topic_path.is_file():
            raise GuideContractError(f"missing topic for refresh-code: {topic_id}")
        topic = _read_json(topic_path)
        if topic.get("id") != topic_id:
            raise GuideContractError(f"topic id mismatch: {topic_path}")
        references = _topic_code_references(topic, topic_id=topic_id)
        for reference in references:
            validate_code_reference(
                root,
                reference,
                symbols,
                check_digest=False,
            )
            symbol_name = str(reference["symbol"])
            reference["source_sha256"] = symbols[symbol_name]["source_sha256"]
        topic["code_references"] = references
        topic_path.write_text(
            json.dumps(topic, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
'''

text = text.replace(marker, compat + marker, 1)
path.write_text(text, encoding="utf-8")
