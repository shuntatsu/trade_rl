contract = __import__("guide.tools.content_contract", fromlist=["*"])


def _assert_contract_error(callable_object, message: str) -> None:
    try:
        callable_object()
    except contract.GuideContractError as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"expected GuideContractError containing {message!r}")


def _topic() -> dict[str, object]:
    return {
        "code_references": [
            {
                "id": "known",
                "symbol": "trade_rl.demo.run",
            }
        ],
        "visualization": {
            "kind": "code-map",
            "nodes": [
                {
                    "id": "data",
                    "label_ja": "データ",
                    "description_ja": "データ責務",
                    "code_ref": "known",
                },
                {
                    "id": "risk",
                    "label_ja": "リスク",
                    "description_ja": "リスク責務",
                },
            ],
            "edges": [
                {
                    "from": "data",
                    "to": "risk",
                    "relation": "data-flow",
                    "label_ja": "渡す",
                }
            ],
        },
    }


def _visualization(topic: dict[str, object]) -> dict[str, object]:
    value = topic["visualization"]
    assert isinstance(value, dict)
    return value


def test_code_map_contract_rejects_unknown_node_reference() -> None:
    topic = _topic()
    visualization = _visualization(topic)
    edges = visualization["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    edge["to"] = "missing"

    _assert_contract_error(
        lambda: contract.validate_code_map_visualization(topic, topic_id="demo"),
        "unknown node reference",
    )


def test_code_map_contract_rejects_unknown_relation_and_code_reference() -> None:
    topic = _topic()
    visualization = _visualization(topic)
    edges = visualization["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    edge["relation"] = "imports"
    _assert_contract_error(
        lambda: contract.validate_code_map_visualization(topic, topic_id="demo"),
        "unknown code-map relation",
    )

    topic = _topic()
    visualization = _visualization(topic)
    nodes = visualization["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["code_ref"] = "missing"
    _assert_contract_error(
        lambda: contract.validate_code_map_visualization(topic, topic_id="demo"),
        "unknown code reference",
    )


def test_code_map_contract_rejects_cycles() -> None:
    topic = _topic()
    visualization = _visualization(topic)
    edges = visualization["edges"]
    assert isinstance(edges, list)
    edges.append(
        {
            "from": "risk",
            "to": "data",
            "relation": "calls",
            "label_ja": "戻る",
        }
    )

    _assert_contract_error(
        lambda: contract.validate_code_map_visualization(topic, topic_id="demo"),
        "cycle",
    )
