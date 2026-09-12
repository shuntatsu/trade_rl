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
            "kind": "sequence",
            "actors": [
                {"id": "caller", "label_ja": "呼び出し元", "code_ref": "known"},
                {"id": "callee", "label_ja": "呼び出し先"},
            ],
            "messages": [
                {
                    "id": "call",
                    "from": "caller",
                    "to": "callee",
                    "label_ja": "処理を呼び出す",
                    "code_ref": "known",
                }
            ],
        },
    }


def test_sequence_contract_rejects_unknown_actor_reference() -> None:
    topic = _topic()
    visualization = topic["visualization"]
    assert isinstance(visualization, dict)
    messages = visualization["messages"]
    assert isinstance(messages, list)
    message = messages[0]
    assert isinstance(message, dict)
    message["to"] = "missing"

    _assert_contract_error(
        lambda: contract.validate_sequence_visualization(topic, topic_id="demo"),
        "unknown actor reference",
    )


def test_sequence_contract_rejects_unknown_code_reference() -> None:
    topic = _topic()
    visualization = topic["visualization"]
    assert isinstance(visualization, dict)
    actors = visualization["actors"]
    assert isinstance(actors, list)
    actor = actors[0]
    assert isinstance(actor, dict)
    actor["code_ref"] = "missing"

    _assert_contract_error(
        lambda: contract.validate_sequence_visualization(topic, topic_id="demo"),
        "unknown code reference",
    )


def test_sequence_contract_rejects_duplicate_message_ids() -> None:
    topic = _topic()
    visualization = topic["visualization"]
    assert isinstance(visualization, dict)
    messages = visualization["messages"]
    assert isinstance(messages, list)
    messages.append(dict(messages[0]))

    _assert_contract_error(
        lambda: contract.validate_sequence_visualization(topic, topic_id="demo"),
        "duplicate message ids",
    )
