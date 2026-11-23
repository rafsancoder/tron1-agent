from unittest.mock import Mock

import pytest

from scripts.verify_text import TextConversation


def test_routes_each_turn_and_keeps_history():
    llm = Mock()
    llm.response.side_effect = [iter([None, "I am ", "TRON 1."]), iter(["Olen TRON 1."])]
    conversation = TextConversation(llm)

    assert conversation.ask("What is your name?", "en") == "I am TRON 1."
    assert conversation.ask("Voitko toistaa sen?", "fi") == "Olen TRON 1."

    first, second = [call.args[0] for call in llm.response.call_args_list]
    assert "Reply in English" in first[0]["content"]
    assert "Reply in Finnish" in second[0]["content"]
    assert "TRON 1" in second[0]["content"]
    assert "ask a clarifying question" in second[0]["content"]
    assert "exact name TRON 1" in second[0]["content"]
    assert "without commenting on the user's wording or grammar" in second[0]["content"]
    assert first[1:] == [{"role": "user", "content": "What is your name?"}]
    assert second[1:] == [
        *first[1:],
        {"role": "assistant", "content": "I am TRON 1."},
        {"role": "user", "content": "Voitko toistaa sen?"},
    ]


@pytest.mark.parametrize(("question", "language"), [("Hello", "zh"), ("  ", "fi")])
def test_invalid_input_never_reaches_adapter(question, language):
    llm = Mock()
    with pytest.raises(ValueError):
        TextConversation(llm).ask(question, language)
    llm.response.assert_not_called()


def test_empty_model_reply_does_not_enter_history():
    llm = Mock()
    llm.response.side_effect = [iter([None, " "]), iter(["Mitä tarkoitat?"])]
    conversation = TextConversation(llm)

    with pytest.raises(RuntimeError, match="empty reply"):
        conversation.ask("First question", "en")

    assert conversation.ask("Entä se?", "fi") == "Mitä tarkoitat?"
    assert llm.response.call_args.args[0][1:] == [{"role": "user", "content": "Entä se?"}]
