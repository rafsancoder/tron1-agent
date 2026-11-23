from pathlib import Path
from unittest.mock import Mock, call

import pytest

from scripts.verify_conversation import run_recordings
from scripts.verify_text import TextConversation


def test_recordings_route_in_order_with_clean_history_and_matching_voices(monkeypatch, capsys):
    monkeypatch.setattr("scripts.verify_conversation.read_wav_chunks", lambda path: [path.name.encode()])
    asr = Mock()
    asr.recognizer.side_effect = [("", None, ""), ("你好", "input.wav", "zh"),
                                  ("What is your name?", "input.wav", "en"), ("Entä nyt?", "input.wav", "fi")]
    llm = Mock()
    llm.response.side_effect = [["<thi", "nk>private</think>", "I am TRON 1.<|im_end|>"], ["Olen TRON 1."]]
    tts = Mock()
    tts.to_tts.side_effect = ["en.wav", "fi.wav"]
    monkeypatch.setattr("scripts.verify_conversation.wav_duration", lambda _: 1.25)

    assert run_recordings(list(map(Path, ("empty.wav", "zh.wav", "en.wav", "fi.wav"))),
                          asr, TextConversation(llm, recorded_audio=True), tts) == 0
    assert tts.to_tts.call_args_list == [call("I am TRON 1.", "en", raise_on_error=True),
                                        call("Olen TRON 1.", "fi", raise_on_error=True)]
    first, second = [entry.args[0] for entry in llm.response.call_args_list]
    assert "Reply in English" in first[0]["content"]
    assert "Reply in Finnish" in second[0]["content"]
    assert second[1:-1] == [first[1], {"role": "assistant", "content": "I am TRON 1."}]
    output = capsys.readouterr().out
    assert "empty recognition" in output and "unsupported language (zh)" in output
    assert "Processing total" in output and "Playback duration total" in output
    assert "private" not in output


@pytest.mark.parametrize("reply", ["<think>unfinished", "private</think>Text", "Hello <|im_", "<think>only reasoning</think>"])
def test_non_speakable_reply_never_reaches_tts(reply, monkeypatch, capsys):
    monkeypatch.setattr("scripts.verify_conversation.read_wav_chunks", lambda _: [b"audio"])
    asr = Mock()
    asr.recognizer.return_value = ("Hello", "input.wav", "en")
    llm = Mock()
    llm.response.return_value = [reply]
    conversation = TextConversation(llm)
    tts = Mock()
    assert run_recordings([Path("input.wav")], asr, conversation, tts) == 1
    tts.to_tts.assert_not_called()
    assert conversation.history == []
    assert "LLM failed" in capsys.readouterr().out
