import wave
from unittest.mock import Mock

import numpy as np
import pytest

from bailing.asr import WhisperASR


@pytest.fixture
def adapter(tmp_path):
    adapter = WhisperASR.__new__(WhisperASR)
    adapter.output_dir = tmp_path
    adapter.pipe = Mock(return_value={"text": "Hello"})
    return adapter


@pytest.mark.parametrize(
    ("transcript", "model_language", "expected_language"),
    [("Hello", "english", "en"), ("Moi", "finnish", "fi"), ("你好", "chinese", "zh")],
)
def test_recognizer_returns_text_path_and_language(
    adapter, transcript, model_language, expected_language
):
    adapter.pipe.return_value = {
        "text": transcript,
        "chunks": [{"text": transcript, "language": model_language}],
    }
    chunks = [b"\x00\x40", b"\x00\xc0"]
    text, path, language = adapter.recognizer(iter(chunks))

    assert text == transcript
    assert language == expected_language
    with wave.open(path, "rb") as recording:
        assert recording.getframerate() == 16000
        assert recording.readframes(2) == b"".join(chunks)

    adapter.pipe.assert_called_once()
    call = adapter.pipe.call_args
    np.testing.assert_array_equal(call.args[0], np.array([0.5, -0.5], dtype=np.float32))
    assert call.kwargs == {
        "return_language": True,
        "generate_kwargs": {
            "task": "transcribe",
            "forced_decoder_ids": None,
            "force_unique_generate_call": True,
        },
    }


def test_recognizer_does_not_guess_missing_language(adapter):
    text, _, language = adapter.recognizer([b"\x01\x00"])

    assert (text, language) == ("Hello", "")


@pytest.mark.parametrize("chunks", [[], [b""]])
def test_recognizer_skips_empty_audio(adapter, chunks):
    assert adapter.recognizer(chunks) == ("", None, "")
    adapter.pipe.assert_not_called()
    assert list(adapter.output_dir.iterdir()) == []


def test_recognizer_returns_empty_result_on_error(adapter):
    adapter.pipe.side_effect = RuntimeError("inference failed")

    assert adapter.recognizer([b"\x01\x00"]) == ("", None, "")
