import sys
from unittest.mock import Mock

import numpy as np
import pytest

from bailing.asr import WhisperASR
from scripts import verify_live
from scripts.verify_live import AudioDevices, capture_frames, pcm16_chunks, run_session


def test_whisper_in_memory_mode_never_writes_capture(tmp_path):
    asr = object.__new__(WhisperASR)
    asr.output_dir = tmp_path
    asr.pipe = Mock(return_value={"text": "Hei", "chunks": [{"language": "finnish"}]})
    assert asr.recognizer([b"\x01\x00"], keep_wav=False, raise_on_error=True) == ("Hei", None, "fi")
    assert not list(tmp_path.iterdir())
    asr.pipe.side_effect = RuntimeError("ASR failed")
    with pytest.raises(RuntimeError, match="ASR failed"):
        asr.recognizer([b"\x01\x00"], keep_wav=False, raise_on_error=True)


def test_capture_closes_stream_and_reads_exact_frame_budget():
    stream = Mock()
    stream.read.side_effect = lambda count: (np.ones((count, 1), dtype=np.float32), False)
    port = Mock()
    port.InputStream.return_value.__enter__ = Mock(return_value=stream)
    close = port.InputStream.return_value.__exit__ = Mock(return_value=False)

    result = capture_frames(port, 7, 48000, 0.051)

    assert len(result) == round(48000 * 0.051)
    assert sum(call.args[0] for call in stream.read.call_args_list) == len(result)
    port.InputStream.assert_called_once_with(device=7, samplerate=48000, channels=1, dtype="float32")
    close.assert_called_once()


def test_capture_overflow_discards_audio_and_closes_stream():
    stream = Mock()
    stream.read.return_value = (np.zeros((1024, 1)), True)
    port = Mock()
    port.InputStream.return_value.__enter__ = Mock(return_value=stream)
    close = port.InputStream.return_value.__exit__ = Mock(return_value=False)
    with pytest.raises(RuntimeError, match="overflow"):
        capture_frames(port, 1, 48000, 8)
    close.assert_called_once()


def test_capture_conversion_produces_16khz_signed_pcm():
    samples = np.full(4800, -0.25, dtype=np.float32)
    chunks = pcm16_chunks(samples, 48000)
    pcm = np.frombuffer(b"".join(chunks), dtype="<i2")
    assert [len(chunk) for chunk in chunks] == [640] * 5
    assert len(pcm) == 1600 and abs(int(pcm[800]) - (-8192)) < 16


def test_missing_audio_device_fails_before_opening_stream():
    port = Mock()
    port.default.device = (-1, 4)
    with pytest.raises(ValueError, match="Missing input or output device"):
        AudioDevices(port, None, None)
    port.query_devices.assert_not_called()


@pytest.mark.parametrize("override_paths", [False, True])
def test_live_check_uses_selected_devices_and_model_paths(tmp_path, monkeypatch, override_paths):
    cache = tmp_path / "whisper cache"
    voices = tmp_path / "piper voices"
    monkeypatch.chdir(tmp_path)
    args = []
    if override_paths:
        args = ["--whisper-cache-dir", str(cache), "--piper-model-dir", str(voices)]
    else:
        monkeypatch.setattr(verify_live, "CACHE_DIR", cache)
        voices = tmp_path / "models" / "tts" / "piper"
    snapshot = cache / "models--openai--whisper-small" / "snapshots" / verify_live.WHISPER_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").touch()
    voices.mkdir(parents=True)
    for voice in ("en_US-ryan-medium", "fi_FI-harri-medium"):
        for extension in (".onnx", ".onnx.json"):
            (voices / (voice + extension)).touch()
    port = Mock()
    port.query_devices.return_value = {"default_samplerate": 48000, "max_output_channels": 2}
    monkeypatch.setitem(sys.modules, "sounddevice", port)
    monkeypatch.setattr(sys, "argv", ["verify_live", "--check", "--seconds", "8",
                                    "--input-device", "7", "--output-device", "9", *args])
    monkeypatch.setattr(verify_live, "check_ollama", Mock())
    monkeypatch.setattr("bailing.llm.OpenAILLM", Mock())
    asr = Mock()
    monkeypatch.setattr(verify_live, "WhisperASR", asr)

    assert verify_live.main() == 0
    asr.assert_called_once_with({"model_dir": str(snapshot)})
    port.check_input_settings.assert_called_once_with(device=7, channels=1, samplerate=48000, dtype="float32")
    port.check_output_settings.assert_called_once_with(device=9, channels=2, samplerate=48000, dtype="float32")
    port.InputStream.assert_not_called()
    port.play.assert_not_called()


def test_strict_llm_propagates_stream_failure(monkeypatch):
    pytest.importorskip("openai")
    from bailing.llm import OpenAILLM

    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("connection dropped")
    monkeypatch.setattr("bailing.llm.openai.OpenAI", lambda **_: client)
    with pytest.raises(RuntimeError, match="connection dropped"):
        list(OpenAILLM({"raise_on_error": True}).response([]))
    assert list(OpenAILLM({}).response([])) == []


def test_routes_two_turns_and_waits_for_playback_before_accepting_enter(tmp_path, monkeypatch):
    events = []
    prompts = iter(["", "", "/reset", "/quit"])
    audio, asr, conversation, piper = (Mock() for _ in range(4))
    conversation.history = [{"role": "user", "content": "old turn"}]
    conversation.reset.side_effect = conversation.history.clear
    audio.capture.side_effect = lambda _: events.append("capture") or np.array([0.1, -0.1])
    monkeypatch.setattr("scripts.verify_live.pcm16_chunks", lambda *args: [b"audio"])
    asr.recognizer.side_effect = [("Hello", None, "en"), ("Hei", None, "fi")]
    conversation.ask.side_effect = ["Hello there", "Hyvää päivää"]
    times = iter(["en.wav", "fi.wav"])

    def synthesize(*args, **kwargs):
        wav = tmp_path / next(times)
        wav.touch()
        return wav

    piper.to_tts.side_effect = synthesize
    audio.play.side_effect = lambda _: events.append("play-complete")

    def read(_):
        events.append("command")
        return next(prompts)

    run_session(audio, asr, conversation, piper, 8, read=read)

    assert events == ["command", "capture", "play-complete", "command", "capture",
                      "play-complete", "command", "command"]
    assert [call.args for call in conversation.ask.call_args_list] == [("Hello", "en"), ("Hei", "fi")]
    assert [call.args for call in piper.to_tts.call_args_list] == [("Hello there", "en"), ("Hyvää päivää", "fi")]
    assert not list(tmp_path.iterdir()) and conversation.history == []


def test_silence_unknown_language_and_service_failure_stay_visible(monkeypatch, capsys):
    audio, asr, conversation, piper = (Mock() for _ in range(4))
    audio.capture.side_effect = [np.zeros(10), *[np.array([0.1, -0.1])] * 3]
    monkeypatch.setattr("scripts.verify_live.pcm16_chunks", lambda *args: [b"audio"])
    asr.recognizer.side_effect = [("", None, ""), ("你好", None, "zh"), ("Hello", None, "en")]
    asr.last_detected_language = "chinese"
    conversation.routing_diagnostics = True
    conversation.ask.side_effect = RuntimeError("Ollama unavailable")
    commands = iter(["", "", "", "", "/quit"])
    run_session(audio, asr, conversation, piper, 8, read=lambda _: next(commands))

    output = capsys.readouterr().out
    for message in ("silence", "no transcript", "unsupported language", "Ollama unavailable"):
        assert message in output
    assert "raw='chinese' | transcript present: yes" in output
    assert "raw='chinese' | transcript present: no" in output
    assert asr.recognizer.call_count == 3
    piper.to_tts.assert_not_called()
    audio.play.assert_not_called()


def test_live_labels_local_knowledge_and_actual_llm_time_separately(tmp_path, monkeypatch, capsys):
    from scripts.savonia_knowledge import SavoniaKnowledge
    from scripts.verify_text import TextConversation

    audio, asr, llm, piper = (Mock() for _ in range(4))
    audio.capture.return_value = np.array([0.1, -0.1])
    asr.recognizer.side_effect = [("Savonia", None, "en"), ("Hello", None, "en")]
    llm.response.return_value = ["Hello."]
    monkeypatch.setattr("scripts.verify_live.pcm16_chunks", lambda *args: [b"audio"])
    piper.to_tts.return_value = tmp_path / "reply.wav"
    conversation = TextConversation(llm, knowledge=SavoniaKnowledge())
    commands = iter(["", "", "/quit"])
    run_session(audio, asr, conversation, piper, 8, read=lambda _: next(commands))

    output = capsys.readouterr().out
    assert "Knowledge routing (no LLM call):" in output
    assert output.count("\nLLM:") == 1
    llm.response.assert_called_once()
