from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock

import pytest
import soundfile as sf

from bailing.tts import PiperTTS


def test_voice_mappings_preserve_defaults_and_allow_finnish(tmp_path):
    defaults = PiperTTS({"model_dir": tmp_path})
    assert {lang: Path(path).stem for lang, path in defaults.model_path.items()} == {
        "en": "en_US-amy-medium", "zh": "zh_CN-huayan-medium"
    }
    configured = PiperTTS({"model_dir": tmp_path, "voice_en": "custom-en",
                           "voice_zh": "custom-zh", "voice_fi": "fi_FI-harri-medium"})
    assert {lang: Path(path).stem for lang, path in configured.model_path.items()} == {
        "en": "custom-en", "zh": "custom-zh", "fi": "fi_FI-harri-medium"
    }


@pytest.fixture
def piper(tmp_path):
    adapter = PiperTTS({"model_dir": tmp_path, "output_file": tmp_path,
                        "voice_fi": "fi_FI-harri-medium"})
    voice = Path(adapter.model_path["fi"])
    voice.touch()
    voice.with_suffix(".onnx.json").touch()
    return adapter


def test_finnish_synthesis_preserves_utf8(piper, monkeypatch):
    def run(command, **kwargs):
        assert command[command.index("--model") + 1] == piper.model_path["fi"]
        assert kwargs["input"] == "Hyvää päivää! Pöllö syö yöllä."
        assert kwargs["encoding"] == "utf-8"
        sf.write(command[command.index("--output_file") + 1], [0.1, 0.2], 22050)
        return CompletedProcess(command, 0, stderr="")

    monkeypatch.setattr("bailing.tts.subprocess.run", run)
    output = piper.to_tts("Hyvää päivää! Pöllö syö yöllä.", "fi", raise_on_error=True)
    assert sf.info(output).frames == 2


def test_missing_voice_config_reports_path(piper, monkeypatch):
    config = Path(piper.model_path["fi"] + ".json")
    config.unlink()
    run = Mock()
    monkeypatch.setattr("bailing.tts.subprocess.run", run)
    with pytest.raises(FileNotFoundError, match="fi_FI-harri-medium.onnx.json"):
        piper.to_tts("Hei", "fi", raise_on_error=True)
    run.assert_not_called()


def test_failure_reports_piper_error_and_preserves_default_return(piper, monkeypatch):
    monkeypatch.setattr("bailing.tts.subprocess.run", Mock(return_value=CompletedProcess([], 9, stderr="bad voice")))
    with pytest.raises(RuntimeError, match="exit 9.*bad voice"):
        piper.to_tts("Hei", "fi", raise_on_error=True)
    assert piper.to_tts("Hei", "fi") is None


@pytest.mark.parametrize("value", [0, -0.5, float("inf"), float("nan")])
def test_invalid_length_scales(value):
    with pytest.raises(ValueError, match="length_scale"):
        PiperTTS({"length_scale": value})


@pytest.mark.parametrize("value", [-0.5, float("inf"), float("nan")])
def test_invalid_sentence_silence(value):
    with pytest.raises(ValueError, match="sentence_silence"):
        PiperTTS({"sentence_silence": value})


@pytest.mark.parametrize("config, scale, silence", [({}, None, "1.5"),
                                                  ({"length_scale": 0.9, "sentence_silence": 0.35}, "0.9", "0.35")])
def test_piper_duration_and_pause_arguments(config, scale, silence, tmp_path, monkeypatch):
    adapter = PiperTTS({"model_dir": tmp_path, "output_file": tmp_path, **config})
    voice = Path(adapter.model_path["en"])
    voice.touch()
    voice.with_suffix(".onnx.json").touch()

    def run(command, **kwargs):
        assert command[command.index("--sentence_silence") + 1] == silence
        if scale is None:
            assert "--length_scale" not in command
        else:
            assert command[command.index("--length_scale") + 1] == scale
        assert command[command.index("--noise_w") + 1] == "1.2"
        sf.write(command[command.index("--output_file") + 1], [0.1, 0.2], 22050)
        return CompletedProcess(command, 0, stderr="")

    monkeypatch.setattr("bailing.tts.subprocess.run", run)
    assert adapter.to_tts("Hello", "en", raise_on_error=True)


def test_verifier_selects_bilingual_profile(monkeypatch, tmp_path):
    from scripts import verify_tts

    adapter_class = Mock()
    adapter_class.return_value.to_tts.return_value = str(tmp_path / "spoken.wav")
    monkeypatch.setattr(verify_tts, "PiperTTS", adapter_class)
    monkeypatch.setattr("sys.argv", ["verify_tts", "--output-dir", str(tmp_path),
                                    "--length-scale", "0.91", "en", "Hello"])
    verify_tts.main()

    config = adapter_class.call_args.args[0]
    assert config["voice_en"] == "en_US-ryan-medium"
    assert config["voice_fi"] == "fi_FI-harri-medium"
    assert config["length_scale"] == 0.91
    assert config["sentence_silence"] == 0.35
    adapter_class.return_value.to_tts.assert_called_once_with("Hello", "en", raise_on_error=True)
