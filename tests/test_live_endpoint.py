from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.live_endpoint import EndpointSettings, capture_utterance


def fixture_stream(blocks):
    port = MagicMock()
    stream = port.InputStream.return_value.__enter__.return_value
    stream.read.side_effect = [(np.full((512, 1), level, dtype=np.float32), False) for level in blocks]
    vad = MagicMock()
    return port, stream, vad


def test_endpoint_keeps_preroll_and_internal_pause_then_trims_trailing_silence():
    port, stream, vad = fixture_stream([0.01, 0.02, 0.3, 0, 0.3, 0, 0, 0, 0.9])
    vad.is_vad.side_effect = [None, None, {"start": 1024}, None, None, None, None, {"end": 2560}]
    settings = EndpointSettings(silence=0.096)
    started = MagicMock()
    result = capture_utterance(port, 1, 16000, vad, settings, started)

    assert result.reason == "pause"
    assert result.endpoint_delay == 0.096
    np.testing.assert_allclose(result.samples[::512], [0.01, 0.02, 0.3, 0, 0.3])
    assert stream.read.call_count == 8
    started.assert_called_once()
    vad.reset_states.assert_called_once()
    port.InputStream.return_value.__exit__.assert_called_once()


def test_background_without_speech_stops_at_timeout():
    port, stream, vad = fixture_stream([0.02] * 4)
    vad.is_vad.return_value = None
    result = capture_utterance(port, 1, 16000, vad, EndpointSettings(no_speech=0.096), MagicMock())
    assert result.reason == "no speech" and result.samples.size == 0
    assert stream.read.call_count == 3


def test_maximum_duration_is_reported_separately():
    port, stream, vad = fixture_stream([0.2] * 5)
    vad.is_vad.side_effect = [{"start": 0}, None, None, None]
    result = capture_utterance(port, 1, 16000, vad, EndpointSettings(max_utterance=0.096), MagicMock())
    assert result.reason == "maximum duration" and result.endpoint_delay is None
    assert len(result.samples) == 1536


def test_real_silero_iterator_ignores_short_pause_with_deterministic_scores():
    silero = pytest.importorskip("silero_vad")
    import torch
    from bailing.vad import SileroVAD

    class FixtureModel:
        def reset_states(self):
            pass

        def __call__(self, audio, rate):
            return torch.tensor(0.9 if audio.abs().max() > 0.1 else 0.05)

    port, stream, _ = fixture_stream([0.02, 0.02, 0.3, 0, 0.3, 0, 0, 0, 0])
    vad = object.__new__(SileroVAD)
    vad.vad_iterator = silero.VADIterator(FixtureModel(), sampling_rate=16000, min_silence_duration_ms=96)
    result = capture_utterance(port, 1, 16000, vad, EndpointSettings(silence=0.096), MagicMock())

    assert result.reason == "pause" and stream.read.call_count == 9
    np.testing.assert_allclose(result.samples[::512][:5], [0.02, 0.02, 0.3, 0, 0.3])
