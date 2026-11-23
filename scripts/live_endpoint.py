"""Native-rate microphone capture with the existing Silero VAD events."""

from collections import deque
from dataclasses import dataclass
from math import ceil

import numpy as np


VAD_RATE = 16000
VAD_FRAMES = 512


@dataclass(frozen=True)
class EndpointSettings:
    silence: float = 1.2
    no_speech: float = 8.0
    max_utterance: float = 30.0
    pre_roll: float = 0.25


@dataclass(frozen=True)
class CaptureResult:
    samples: np.ndarray
    reason: str
    endpoint_delay: float | None = None


def capture_utterance(port, device, rate, vad, settings, started):
    block_size = round(rate * VAD_FRAMES / VAD_RATE)
    block_seconds = block_size / rate
    preroll = deque(maxlen=ceil(settings.pre_roll / block_seconds) + 1)
    blocks = []
    seen = 0
    speech_start = None
    if rate != VAD_RATE:
        import torch
        from torchaudio.transforms import Resample

        resample = Resample(rate, VAD_RATE)

        def vad_samples(block):
            converted = resample(torch.from_numpy(block)).numpy()[:VAD_FRAMES]
            return np.pad(converted, (0, VAD_FRAMES - len(converted)))
    else:
        def vad_samples(block):
            return block

    vad.reset_states()
    with port.InputStream(device=device, samplerate=rate, channels=1, dtype="float32") as stream:
        while True:
            frames, overflow = stream.read(block_size)
            if overflow or len(frames) != block_size or not np.isfinite(frames).all():
                raise RuntimeError("Input overflow, short read, or invalid samples; recording was discarded")
            block = frames[:, 0]
            seen += 1
            pcm = np.clip(np.rint(vad_samples(block) * 32768), -32768, 32767).astype("<i2").tobytes()
            event = vad.is_vad(pcm, raise_on_error=True) or {}
            if speech_start is None:
                preroll.append(block)
                if "start" in event:
                    speech_start = (seen - 1) * block_seconds
                    blocks.extend(preroll)
                    started()
                elif seen * block_seconds >= settings.no_speech:
                    return CaptureResult(np.empty(0, dtype=np.float32), "no speech")
            else:
                blocks.append(block)

            if "end" in event and blocks:
                # Silero's end timestamp includes its small speech pad.
                delay = max(0, seen * VAD_FRAMES - event["end"]) / VAD_RATE
                captured = np.concatenate(blocks)
                keep = max(1, len(captured) - round(delay * VAD_RATE * block_size / VAD_FRAMES))
                return CaptureResult(captured[:keep], "pause", delay)
            if speech_start is not None and seen * block_seconds - speech_start >= settings.max_utterance:
                return CaptureResult(np.concatenate(blocks), "maximum duration")
