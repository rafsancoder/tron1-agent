"""Verify WhisperASR in isolation with a WAV recording."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import soundfile as sf

from bailing.asr import WhisperASR


MODEL_ID = "openai/whisper-small"
CACHE_DIR = Path(__file__).resolve().parents[1] / "models" / "huggingface"
SAMPLE_RATE = 16000
CHUNK_BYTES = 640  # 20 ms of mono signed 16-bit PCM.


def read_wav_chunks(path):
    with sf.SoundFile(path) as recording:
        if recording.format not in {"WAV", "WAVEX", "RF64"}:
            raise ValueError("Expected a WAV file.")
        sample_rate = recording.samplerate
        samples = recording.read(dtype="float32", always_2d=True)

    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("The WAV must contain nonempty, finite audio samples.")

    return pcm16_chunks(samples.mean(axis=1), sample_rate)


def resample_mono(samples, source_rate, target_rate):
    if source_rate == target_rate:
        return samples
    import torch
    from torchaudio.functional import resample

    return resample(torch.from_numpy(samples), source_rate, target_rate).numpy()


def pcm16_chunks(samples, sample_rate):
    mono = resample_mono(samples, sample_rate, SAMPLE_RATE)
    pcm = np.clip(np.rint(mono * 32768), -32768, 32767).astype("<i2").tobytes()
    return [pcm[offset : offset + CHUNK_BYTES] for offset in range(0, len(pcm), CHUNK_BYTES)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="Path to a WAV recording")
    parser.add_argument("--revision", default="main", help="Cached Hugging Face revision or commit SHA")
    args = parser.parse_args()

    try:
        chunks = read_wav_chunks(args.wav)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(f"Could not read {args.wav}: {error}")

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot = Path(
            snapshot_download(MODEL_ID, revision=args.revision, cache_dir=CACHE_DIR, local_files_only=True)
        )
    except LocalEntryNotFoundError:
        parser.error(f"No cached {MODEL_ID} revision {args.revision!r} in {CACHE_DIR}.")
    if not (snapshot / "model.safetensors").is_file():
        parser.error(f"The cached {MODEL_ID} snapshot {snapshot.name} is missing model.safetensors.")

    with TemporaryDirectory(prefix="tron1-whisper-") as output_dir:
        adapter = WhisperASR({"model_dir": str(snapshot), "output_file": output_dir})
        start = perf_counter()
        text, _, language = adapter.recognizer(chunks)
        duration = perf_counter() - start

    print(f"Model: {MODEL_ID}")
    print(f"Revision: {snapshot.name}")
    print(f"Device: {adapter.pipe.device}")
    print(f"Inference duration: {duration:.3f} s")
    print(f"Detected language: {language or 'unknown'}")
    print(f"Recognized text: {text.strip()}")
    return 0 if text.strip() and language else 1


if __name__ == "__main__":
    raise SystemExit(main())
