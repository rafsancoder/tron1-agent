"""Synthesize an English or Finnish WAV with the existing Piper adapter."""

import argparse
from pathlib import Path
import sys

from bailing.tts import PiperTTS


def bilingual_piper(model_dir, output_dir, *, length_scale=1.0, sentence_silence=0.35):
    return PiperTTS({
        "model_dir": model_dir,
        "output_file": output_dir,
        "voice_en": "en_US-ryan-medium",
        "voice_fi": "fi_FI-harri-medium",
        "length_scale": length_scale,
        "sentence_silence": sentence_silence,
        "piper_command": [sys.executable, "-m", "piper"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=("en", "fi"))
    parser.add_argument("text", help="Text to speak, quoted if it contains spaces")
    parser.add_argument("--model-dir", type=Path, default=Path("models/tts/piper"))
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/tts-verification"))
    parser.add_argument("--length-scale", type=float, default=1.0, help="Piper phoneme duration scale (default: 1.0)")
    parser.add_argument("--sentence-silence", type=float, default=0.35, help="Seconds between sentences (default: 0.35)")
    args = parser.parse_args()

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        adapter = bilingual_piper(args.model_dir, args.output_dir,
                                  length_scale=args.length_scale, sentence_silence=args.sentence_silence)
        wav = adapter.to_tts(args.text, args.language, raise_on_error=True)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(Path(wav).resolve())


if __name__ == "__main__":
    main()
