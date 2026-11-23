"""Verify bilingual conversation from recorded WAV files using local models."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from time import perf_counter
import wave

from scripts.verify_text import LANGUAGES, TextConversation
from scripts.verify_tts import bilingual_piper
from scripts.verify_whisper import CACHE_DIR, MODEL_ID, WhisperASR, read_wav_chunks


WHISPER_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
OLLAMA_MODEL = "qwen3:4b"


def wav_duration(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def run_recordings(paths, asr, conversation, tts):
    started = perf_counter()
    playback_total = 0.0
    produced = 0
    failed = False
    for path in paths:
        print(f"\nRecording: {path}", flush=True)
        turn_start = perf_counter()
        stage = "ASR"
        try:
            transcript, _, language = asr.recognizer(read_wav_chunks(path))
            asr_end = perf_counter()
            print(f"ASR: {asr_end - turn_start:.3f} s")
            if not (transcript or "").strip():
                print("Skipped: empty recognition")
                continue
            print(f"Transcript: {transcript.strip()}\nLanguage: {language or 'unknown'}")
            if language not in LANGUAGES:
                print(f"Skipped: unsupported language ({language or 'unknown'})")
                continue

            stage = "LLM"
            reply = conversation.ask(transcript, language)
            llm_end = perf_counter()
            print(f"Reply: {reply}\nLLM: {llm_end - asr_end:.3f} s")

            stage = "TTS"
            output = tts.to_tts(reply, language, raise_on_error=True)
            tts_end = perf_counter()
            duration = wav_duration(output)
            print(f"Output WAV: {output}\nTTS: {tts_end - llm_end:.3f} s")
            print(f"Processing: {tts_end - turn_start:.3f} s\nPlayback duration: {duration:.3f} s")
            playback_total += duration
            produced += 1
        except (OSError, ValueError, RuntimeError, wave.Error) as error:
            print(f"{stage} failed: {error}", flush=True)
            failed = True

    print(f"\nProcessing total: {perf_counter() - started:.3f} s")
    print(f"Playback duration total: {playback_total:.3f} s ({produced} replies)")
    return int(failed or not produced)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, nargs="+", help="Recordings in conversation order")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        from bailing.llm import OpenAILLM

        snapshot = Path(snapshot_download(MODEL_ID, revision=WHISPER_REVISION,
                                          cache_dir=CACHE_DIR, local_files_only=True))
        if not (snapshot / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing Whisper weights in {snapshot}")
        output_root = Path("asset/conversation-verification").resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        output_dir = mkdtemp(prefix="replies-", dir=output_root)
        tts = bilingual_piper(Path("models/tts/piper"), output_dir)
        llm = OpenAILLM({"model_name": OLLAMA_MODEL, "api_key": "ollama", "url": "http://127.0.0.1:11434/v1"})
        with TemporaryDirectory(prefix="tron1-asr-") as asr_dir:
            asr = WhisperASR({"model_dir": str(snapshot), "output_file": asr_dir})
            print(f"ASR: {MODEL_ID} @ {WHISPER_REVISION} on {asr.pipe.device}")
            print(f"LLM: {OLLAMA_MODEL}\nReply directory: {output_dir}", flush=True)
            return run_recordings(args.wav, asr, TextConversation(llm, recorded_audio=True), tts)
    except (ImportError, LocalEntryNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Could not initialize the conversation verifier: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
