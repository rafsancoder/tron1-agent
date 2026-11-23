"""Opt-in, half-duplex Windows microphone conversation verifier."""

import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import soundfile as sf

from scripts.evaluate_conversation import OPTIONS, URL, api
from scripts.live_endpoint import EndpointSettings, capture_utterance
from scripts.savonia_people import speak_work_email
from scripts.verify_conversation import WHISPER_REVISION
from scripts.verify_text import LANGUAGES, TextConversation
from scripts.verify_tts import bilingual_piper
from scripts.verify_whisper import CACHE_DIR, WhisperASR, pcm16_chunks, resample_mono


MODEL = "qwen3:4b-instruct-2507-q4_K_M"
MODEL_DIGEST = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
ALIAS = "tron1-instruct-2507-bench-2048:latest"
SILENCE_RMS = 10 ** (-55 / 20)


def list_devices(port):
    devices = port.query_devices()
    host_apis = port.query_hostapis()
    for direction, default in zip(("input", "output"), port.default.device):
        print(f"{direction.capitalize()} devices (default {default}):")
        for index, device in enumerate(devices):
            if device[f"max_{direction}_channels"]:
                print(f"  {index}: {device['name']} [{host_apis[device['hostapi']]['name']}] "
                      f"{device['default_samplerate']:g} Hz")


def capture_frames(port, device, rate, seconds):
    remaining = round(rate * seconds)
    if remaining < 1:
        raise ValueError("Capture duration must cover at least one audio frame")
    blocks = []
    with port.InputStream(device=device, samplerate=rate, channels=1, dtype="float32") as stream:
        while remaining:
            count = min(remaining, 1024)
            block, overflow = stream.read(count)
            if overflow or len(block) != count:
                raise RuntimeError("Input overflow or short read; recording was discarded")
            blocks.append(block[:, 0])
            remaining -= count
    return np.concatenate(blocks)


class AudioDevices:
    def __init__(self, port, input_id, output_id):
        self.port = port
        self.input_id = port.default.device[0] if input_id is None else input_id
        self.output_id = port.default.device[1] if output_id is None else output_id
        if self.input_id < 0 or self.output_id < 0:
            raise ValueError("Missing input or output device; use --list-devices and select both IDs")
        input_device = port.query_devices(self.input_id, "input")
        output_device = port.query_devices(self.output_id, "output")
        self.input_rate = round(input_device["default_samplerate"])
        self.output_rate = round(output_device["default_samplerate"])
        self.output_channels = min(2, output_device["max_output_channels"])
        port.check_input_settings(device=self.input_id, channels=1, samplerate=self.input_rate, dtype="float32")
        port.check_output_settings(device=self.output_id, channels=self.output_channels,
                                   samplerate=self.output_rate, dtype="float32")

    def capture(self, seconds):
        return capture_frames(self.port, self.input_id, self.input_rate, seconds)

    def listen(self, vad, settings):
        return capture_utterance(self.port, self.input_id, self.input_rate, vad, settings,
                                 lambda: print("RECORDING", flush=True))

    def prepare_playback(self, path):
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if not samples.size or not np.isfinite(samples).all():
            raise RuntimeError("Piper returned no usable audio")
        mono = resample_mono(samples.mean(axis=1), rate, self.output_rate)
        return np.repeat(mono[:, None], self.output_channels, axis=1)

    def play(self, samples):
        status = self.port.play(samples, samplerate=self.output_rate, device=self.output_id, blocking=True)
        if status:
            raise RuntimeError(f"Audio playback reported {status}")


def run_turn(audio, asr, conversation, piper, seconds, *, vad=None, endpoint=None):
    stage = "capture"
    reply_path = None
    try:
        if seconds is None:
            print("LISTENING", flush=True)
            capture = audio.listen(vad, endpoint)
            capture_end = perf_counter()
            if capture.reason == "no speech":
                print("Skipped: no speech detected before timeout.")
                return
            if capture.endpoint_delay is not None:
                print(f"VAD boundary to pause decision: {capture.endpoint_delay:.3f} s")
            else:
                print("Stopped at maximum utterance duration (safeguard).")
            samples = capture.samples
        else:
            print(f"RECORDING ({seconds:g} s, fixed debug capture)", flush=True)
            samples = audio.capture(seconds)
            capture_end = perf_counter()
        print("PROCESSING", flush=True)
        if not samples.size or not np.isfinite(samples).all():
            raise RuntimeError("Capture returned no usable audio")
        if np.sqrt(np.mean((samples - samples.mean()) ** 2)) < SILENCE_RMS:
            print("Skipped: silence or signal below -55 dBFS RMS.")
            return

        stage = "ASR"
        start = perf_counter()
        text, _, language = asr.recognizer(pcm16_chunks(samples, audio.input_rate),
                                            keep_wav=False, raise_on_error=True)
        print(f"ASR: {perf_counter() - start:.3f} s")
        has_transcript = bool((text or "").strip())
        if language not in LANGUAGES and getattr(conversation, "routing_diagnostics", False):
            raw_language = getattr(asr, "last_detected_language", language)
            print(f"ASR language diagnostic: raw={raw_language!r} | transcript present: {'yes' if has_transcript else 'no'}")
        if not has_transcript:
            print("Skipped: Whisper returned no transcript.")
            return
        if language not in LANGUAGES:
            print(f"Skipped: unsupported language ({language or 'unknown'}). Only en and fi are supported.")
            return
        print(f"Transcript [{language}]: {text}")

        stage = "reply"
        start = perf_counter()
        reply = conversation.ask(text, language)
        reply_stage = "Knowledge routing (no LLM call)" if conversation.last_reply_source == "knowledge" else "LLM"
        print(f"{reply_stage}: {perf_counter() - start:.3f} s\nReply: {reply}")

        stage = "TTS"
        start = perf_counter()
        speech = speak_work_email(reply, language) if conversation.last_reply_source == "knowledge" else reply
        reply_path = Path(piper.to_tts(speech, language, raise_on_error=True))
        print(f"TTS: {perf_counter() - start:.3f} s")
        stage = "playback"
        output = audio.prepare_playback(reply_path)
        print(f"SPEAKING (capture close to playback submission: {perf_counter() - capture_end:.3f} s)", flush=True)
        audio.play(output)
    except Exception as error:
        print(f"{stage} failed: {error}", flush=True)
    finally:
        if reply_path is not None:
            try:
                reply_path.unlink(missing_ok=True)
            except OSError as error:
                print(f"Temporary reply cleanup failed: {error}", flush=True)


def run_session(audio, asr, conversation, piper, seconds, *, vad=None, endpoint=None, read=input):
    print("Use headphones for the first tests. Press Enter to record, /reset to clear history, or /quit to exit.")
    while True:
        try:
            command = read("> ").strip().lower()
        except EOFError:
            return
        if command == "/quit":
            return
        if command == "/reset":
            conversation.reset()
            print("Conversation history and topic cleared.")
        elif not command:
            run_turn(audio, asr, conversation, piper, seconds, vad=vad, endpoint=endpoint)
        else:
            print("Use Enter, /reset, or /quit.")


def check_ollama():
    installed = {item["name"]: item for item in api("tags")["models"]}
    if installed.get(MODEL, {}).get("digest") != MODEL_DIGEST or ALIAS not in installed:
        raise RuntimeError(f"Expected the verified {MODEL} and its local alias {ALIAS}")
    original = api("show", {"model": MODEL})
    alias = api("show", {"model": ALIAS})
    original_blob = next(line for line in original["modelfile"].splitlines() if line.startswith("FROM "))
    if original_blob not in alias["modelfile"].splitlines() or original["template"] != alias["template"]:
        raise RuntimeError("Ollama alias does not match the verified weights and template")
    settings = dict(line.split(maxsplit=1) for line in alias["parameters"].splitlines() if line.strip())
    if any(float(settings.get(key, "nan")) != value for key, value in OPTIONS.items()):
        raise RuntimeError("Ollama alias settings do not match the verified benchmark")


def load_knowledge(enabled, profile_name=None):
    if not enabled:
        print("Savonia knowledge: disabled (no file loaded, 0 facts)", flush=True)
        print("Savonia staff: disabled (no file loaded, 0 records)", flush=True)
        print("Personal profile: not selected", flush=True)
        return None
    from scripts.savonia_knowledge import SavoniaKnowledge
    from scripts.savonia_people import load_profile

    provider = SavoniaKnowledge(profile=load_profile(profile_name) if profile_name else None)
    print(f"Savonia knowledge: enabled | file: {provider.path} | {len(provider.facts)} facts", flush=True)
    print(f"Savonia staff: enabled | file: {provider.people.path} | {len(provider.people.staff)} records", flush=True)
    print(f"Personal profile: selected (local, user-supplied) | name: {profile_name}" if profile_name else
          "Personal profile: not selected", flush=True)
    return provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true", help="List devices and exit")
    parser.add_argument("--input-device", type=int, help="Input device ID from --list-devices")
    parser.add_argument("--output-device", type=int, help="Output device ID from --list-devices")
    parser.add_argument("--seconds", type=float, help="Use fixed-duration debug capture instead of VAD")
    parser.add_argument("--silence", type=float, default=1.2, help="End-of-speech pause in seconds (default: 1.2)")
    parser.add_argument("--no-speech-timeout", type=float, default=8.0, help="Seconds to wait for speech (default: 8)")
    parser.add_argument("--max-utterance", type=float, default=30.0, help="Utterance safeguard in seconds (default: 30)")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero speech threshold (default: 0.5)")
    parser.add_argument("--check", action="store_true", help="Check startup without recording or playback")
    parser.add_argument("--savonia", action="store_true", help="Enable the local, verified Savonia knowledge file")
    parser.add_argument("--profile", help="Select a Git-ignored local profile by name; requires --savonia")
    args = parser.parse_args()
    if args.profile and not args.savonia:
        parser.error("--profile requires --savonia")
    durations = (args.seconds, args.silence, args.no_speech_timeout, args.max_utterance)
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in durations):
        parser.error("Capture durations must be finite and positive")
    if not 0 < args.vad_threshold < 1:
        parser.error("--vad-threshold must be between 0 and 1")

    try:
        import sounddevice as port

        if args.list_devices:
            list_devices(port)
            return 0
        knowledge = load_knowledge(args.savonia, args.profile)
        audio = AudioDevices(port, args.input_device, args.output_device)
        endpoint = EndpointSettings(args.silence, args.no_speech_timeout, args.max_utterance)
        vad = None
        if args.seconds is None:
            from bailing.vad import SileroVAD

            vad = SileroVAD({"sampling_rate": 16000, "threshold": args.vad_threshold,
                             "min_silence_duration_ms": endpoint.silence * 1000})
        check_ollama()
        from bailing.llm import OpenAILLM

        snapshot = CACHE_DIR / "models--openai--whisper-small" / "snapshots" / WHISPER_REVISION
        if not (snapshot / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing cached Whisper weights: {snapshot}")
        with TemporaryDirectory(prefix="tron1-live-") as temporary:
            piper = bilingual_piper(Path("models/tts/piper"), temporary)
            for language in LANGUAGES:
                model = piper.model_path[language]
                for path in (model, model + ".json"):
                    if not Path(path).is_file():
                        raise FileNotFoundError(f"Missing Piper voice file: {path}")
            llm = OpenAILLM({"model_name": ALIAS, "api_key": "ollama", "url": URL + "/v1",
                             "raise_on_error": True})
            llm.client = llm.client.with_options(timeout=45.0, max_retries=0)
            conversation = TextConversation(llm, recorded_audio=True, knowledge=knowledge, routing_diagnostics=True,
                                            personal_questions=True)
            conversation.capabilities = (
                "This session can listen during a user-started recording and synthesize spoken replies. "
                "It has no continuous listening, robot controls, or external information access."
                " Do not say you searched the web, checked a database, or used a live lookup."
            )
            asr = WhisperASR({"model_dir": str(snapshot)})
            print(f"Ready: input {audio.input_id} @ {audio.input_rate} Hz, output {audio.output_id} "
                  f"@ {audio.output_rate} Hz, Whisper {asr.pipe.device}, {ALIAS}.", flush=True)
            if not args.check:
                run_session(audio, asr, conversation, piper, args.seconds, vad=vad, endpoint=endpoint)
    except KeyboardInterrupt:
        print("\nSession ended.")
    except ModuleNotFoundError as error:
        hint = " Install sounddevice==0.5.0 in .venv." if error.name == "sounddevice" else ""
        parser.exit(1, f"Startup failed: {error}.{hint}\n")
    except Exception as error:
        parser.exit(1, f"Startup failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
