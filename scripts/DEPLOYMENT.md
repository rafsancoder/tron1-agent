# TRON 1 bilingual voice verifier deployment

Run commands from the repository root. The isolated verifier uses `python -m scripts.verify_live`. Its configuration is separate from `config/config.yaml` and it does not start robot actions. Linux deployment remains unverified.

## Model manifest

| Component | Verified identity and location |
| --- | --- |
| Whisper ASR | Multilingual `openai/whisper-small` at revision `973afd24965f72e36ca33b3055d56a652f456b4d`. Install the complete Transformers snapshot, including `model.safetensors`, config, tokenizer and processor files, under `models/huggingface/models--openai--whisper-small/snapshots/<revision>/`. The live and recorded verifiers use the local snapshot. Language detection is automatic. |
| Live Ollama | `qwen3:4b-instruct-2507-q4_K_M`, digest `sha256:0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`. The service must listen at `http://127.0.0.1:11434`. Windows benchmark version was Ollama 0.40.17. The SDK uses a local placeholder key, so no hosted API credential is required. |
| Live alias | `tron1-instruct-2507-bench-2048:latest`. The verifier checks that the alias uses the same weights and template as the base tag and matches all settings in [Modelfile.live](Modelfile.live). |
| Recorded verifier | `scripts.verify_conversation` uses the separate `qwen3:4b` baseline. Use the live verifier when validating the Instruct-2507 alias. |
| Piper | Ryan Medium (`en_US-ryan-medium`) and Harri Medium (`fi_FI-harri-medium`). Keep each ONNX file beside its matching JSON in `models/tts/piper/`. The live profile uses length scale 1.0 and sentence silence 0.35 seconds. |

Once the verified Ollama base tag is provisioned locally, create the alias:

```text
ollama create tron1-instruct-2507-bench-2048:latest -f scripts/Modelfile.live
```

Use these SHA-256 hashes to verify the Piper files. They identify the exact installed artifacts independently of any upstream branch name.

| Filename | SHA-256 |
| --- | --- |
| `en_US-ryan-medium.onnx` | `33b53106e049f68a51f014e4fa10bf5a04e0f7d6668692dc05228cc4554c1d71` |
| `en_US-ryan-medium.onnx.json` | `d29061675e660113d51f81650e922134d94cab750271f9ce2bcb05121f925a60` |
| `fi_FI-harri-medium.onnx` | `315cd970fc5e3dbae98998cbe9926527196c0a3ebdc7e02e21bd9e7325fd085b` |
| `fi_FI-harri-medium.onnx.json` | `716ad6fd8cadcdf2f926e76d6f5ce757a93a3fe55e78077319006ea27e60777f` |

Voice source is the `rhasspy/piper-voices` repository, under `en/en_US/ryan/medium` and `fi/fi_FI/harri/medium`. The locally saved voice cards identify the Ryan dataset as CC BY-NC-SA 4.0 and the Harri dataset as CC0. Review the cards and applicable terms before redistribution or commercial use.

## Runtime baseline

Windows verification used Python 3.10.9, Torch and Torchaudio 2.6.0+cu124, Transformers 4.51.1, Hugging Face Hub 0.36.2, NumPy 1.26.4, OpenAI SDK 1.76.0, Piper 1.8.0, ONNX Runtime 1.23.2, SoundDevice 0.5.0, SoundFile 0.12.1 and Silero VAD 5.1. Tests used pytest 8.3.5. The recorded comparison on the GTX 1660 SUPER sampled a system GPU peak of 5,407 MiB out of 6,144 MiB. That measurement is specific to the Windows run.

## Windows startup

Run from the checkout root in PowerShell. Devices 1 and 4 were used on the verification PC. Use the IDs reported on your machine.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_live --list-devices
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_live --input-device 1 --output-device 4 --check
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_live --input-device 1 --output-device 4
```

## External files and startup

- The live verifier defaults to the model locations above. Use `--whisper-cache-dir` to point to a different Hugging Face cache root that contains the same pinned snapshot layout, and `--piper-model-dir` for a different voice directory. For a standalone Piper check, `scripts.verify_tts --model-dir <directory>` accepts a voice directory.
- Public Savonia facts and staff records ship with the checkout. Enable them with `--savonia`. An optional private profile goes in `asset/personal-profiles/<name>.json`. Use [the example](personal_profile.example.json) as a template and select it with `--savonia --profile <name>`. A selected profile is visible to anyone using that session.
- Supply test WAVs by path to `scripts.verify_whisper` or `scripts.verify_conversation`. Keep personal recordings, generated audio, benchmark reports, model files, virtual environments and credentials local. Live replies use the platform temporary directory.

## Linux transfer checklist

Every Linux step below is unverified on TRON 1.

1. **Inventory the target.** Record distribution, CPU architecture, Python, CPU, RAM, swap, free disk, and any accelerator and driver already present. Identify the audio backend, capture and playback devices, and the user account that can access them.
2. **Provision the runtime (hardware-dependent).** Select a compatible Python environment, Torch/Torchaudio build, Ollama runtime, Piper/ONNX Runtime, PortAudio and libsndfile only after inventory. Start from the Windows version record for comparison. Verify compatible builds, especially on ARM. No Linux Torch or CUDA package is prescribed. Check Whisper and Ollama device placement separately, including CPU fallback and available memory.
3. **Provision the artifacts.** Transfer or download the pinned Whisper snapshot, both checked Piper voice pairs, and the exact Ollama tag. Verify the Piper SHA-256 hashes and create the local Ollama alias from the supplied Modelfile. Keep assets outside version control.
4. **Test from the intended account.** With the target environment active, enumerate devices, run the startup check, then verify an English and a Finnish WAV and real Piper playback before recording live. Measure latency and peak memory with Whisper and Ollama loaded together.

```text
python -m scripts.verify_live --list-devices
python -m scripts.verify_live --input-device INPUT_ID --output-device OUTPUT_ID --whisper-cache-dir "/path/to/Hugging Face cache" --piper-model-dir "/path/to/Piper voices" --check
```

Replace the example paths and IDs before running. Omit the path flags when using the repository defaults. The startup check loads Whisper and checks device settings, model files and Ollama metadata. After the separate audio checks, remove `--check` for a supervised live pilot.
