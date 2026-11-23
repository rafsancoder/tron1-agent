# TRON 1 live verifier on Windows

See the [deployment manifest](DEPLOYMENT.md) for exact model artifacts, the Ollama alias, the verified Windows environment and Linux prerequisites.

From the repository root, list audio device IDs and check the selected devices without recording or playback.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_live --list-devices
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_live --input-device 1 --output-device 4 --check
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_live --input-device 1 --output-device 4 --silence 1.2 --no-speech-timeout 8 --max-utterance 30
```

Replace the IDs with your microphone and headphone output. The verifier uses the repository-pinned `sounddevice==0.5.0` dependency and expects local Ollama with the verified Instruct-2507 tag and `tron1-instruct-2507-bench-2048:latest` alias. It uses the cached Whisper Small revision and the installed Ryan and Harri voices.

Press Enter to start `LISTENING`, speak, then wait for `SPEAKING` to finish before pressing Enter again. The repository-pinned `silero-vad==5.1` detects speech, keeps about 250 ms before onset, and stops on a sustained pause. The 1.2-second setting measures a pause, not completion of a thought. Try brief internal pauses, quiet speech, and background noise when tuning it. The no-speech timeout skips the turn, and the maximum duration is a safeguard. Use `--seconds 8` for fixed-duration debugging.

`/reset` clears session history and `/quit` exits. Capture stays in memory. Piper's reply WAV is temporary. The VAD boundary-to-pause decision time is separate from ASR, LLM and TTS times and from capture-close to playback submission.

## Optional Savonia facts

Add `--savonia` to the live command. The [local fact file](savonia_facts.json) contains bilingual facts, official URLs, and verification dates. Supported questions use those facts directly, with URLs printed in the terminal. The provider handles a limited set of topics. It uses no embeddings or live lookup and has no schedule, staff, room, or navigation data.

Startup prints whether Savonia is enabled and the loaded file and fact count. The terminal shows the original transcript, proposed entity, match evidence, routing decision and fact IDs. Local answers use a separate timing label from LLM calls.

Exact names, letter-by-letter spelling and `SAVO NIA` identify Savonia. Observed phonetic variants such as Sabuna, Sabunya, Saboenia, Saboven and Sabo Nia require an institution term and confirmation. Other close names can prompt clarification only when they are adjacent to an institution term and the recent topic is already Savonia. Confirmation uses the original question and the current reply language. A rejection, an explicit different institution, or `/reset` prevents the old pending interpretation from carrying forward. A confirmed topic is used for immediate related follow-ups. No Whisper vocabulary hint is enabled.

The saved [Qwen benchmark](../asset/llm-benchmark/comparison-p5wazpwi/comparisons.jsonl) documented awkward Finnish such as “Minun nimi,” a repeated follow-up question, and unsupported location claims in the café and bus-stop cases. Finnish naturalness and microphone quality still need human review.

## Optional personal profile and public staff facts

Startup prints whether the public staff file and personal profile loaded. In Savonia mode, extracted name spans can be compared against Savonia, loaded staff records or the selected profile. Institution terms, possible campus names and recent topics can support a suggestion. Unmatched institution names prompt for clarification. Recognized alternative institutions stay outside the Savonia facts. The terminal shows the raw transcript, proposed candidates, supporting evidence and routing decision. Similarity scores are heuristic. Confirmation answers the original question in the current language. Rejection and `/reset` clear pending matches. Skipped unsupported-language turns report Whisper's raw language label and whether a transcript was present.

`--savonia` enables the [public staff records](savonia_staff.json). The terminal prints official references and verification dates. Staff claims and missing fields are handled locally. A given name or surname can identify a record, while an ambiguous teacher or email follow-up asks for clarification. The terminal shows the exact address. The speech layer expands its separators and reads the domain suffix as letters.

Use `--savonia --profile <name>` to select `asset/personal-profiles/<name>.json` for local testing. Copy [the placeholder example](personal_profile.example.json) when adding a profile. The profile directory is Git-ignored. Profile selection is not authentication. Anyone using that session can ask about the selected profile, so keep it limited to non-sensitive project information. The supervisor relationships and any unpublished work email in the profile are user-supplied. `/reset` clears history and staff referents and keeps the selected profile until you exit.

To test before opening the microphone, run the text verifier using the same flags. It accepts an English or Finnish question with a language prefix in interactive mode.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.verify_text --model tron1-instruct-2507-bench-2048:latest --savonia --profile <name>
```
