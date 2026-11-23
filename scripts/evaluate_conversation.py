"""Run the frozen TRON 1 prompt comparison against an installed Ollama model."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory, mkdtemp
from time import perf_counter, sleep
from urllib.request import Request, urlopen

from scripts.verify_text import TextConversation


URL = "http://127.0.0.1:11434"
RESULTS = Path(__file__).resolve().parents[1] / "asset" / "llm-benchmark"
PROFILES = Path(__file__).with_name("conversation_benchmark_profiles.json")
OPTIONS = {
    "num_ctx": 2048, "num_predict": 256, "seed": 42, "temperature": 0.6,
    "top_p": 0.95, "top_k": 20, "min_p": 0, "repeat_penalty": 1,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def api(endpoint, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(f"{URL}/api/{endpoint}", body, {"Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def residency(model):
    return next((entry for entry in api("ps")["models"] if entry["name"] == model), {})


def unload(model):
    if residency(model):
        api("generate", {"model": model, "keep_alive": 0})
    for _ in range(50):
        if not residency(model):
            return
        sleep(0.2)
    raise RuntimeError(f"Still resident after unload: {model}")


def prepare(model, alias, digest_prefix, output, ollama):
    tags = {item["name"]: item for item in api("tags")["models"]}
    if model not in tags or not tags[model]["digest"].startswith(digest_prefix):
        raise ValueError(f"Expected installed model {model} with digest beginning {digest_prefix}")
    if alias == model:
        raise ValueError("Use a separate local benchmark alias")
    original = api("show", {"model": model})
    if alias not in tags:
        modelfile = output / "Modelfile.benchmark"
        parameters = "".join(f"PARAMETER {key} {value}\n" for key, value in OPTIONS.items())
        modelfile.write_text(f"FROM {model}\n{parameters}", encoding="utf-8")
        subprocess.run([ollama, "create", alias, "-f", str(modelfile)], check=True)
    configured = api("show", {"model": alias})
    from_line = lambda show: next(line for line in show["modelfile"].splitlines() if line.startswith("FROM "))
    if from_line(original) != from_line(configured) or original["template"] != configured["template"]:
        raise ValueError(f"{alias} does not share the weights and template of {model}")
    actual = dict(line.split(maxsplit=1) for line in configured["parameters"].splitlines() if line.strip())
    if any(float(actual.get(key, "nan")) != value for key, value in OPTIONS.items()):
        raise ValueError(f"{alias} does not match the frozen benchmark settings")
    metadata = {"ollama": api("version"), "tag": tags[model], "alias": alias, "settings": OPTIONS,
                "template": original["template"], "model_info": original["model_info"],
                "capabilities": original.get("capabilities"), "thinking": original.get("thinking"),
                "original_parameters": original.get("parameters"), "alias_parameters": configured["parameters"],
                "prompt_sha256": sha256(PROFILES.read_bytes()).hexdigest()}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for name in ("qwen3:4b", model, alias):
        unload(name)


class MeasuredConversation:
    def __init__(self, adapter, prompts, alias):
        self.adapter = adapter
        self.prompts = prompts
        self.alias = alias
        self.create = adapter.client.chat.completions.create
        adapter.client.chat.completions.create = self.stream

    def response(self, messages):
        self.messages = [{"role": "system", "content": self.prompts[self.language]}, *messages[1:]]
        return self.adapter.response(self.messages)

    def stream(self, **request):
        start = perf_counter()
        first = finish = usage = None
        reasoning_chars = 0
        request.update(max_tokens=OPTIONS["num_predict"], seed=OPTIONS["seed"], temperature=OPTIONS["temperature"],
                       top_p=OPTIONS["top_p"], stream_options={"include_usage": True})
        with self.create(**request) as stream:
            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage.model_dump(exclude_none=True)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                reasoning_chars += sum(len(getattr(choice.delta, key, "") or "")
                                       for key in ("reasoning", "reasoning_content"))
                if first is None and (choice.delta.content or "").strip():
                    first = perf_counter() - start
                finish = choice.finish_reason or finish
                yield chunk
        elapsed = perf_counter() - start
        resident = residency(self.alias)
        if resident.get("context_length") != OPTIONS["num_ctx"] or usage is None or finish is None:
            raise RuntimeError("Missing telemetry or incorrect running context size")
        self.metrics = dict(first_answer_s=first, total_s=elapsed, usage=usage, finish=finish,
                            reasoning_chars=reasoning_chars, vram_bytes=resident["size_vram"],
                            model_bytes=resident["size"], context=resident["context_length"])


def evaluate(adapter, prompts, alias, datasets, variant):
    measured = MeasuredConversation(adapter, prompts, alias)
    measured.language = "en"
    reply = TextConversation(measured, recorded_audio=True).ask("Acknowledge this setup briefly.", "en")
    yield dict(variant=variant, dataset="warmup", case="setup", turn=1, reply=reply, metrics=measured.metrics)
    for dataset, cases in datasets.items():
        for case, turns in cases.items():
            conversation = TextConversation(measured, recorded_audio=True)
            expected = []
            for index, (lang, question) in enumerate(turns, 1):
                measured.language = lang
                measured.metrics = None
                error = None
                try:
                    reply = conversation.ask(question, lang)
                except RuntimeError as failure:
                    reply, error = None, str(failure)
                user = {"role": "user", "content": question}
                if measured.metrics is None or measured.messages[1:] != [*expected, user]:
                    raise RuntimeError(f"Missing telemetry or unexpected history in {case}.{index}")
                if reply is not None:
                    expected.extend([user, {"role": "assistant", "content": reply}])
                yield dict(variant=variant, dataset=dataset, case=case, turn=index, lang=lang,
                           question=question, reply=reply, error=error, messages=measured.messages,
                           metrics=measured.metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact installed model tag")
    parser.add_argument("--digest-prefix", required=True, help="Expected installed model digest prefix")
    parser.add_argument("--alias", required=True, help="Local alias with the frozen settings, including its tag")
    parser.add_argument("--cases", required=True, type=Path, help="Local JSON case file")
    parser.add_argument("--prompt", choices=("old", "revised", "both"), default="both")
    parser.add_argument("--with-whisper", type=Path, metavar="WAV", help="Keep Whisper on CUDA after warming it with this WAV")
    default_ollama = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe"
    parser.add_argument("--ollama", default=str(default_ollama) if default_ollama.is_file() else "ollama")
    args = parser.parse_args()

    if not args.digest_prefix or not all(character in "0123456789abcdef" for character in args.digest_prefix):
        parser.error("Expected a lowercase hexadecimal model digest prefix")
    datasets = read_json(args.cases)
    profiles = read_json(PROFILES)
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="comparison-", dir=RESULTS))
    print(f"Results directory: {output}", flush=True)
    prepare(args.model, args.alias, args.digest_prefix, output, args.ollama)

    from bailing.llm import OpenAILLM

    try:
        with ExitStack() as stack:
            if args.with_whisper:
                from scripts.verify_conversation import WHISPER_REVISION
                from scripts.verify_whisper import CACHE_DIR, WhisperASR, read_wav_chunks

                snapshot = CACHE_DIR / "models--openai--whisper-small" / "snapshots" / WHISPER_REVISION
                temporary = stack.enter_context(TemporaryDirectory(prefix="whisper-benchmark-"))
                asr = WhisperASR({"model_dir": str(snapshot), "output_file": temporary})
                if asr.pipe.device.type != "cuda":
                    raise RuntimeError("Whisper must be on CUDA for the shared-GPU comparison")
                asr.recognizer(read_wav_chunks(args.with_whisper))
            log = stack.enter_context((output / "comparisons.jsonl").open("w", encoding="utf-8"))
            telemetry = stack.enter_context((output / "telemetry.jsonl").open("w", encoding="utf-8"))
            variants = ("old", "revised") if args.prompt == "both" else (args.prompt,)
            for variant in variants:
                unload(args.alias)
                adapter = OpenAILLM({"model_name": args.alias, "api_key": "ollama", "url": URL + "/v1"})
                for row in evaluate(adapter, profiles[variant], args.alias, datasets, variant):
                    metrics = row.pop("metrics")
                    key = {field: row[field] for field in ("variant", "dataset", "case", "turn")}
                    telemetry.write(json.dumps({**key, "reply": row["reply"], **metrics}, ensure_ascii=False) + "\n")
                    telemetry.flush()
                    if row["dataset"] != "warmup":
                        log.write(json.dumps(row, ensure_ascii=False) + "\n")
                        log.flush()
                    print(f"{variant}/{row['dataset']}/{row['case']}.{row['turn']}: {row['reply'] or row['error']}", flush=True)
    finally:
        unload(args.alias)


if __name__ == "__main__":
    main()
