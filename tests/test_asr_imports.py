import subprocess
import sys
from pathlib import Path


def test_whisper_adapter_imports_without_optional_engines():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from bailing.asr import WhisperASR; assert callable(WhisperASR)",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "The Whisper adapter must import without unused speech engines.\n"
        + result.stderr
    )
