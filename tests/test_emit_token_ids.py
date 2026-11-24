"""Verify --emit-token-id output against the unmodified generation transcript.

Set KOBOLD_TEST_MODEL to a GGUF and build ./main to run the live test. The test
compares off/on output and reconstructs the assistant text from TOKEN records.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _assistant_text(output: str) -> str:
    match = re.search(r"assistant\n(.*?)\n> EOF by user", output, re.DOTALL)
    if match is None:
        raise AssertionError(f"assistant transcript not found:\n{output}")
    return match.group(1)


def _run(model: Path, *, emit: bool) -> str:
    command = [str(ROOT / "main"), "-m", str(model), "-p", "What is 17 multiplied by 23? The answer is", "-n", "4", "--temp", "0", "-ngl", "0"]
    if emit:
        command.append("--emit-token-ids")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise AssertionError(f"generation failed rc={result.returncode}:\n{output}")
    return output


@pytest.mark.skipif(not os.environ.get("KOBOLD_TEST_MODEL"), reason="set KOBOLD_TEST_MODEL to a GGUF for the live CLI test")
def test_emit_token_ids_preserves_text_and_reconstructs_tokens() -> None:
    model = Path(os.environ["KOBOLD_TEST_MODEL"])
    if not model.is_file():
        pytest.fail(f"model not found: {model}")
    if not (ROOT / "main").is_file():
        pytest.fail("build ./main first")

    baseline = _run(model, emit=False)
    emitted = _run(model, emit=True)
    baseline_text = _assistant_text(baseline)
    emitted_text = _assistant_text(emitted)
    assert emitted_text == baseline_text
    assert "TOKEN\t" not in baseline

    records = [json.loads(line.removeprefix("TOKEN\t")) for line in emitted.splitlines() if line.startswith("TOKEN\t")]
    assert len(records) == 4
    assert all(isinstance(record["id"], int) and isinstance(record["text"], str) for record in records)
    assert "".join(record["text"] for record in records) == emitted_text
