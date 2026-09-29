"""Helpers for resuming JSONL outputs."""

from pathlib import Path


def count_nonempty_lines(path: Path) -> int:
    """Count non-empty lines in a text file."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as infile:
        return sum(1 for line in infile if line.strip())


def recover_newer_temp_output(output_jsonl: Path) -> bool:
    """Promote an interrupted-run temp file when it contains more completed rows."""
    temp_output_jsonl = output_jsonl.with_name(f"{output_jsonl.name}.tmp")
    if not temp_output_jsonl.exists():
        return False

    output_lines = count_nonempty_lines(output_jsonl)
    temp_lines = count_nonempty_lines(temp_output_jsonl)
    if temp_lines <= output_lines:
        print(
            f"[resume] keeping {output_jsonl}: lines={output_lines}; "
            f"temporary lines={temp_lines}",
            flush=True,
        )
        return False

    print(
        f"[resume] promoting newer temporary output: {temp_output_jsonl} "
        f"({temp_lines} lines) -> {output_jsonl} ({output_lines} lines)",
        flush=True,
    )
    temp_output_jsonl.replace(output_jsonl)
    return True
