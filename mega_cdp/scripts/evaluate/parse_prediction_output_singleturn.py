#!/usr/bin/env python3
"""Parse single-turn predictions for evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results" / "prediction_singleturn_sglang"

SECTION_RE_EN = re.compile(r"(?is)^\s*#+\s*Clinical Decision Process\s*:?\s*(.*?)\s*#+\s*Clinical Decision Outcome\s*:?\s*(.+)\s*$")
SECTION_RE_ZH = re.compile(r"(?is)^\s*#+\s*临床决策过程\s*：?\s*(.*?)\s*#+\s*临床决策结论\s*：?\s*(.+)\s*$")
STEP_RE = re.compile(r"(?ms)^\s*(\d+)\.\s+(.*?)(?=(?:\n\s*\d+\.|\Z))")
HUATUO_FINAL_HEADER = re.compile(r"(?im)^\s*#+\s*Final Response\s*$")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", "--model-name", dest="experiment_name", required=True)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def split_thinking_text(text: str) -> str:
    """Remove common thinking wrappers while preserving the final answer."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    match = HUATUO_FINAL_HEADER.search(text)
    if match:
        return text[match.end():].strip()
    marker = "assistantfinal"
    idx = text.rfind(marker)
    if idx != -1:
        return text[idx + len(marker):].strip()
    return text.strip()


def normalize_text(text: str) -> str:
    """Collapse whitespace for stable downstream evaluation."""
    return re.sub(r"\s+", " ", text).strip()


def parse_sections(text: str, language: str) -> tuple[list[str], str]:
    """Extract numbered decision-process steps and the final clinical outcome."""
    regex = SECTION_RE_ZH if language == "zh" else SECTION_RE_EN
    match = regex.search(text)
    if not match:
        raise ValueError("Missing Clinical Decision Process/Outcome sections")
    process_block, outcome_block = match.groups()
    steps = [normalize_text(step) for _, step in STEP_RE.findall(process_block)]
    steps = [step for step in steps if step]
    if not steps:
        raise ValueError("No numbered decision-process steps found")
    outcome = normalize_text(outcome_block)
    if not outcome:
        raise ValueError("Clinical Decision Outcome is empty")
    return steps, outcome


def parse_record(record: dict) -> dict:
    """Parse one raw single-turn prediction record."""
    for key in ["id", "case_path_id", "path_index", "filename", "language", "llm_output"]:
        if key not in record:
            raise KeyError(f"Missing key: {key}")
    if not isinstance(record["llm_output"], str):
        raise TypeError("llm_output must be a string")
    answer_text = split_thinking_text(record["llm_output"])
    steps, outcome = parse_sections(answer_text, record.get("language", "en"))
    return {
        "id": record.get("id"),
        "case_path_id": record.get("case_path_id"),
        "path_index": record.get("path_index"),
        "filename": record.get("filename"),
        "language": record.get("language"),
        "llm_output": {
            "clinical_decision_process": steps,
            "clinical_decision_outcome": outcome,
        },
        "parsed_success": True,
    }


def main() -> int:
    """Parse a JSONL file and write one structured record per input case."""
    args = parse_args()
    input_jsonl = args.input_jsonl or RESULTS_ROOT / args.experiment_name / f"{args.experiment_name}_prediction_output.jsonl"
    output_jsonl = args.output_jsonl or RESULTS_ROOT / args.experiment_name / f"{args.experiment_name}_prediction_output_parsed.jsonl"
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    error_jsonl = output_jsonl.with_suffix(output_jsonl.suffix + ".errors.jsonl")
    errors = 0
    with input_jsonl.open("r", encoding="utf-8") as infile, output_jsonl.open("w", encoding="utf-8") as outfile, error_jsonl.open("w", encoding="utf-8") as errfile:
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                parsed = parse_record(record)
            except Exception as exc:
                errors += 1
                errfile.write(json.dumps({"line_number": line_number, "error": str(exc)}, ensure_ascii=False) + "\n")
                parsed = {
                    "id": record.get("id"),
                    "case_path_id": record.get("case_path_id"),
                    "path_index": record.get("path_index"),
                    "filename": record.get("filename"),
                    "language": record.get("language"),
                    "llm_output": record.get("llm_output"),
                    "parsed_success": False,
                }
            outfile.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            if args.debug:
                break
    print(f"[done] parsed={output_jsonl} errors={errors}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
