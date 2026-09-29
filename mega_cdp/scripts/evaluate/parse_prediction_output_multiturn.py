"""Parse multi-turn predictions for evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    from .resume_utils import recover_newer_temp_output
except ImportError:
    from resume_utils import recover_newer_temp_output


FINAL_CONCLUSION_PATTERN = re.compile(
    r"(Final Conclusion|最终结论)\s*[:：]?\s*",
    re.IGNORECASE,
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results" / "prediction_multiturn_sglang"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(
        description="Parse a multiturn inference experiment into evaluation-ready JSONL."
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Directory name under the selected prediction_multiturn_sglang result root.",
    )
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Only parse the first non-empty record.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the newest valid output or temporary output.",
    )
    return parser.parse_args()


def parse_conversation(conversation: object) -> list[str]:
    """Extract doctor messages from a conversation state."""
    if not isinstance(conversation, list):
        raise TypeError("conversation must be a list")

    doctor_messages: list[str] = []
    for entry in conversation:
        if not isinstance(entry, dict):
            raise TypeError("conversation entries must be objects")
        if entry.get("role") != "doctor":
            continue
        content = entry.get("content")
        if not isinstance(content, str):
            raise TypeError("doctor content must be a string")
        doctor_messages.append(content)

    if not doctor_messages:
        raise ValueError("No doctor messages found in conversation")
    return doctor_messages


def extract_final_conclusion(messages: list[str]) -> tuple[list[str], str | None, bool]:
    """Split final conclusion from the last doctor message."""
    last_message = messages[-1]
    match = FINAL_CONCLUSION_PATTERN.search(last_message)
    if match is None:
        return messages, None, False

    outcome = last_message[match.end() :].strip()
    if not outcome:
        return messages, None, False

    return messages[:-1], outcome, True


def build_parsed_record(record: dict) -> dict:
    """Build one parsed prediction record."""
    required = ["id", "case_path_id", "path_index", "filename", "language", "conversation"]
    missing = [key for key in required if key not in record]
    if missing:
        raise KeyError(f"Missing keys: {', '.join(missing)}")

    doctor_messages = parse_conversation(record["conversation"])
    pathway, outcome, has_final = extract_final_conclusion(doctor_messages)
    return {
        "id": record["id"],
        "case_path_id": record["case_path_id"],
        "path_index": record["path_index"],
        "filename": record["filename"],
        "language": record["language"],
        "llm_output": pathway,
        "clinical_decision_outcome": outcome,
        "parsed_success": bool(record.get("ended", True)) and has_final and len(doctor_messages) > 1,
    }


def get_parse_failure_reason(record: dict, parsed: dict) -> str:
    """Explain why a parsed record failed validation."""
    if record.get("ended") is False:
        return "Inference did not finish successfully: ended is false"
    if not parsed.get("clinical_decision_outcome"):
        return "No Final Conclusion marker with non-empty content was found"
    llm_output = parsed.get("llm_output")
    if not isinstance(llm_output, list) or len(llm_output) < 1:
        return "Fewer than two doctor messages were found"
    return "Parsed record did not satisfy the success criteria"


def resolve_input_path(experiment_dir: Path, experiment_name: str) -> Path:
    """Resolve the raw prediction file for an experiment."""
    suffix = "_prediction_state.jsonl"
    preferred_names = [
        f"{experiment_name}{suffix}",
        f"{experiment_name}_Qwen3_8B{suffix}",
    ]
    for filename in preferred_names:
        candidate = experiment_dir / filename
        if candidate.is_file():
            return candidate

    candidates = sorted(
        path
        for path in experiment_dir.glob(f"*{suffix}")
        if "_parsed" not in path.name and not path.name.endswith(".tmp")
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No prediction JSONL found under {experiment_dir}")
    names = "\n".join(f"  - {path.name}" for path in candidates)
    raise RuntimeError(
        f"Multiple prediction JSONL files found under {experiment_dir}; "
        f"specify --input-jsonl explicitly:\n{names}"
    )


def failed_record(record: dict, llm_output: object = None) -> dict:
    """Build a parsed-failure placeholder record."""
    return {
        "id": record.get("id"),
        "case_path_id": record.get("case_path_id"),
        "path_index": record.get("path_index"),
        "filename": record.get("filename"),
        "language": record.get("language"),
        "llm_output": llm_output if llm_output is not None else [],
        "clinical_decision_outcome": None,
        "parsed_success": False,
    }


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    results_root = RESULTS_ROOT
    experiment_dir = results_root / args.experiment_name
    if args.input_jsonl is None or args.output_jsonl is None:
        if not experiment_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    if args.input_jsonl is not None:
        input_jsonl = args.input_jsonl
    else:
        input_jsonl = resolve_input_path(experiment_dir, args.experiment_name)
    parsed_suffix = "_prediction_state_parsed.jsonl"
    if args.output_jsonl is not None:
        output_jsonl = args.output_jsonl
    else:
        output_jsonl = experiment_dir / f"{args.experiment_name}{parsed_suffix}"
    error_jsonl = output_jsonl.with_suffix(".errors.jsonl")
    temp_output = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and not args.debug:
        recover_newer_temp_output(output_jsonl)

    existing_output_lines: list[str] = []
    if args.resume and not args.debug and output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as existing_file:
            existing_output_lines = [line for line in existing_file if line.strip()]

    completed = len(existing_output_lines)
    processed = completed
    parsed_success = 0
    for line in existing_output_lines:
        try:
            if json.loads(line).get("parsed_success") is True:
                parsed_success += 1
        except json.JSONDecodeError:
            pass
    parse_failures = 0

    print(f"[input] {input_jsonl}", flush=True)
    print(f"[output] {output_jsonl}", flush=True)
    if completed:
        print(f"[resume] continuing after {completed} parsed records", flush=True)
    with (
        input_jsonl.open("r", encoding="utf-8") as infile,
        temp_output.open("w", encoding="utf-8") as outfile,
        error_jsonl.open("a" if completed else "w", encoding="utf-8") as errfile,
    ):
        outfile.writelines(existing_output_lines)
        input_record_index = 0
        for line_number, raw_line in enumerate(infile, start=1):
            if not raw_line.strip():
                continue
            input_record_index += 1
            if input_record_index <= completed:
                continue
            processed += 1
            record: dict = {}
            try:
                record = json.loads(raw_line)
                parsed = build_parsed_record(record)
                if parsed["parsed_success"]:
                    parsed_success += 1
                else:
                    errfile.write(
                        json.dumps(
                            {
                                "line_number": line_number,
                                "id": parsed.get("id"),
                                "case_path_id": parsed.get("case_path_id"),
                                "path_index": parsed.get("path_index"),
                                "filename": parsed.get("filename"),
                                "error_type": "parsed_failed",
                                "reason": get_parse_failure_reason(record, parsed),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                outfile.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            except Exception as exc:
                parse_failures += 1
                try:
                    doctor_messages = parse_conversation(record.get("conversation"))
                except Exception:
                    doctor_messages = []
                outfile.write(json.dumps(failed_record(record, doctor_messages), ensure_ascii=False) + "\n")
                errfile.write(
                    json.dumps(
                        {
                            "line_number": line_number,
                            "id": record.get("id"),
                            "case_path_id": record.get("case_path_id"),
                            "path_index": record.get("path_index"),
                            "filename": record.get("filename"),
                            "error_type": "exception",
                            "reason": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if processed % 100 == 0 or args.debug:
                print(
                    f"[progress] processed={processed} parsed_success={parsed_success} "
                    f"errors={parse_failures}",
                    flush=True,
                )
            if args.debug:
                break

    temp_output.replace(output_jsonl)
    print(
        f"[done] processed={processed} parsed_success={parsed_success} "
        f"parsed_failed={processed - parsed_success} errors={parse_failures}",
        flush=True,
    )


if __name__ == "__main__":
    main()
