#!/usr/bin/env python3
"""Evaluate final clinical decision conclusions for single-turn predictions."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from .gt_path_references import load_gt_path_references, resolve_reference, load_reference_conclusion
    from .resume_utils import recover_newer_temp_output
except ImportError:
    from gt_path_references import load_gt_path_references, resolve_reference, load_reference_conclusion
    from resume_utils import recover_newer_temp_output


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ROOT = PROJECT_ROOT / "mega_cdp" / "template"
DEFAULT_TEST_ROOT = PROJECT_ROOT / "datasets" / "mega_cdp" / "test"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "prediction_singleturn_sglang"
TEMPLATE_EN = (TEMPLATE_ROOT / "template_en" / "ACCURACY_EVAL_TEMPLATE.txt").read_text(encoding="utf-8")
TEMPLATE_ZH = (TEMPLATE_ROOT / "template_zh" / "ZH_ACCURACY_EVAL_TEMPLATE.txt").read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for single-turn accuracy evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--qwen-base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--qwen-model-name", default="Qwen3_8B")
    parser.add_argument("--qwen-max-tokens", type=int, default=4096)
    parser.add_argument("--qwen-request-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--request-workers", type=int, default=1)
    parser.add_argument("--expected-results", type=int, default=5)
    parser.add_argument("--parsed-jsonl", type=Path, default=None)
    parser.add_argument("--gt-paths-jsonl", type=Path, default=DEFAULT_TEST_ROOT / "gt_paths.jsonl")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def render_template(template: str, values: dict[str, Any]) -> str:
    """Fill prompt placeholders and reject unresolved variables."""
    output = template
    for key, value in values.items():
        output = output.replace("{" + key + "}", str(value))
    missing = sorted(set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", output)))
    if missing:
        raise KeyError(f"Unresolved template variables: {missing}")
    return output


def clean_response(text: str) -> str:
    """Normalize a judge response before score parsing."""
    text = (text or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1:] if first_newline != -1 else ""
    if text.endswith("```"):
        text = text[:-3].strip()
    return text.strip()


def parse_accuracy_score(text: str) -> int:
    """Parse a 1-5 final-conclusion consistency score."""
    cleaned = clean_response(text)
    if re.fullmatch(r"[1-5]", cleaned) is None:
        raise ValueError(f"Invalid accuracy score: {cleaned!r}")
    return int(cleaned)


def candidate_conclusion(record: dict[str, Any]) -> str:
    """Extract the parsed predicted final conclusion."""
    if record.get("parsed_success") is not True:
        raise ValueError("parsed_success is not true")
    llm_output = record.get("llm_output")
    if not isinstance(llm_output, dict):
        raise ValueError("llm_output must be a parsed object")
    conclusion = llm_output.get("clinical_decision_outcome")
    if not isinstance(conclusion, str) or not conclusion.strip():
        raise ValueError("clinical_decision_outcome is missing")
    return conclusion.strip()


def qwen_chat(client: OpenAI, args: argparse.Namespace, prompt: str) -> str:
    """Call the Qwen judge through an OpenAI-compatible endpoint."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=args.qwen_model_name,
                messages=[{"role": "system", "content": prompt}],
                max_tokens=args.qwen_max_tokens,
                temperature=0.7,
                top_p=0.8,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            )
            return response.choices[0].message.content or ""
        except Exception:
            if attempt >= 2:
                raise
            time.sleep(2**attempt)
    return ""


def evaluate_record(record: dict[str, Any], references: dict, client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one parsed prediction record."""
    output = {
        "id": record.get("id"),
        "case_path_id": record.get("case_path_id"),
        "path_index": record.get("path_index"),
        "filename": record.get("filename"),
        "language": record.get("language"),
        "eval_method": "qwen",
        "reference_source": "gt_paths",
    }
    try:
        reference = resolve_reference(record, references)
        ref = load_reference_conclusion(reference)
        cand = candidate_conclusion(record)
        template = TEMPLATE_ZH if (output.get("language") or reference.get("language")) == "zh" else TEMPLATE_EN
        prompt = render_template(template, {"Reference_Conclusion": ref, "Candidate_Conclusion": cand})
        scores = []
        for _ in range(args.expected_results):
            scores.append(parse_accuracy_score(qwen_chat(client, args, prompt)))
        output["results"] = scores
    except Exception as exc:
        output["skipped"] = True
        output["reason"] = str(exc)
    return output


def load_existing(path: Path) -> dict[int, dict[str, Any]]:
    """Load existing records by 1-based line number for resume mode."""
    if not path.exists():
        return {}
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records[line_number] = json.loads(line)
                except json.JSONDecodeError:
                    pass
    return records


def main() -> int:
    """Run batch single-turn final-answer evaluation."""
    args = parse_args()
    parsed_jsonl = args.parsed_jsonl or DEFAULT_RESULTS_ROOT / args.model_name / f"{args.model_name}_prediction_output_parsed.jsonl"
    output_jsonl = args.output_jsonl or DEFAULT_RESULTS_ROOT / args.model_name / f"{args.model_name}_accuracy_eval_qwen.jsonl"
    if not parsed_jsonl.exists():
        raise FileNotFoundError(f"Parsed JSONL not found: {parsed_jsonl}")
    if not args.gt_paths_jsonl.exists():
        raise FileNotFoundError(f"gt_paths JSONL not found: {args.gt_paths_jsonl}")
    references = load_gt_path_references(args.gt_paths_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    recover_newer_temp_output(output_jsonl)
    existing = load_existing(output_jsonl) if args.resume else {}
    timeout = None if args.qwen_request_timeout_seconds <= 0 else args.qwen_request_timeout_seconds
    client = OpenAI(api_key=os.getenv("API_KEY", "EMPTY"), base_url=args.qwen_base_url, timeout=timeout)

    with parsed_jsonl.open("r", encoding="utf-8") as infile:
        parsed_records = [json.loads(line) for line in infile if line.strip()]
    outputs: list[dict[str, Any] | None] = [None] * len(parsed_records)
    pending = []
    for idx, record in enumerate(parsed_records, start=1):
        existing_record = existing.get(idx)
        if existing_record and (existing_record.get("skipped") or existing_record.get("results")):
            outputs[idx - 1] = existing_record
        else:
            pending.append((idx, record))

    if args.request_workers <= 1:
        for idx, record in pending:
            print(f"[accuracy] processing {idx}/{len(parsed_records)}", flush=True)
            outputs[idx - 1] = evaluate_record(record, references, client, args)
    else:
        with ThreadPoolExecutor(max_workers=args.request_workers) as executor:
            futures = {executor.submit(evaluate_record, record, references, client, args): idx for idx, record in pending}
            for done, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                outputs[idx - 1] = future.result()
                print(f"[accuracy] completed {done}/{len(pending)} index={idx}", flush=True)

    temp = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as outfile:
        for output in outputs:
            outfile.write(json.dumps(output, ensure_ascii=False) + "\n")
    temp.replace(output_jsonl)
    print(f"[done] wrote {output_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
