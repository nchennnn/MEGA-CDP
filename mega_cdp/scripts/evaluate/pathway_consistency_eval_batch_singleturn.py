#!/usr/bin/env python3
"""Evaluate pathway consistency for single-turn predictions."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from .gt_path_references import load_gt_path_references, resolve_reference, load_reference_steps
    from .pairwise_score_parser import parse_pairwise_score_matrix
    from .resume_utils import recover_newer_temp_output
except ImportError:
    from gt_path_references import load_gt_path_references, resolve_reference, load_reference_steps
    from pairwise_score_parser import parse_pairwise_score_matrix
    from resume_utils import recover_newer_temp_output


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ROOT = PROJECT_ROOT / "mega_cdp" / "template"
DEFAULT_TEST_ROOT = PROJECT_ROOT / "datasets" / "mega_cdp" / "test"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "prediction_singleturn_sglang"
TEMPLATE_EN = (TEMPLATE_ROOT / "template_en" / "PATHWAY_CONSISTENCY_EVAL_TEMPLATE_PATHWAY.txt").read_text(encoding="utf-8")
TEMPLATE_ZH = (TEMPLATE_ROOT / "template_zh" / "ZH_PATHWAY_CONSISTENCY_EVAL_TEMPLATE_PATHWAY.txt").read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for single-turn pathway evaluation."""
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
    """Validate template placeholders and insert values without parsing their contents."""
    pattern = r"\{([A-Za-z_][A-Za-z0-9_]*)\}"
    missing = sorted(set(re.findall(pattern, template)).difference(values))
    if missing:
        raise KeyError(f"Unresolved template variables: {missing}")
    return re.sub(pattern, lambda match: str(values[match.group(1)]), template)


def stringify_step(step: Any) -> str:
    """Convert a pathway step into text for judge prompts."""
    if isinstance(step, str):
        return step
    return json.dumps(step, ensure_ascii=False, sort_keys=True)


def format_pathway_lines(pathway: list[Any], prefix: str) -> str:
    """Format pathway steps as R1/R2 or C1/C2 prompt lines."""
    return "\n".join(f"{prefix}{idx}: {stringify_step(step)}" for idx, step in enumerate(pathway, start=1))


def clean_response(text: str) -> str:
    """Remove markdown fences around judge output."""
    text = (text or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1:] if first_newline != -1 else ""
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def get_candidate_steps(record: dict[str, Any]) -> list[str]:
    """Extract predicted single-turn decision-process steps from a parsed record."""
    if record.get("parsed_success") is not True:
        raise ValueError("parsed_success is not true")
    llm_output = record.get("llm_output")
    if not isinstance(llm_output, dict):
        raise ValueError("llm_output must be a parsed object")
    steps = llm_output.get("clinical_decision_process")
    if not isinstance(steps, list) or not steps:
        raise ValueError("clinical_decision_process must be a non-empty list")
    return [stringify_step(step) for step in steps]


def build_prompt(language: str, reference_steps: list[str], candidate_steps: list[str]) -> str:
    """Create the language-specific step-level consistency prompt."""
    template = TEMPLATE_ZH if language == "zh" else TEMPLATE_EN
    return render_template(
        template,
        {
            "Reference_Pathway": format_pathway_lines(reference_steps, "R"),
            "Candidate_Pathway": format_pathway_lines(candidate_steps, "C"),
            "Reference_Pathway_Length": len(reference_steps),
            "Candidate_Pathway_Length": len(candidate_steps),
        },
    )


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
        reference_steps = load_reference_steps(reference)
        candidate_steps = get_candidate_steps(record)
        prompt = build_prompt(output.get("language") or reference.get("language") or "en", reference_steps, candidate_steps)
        results = []
        for _ in range(args.expected_results):
            response_text = qwen_chat(client, args, prompt)
            results.append(
                parse_pairwise_score_matrix(response_text, reference_steps, candidate_steps)
            )
        output["results"] = results
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
    """Run batch single-turn pathway evaluation."""
    args = parse_args()
    parsed_jsonl = args.parsed_jsonl or DEFAULT_RESULTS_ROOT / args.model_name / f"{args.model_name}_prediction_output_parsed.jsonl"
    output_jsonl = args.output_jsonl or DEFAULT_RESULTS_ROOT / args.model_name / f"{args.model_name}_pathway_consistency_eval_qwen.jsonl"
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
            print(f"[pathway] processing {idx}/{len(parsed_records)}", flush=True)
            outputs[idx - 1] = evaluate_record(record, references, client, args)
    else:
        with ThreadPoolExecutor(max_workers=args.request_workers) as executor:
            futures = {executor.submit(evaluate_record, record, references, client, args): idx for idx, record in pending}
            for done, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                outputs[idx - 1] = future.result()
                print(f"[pathway] completed {done}/{len(pending)} index={idx}", flush=True)

    temp = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as outfile:
        for output in outputs:
            outfile.write(json.dumps(output, ensure_ascii=False) + "\n")
    temp.replace(output_jsonl)
    print(f"[done] wrote {output_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
