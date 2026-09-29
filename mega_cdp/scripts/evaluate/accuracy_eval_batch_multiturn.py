"""Evaluate final clinical decision conclusions for multi-turn predictions."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import re
import threading
import time



def render_template(template_text: str, vars: dict, strict: bool = True) -> str:
    """Fill prompt placeholders and optionally reject unresolved variables."""
    rendered = template_text
    for key, value in vars.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    if strict:
        import re as _re
        missing = sorted(set(_re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", rendered)))
        if missing:
            raise KeyError(f"Unresolved template variables: {missing}")
    return rendered
from openai import OpenAI
try:
    from .resume_utils import recover_newer_temp_output
    from .case_path_references import (
        build_aligned_reference_lines,
        load_case_path_references,
        load_reference_conclusion,
    )
except ImportError:
    from resume_utils import recover_newer_temp_output
    from case_path_references import (
        build_aligned_reference_lines,
        load_case_path_references,
        load_reference_conclusion,
    )


parser = argparse.ArgumentParser(description="Batch multiturn accuracy evaluation")
parser.add_argument("--model-name", type=str, default="Qwen3_8B", help="Model/run name used for default output paths")
parser.add_argument(
    "--eval-method",
    type=str,
    default="qwen",
    choices=["gpt", "qwen", "bertscore"],
    help="Evaluation method to use",
)
parser.add_argument("--tensor-parallel-size", type=int, default=1)
parser.add_argument(
    "--qwen-base-url",
    type=str,
    default="http://127.0.0.1:30000/v1",
    help="OpenAI-compatible base URL for Qwen3-8B.",
)
parser.add_argument(
    "--qwen-model-name",
    type=str,
    default="Qwen3_8B",
    help="Model name to send in OpenAI-compatible requests.",
)
parser.add_argument(
    "--qwen-max-tokens",
    type=int,
    default=4096,
    help="Max tokens for Qwen3-8B generation.",
)
parser.add_argument(
    "--request-workers",
    type=int,
    default=1,
    help="Number of Qwen evaluator cases to process concurrently. Default 1 keeps serial behavior.",
)
parser.add_argument(
    "--qwen-request-timeout-seconds",
    type=float,
    default=0.0,
    help="OpenAI-compatible request timeout in seconds for Qwen evaluator. <=0 disables client-side timeout.",
)
parser.add_argument("--parsed-jsonl", type=Path, default=None)
parser.add_argument("--case-paths-jsonl", type=Path, default=None)
parser.add_argument("--output-jsonl", type=Path, default=None)
parser.add_argument("--results-root", type=Path, default=None)
parser.add_argument(
    "--debug",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Debug mode: only process a few samples and write detailed inputs/outputs.",
)
parser.add_argument(
    "--resume",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Resume from existing output jsonl if it exists (default: True)",
)
args = parser.parse_args()

MODEL_NAME = args.model_name
EVAL_METHOD = args.eval_method
MAX_RETRIES = 5
QWEN_BASE_URL = args.qwen_base_url
QWEN_REQUEST_MODEL_NAME = args.qwen_model_name
QWEN_MAX_TOKENS = args.qwen_max_tokens
QWEN_REQUEST_TIMEOUT_SECONDS = args.qwen_request_timeout_seconds
QWEN_REQUEST_RETRIES = 3
BERTSCORE_LANG_EN = "en"
BERTSCORE_LANG_ZH = "zh"
BERTSCORE_MODEL_EN = "roberta-large"
BERTSCORE_MODEL_ZH = "bert-base-chinese"
BERTSCORE_SCALE = 1.0
BERTSCORE_BATCH_SIZE = 64
TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
DEBUG_MODE = args.debug
DEBUG_MAX_SAMPLES = 3

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = PROJECT_ROOT / "mega_cdp" / "template"
TEMPLATE_PATH = TEMPLATE_ROOT / "template_en" / "ACCURACY_EVAL_TEMPLATE.txt"
TEMPLATE_PATH_ZH = TEMPLATE_ROOT / "template_zh" / "ZH_ACCURACY_EVAL_TEMPLATE.txt"
MODEL_MULTITURN_NAME = MODEL_NAME

DEFAULT_TEST_ROOT = PROJECT_ROOT / "datasets" / "mega_cdp" / "test"
default_results_root = PROJECT_ROOT / "results" / "prediction_multiturn_sglang"
RESULTS_ROOT = args.results_root or default_results_root
DEFAULT_PARSED_JSONL = (
    RESULTS_ROOT
    / MODEL_MULTITURN_NAME
    / f"{MODEL_MULTITURN_NAME}_prediction_state_parsed.jsonl"
)
DEFAULT_CASE_PATHS_JSONL = DEFAULT_TEST_ROOT / "case_paths.jsonl"


with open(TEMPLATE_PATH, "r", encoding="utf-8") as file:
    TEMPLATE = file.read()
with open(TEMPLATE_PATH_ZH, "r", encoding="utf-8") as file:
    TEMPLATE_ZH = file.read()

def append_error_record(error_path: Path, record: dict) -> None:
    """Append one structured error record to an errors JSONL file."""
    if "time" not in record:
        record["time"] = datetime.now().isoformat(timespec="seconds")
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with _ERROR_LOCK:
        with error_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stringify_step(step):
    """Convert a pathway step into prompt text."""
    if isinstance(step, str):
        return step
    return json.dumps(step, ensure_ascii=False, sort_keys=True)


def generate_consistency_matrix_bertscore(
    first_pathway, second_pathway, bertscore_lang, bertscore_model
):
    """Compute a BERTScore similarity matrix for a reference and candidate pathway."""
    first_steps = [_stringify_step(step) for step in first_pathway]
    second_steps = [_stringify_step(step) for step in second_pathway]
    if not first_steps or not second_steps:
        return []

    candidates = []
    references = []
    for first_step in first_steps:
        for second_step in second_steps:
            candidates.append(second_step)
            references.append(first_step)

    scorer = _get_bertscore_scorer(bertscore_lang, bertscore_model)
    _, _, f1 = scorer.score(candidates, references, batch_size=BERTSCORE_BATCH_SIZE)
    f1_scores = f1.tolist()
    matrix = []
    idx = 0
    for _ in first_steps:
        row = []
        for _ in second_steps:
            row.append(round(f1_scores[idx] * BERTSCORE_SCALE, 6))
            idx += 1
        matrix.append(row)
    return matrix


_QWEN_CLIENT_LOCAL = threading.local()
_ERROR_LOCK = threading.Lock()
_BERT_SCORERS = {}
_BERTSCORER_CLASS = None


def _get_bertscore_scorer(bertscore_lang, bertscore_model):
    """Return a cached BERTScore scorer."""
    if _BERTSCORER_CLASS is None:
        raise RuntimeError(
            "BERTScorer is not initialized. Use --eval-method bertscore to enable it."
        )
    key = (bertscore_lang, bertscore_model or "")
    scorer = _BERT_SCORERS.get(key)
    if scorer is not None:
        return scorer
    scorer_kwargs = {"lang": bertscore_lang, "rescale_with_baseline": True}
    if bertscore_model:
        scorer_kwargs["model_type"] = bertscore_model
    scorer = _BERTSCORER_CLASS(**scorer_kwargs)
    _BERT_SCORERS[key] = scorer
    return scorer


def _get_qwen_client():
    """Return a thread-local judge client."""
    client = getattr(_QWEN_CLIENT_LOCAL, "client", None)
    if client is not None:
        return client
    timeout = None if QWEN_REQUEST_TIMEOUT_SECONDS <= 0 else QWEN_REQUEST_TIMEOUT_SECONDS
    client = OpenAI(api_key=os.getenv("API_KEY", "EMPTY"), base_url=QWEN_BASE_URL, timeout=timeout)
    _QWEN_CLIENT_LOCAL.client = client
    return client


def format_pathway_lines(pathway, prefix):
    """Format structured data into the prompt or label representation expected downstream."""
    lines = []
    for idx, step in enumerate(pathway, start=1):
        step_text = _stringify_step(step)
        lines.append(f"{prefix}{idx}: {step_text}")
    return "\n".join(lines)


def generate_consistency_matrix_qwen(
    template_text,
    reference_conclusion_text,
    candidate_conclusion_text,
):
    """Request final-conclusion scores from the judge model."""
    prompt = render_template(
        template_text,
        vars={
            "Reference_Conclusion": reference_conclusion_text,
            "Candidate_Conclusion": candidate_conclusion_text,
        },
        strict=True,
    )
    message = [{"role": "system", "content": prompt}]
    client = _get_qwen_client()
    for attempt in range(QWEN_REQUEST_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=QWEN_REQUEST_MODEL_NAME,
                messages=message,
                max_tokens=QWEN_MAX_TOKENS,
                temperature=0.7,
                top_p=0.8,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            return resp.choices[0].message.content
        except Exception:
            if attempt < QWEN_REQUEST_RETRIES - 1:
                time.sleep(2 ** attempt)
        return None


def clean_response(response):
    """Remove wrappers such as markdown fences from judge model output."""
    response = response.strip()
    if response.startswith("```"):
        first_newline = response.find("\n")
        response = response[first_newline + 1 :] if first_newline != -1 else ""
    if response.endswith("```"):
        response = response[:-3].strip()
    return response


def validate_response_matrix(response_json, first_pathway, second_pathway):
    """Validate that a score matrix matches the expected pathway dimensions."""
    if len(response_json) != len(first_pathway):
        raise ValueError(
            "The number of consistency probabilities does not match the number of steps "
            "of second pathway."
        )
    for row in response_json:
        if len(row) != len(second_pathway):
            raise ValueError(
                "The number of consistency probabilities does not match the number of steps "
                "of first pathway."
            )


def parse_accuracy_score(response_text):
    """Parse one or more judge accuracy scores from a response string."""
    cleaned = clean_response(response_text)
    if not cleaned:
        raise ValueError("Empty response after cleaning.")
    match = re.fullmatch(r"[1-5]", cleaned)
    if match is None:
        raise ValueError(f"Invalid accuracy score format: {cleaned!r}")
    return int(cleaned)

def _extract_gt_prompt_meta(gt_record: dict) -> dict:
    """Extract reference case identity metadata."""
    prompt_meta = gt_record.get("prompt")
    if isinstance(prompt_meta, dict):
        return prompt_meta
    if "guideline_id" in gt_record or "path_id" in gt_record:
        return {
            "case_path_id": gt_record.get("guideline_id"),
            "path_index": gt_record.get("path_id"),
            "filename": gt_record.get("filename"),
            "language": gt_record.get("language"),
        }
    return gt_record


def _assert_same_case_fields(parsed_record: dict, gt_record: dict, line_number: int) -> dict:
    """Validate that prediction and reference rows describe the same case."""
    gt_meta = _extract_gt_prompt_meta(gt_record)
    fields = ["case_path_id", "path_index", "filename"]
    for field in fields:
        if parsed_record.get(field) != gt_meta.get(field):
            raise ValueError(
                f"Mismatch at line {line_number} for field '{field}': "
                f"parsed={parsed_record.get(field)!r}, gt={gt_meta.get(field)!r}"
            )
    return gt_meta


def _load_pred_outcome(parsed_record: dict) -> str:
    """Extract the predicted final conclusion."""
    outcome = parsed_record.get("clinical_decision_outcome")
    if isinstance(outcome, str):
        outcome = outcome.strip()
        if outcome:
            return outcome

    llm_output = parsed_record.get("llm_output")
    if isinstance(llm_output, list):
        if llm_output:
            last_item = llm_output[-1]
            if isinstance(last_item, str):
                match = re.search(r"(Final Conclusion|最终结论)\s*[:：]?\s*(.+)", last_item, re.IGNORECASE)
                if match and match.group(2).strip():
                    return match.group(2).strip()
        raise ValueError("Missing clinical_decision_outcome in llm_output list")
    if isinstance(llm_output, dict):
        outcome = llm_output.get("clinical_decision_outcome")
        if isinstance(outcome, str) and outcome.strip():
            return outcome.strip()
        raise ValueError("Missing clinical_decision_outcome in llm_output")
    if isinstance(llm_output, str):
        try:
            parsed = json.loads(llm_output)
        except json.JSONDecodeError as exc:
            raise ValueError("llm_output is not valid JSON") from exc
        if isinstance(parsed, dict):
            outcome = parsed.get("clinical_decision_outcome")
            if isinstance(outcome, str) and outcome.strip():
                return outcome.strip()
            raise ValueError("Missing clinical_decision_outcome in llm_output JSON")
    raise ValueError("Unsupported llm_output format")


def _load_gt_conclusion(reference_record: dict) -> str:
    """Extract the reference final conclusion."""
    return load_reference_conclusion(reference_record)


def _resolve_language_settings(language: str) -> dict:
    """Select templates and scorer settings for a language."""
    if language == "zh":
        return {
            "template_text": TEMPLATE_ZH,
            "bertscore_lang": BERTSCORE_LANG_ZH,
            "bertscore_model": BERTSCORE_MODEL_ZH,
        }
    return {
        "template_text": TEMPLATE,
        "bertscore_lang": BERTSCORE_LANG_EN,
        "bertscore_model": BERTSCORE_MODEL_EN,
    }


def _build_debug_record(
    parsed_line: str,
    gt_line: str,
    parsed_record: dict,
    gt_record: dict,
    gt_outcome=None,
    pred_outcome=None,
    prompt=None,
    response_text=None,
):
    """Build a diagnostic record for failed evaluations."""
    return {
        "parsed_jsonl_line": parsed_line.rstrip("\n"),
        "gt_jsonl_line": gt_line.rstrip("\n"),
        "parsed_record": parsed_record,
        "gt_record": gt_record,
        "gt_outcome": gt_outcome,
        "pred_outcome": pred_outcome,
        "prompt": prompt,
        "response_text": [] if response_text is None else response_text,
    }


def evaluate_qwen_line(
    *,
    line_number: int,
    parsed_line: str,
    gt_line: str,
    existing_line: str | None,
    error_path: Path,
) -> dict:
    """Evaluate one parsed prediction record with the Qwen judge."""
    print(f"processing line {line_number}", flush=True)
    if parsed_line is None or gt_line is None:
        raise ValueError("Parsed jsonl and GT jsonl lengths do not match")
    if not parsed_line.strip():
        raise ValueError(f"Empty parsed jsonl line at {line_number}")
    if not gt_line.strip():
        raise ValueError(f"Empty GT jsonl line at {line_number}")

    parsed_record = json.loads(parsed_line)
    gt_record = json.loads(gt_line)
    gt_meta = _assert_same_case_fields(parsed_record, gt_record, line_number)

    language = parsed_record.get("language") or gt_meta.get("language")
    if language is None:
        raise ValueError(f"Missing language at line {line_number}")
    if gt_meta.get("language") and parsed_record.get("language"):
        if gt_meta.get("language") != parsed_record.get("language"):
            raise ValueError(f"Language mismatch at line {line_number}")

    output_record = {
        "id": parsed_record.get("id"),
        "case_path_id": parsed_record.get("case_path_id"),
        "path_index": parsed_record.get("path_index"),
        "filename": parsed_record.get("filename"),
        "language": language,
        "eval_method": EVAL_METHOD,
        "reference_source": "case_paths",
    }

    parsed_success = parsed_record.get("parsed_success")
    if parsed_success is not True:
        output_record["skipped"] = True
        output_record["reason"] = "parsed_success is not true"
        return output_record

    try:
        gt_outcome = _load_gt_conclusion(gt_record)
        pred_outcome = _load_pred_outcome(parsed_record)
    except Exception as exc:
        output_record["skipped"] = True
        output_record["reason"] = str(exc)
        append_error_record(
            error_path,
            {
                "model": MODEL_NAME,
                "case_index": line_number,
                "language": language,
                "reason": str(exc),
            },
        )
        return output_record

    existing_record = None
    if existing_line:
        try:
            existing_record = json.loads(existing_line)
        except json.JSONDecodeError:
            existing_record = None
    if not should_reuse_existing_eval_record(existing_record, parsed_record):
        existing_record = None
    if existing_record is not None:
        output_record = existing_record

    cur_response_list = []
    if existing_record is not None:
        existing_results = output_record.get("results")
        if isinstance(existing_results, list):
            cur_response_list = list(existing_results)

    if len(cur_response_list) >= MAX_RETRIES:
        output_record["results"] = cur_response_list[:MAX_RETRIES]
        return output_record

    template_text = _resolve_language_settings(language)["template_text"]
    attempts = 0
    while len(cur_response_list) < MAX_RETRIES and attempts < MAX_RETRIES * 5:
        attempts += 1
        response_text = None
        try:
            response_text = generate_consistency_matrix_qwen(
                template_text,
                gt_outcome,
                pred_outcome,
            )
            if not response_text:
                raise ValueError("Empty Qwen evaluator response")
            score = parse_accuracy_score(response_text)
            cur_response_list.append(score)
        except Exception as exc:
            append_error_record(
                error_path,
                {
                    "model": MODEL_NAME,
                    "case_index": line_number,
                    "language": language,
                    "reason": str(exc),
                    "attempt": attempts,
                    "response": response_text,
                },
            )
            continue

    output_record["results"] = cur_response_list
    return output_record


def should_reuse_existing_eval_record(existing_record: dict | None, parsed_record: dict) -> bool:
    """Return whether a resumed eval row is still consistent with the latest parse."""
    if existing_record is None:
        return False
    if existing_record.get("reference_source") != "case_paths":
        return False

    for key in ("case_path_id", "path_index", "filename"):
        parsed_value = parsed_record.get(key)
        existing_value = existing_record.get(key)
        if parsed_value is not None and existing_value is not None and parsed_value != existing_value:
            return False

    parsed_success = parsed_record.get("parsed_success") is True
    stale_parse_failure = (
        existing_record.get("skipped") is True
        and existing_record.get("reason") == "parsed_success is not true"
    )
    if parsed_success and stale_parse_failure:
        return False

    return True


def run_qwen_parallel_eval(
    *,
    parsed_lines: list[str],
    gt_lines: list[str],
    existing_output_lines: list[str],
    temp_output_jsonl: Path,
    output_jsonl: Path,
    error_path: Path,
    resume_enabled: bool,
) -> None:
    """Run Qwen judge evaluation concurrently while preserving resumability."""
    if len(parsed_lines) != len(gt_lines):
        raise ValueError("Parsed jsonl and GT jsonl lengths do not match")

    print(
        f"[parallel] request_workers={args.request_workers}, lines={len(parsed_lines)}, "
        f"timeout={'disabled' if QWEN_REQUEST_TIMEOUT_SECONDS <= 0 else QWEN_REQUEST_TIMEOUT_SECONDS}",
        flush=True,
    )
    next_write = 1
    pending: dict[int, dict] = {}
    with temp_output_jsonl.open("w", encoding="utf-8") as output_file:
        with ThreadPoolExecutor(max_workers=args.request_workers) as executor:
            future_to_line: dict = {}
            for line_number, (parsed_line, gt_line) in enumerate(zip(parsed_lines, gt_lines), start=1):
                existing_line = (
                    existing_output_lines[line_number - 1].strip()
                    if resume_enabled and line_number <= len(existing_output_lines)
                    else None
                )
                future = executor.submit(
                    evaluate_qwen_line,
                    line_number=line_number,
                    parsed_line=parsed_line,
                    gt_line=gt_line,
                    existing_line=existing_line,
                    error_path=error_path,
                )
                future_to_line[future] = line_number

            completed = 0
            for future in as_completed(future_to_line):
                line_number = future_to_line[future]
                record = future.result()
                completed += 1
                pending[line_number] = record
                print(f"[parallel] completed {completed}/{len(future_to_line)} line={line_number}", flush=True)
                while next_write in pending:
                    output_file.write(json.dumps(pending.pop(next_write), ensure_ascii=False) + "\n")
                    output_file.flush()
                    next_write += 1

    if resume_enabled:
        temp_output_jsonl.replace(output_jsonl)


def main():
    """Run the command-line entry point."""
    if args.request_workers < 1:
        raise ValueError("--request-workers must be >= 1")
    parsed_jsonl = args.parsed_jsonl or DEFAULT_PARSED_JSONL
    case_paths_jsonl = args.case_paths_jsonl or DEFAULT_CASE_PATHS_JSONL
    output_jsonl = args.output_jsonl
    if output_jsonl is None:
        if EVAL_METHOD == "gpt":
            output_jsonl = (
                RESULTS_ROOT / MODEL_MULTITURN_NAME / f"prompt_{MODEL_MULTITURN_NAME}_accuracy_eval.jsonl"
            )
        else:
            output_jsonl = (
                RESULTS_ROOT
                / MODEL_MULTITURN_NAME
                / f"{MODEL_MULTITURN_NAME}_accuracy_eval_{EVAL_METHOD}.jsonl"
            )

    if not parsed_jsonl.exists():
        raise FileNotFoundError(f"Parsed jsonl not found: {parsed_jsonl}")
    if not case_paths_jsonl.exists():
        raise FileNotFoundError(f"case_paths jsonl not found: {case_paths_jsonl}")

    with parsed_jsonl.open("r", encoding="utf-8") as parsed_file:
        parsed_lines = parsed_file.readlines()
    references = load_case_path_references(case_paths_jsonl)
    gt_lines = build_aligned_reference_lines(parsed_lines, references)

    if EVAL_METHOD == "bertscore":
        try:
            from bert_score import BERTScorer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "BERTScore is not available. Install bert-score to use --eval-method bertscore."
            ) from exc
        global _BERTSCORER_CLASS
        _BERTSCORER_CLASS = BERTScorer

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    error_path = output_jsonl.with_suffix(".errors.jsonl")

    if args.resume and not DEBUG_MODE and EVAL_METHOD in {"qwen", "bertscore"}:
        recover_newer_temp_output(output_jsonl)
    resume_enabled = (
        args.resume
        and not DEBUG_MODE
        and EVAL_METHOD in {"qwen", "bertscore"}
        and output_jsonl.exists()
    )
    existing_output_lines = []
    if resume_enabled:
        with output_jsonl.open("r", encoding="utf-8") as existing_file:
            existing_output_lines = existing_file.readlines()
        temp_output_jsonl = output_jsonl.with_name(f"{output_jsonl.name}.tmp")
    else:
        temp_output_jsonl = output_jsonl

    if EVAL_METHOD == "qwen" and args.request_workers > 1 and not DEBUG_MODE:
        run_qwen_parallel_eval(
            parsed_lines=parsed_lines,
            gt_lines=gt_lines,
            existing_output_lines=existing_output_lines,
            temp_output_jsonl=temp_output_jsonl,
            output_jsonl=output_jsonl,
            error_path=error_path,
            resume_enabled=resume_enabled,
        )
        return

    with temp_output_jsonl.open("w", encoding="utf-8") as output_file:
        output_line_number = 0
        for line_number, (parsed_line, gt_line) in enumerate(
            zip(parsed_lines, gt_lines), start=1
        ):
            if DEBUG_MODE and line_number > DEBUG_MAX_SAMPLES:
                break
            print(f"processing line {line_number}")
            if parsed_line is None or gt_line is None:
                raise ValueError("Parsed jsonl and GT jsonl lengths do not match")
            if not parsed_line.strip():
                raise ValueError(f"Empty parsed jsonl line at {line_number}")
            if not gt_line.strip():
                raise ValueError(f"Empty GT jsonl line at {line_number}")

            parsed_record = json.loads(parsed_line)
            gt_record = json.loads(gt_line)
            gt_meta = _assert_same_case_fields(parsed_record, gt_record, line_number)

            language = parsed_record.get("language") or gt_meta.get("language")
            if language is None:
                raise ValueError(f"Missing language at line {line_number}")
            if gt_meta.get("language") and parsed_record.get("language"):
                if gt_meta.get("language") != parsed_record.get("language"):
                    raise ValueError(f"Language mismatch at line {line_number}")

            output_record = {
                "id": parsed_record.get("id"),
                "case_path_id": parsed_record.get("case_path_id"),
                "path_index": parsed_record.get("path_index"),
                "filename": parsed_record.get("filename"),
                "language": language,
                "eval_method": EVAL_METHOD,
                "reference_source": "case_paths",
            }
            if DEBUG_MODE:
                output_record["debug"] = _build_debug_record(
                    parsed_line=parsed_line,
                    gt_line=gt_line,
                    parsed_record=parsed_record,
                    gt_record=gt_record,
                )

            parsed_success = parsed_record.get("parsed_success")
            if parsed_success is not True:
                if EVAL_METHOD == "gpt":
                    continue
                output_record["skipped"] = True
                output_record["reason"] = "parsed_success is not true"
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                output_file.flush()
                continue

            try:
                gt_outcome = _load_gt_conclusion(gt_record)
                pred_outcome = _load_pred_outcome(parsed_record)
                if DEBUG_MODE:
                    output_record["debug"]["gt_outcome"] = gt_outcome
                    output_record["debug"]["pred_outcome"] = pred_outcome
            except Exception as exc:
                output_record["skipped"] = True
                output_record["reason"] = str(exc)
                append_error_record(
                    error_path,
                    {
                        "model": MODEL_NAME,
                        "case_index": line_number,
                        "language": language,
                        "reason": str(exc),
                    },
                )
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                output_file.flush()
                continue

            lang_settings = _resolve_language_settings(language)
            template_text = lang_settings["template_text"]
            gt_items = [gt_outcome]
            pred_items = [pred_outcome]

            if EVAL_METHOD == "gpt":
                first_pathway_text = format_pathway_lines(gt_items, "R")
                second_pathway_text = format_pathway_lines(pred_items, "C")
                prompt = render_template(
                    template_text,
                    vars={
                        "Reference_Conclusion": gt_outcome,
                        "Candidate_Conclusion": pred_outcome,
                    },
                    strict=True,
                )
                output_record["text"] = prompt
                if DEBUG_MODE:
                    output_record["debug"]["prompt"] = prompt
                for _ in range(MAX_RETRIES):
                    output_line_number += 1
                    output_record["id"] = output_line_number
                    output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                    output_file.flush()
                continue

            existing_record = None
            if resume_enabled and line_number <= len(existing_output_lines):
                existing_line = existing_output_lines[line_number - 1].strip()
                if existing_line:
                    try:
                        existing_record = json.loads(existing_line)
                    except json.JSONDecodeError:
                        existing_record = None
            if not should_reuse_existing_eval_record(existing_record, parsed_record):
                existing_record = None

            if EVAL_METHOD == "bertscore":
                print(f"processing line {line_number}")
                if existing_record is not None and not existing_record.get("skipped"):
                    existing_results = existing_record.get("results")
                    if isinstance(existing_results, list) and existing_results:
                        output_file.write(json.dumps(existing_record, ensure_ascii=False) + "\n")
                        output_file.flush()
                        continue
                try:
                    matrix = generate_consistency_matrix_bertscore(
                        gt_items,
                        pred_items,
                        lang_settings["bertscore_lang"],
                        lang_settings["bertscore_model"],
                    )
                    validate_response_matrix(matrix, gt_items, pred_items)
                    output_record["results"] = [matrix]
                    if DEBUG_MODE:
                        output_record["debug"]["response_text"] = matrix
                except Exception as exc:
                    output_record["skipped"] = True
                    output_record["reason"] = str(exc)
                    append_error_record(
                        error_path,
                        {
                            "model": MODEL_NAME,
                            "case_index": line_number,
                            "language": language,
                            "reason": str(exc),
                        },
                    )
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                output_file.flush()
                continue

            if existing_record is not None:
                output_record = existing_record

            cur_response_list = []
            if existing_record is not None:
                existing_results = output_record.get("results")
                if isinstance(existing_results, list):
                    cur_response_list = list(existing_results)

            if len(cur_response_list) >= MAX_RETRIES:
                output_record["results"] = cur_response_list[:MAX_RETRIES]
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                output_file.flush()
                continue

            attempts = 0
            if DEBUG_MODE:
                output_record["debug"]["prompt"] = render_template(
                    template_text,
                    vars={
                        "Reference_Conclusion": gt_outcome,
                        "Candidate_Conclusion": pred_outcome,
                    },
                    strict=True,
                )
            while len(cur_response_list) < MAX_RETRIES and attempts < MAX_RETRIES * 5:
                attempts += 1
                response_text = None
                try:
                    response_text = generate_consistency_matrix_qwen(
                        template_text,
                        gt_outcome,
                        pred_outcome,
                    )
                    score = parse_accuracy_score(response_text)
                    cur_response_list.append(score)
                    if DEBUG_MODE:
                        output_record["debug"].setdefault("response_text", [])
                        output_record["debug"]["response_text"].append(response_text)
                except Exception as exc:
                    append_error_record(
                        error_path,
                        {
                            "model": MODEL_NAME,
                            "case_index": line_number,
                            "language": language,
                            "reason": str(exc),
                            "attempt": attempts,
                            "response": response_text
                        },
                    )
                    continue

            output_record["results"] = cur_response_list
            output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            output_file.flush()

    if resume_enabled:
        temp_output_jsonl.replace(output_jsonl)


if __name__ == "__main__":
    main()
