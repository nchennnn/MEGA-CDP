#!/usr/bin/env python3
"""Run single-turn MEGA-CDP inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from anthropic import Anthropic

BACKENDS = ("auto", "sglang", "openai", "anthropic")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEST_ROOT = PROJECT_ROOT / "datasets" / "mega_cdp" / "test"
TEMPLATE_ROOT = PROJECT_ROOT / "mega_cdp" / "template"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "prediction_singleturn_sglang"


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """Add dataset, output, and resume arguments."""
    parser.add_argument("--experiment-name", default=None, help="Filesystem-safe run name.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_TEST_ROOT / "vignette_cases.jsonl",
    )
    parser.add_argument(
        "--guidelines-root",
        type=Path,
        default=DEFAULT_TEST_ROOT / "guideline_md",
    )
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--line-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="1-based inclusive range of non-empty input records.",
    )


def load_template(language: str) -> str:
    """Load the original MEGA-CDP single-turn prompt for one language."""
    if language == "zh":
        path = TEMPLATE_ROOT / "template_zh" / "ZH_DECISION_MAKING_PREDICTION_TEMPLATE.txt"
    else:
        path = TEMPLATE_ROOT / "template_en" / "DECISION_MAKING_PREDICTION_TEMPLATE.txt"
    return path.read_text(encoding="utf-8")


def split_output_marker(prompt: str) -> tuple[str, str]:
    """Separate the output request for providers requiring a user message."""
    for marker in ("## Your Output:", "## 你的输出："):
        position = prompt.rfind(marker)
        if position >= 0:
            return prompt[:position].rstrip(), prompt[position:]
    return prompt, "Please provide the final answer based on the system instructions."


def split_system_for_claude_cache(system_prompt: str) -> tuple[str, str]:
    """Move case-specific text into the user message for providers that require it."""
    markers = ("## Case Vignette:", "## 病例摘要：")
    positions = [(marker, system_prompt.find(marker)) for marker in markers]
    valid = [(marker, position) for marker, position in positions if position != -1]
    if not valid:
        return system_prompt, ""
    _, position = min(valid, key=lambda item: item[1])
    return system_prompt[:position].rstrip(), system_prompt[position:].lstrip()


def extract_anthropic_text(response: Any) -> str:
    """Extract concatenated text blocks from an Anthropic response."""
    chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks).strip()


def load_guideline_text(
    guidelines_root: Path,
    filename: str,
    cache: dict[str, str],
) -> str:
    """Load one guideline document with a per-process cache."""
    if filename not in cache:
        path = guidelines_root / filename / f"{filename}.md"
        if not path.exists():
            raise FileNotFoundError(f"Guideline markdown not found: {path}")
        cache[filename] = path.read_text(encoding="utf-8")
    return cache[filename]


def record_identity(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Return the stable case identity used for exact resume matching."""
    return (
        record.get("case_path_id", record.get("guideline_id")),
        record.get("path_index", record.get("path_id")),
        record.get("filename"),
    )


def load_existing_records(path: Path) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
    """Load successful records from an existing output file."""
    records: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "error" or not isinstance(record.get("llm_output"), str):
                continue
            records[record_identity(record)] = record
    return records


def recover_newer_temp_output(output_path: Path) -> None:
    """Promote an interrupted temporary output when it has more rows."""
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if not temp_path.exists():
        return

    def count_rows(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    if count_rows(temp_path) > count_rows(output_path):
        temp_path.replace(output_path)


def run_prediction_dataset(
    args: argparse.Namespace,
    *,
    request_model_name: str,
    predict: Callable[[str, str], str],
) -> int:
    """Render selected cases, call the backend, and write resumable JSONL."""
    experiment_name = args.experiment_name or request_model_name
    experiment_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_name).strip("_") or "model"
    output_path = args.output_jsonl or (
        DEFAULT_RESULTS_ROOT / experiment_tag / f"{experiment_tag}_prediction_output.jsonl"
    )
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")
    if not args.guidelines_root.is_dir():
        raise FileNotFoundError(f"Guideline directory not found: {args.guidelines_root}")
    if args.line_range and args.line_range[0] > args.line_range[1]:
        raise ValueError("line-range START must not exceed END")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        recover_newer_temp_output(output_path)
    existing = load_existing_records(output_path) if args.resume else {}
    write_path = (
        output_path.with_suffix(output_path.suffix + ".tmp")
        if output_path.exists()
        else output_path
    )
    error_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    guideline_cache: dict[str, str] = {}
    templates = {"en": load_template("en"), "zh": load_template("zh")}
    selected_index = 0
    completed = 0

    with (
        args.input_jsonl.open("r", encoding="utf-8") as infile,
        write_path.open("w", encoding="utf-8") as outfile,
        error_path.open("w", encoding="utf-8") as errfile,
    ):
        for source_line, raw_line in enumerate(infile, start=1):
            if not raw_line.strip():
                continue
            selected_index += 1
            if args.line_range and not (
                args.line_range[0] <= selected_index <= args.line_range[1]
            ):
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {args.input_jsonl}:{source_line}") from exc
            filename = record.get("filename")
            case_text = record.get("text")
            language = record.get("language", "en")
            if not isinstance(filename, str) or not isinstance(case_text, str):
                raise ValueError(f"Missing filename or text at {args.input_jsonl}:{source_line}")

            previous = existing.get(record_identity(record))
            if previous is not None:
                output_record = previous
            else:
                output_id = record.get("id", selected_index)
                case_path_id = record.get("case_path_id", record.get("guideline_id"))
                path_index = record.get("path_index", record.get("path_id"))
                if case_path_id is None or path_index is None:
                    raise ValueError(
                        f"Missing case_path_id/guideline_id or path_index/path_id at "
                        f"{args.input_jsonl}:{source_line}"
                    )
                try:
                    guideline = load_guideline_text(
                        args.guidelines_root, filename, guideline_cache
                    )
                    template = templates["zh" if language == "zh" else "en"]
                    prompt = template.replace("{Guideline}", guideline).replace("{Case}", case_text)
                    response = predict(prompt, filename)
                    if not isinstance(response, str) or not response.strip():
                        raise ValueError("Model returned an empty response")
                    output_record = {
                        "id": output_id,
                        "case_path_id": case_path_id,
                        "path_index": path_index,
                        "filename": filename,
                        "llm_output": response,
                        "language": language,
                        "model": request_model_name,
                        "status": "ok",
                    }
                except Exception as exc:
                    output_record = {
                        "id": output_id,
                        "case_path_id": case_path_id,
                        "path_index": path_index,
                        "filename": filename,
                        "language": language,
                        "model": request_model_name,
                        "status": "error",
                        "error": str(exc),
                    }
                    errfile.write(
                        json.dumps(
                            {
                                "source_line": source_line,
                                "case_identity": record_identity(record),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    errfile.flush()

            outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            outfile.flush()
            completed += 1
            print(
                f"[singleturn] completed={completed} case_index={selected_index}",
                flush=True,
            )

    if write_path != output_path:
        write_path.replace(output_path)
    print(f"[done] wrote {output_path}", flush=True)
    return 0


def resolve_backend(model_name: str, backend: str) -> str:
    """Resolve the backend from user settings and model name."""
    normalized = backend.lower()
    if normalized != "auto":
        return normalized
    name = model_name.lower()
    if "gpt-oss" in name or "gpt_oss" in name:
        return "sglang"
    if name.startswith(("gpt-", "gpt_")) or "/gpt-" in name or "gemini" in name:
        return "openai"
    if "claude" in name:
        return "anthropic"
    return "sglang"


def require_api_key(variable: str) -> str:
    """Read one provider credential without accepting keys on the command line."""
    value = os.getenv(variable)
    if not value:
        raise ValueError(f"Environment variable {variable} is required")
    return value


def provider_stripped_model_name(model_name: str) -> str:
    """Return the model id without a provider prefix such as google/."""
    return model_name.rsplit("/", 1)[-1]


def retrying_predict(
    call: Callable[[str, str], str],
    retries: int,
) -> Callable[[str, str], str]:
    """Wrap a provider call with bounded exponential-backoff retries."""
    def predict(prompt: str, filename: str) -> str:
        for attempt in range(retries):
            try:
                return call(prompt, filename)
            except Exception:
                if attempt + 1 >= retries:
                    raise
                time.sleep(2**attempt)
        return ""

    return predict


def parse_args() -> argparse.Namespace:
    """Collect backend, generation, and dataset options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        "--served-model-name",
        dest="model_name",
        required=True,
        help="Provider model ID or model name exposed by SGLang.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help="Infer common remote model names as OpenAI-compatible models, otherwise use SGLang.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Remote endpoint override. For SGLang, defaults to "
            "http://127.0.0.1:30000/v1; for openai, defaults to BASE_URL; "
            "for anthropic, defaults to ANTHROPIC_BASE_URL or the official API."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Qwen thinking mode; omit to use the model template default.",
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--enable-prompt-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--claude-cache-ttl", type=str, default="")
    add_dataset_arguments(parser)
    return parser.parse_args()


def build_predictor(args: argparse.Namespace, backend: str) -> Callable[[str, str], str]:
    """Create the selected local or remote inference client."""
    timeout = None if args.request_timeout_seconds <= 0 else args.request_timeout_seconds

    if backend == "sglang":
        sglang_base_url = args.base_url or "http://127.0.0.1:30000/v1"
        client = OpenAI(api_key="EMPTY", base_url=sglang_base_url, timeout=timeout)

        def call(prompt: str, filename: str) -> str:
            request: dict = {
                "model": args.model_name,
                "messages": [{"role": "system", "content": prompt}],
                "max_tokens": args.max_tokens,
            }
            if args.temperature is not None:
                request["temperature"] = args.temperature
            if args.top_p is not None:
                request["top_p"] = args.top_p
            extra_body = {}
            if args.top_k is not None:
                extra_body["top_k"] = args.top_k
            if args.enable_thinking is not None:
                extra_body["chat_template_kwargs"] = {
                    "enable_thinking": args.enable_thinking
                }
            if extra_body:
                request["extra_body"] = extra_body
            response = client.chat.completions.create(**request)
            return response.choices[0].message.content or ""

    elif backend == "openai":
        api_base_url = args.base_url or os.getenv("BASE_URL")
        if not api_base_url:
            raise ValueError("BASE_URL or --base-url is required for OpenAI-compatible inference")
        client = OpenAI(
            api_key=require_api_key("API_KEY"),
            base_url=api_base_url,
            timeout=timeout,
        )

        def call(prompt: str, filename: str) -> str:
            system_prompt, user_prompt = split_output_marker(prompt)
            messages = [{"role": "system", "content": prompt}]
            if "gemini" in args.model_name.lower():
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            request = {
                "model": args.model_name,
                "messages": messages,
                "max_completion_tokens": args.max_tokens,
            }
            if provider_stripped_model_name(args.model_name) == "gpt-5.4":
                request["extra_body"] = {
                    "prompt_cache_key": filename,
                    "prompt_cache_retention": "24h",
                }
            try:
                response = client.chat.completions.create(**request)
            except Exception:
                request.pop("max_completion_tokens", None)
                request["max_tokens"] = args.max_tokens
                response = client.chat.completions.create(**request)
            return response.choices[0].message.content or ""

    elif backend == "anthropic":
        api_base_url = args.base_url or os.getenv("ANTHROPIC_BASE_URL")
        client_kwargs: dict[str, Any] = {
            "api_key": require_api_key("API_KEY"),
            "timeout": timeout,
        }
        if api_base_url:
            client_kwargs["base_url"] = api_base_url
        client = Anthropic(**client_kwargs)

        def call(prompt: str, filename: str) -> str:
            system_prompt, user_prompt = split_output_marker(prompt)
            system_prefix, dynamic_suffix = split_system_for_claude_cache(system_prompt)
            final_user_prompt = (
                f"{dynamic_suffix}\n\n{user_prompt}".strip()
                if dynamic_suffix
                else user_prompt
            )
            system_block: dict[str, Any] = {"type": "text", "text": system_prefix}
            if args.enable_prompt_caching:
                cache_control: dict[str, Any] = {"type": "ephemeral"}
                if args.claude_cache_ttl:
                    cache_control["ttl"] = args.claude_cache_ttl
                system_block["cache_control"] = cache_control
            response = client.messages.create(
                model=args.model_name,
                max_tokens=args.max_tokens,
                system=[system_block],
                messages=[{"role": "user", "content": final_user_prompt}],
            )
            return extract_anthropic_text(response)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return retrying_predict(call, args.max_retries)


def main() -> int:
    """Select a backend and preserve one output schema across model providers."""
    args = parse_args()
    backend = resolve_backend(args.model_name, args.backend)
    print(f"[config] backend={backend} model={args.model_name}", flush=True)
    return run_prediction_dataset(
        args,
        request_model_name=args.model_name,
        predict=build_predictor(args, backend),
    )


if __name__ == "__main__":
    raise SystemExit(main())
