"""Run multi-turn MEGA-CDP inference."""

import os
import time
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Dict
import re

from openai import OpenAI
from anthropic import Anthropic


parser = argparse.ArgumentParser(description="Stepwise multi-turn doctor-patient prediction script")
parser.add_argument("--doctor-model-name", type=str, required=True, help="Doctor model name or path")
parser.add_argument("--patient-model-name", type=str, default="Qwen3_8B", help="Patient model name or path")
parser.add_argument(
    "--doctor-backend",
    choices=("auto", "sglang", "openai", "anthropic"),
    default="auto",
    help="Infer GPT/Gemini/Claude remote backends from model names; otherwise use SGLang.",
)
parser.add_argument("--base-host", type=str, default="127.0.0.1")
parser.add_argument(
    "--doctor-port",
    type=int,
    default=30000,
    help="Port for the local doctor SGLang service; ignored by remote doctor backends.",
)
parser.add_argument(
    "--patient-port",
    type=int,
    required=True,
    help="Port for patient OpenAI-compatible service",
)
parser.add_argument("--max-turns", type=int, default=30)
parser.add_argument("--max-tokens", type=int, default=2*1024)
parser.add_argument(
    "--max-retries",
    type=int,
    default=5,
    help="Maximum retry attempts for each doctor/patient generation call.",
)
parser.add_argument("--input-jsonl", type=Path, default=None, help="Path to vignette_cases.jsonl")
parser.add_argument("--output-jsonl", type=Path, default=None, help="Path to output jsonl (stateful)")
parser.add_argument(
    "--case-workers",
    type=int,
    default=1,
    help="Number of cases to process concurrently. Default 1 keeps the original serial behavior.",
)
parser.add_argument(
    "--request-timeout-seconds",
    type=float,
    default=0.0,
    help="OpenAI-compatible request timeout in seconds. <=0 disables the client-side timeout.",
)
parser.add_argument(
    "--guidelines-root",
    type=Path,
    default=None,
    help="Directory containing guideline_md/<filename>/<filename>.md.",
)
parser.add_argument(
    "--decision-trees-jsonl",
    type=Path,
    default=None,
    help="Path to decision_trees.jsonl used to load primary_clinical_focus.",
)
parser.add_argument(
    "--line-range",
    type=int,
    nargs=2,
    default=None,
    metavar=("START", "END"),
    help="1-based inclusive range of non-empty input records to process",
)
parser.add_argument("--enable-prompt-caching", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--claude-cache-ttl", type=str, default="")
args = parser.parse_args()

MAX_RETRIES = args.max_retries
MAX_ROUNDS_PER_LINE = args.max_turns
RUNNER_LOG_LOCK = threading.Lock()
RUNNER_LOGGED_KEYS: set[tuple[str, str, bool]] = set()

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent.parent
DEFAULT_TEST_ROOT = PROJECT_ROOT / "datasets" / "mega_cdp" / "test"
RESULTS_ROOT = PROJECT_ROOT / "results" / "prediction_multiturn_sglang"

GUIDELINES_ROOT = args.guidelines_root or (DEFAULT_TEST_ROOT / "guideline_md")
TEMPLATE_PATH = PROJECT_ROOT / "mega_cdp" / "template" / "template_en"
DOCTOR_TEMPLATE_PATH = TEMPLATE_PATH / "MULTITURN_DOCTOR_SYSTEM_TEMPLATE.txt"
PATIENT_TEMPLATE_PATH = TEMPLATE_PATH / "MULTITURN_PATIENT_SYSTEM_TEMPLATE.txt"
TEMPLATE_PATH_ZH = PROJECT_ROOT / "mega_cdp" / "template" / "template_zh"
DOCTOR_TEMPLATE_PATH_ZH = TEMPLATE_PATH_ZH / "ZH_MULTITURN_DOCTOR_SYSTEM_TEMPLATE.txt"
PATIENT_TEMPLATE_PATH_ZH = TEMPLATE_PATH_ZH / "ZH_MULTITURN_PATIENT_SYSTEM_TEMPLATE.txt"

doctor_template_path = DOCTOR_TEMPLATE_PATH
doctor_template_path_zh = DOCTOR_TEMPLATE_PATH_ZH

with open(doctor_template_path, "r", encoding="utf-8") as f:
    DOCTOR_SYSTEM_TEMPLATE = f.read()

with open(PATIENT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
    PATIENT_SYSTEM_TEMPLATE = f.read()

with open(doctor_template_path_zh, "r", encoding="utf-8") as f:
    DOCTOR_SYSTEM_TEMPLATE_ZH = f.read()

with open(PATIENT_TEMPLATE_PATH_ZH, "r", encoding="utf-8") as f:
    PATIENT_SYSTEM_TEMPLATE_ZH = f.read()

DEFAULT_INPUT_JSONL = DEFAULT_TEST_ROOT / "vignette_cases.jsonl"
DECISION_TREES_JSONL = args.decision_trees_jsonl or (DEFAULT_TEST_ROOT / "decision_trees.jsonl")
DEFAULT_OUTPUT_JSONL = (
    RESULTS_ROOT
    / args.doctor_model_name
    / (
        f"{args.doctor_model_name}_{args.patient_model_name}"
        f"_prediction_state.jsonl"
    )
)


@dataclass
class ChatModelRunner:
    """Common interface for chat-generation backends."""
    def generate(self, messages: List[Dict[str, str]], filename: str | None = None) -> str:
        """Generate one chat response."""
        raise NotImplementedError


class SGLangChatRunner(ChatModelRunner):
    """Call a locally deployed SGLang OpenAI-compatible endpoint."""
    def __init__(
        self,
        base_url: str,
        model_name: str,
        is_patient: bool,
    ):
        """Initialize the writer."""
        timeout = None if args.request_timeout_seconds <= 0 else args.request_timeout_seconds
        self.llm = OpenAI(api_key="EMPTY", base_url=base_url, timeout=timeout)
        self.is_patient = is_patient
        self.model_name = model_name
        log_key = ("sglang", model_name, is_patient)
        with RUNNER_LOG_LOCK:
            should_log = log_key not in RUNNER_LOGGED_KEYS
            if should_log:
                RUNNER_LOGGED_KEYS.add(log_key)
        if should_log:
            print(
                f"Using backend=sglang model={model_name}; is_patient={is_patient}",
                flush=True,
            )

    def generate(self, messages: List[Dict[str, str]], filename: str | None = None) -> str:
        """Generate one chat response."""
        prompt = messages
        model_role = "patient" if self.is_patient else "doctor"
        for attempt in range(MAX_RETRIES):
            try:
                if self.is_patient:
                    if self.model_name == "Qwen3_8B":
                        resp = self.llm.chat.completions.create(
                            model=model_role,
                            messages=prompt,
                            temperature=0.7,
                            top_p=0.8,
                            max_tokens=512,
                            extra_body={
                                "top_k": 20,
                                "chat_template_kwargs": {"enable_thinking": False},
                            },
                        )
                    else:
                        resp = self.llm.chat.completions.create(
                            model=model_role,
                            messages=prompt,
                            max_tokens=512,
                        )
                else:
                    resp = self.llm.chat.completions.create(
                        model=model_role,
                        messages=prompt,
                        max_tokens=args.max_tokens,
                    )
                response_message = resp.choices[0].message
                answer = response_message.content or ""
                reasoning = getattr(response_message, "reasoning_content", None)
                if reasoning is None:
                    model_extra = getattr(response_message, "model_extra", None) or {}
                    reasoning = model_extra.get("reasoning_content")
                if reasoning:
                    return f"<think>\n{reasoning.strip()}\n</think>\n\n{answer.strip()}"
                return answer
            except Exception as e:
                print(f"Error when generating {model_role} response: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(attempt)
        return None


def resolve_doctor_backend(model_name: str, backend: str) -> str:
    """Resolve the doctor backend from user settings and model name."""
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


def split_system_for_claude_cache(system_text: str) -> tuple[str, str]:
    """Move case-specific text into the user message for providers that require it."""
    markers = ("## Case Vignette:", "## 病例摘要：")
    positions = [(marker, system_text.find(marker)) for marker in markers]
    valid = [(marker, position) for marker, position in positions if position != -1]
    if not valid:
        return system_text, ""
    _, position = min(valid, key=lambda item: item[1])
    return system_text[:position].rstrip(), system_text[position:].lstrip()


def extract_anthropic_text(response: Any) -> str:
    """Extract concatenated text blocks from an Anthropic response."""
    chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks).strip()


class OpenAICompatibleChatRunner(ChatModelRunner):
    """Call a remote doctor through an OpenAI-compatible chat completion API."""

    def __init__(self, model_name: str):
        timeout = None if args.request_timeout_seconds <= 0 else args.request_timeout_seconds
        base_url = os.getenv("BASE_URL")
        if not base_url:
            raise ValueError("BASE_URL is required for OpenAI-compatible doctor inference")
        self.llm = OpenAI(
            api_key=require_api_key("API_KEY"),
            base_url=base_url,
            timeout=timeout,
        )
        self.model_name = model_name
        log_key = ("openai", model_name, False, False)
        with RUNNER_LOG_LOCK:
            should_log = log_key not in RUNNER_LOGGED_KEYS
            if should_log:
                RUNNER_LOGGED_KEYS.add(log_key)
        if should_log:
            print(f"Using backend=openai model={model_name}; is_patient=False", flush=True)

    def _create_completion(self, messages: List[Dict[str, str]], filename: str | None):
        """Call chat completion with a compatibility fallback."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": args.max_tokens,
        }
        if provider_stripped_model_name(self.model_name) == "gpt-5.4" and filename:
            kwargs["extra_body"] = {
                "prompt_cache_key": filename,
                "prompt_cache_retention": "24h",
            }
        try:
            return self.llm.chat.completions.create(**kwargs)
        except Exception:
            fallback_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": args.max_tokens,
            }
            return self.llm.chat.completions.create(**fallback_kwargs)

    def generate(self, messages: List[Dict[str, str]], filename: str | None = None) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                response = self._create_completion(messages, filename)
                return response.choices[0].message.content or ""
            except Exception as exc:
                print(f"Error when generating doctor response via openai: {exc}")
                if attempt + 1 < MAX_RETRIES:
                    time.sleep(2**attempt)
        return None


class AnthropicDoctorRunner(ChatModelRunner):
    """Call a Claude doctor model through the Anthropic Messages API."""

    def __init__(self, model_name: str):
        timeout = None if args.request_timeout_seconds <= 0 else args.request_timeout_seconds
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        client_kwargs: dict[str, Any] = {
            "api_key": require_api_key("API_KEY"),
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.llm = Anthropic(**client_kwargs)
        self.model_name = model_name
        log_key = ("anthropic", model_name, False, False)
        with RUNNER_LOG_LOCK:
            should_log = log_key not in RUNNER_LOGGED_KEYS
            if should_log:
                RUNNER_LOGGED_KEYS.add(log_key)
        if should_log:
            print(f"Using backend=anthropic model={model_name}; is_patient=False", flush=True)

    def generate(self, messages: List[Dict[str, str]], filename: str | None = None) -> str:
        system_text = "\n".join(
            message.get("content", "") for message in messages if message.get("role") == "system"
        ).strip()
        non_system = [
            {"role": message.get("role", "user"), "content": message.get("content", "")}
            for message in messages
            if message.get("role") != "system"
        ]
        system_prefix, dynamic_suffix = split_system_for_claude_cache(system_text)
        if dynamic_suffix:
            if non_system and non_system[0].get("role") == "user":
                first_content = non_system[0].get("content", "")
                non_system[0]["content"] = f"{dynamic_suffix}\n\n{first_content}".strip()
            else:
                non_system.insert(0, {"role": "user", "content": dynamic_suffix})

        for attempt in range(MAX_RETRIES):
            try:
                system_block: dict[str, Any] = {"type": "text", "text": system_prefix}
                if args.enable_prompt_caching:
                    cache_control: dict[str, Any] = {"type": "ephemeral"}
                    if args.claude_cache_ttl:
                        cache_control["ttl"] = args.claude_cache_ttl
                    system_block["cache_control"] = cache_control
                response = self.llm.messages.create(
                    model=self.model_name,
                    max_tokens=args.max_tokens,
                    system=[system_block],
                    messages=non_system,
                )
                return extract_anthropic_text(response)
            except Exception as exc:
                print(f"Error when generating doctor response via anthropic: {exc}")
                if attempt + 1 < MAX_RETRIES:
                    time.sleep(2**attempt)
        return None


HUATUO_THINKING_HEADER = re.compile(r"(?im)^\s*#+\s*Thinking\s*$")
HUATUO_FINAL_HEADER = re.compile(r"(?im)^\s*#+\s*Final Response\s*$")

def build_runner_from_name(
    model_name: str,
    base_url: str,
    is_patient: bool = False,
    backend: str = "sglang",
) -> ChatModelRunner:
    """Build a local patient runner or the selected doctor provider runner."""
    if is_patient or backend == "sglang":
        return SGLangChatRunner(
            base_url,
            model_name,
            is_patient=is_patient,
        )
    if backend == "openai":
        return OpenAICompatibleChatRunner(model_name)
    if backend == "anthropic":
        return AnthropicDoctorRunner(model_name)
    raise ValueError(f"Unsupported doctor backend: {backend}")


def should_stop_doctor(response_text: str) -> bool:
    """Return whether the doctor produced a final conclusion."""
    if "Final Conclusion:" in response_text or "最终结论：" in response_text:
        return True
    return False


def split_think(text: str):
    """Separate hidden reasoning wrappers from the final answer."""
    if "</think>" not in text:
        return "", text.strip()
    think, answer = text.rsplit("</think>", 1)
    if "<think>" in think:
        think = think.rsplit("<think>", 1)[1]
    return think.strip(), answer.strip()


HUATUO_O1_MODELS = {
    "HuatuoGPT_o1_7B",
    "HuatuoGPT_o1_8B",
    "HuatuoGPT_o1_70B",
    "HuatuoGPT_o1_72B",
}

def split_gpt(text: str) -> tuple[str, str]:
    """Split GPT-style reasoning and final-answer markers."""
    marker = "<|start|>assistant<|channel|>final<|message|>"
    idx = text.rfind(marker)
    if idx == -1:
        marker = "assistantfinal"
        idx = text.rfind(marker)
        if idx == -1:
            return "", text.strip()
    thinking = text[:idx].strip()
    response = text[idx + len(marker):].strip()
    return thinking, response


def split_huatuo_think(text: str):
    """Split Huatuo-style reasoning and final-answer sections."""
    text = text.strip()
    final_match = HUATUO_FINAL_HEADER.search(text)
    if not final_match:
        return "", text

    response = text[final_match.end():].strip()
    thinking = ""
    for match in HUATUO_THINKING_HEADER.finditer(text):
        if match.start() < final_match.start():
            thinking = text[match.end():final_match.start()].strip()
        else:
            break
    return thinking, response


def split_model_response(model_name: str, text: str):
    """Normalize provider-specific response wrappers."""
    if model_name == "gpt_oss_120b":
        return split_gpt(text)
    if model_name in HUATUO_O1_MODELS:
        return split_huatuo_think(text)
    return split_think(text)


def load_guideline_text(filename: str, cache: dict[str, str]) -> str:
    """Load source records."""
    if filename in cache:
        return cache[filename]
    guideline_path = GUIDELINES_ROOT / filename / f"{filename}.md"
    if not guideline_path.exists():
        raise FileNotFoundError(f"Guideline markdown not found: {guideline_path}")
    guideline_text = guideline_path.read_text(encoding="utf-8")
    cache[filename] = guideline_text
    return guideline_text


def load_primary_clinical_focus_cache(decision_trees_path: Path) -> dict[str, str]:
    """Load source records."""
    if not decision_trees_path.exists():
        raise FileNotFoundError(f"Decision trees jsonl not found: {decision_trees_path}")
    focus_cache: dict[str, str] = {}
    with decision_trees_path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in decision trees at line {line_number}") from exc
            filename = record.get("filename")
            tree_text = record.get("text")
            if not filename or not tree_text:
                raise ValueError(
                    f"Decision trees line {line_number} missing filename or text"
                )
            try:
                tree_data = json.loads(tree_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Decision trees line {line_number} has invalid text JSON"
                ) from exc
            primary_clinical_focus = tree_data.get("primary_clinical_focus")
            if not primary_clinical_focus:
                raise ValueError(
                    f"Decision trees line {line_number} missing filename or primary_clinical_focus"
                )
            existing = focus_cache.get(filename)
            if existing and existing != primary_clinical_focus:
                raise ValueError(
                    f"Decision trees conflict for {filename}: {existing} vs {primary_clinical_focus}"
                )
            focus_cache[filename] = primary_clinical_focus
    return focus_cache


def parse_line_range(line_range: list[int] | None) -> tuple[int, int] | None:
    """Validate a 1-based inclusive line range."""
    if not line_range:
        return None
    if len(line_range) != 2:
        raise ValueError("line range must provide START and END")
    start, end = line_range
    if start <= 0 or end <= 0:
        raise ValueError("line range must be 1-based positive integers")
    if start > end:
        raise ValueError("line range START must be <= END")
    return start, end


def count_nonempty_lines(path: Path) -> int:
    """Count non-empty lines in a text file."""
    with path.open("r", encoding="utf-8") as infile:
        return sum(1 for line in infile if line.strip())


def recover_newer_temp_output(output_jsonl: Path) -> bool:
    """Promote a more advanced interrupted-run temp file before resuming."""
    temp_output_jsonl = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    if not output_jsonl.exists() or not temp_output_jsonl.exists():
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


def build_initial_conversation(language: str) -> List[Dict[str, str]]:
    """Build an inference record."""
    greeting = (
        "你好医生，可以开始问我问题了。"
        if language == "zh"
        else "Hello doctor, please feel free to ask me any questions you need."
    )
    return [{"role": "patient", "content": greeting}]


def completed_rounds_from_conversation(conversation: List[Dict[str, str]]) -> int:
    """Count completed doctor turns."""
    return sum(1 for message in conversation if message.get("role") == "doctor")


def to_chat_messages(
    conversation: List[Dict[str, str]],
    doctor_system_prompt: str,
    patient_system_prompt: str,
    *,
    for_role: str,
) -> List[Dict[str, str]]:
    """Convert the conversation state into chat messages."""
    system_prompt = doctor_system_prompt if for_role == "doctor" else patient_system_prompt
    messages = [{"role": "system", "content": system_prompt}]
    for message in conversation:
        role = message.get("role")
        content = message.get("content", "")
        messages.append({"role": "assistant" if role==for_role else "user", "content": content})
    return messages


def build_system_prompts(
    record: dict,
    guideline_cache: dict[str, str],
    focus_cache: dict[str, str],
) -> tuple[str, str]:
    """Build an inference record."""
    filename = record.get("filename")
    case_vignette = record.get("text")
    language = record.get("language")
    if case_vignette is None or not filename:
        raise ValueError("Missing text or filename in input record")

    guideline_text = load_guideline_text(filename, guideline_cache)
    primary_clinical_focus = focus_cache.get(filename)
    if not primary_clinical_focus:
        raise KeyError(f"primary_clinical_focus not found for {filename}")

    if language == "zh":
        doctor_system_prompt = DOCTOR_SYSTEM_TEMPLATE_ZH.replace("{Guideline}", guideline_text).replace(
            "{Clinical_Inquiry_Goal}", primary_clinical_focus
        )
        patient_system_prompt = PATIENT_SYSTEM_TEMPLATE_ZH.replace("{Case}", case_vignette)
    else:
        doctor_system_prompt = DOCTOR_SYSTEM_TEMPLATE.replace("{Guideline}", guideline_text).replace(
            "{Clinical_Inquiry_Goal}", primary_clinical_focus
        )
        patient_system_prompt = PATIENT_SYSTEM_TEMPLATE.replace("{Case}", case_vignette)
    return doctor_system_prompt, patient_system_prompt


def build_initial_record(record: dict, record_index: int | None = None) -> dict:
    """Build an inference record."""
    filename = record.get("filename")
    case_vignette = record.get("text")
    language = record.get("language")
    if case_vignette is None or not filename:
        raise ValueError("Missing text or filename in input record")

    return {
        "id": record.get("id", record_index + 1 if record_index is not None else None),
        "case_path_id": record.get("case_path_id", record.get("guideline_id")),
        "path_index": record.get("path_index", record.get("path_id")),
        "filename": filename,
        "language": language,
        "case_vignette": case_vignette,
        "conversation": build_initial_conversation(language),
        "ended": False,
    }



def run_stepwise_generation(
    record: dict,
    index: int,
    doctor_system_prompt: str,
    patient_system_prompt: str,
    doctor_runner: ChatModelRunner,
    patient_runner: ChatModelRunner,
    error_file,
):
    """Run stepwise inference for selected cases."""
    if record.get("ended"):
        return record

    conversation = record.get("conversation") or []
    if not doctor_system_prompt or not patient_system_prompt:
        raise ValueError(f"Missing system prompts at index {index}")
    if not conversation:
        raise ValueError(f"Missing conversation at index {index}")

    doctor_messages = to_chat_messages(
        conversation, doctor_system_prompt, patient_system_prompt, for_role="doctor"
    )
    patient_messages = to_chat_messages(
        conversation, doctor_system_prompt, patient_system_prompt, for_role="patient"
    )

    rounds_completed = completed_rounds_from_conversation(conversation)
    filename = record.get("filename")
    while rounds_completed < MAX_ROUNDS_PER_LINE and not record.get("ended"):
        last_role = conversation[-1]["role"] if conversation else "patient"
        if last_role == "doctor":
            patient_response = None
            patient_answer = None

            success = False
            for attempt in range(MAX_RETRIES):
                try:
                    patient_response = patient_runner.generate(patient_messages)
                except Exception as exc:
                    error_file.write(json.dumps({
                        "index": index,
                        "stage": "patient",
                        "error": str(exc),
                        "attempt": attempt + 1,
                    }, ensure_ascii=False) + "\n")
                    error_file.flush()
                    patient_response = None
                if not patient_response:
                    continue
                patient_thinking, patient_answer = split_model_response(
                    args.patient_model_name, patient_response
                )
                if patient_thinking or "<think>" in patient_answer:
                    print(
                        f"patient_thinking error: patient model {patient_runner.model_name} should not have thinking process",
                        flush=True,
                    )
                    error_file.write(json.dumps({
                        "index": index,
                        "stage": "patient",
                        "turn": completed_rounds_from_conversation(conversation),
                        "error": "extra_thinking",
                        "attempt": attempt + 1,
                        "model_name": args.patient_model_name,
                        "response": patient_response,
                    }, ensure_ascii=False) + "\n")
                    error_file.flush()
                    patient_answer = None
                    continue
                success = True
                break
            if success:
                conversation.append({"role": "patient", "content": patient_answer})
                record["conversation"] = conversation
                doctor_messages.append({"role": "user", "content": patient_answer})
                patient_messages.append({"role": "assistant", "content": patient_answer})
            else:
                break

        doctor_response = None
        doctor_answer = None
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                doctor_response = doctor_runner.generate(doctor_messages, filename=filename)
            except Exception as exc:
                error_file.write(json.dumps({
                    "index": index,
                    "stage": "doctor",
                    "error": str(exc),
                    "attempt": attempt + 1,
                }, ensure_ascii=False) + "\n")
                error_file.flush()
                doctor_response = None
            if not doctor_response:
                continue
            _, doctor_answer = split_model_response(
                args.doctor_model_name, doctor_response
            )
            if not doctor_response or not doctor_answer:
                record["error"] = {
                    "stage": "doctor",
                    "turn": completed_rounds_from_conversation(conversation),
                    "message": "generation_failed",
                }
                error_file.write(json.dumps({
                    "index": index,
                    "stage": "doctor",
                    "error": "generation_failed",
                }, ensure_ascii=False) + "\n")
                error_file.flush()
                success = False
                continue
            success = True
            break
        if success:
            conversation.append({"role": "doctor", "content": doctor_answer})
            record["conversation"] = conversation
            doctor_messages.append({"role": "assistant", "content": doctor_answer})
            patient_messages.append({"role": "user", "content": doctor_answer})
            rounds_completed += 1
            if "error" in record:
                record.pop("error")
                
            if should_stop_doctor(doctor_answer):
                record["ended"] = True
                break
        else:
            break
        
    return record


class LockedTextWriter:
    """Thread-safe line writer used by concurrent inference workers."""
    def __init__(self, file_obj):
        """Initialize the writer."""
        self.file_obj = file_obj
        self.lock = threading.Lock()

    def write(self, text: str):
        """Write text with a lock."""
        with self.lock:
            return self.file_obj.write(text)

    def flush(self):
        """Flush with a lock."""
        with self.lock:
            return self.file_obj.flush()


def load_existing_records(path: Path) -> List[dict]:
    """Load source records."""
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            if not raw_line.strip():
                continue
            try:
                records.append(json.loads(raw_line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in existing output at line {line_number}") from exc
    return records


def load_input_records(input_jsonl: Path) -> List[tuple[int, dict]]:
    """Load source records."""
    records: List[tuple[int, dict]] = []
    record_index = 0
    with input_jsonl.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            if not raw_line.strip():
                continue
            try:
                input_record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in input at line {line_number}") from exc
            records.append((record_index, input_record))
            record_index += 1
    return records


def run_parallel_stepwise(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    error_path: Path,
    line_range: tuple[int, int] | None,
    existing_records: List[dict],
    doctor_url: str,
    patient_url: str,
    focus_cache: dict[str, str],
) -> None:
    """Run stepwise inference for selected cases."""
    all_inputs = load_input_records(input_jsonl)
    if line_range:
        start, end = line_range
        selected_inputs = [
            (record_index, input_record)
            for record_index, input_record in all_inputs
            if start <= record_index + 1 <= end
        ]
    else:
        selected_inputs = all_inputs

    if not selected_inputs:
        print("[parallel] no input records selected", flush=True)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl.write_text("", encoding="utf-8")
        return

    first_index = selected_inputs[0][0]

    def existing_for(record_index: int) -> dict | None:
        """Return the existing output row for a record index."""
        if line_range:
            local_index = record_index - first_index
        else:
            local_index = record_index
        if 0 <= local_index < len(existing_records):
            return existing_records[local_index]
        return None

    worker_state = threading.local()

    def get_worker_runners() -> tuple[ChatModelRunner, ChatModelRunner]:
        """Create one doctor/patient runner pair per worker thread."""
        if not hasattr(worker_state, "doctor_runner"):
            worker_state.doctor_runner = build_runner_from_name(
                args.doctor_model_name,
                doctor_url,
                is_patient=False,
                backend=resolve_doctor_backend(
                    args.doctor_model_name, args.doctor_backend
                ),
            )
            worker_state.patient_runner = build_runner_from_name(
                args.patient_model_name,
                patient_url,
                is_patient=True,
            )
        return worker_state.doctor_runner, worker_state.patient_runner

    def process_one(record_index: int, input_record: dict, error_writer: LockedTextWriter) -> tuple[int, dict]:
        """Run inference for one selected case."""
        record = existing_for(record_index)
        if record is None:
            record = build_initial_record(input_record, record_index)
        local_guideline_cache: dict[str, str] = {}
        doctor_system_prompt, patient_system_prompt = build_system_prompts(
            input_record, local_guideline_cache, focus_cache
        )
        if not record.get("ended"):
            doctor_runner, patient_runner = get_worker_runners()
            record = run_stepwise_generation(
                record,
                record_index,
                doctor_system_prompt,
                patient_system_prompt,
                doctor_runner,
                patient_runner,
                error_writer,
            )
        record.pop("doctor_system_prompt", None)
        record.pop("patient_system_prompt", None)
        return record_index, record

    temp_output_jsonl = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    write_path = temp_output_jsonl if output_jsonl.exists() else output_jsonl
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[parallel] case_workers={args.case_workers}, selected_cases={len(selected_inputs)}, "
        f"write_path={write_path}",
        flush=True,
    )

    next_write = first_index
    pending: dict[int, dict] = {}
    selected_indices = {record_index for record_index, _ in selected_inputs}
    max_selected_index = selected_inputs[-1][0]

    # Workers finish out of order, so completed rows are buffered until their original index is next.
    with error_path.open("w", encoding="utf-8") as raw_error_file, write_path.open("w", encoding="utf-8") as outfile:
        error_writer = LockedTextWriter(raw_error_file)
        with ThreadPoolExecutor(max_workers=args.case_workers) as executor:
            futures = [
                executor.submit(process_one, record_index, input_record, error_writer)
                for record_index, input_record in selected_inputs
            ]
            completed = 0
            for future in as_completed(futures):
                record_index, record = future.result()
                completed += 1
                pending[record_index] = record
                print(
                    f"[parallel] completed {completed}/{len(selected_inputs)} index={record_index}",
                    flush=True,
                )
                while next_write <= max_selected_index:
                    if next_write not in selected_indices:
                        next_write += 1
                        continue
                    if next_write not in pending:
                        break
                    outfile.write(json.dumps(pending.pop(next_write), ensure_ascii=False) + "\n")
                    outfile.flush()
                    next_write += 1

    # Replace the final output only after all requested rows were written successfully.
    if write_path == temp_output_jsonl:
        temp_output_jsonl.replace(output_jsonl)


def main():
    """Run the command-line entry point."""
    if args.case_workers < 1:
        raise ValueError("--case-workers must be >= 1")
    # Resolve paths late so environment-variable defaults from the shell wrapper are honored.
    input_jsonl = args.input_jsonl or DEFAULT_INPUT_JSONL
    base_output_jsonl = args.output_jsonl or DEFAULT_OUTPUT_JSONL
    output_jsonl = base_output_jsonl
    line_range = parse_line_range(args.line_range)
    if line_range:
        start, end = line_range
        output_jsonl = base_output_jsonl.with_name(
            f"{base_output_jsonl.stem}.lines_{start}-{end}{base_output_jsonl.suffix}"
        )
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input jsonl not found: {input_jsonl}")

    # A killed resume run can leave a .tmp with more completed records than the
    # last finalized JSONL. Promote it first; this run will then write a fresh
    # .tmp based on the recovered file.
    recover_newer_temp_output(output_jsonl)

    doctor_backend = resolve_doctor_backend(args.doctor_model_name, args.doctor_backend)
    doctor_url = f"http://{args.base_host}:{args.doctor_port}/v1"
    patient_url = f"http://{args.base_host}:{args.patient_port}/v1"

    guideline_cache: dict[str, str] = {}
    focus_cache: dict[str, str] | None = None

    existing_records: List[dict] = []
    resume_source = base_output_jsonl
    if resume_source.exists():
        with resume_source.open("r", encoding="utf-8") as infile:
            for line_number, raw_line in enumerate(infile, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in existing output at line {line_number}") from exc
                existing_records.append(record)

    print(f"Doctor backend: {doctor_backend}", flush=True)
    print(f"Doctor template EN: {doctor_template_path}", flush=True)
    print(f"Doctor template ZH: {doctor_template_path_zh}", flush=True)
    print(f"Case workers: {args.case_workers}", flush=True)
    print(
        "Request timeout seconds: "
        f"{'disabled' if args.request_timeout_seconds <= 0 else args.request_timeout_seconds}",
        flush=True,
    )
    error_path = output_jsonl.with_suffix(output_jsonl.suffix + ".stepwise.err")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.case_workers > 1:
        if focus_cache is None:
            focus_cache = load_primary_clinical_focus_cache(DECISION_TREES_JSONL)
        parallel_existing_records = (
            load_existing_records(output_jsonl)
            if output_jsonl.exists()
            else ([] if line_range else existing_records)
        )
        run_parallel_stepwise(
            input_jsonl=input_jsonl,
            output_jsonl=output_jsonl,
            error_path=error_path,
            line_range=line_range,
            existing_records=parallel_existing_records,
            doctor_url=doctor_url,
            patient_url=patient_url,
            focus_cache=focus_cache,
        )
        return

    doctor_runner = build_runner_from_name(
        args.doctor_model_name,
        doctor_url,
        is_patient=False,
        backend=doctor_backend,
    )
    patient_runner = build_runner_from_name(args.patient_model_name, patient_url, is_patient=True)

    if line_range:
        start, end = line_range
        if focus_cache is None:
            focus_cache = load_primary_clinical_focus_cache(DECISION_TREES_JSONL)
        with error_path.open("w", encoding="utf-8") as error_file, output_jsonl.open("w", encoding="utf-8") as outfile, input_jsonl.open("r", encoding="utf-8") as infile:
            record_index = 0
            for line_number, raw_line in enumerate(infile, start=1):
                if not raw_line.strip():
                    continue
                record_index += 1
                if not (start <= record_index <= end):
                    continue
                try:
                    input_record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in input at line {line_number}") from exc

                if record_index - 1 < len(existing_records):
                    record = existing_records[record_index - 1]
                else:
                    record = build_initial_record(input_record, record_index - 1)

                doctor_system_prompt, patient_system_prompt = build_system_prompts(
                    input_record, guideline_cache, focus_cache
                )

                if not record.get("ended"):
                    record = run_stepwise_generation(
                        record,
                        record_index - 1,
                        doctor_system_prompt,
                        patient_system_prompt,
                        doctor_runner,
                        patient_runner,
                        error_file,
                    )

                record.pop("doctor_system_prompt", None)
                record.pop("patient_system_prompt", None)
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                outfile.flush()
    elif output_jsonl.exists():
        temp_output_jsonl = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
        with error_path.open("w", encoding="utf-8") as error_file, temp_output_jsonl.open("w", encoding="utf-8") as outfile, input_jsonl.open("r", encoding="utf-8") as infile:
            record_index = 0
            for line_number, raw_line in enumerate(infile, start=1):
                if not raw_line.strip():
                    continue
                try:
                    input_record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in input at line {line_number}") from exc

                if record_index < len(existing_records):
                    record = existing_records[record_index]
                else:
                    record = build_initial_record(input_record, record_index)

                if focus_cache is None:
                    focus_cache = load_primary_clinical_focus_cache(DECISION_TREES_JSONL)
                doctor_system_prompt, patient_system_prompt = build_system_prompts(
                    input_record, guideline_cache, focus_cache
                )

                if not record.get("ended"):
                    record = run_stepwise_generation(
                        record,
                        record_index,
                        doctor_system_prompt,
                        patient_system_prompt,
                        doctor_runner,
                        patient_runner,
                        error_file,
                    )

                record.pop("doctor_system_prompt", None)
                record.pop("patient_system_prompt", None)
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                outfile.flush()
                record_index += 1

            if record_index < len(existing_records):
                for record in existing_records[record_index:]:
                    record.pop("doctor_system_prompt", None)
                    record.pop("patient_system_prompt", None)
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                    outfile.flush()
        temp_output_jsonl.replace(output_jsonl)
    else:
        if focus_cache is None:
            focus_cache = load_primary_clinical_focus_cache(DECISION_TREES_JSONL)
        with error_path.open("w", encoding="utf-8") as error_file, output_jsonl.open("w", encoding="utf-8") as outfile, input_jsonl.open("r", encoding="utf-8") as infile:
            record_index = 0
            for line_number, raw_line in enumerate(infile, start=1):
                if not raw_line.strip():
                    continue
                try:
                    input_record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in input at line {line_number}") from exc

                record = build_initial_record(input_record, record_index)
                doctor_system_prompt, patient_system_prompt = build_system_prompts(
                    input_record, guideline_cache, focus_cache
                )
                record = run_stepwise_generation(
                    record,
                    record_index,
                    doctor_system_prompt,
                    patient_system_prompt,
                    doctor_runner,
                    patient_runner,
                    error_file,
                )
                record.pop("doctor_system_prompt", None)
                record.pop("patient_system_prompt", None)
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                outfile.flush()
                record_index += 1


if __name__ == "__main__":
    main()
