#!/usr/bin/env bash
# Run multi-turn MEGA-CDP inference.
set -euo pipefail

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/../../.." && pwd)"

DOCTOR_MODEL_PATH="${DOCTOR_MODEL_PATH:-${MODEL_PATH:-}}"
DOCTOR_MODEL="${DOCTOR_MODEL:-}"
DOCTOR_BACKEND="${DOCTOR_BACKEND:-auto}"
PATIENT_MODEL_PATH="${PATIENT_MODEL_PATH:-}"
PATIENT_MODEL="${PATIENT_MODEL:-${PATIENT_MODEL_PATH:-/path/to/Qwen3-8B}}"
# Run the doctor and patient as two independent services when both are local.
DOCTOR_TP="${DOCTOR_TP:-1}"
PATIENT_TP="${PATIENT_TP:-1}"
DOCTOR_GPUS="${DOCTOR_GPUS:-0}"
PATIENT_GPUS="${PATIENT_GPUS:-}"
INPUT_JSONL="${INPUT_JSONL:-}"
OUTPUT_JSONL="${OUTPUT_JSONL:-}"
GUIDELINES_ROOT="${GUIDELINES_ROOT:-}"
DECISION_TREES_JSONL="${DECISION_TREES_JSONL:-}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_RETRIES="${MAX_RETRIES:-5}"
BASE_PORT="${BASE_PORT:-30000}"
PATIENT_BASE_PORT="${PATIENT_BASE_PORT:-}"
RUN_PATIENT_SERVER="${RUN_PATIENT_SERVER:-1}"
DOCTOR_GPU_MEM_UTIL="${DOCTOR_GPU_MEM_UTIL:-0.9}"
PATIENT_GPU_MEM_UTIL="${PATIENT_GPU_MEM_UTIL:-0.9}"
LINE_RANGE_START="${LINE_RANGE_START:-}"
LINE_RANGE_END="${LINE_RANGE_END:-}"
CASE_WORKERS="${CASE_WORKERS:-1}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-0}"
SGLANG_SHUTDOWN_TIMEOUT="${SGLANG_SHUTDOWN_TIMEOUT:-120}"

DOCTOR_PIDS=()
PATIENT_PIDS=()
DOCTOR_PORTS=()
PATIENT_PORTS=()
SERVICES_STOPPED=0

declare -A MODEL_REGISTRY=(
  ["Qwen3_8B"]="${PATIENT_MODEL_PATH:-/path/to/Qwen3-8B}"
)

if [[ -z "${DOCTOR_MODEL}" && -z "${DOCTOR_MODEL_PATH}" ]]; then
  echo "[error] either DOCTOR_MODEL or DOCTOR_MODEL_PATH is required" >&2
  exit 1
fi
if [[ -n "${DOCTOR_MODEL_PATH}" ]]; then
  if [[ ! -d "${DOCTOR_MODEL_PATH}" ]]; then
    echo "[error] doctor model directory not found: ${DOCTOR_MODEL_PATH}" >&2
    exit 1
  fi
  DOCTOR_MODEL_PATH_FOR_NAME="${DOCTOR_MODEL_PATH}"
  DOCTOR_MODEL_PATH="$(realpath "${DOCTOR_MODEL_PATH}")"
fi
infer_experiment_name() {
  local model_path="$1"
  local logical_model_path
  local outputs_root

  outputs_root="$(realpath -m "${PROJECT_ROOT}/outputs")"
  if [[ "${model_path}" = /* ]]; then
    logical_model_path="${model_path}"
  else
    logical_model_path="${PROJECT_ROOT}/${model_path}"
  fi
  logical_model_path="$(realpath -ms "${logical_model_path}")"

  if [[ "${logical_model_path}" == "${outputs_root}/"* ]]; then
    local relative_path="${logical_model_path#"${outputs_root}/"}"
    printf '%s\n' "${relative_path%%/*}"
  else
    printf '%s\n' "$(basename "${logical_model_path}")"
  fi
}

if [[ -z "${DOCTOR_MODEL}" ]]; then
  DOCTOR_MODEL="$(infer_experiment_name "${DOCTOR_MODEL_PATH_FOR_NAME:-${DOCTOR_MODEL_PATH}}")"
  DOCTOR_MODEL="$(printf '%s' "${DOCTOR_MODEL}" | tr -c 'A-Za-z0-9_.-' '_')"
fi

resolve_doctor_backend() {
  local backend="${DOCTOR_BACKEND,,}"
  local model_name="${DOCTOR_MODEL,,}"
  if [[ "${backend}" != "auto" ]]; then
    printf '%s\n' "${backend}"
    return
  fi
  case "${model_name}" in
    *gpt-oss*|*gpt_oss*) printf '%s\n' sglang ;;
    gpt-*|gpt_*|*/gpt-*|*gemini*) printf '%s\n' openai ;;
    *claude*) printf '%s\n' anthropic ;;
    *) printf '%s\n' sglang ;;
  esac
}

RESOLVED_DOCTOR_BACKEND="$(resolve_doctor_backend)"
case "${RESOLVED_DOCTOR_BACKEND}" in
  sglang|openai|anthropic) ;;
  *)
    echo "[error] DOCTOR_BACKEND must be auto, sglang, openai, or anthropic" >&2
    exit 1
    ;;
esac
if [[ -z "${PATIENT_GPUS}" ]]; then
  if [[ "${RESOLVED_DOCTOR_BACKEND}" == "sglang" ]]; then
    PATIENT_GPUS="1"
  else
    PATIENT_GPUS="0"
  fi
fi
if [[ -n "${LINE_RANGE_START}" || -n "${LINE_RANGE_END}" ]]; then
  if [[ -z "${LINE_RANGE_START}" || -z "${LINE_RANGE_END}" ]]; then
    echo "[error] LINE_RANGE_START and LINE_RANGE_END must be set together" >&2
    exit 1
  fi
fi

normalize_output_jsonl() {
  local output_path="$1"
  local output_name
  output_name="$(basename "${output_path}")"
  if [[ "${output_name}" == *.jsonl ]]; then
    printf '%s\n' "${output_path}"
    return 0
  fi
  printf '%s/%s_prediction_state.jsonl\n' "${output_path}" "${output_name}"
}

check_cuda_devices() {
  local devices="$1"
  local served_name="$2"
  local python_bin="${PYTHON_BIN:-python}"

  if ! CUDA_VISIBLE_DEVICES="${devices}" "${python_bin}" - <<'PY' >/dev/null 2>&1
import torch
if torch.cuda.device_count() <= 0:
    raise SystemExit(1)
PY
  then
    echo "[error] ${served_name} cannot see any CUDA device with CUDA_VISIBLE_DEVICES=${devices}" >&2
    echo "[hint] Set DOCTOR_GPUS/PATIENT_GPUS to GPU ids visible in your current environment." >&2
    return 1
  fi
}

resolve_model_path() {
  local name="$1"
  if [[ "${name}" == "${DOCTOR_MODEL}" && -n "${DOCTOR_MODEL_PATH}" ]]; then
    echo "${DOCTOR_MODEL_PATH}"
    return
  fi
  local path="${MODEL_REGISTRY[$name]:-}"
  if [[ -n "${path}" ]]; then
    echo "${path}"
  else
    echo "${name}"
  fi
}

start_services() {
  local model_name="$1"
  local served_name="$2"
  local max_model_len="$3"
  local base_port="$4"
  local model_path
  model_path="$(resolve_model_path "${model_name}")"
  local tp_local="${DOCTOR_TP}"
  local gpu_devices="${DOCTOR_GPUS}"
  if [[ "${served_name}" == "patient" ]]; then
    tp_local="${PATIENT_TP}"
    gpu_devices="${PATIENT_GPUS}"
  fi

  local chat_template_arg=()
  if [[ "${served_name}" == "patient" ]]; then
    chat_template_arg=(--chat-template "${PROJECT_ROOT}/mega_cdp/template/qwen3_nonthinking.jinja")
  fi
  local rope_scaling_arg=()
  if [[ "${served_name}" == "doctor" && ( "${model_name}" == "Qwen3_8B" || "${model_name}" == "Qwen3_32B" || "${model_name}" == "Qwen3_235B_A22B" ) ]]; then
    rope_scaling_arg=(--json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}}')
  fi
  local mem_fraction="${DOCTOR_GPU_MEM_UTIL}"
  if [[ "${served_name}" == "patient" ]]; then
    mem_fraction="${PATIENT_GPU_MEM_UTIL}"
  fi

  local ports=()
  local pids=()

  local timestamp
  timestamp="$(date +"%Y%m%d_%H%M%S")"
  local log_dir="${PROJECT_ROOT}/logs/${timestamp}_${served_name}"
  mkdir -p "${log_dir}"

  local port="${base_port}"
  local log_file="${log_dir}/sglang_${served_name}_${port}.log"
  echo "[start] sglang ${served_name} port=${port} tp=${tp_local} gpus=${gpu_devices} gpu_mem_util=${mem_fraction}"
  check_cuda_devices "${gpu_devices}" "${served_name}"

  local server_cmd=(
    env
    SGLANG_ENABLE_JIT_DEEPGEMM=0
    CUDA_VISIBLE_DEVICES="${gpu_devices}"
    "${PYTHON_BIN:-python}"
    -m sglang.launch_server
    --model-path "${model_path}"
    --served-model-name "${served_name}"
    --host 0.0.0.0
    --port "${port}"
    --tensor-parallel-size "${tp_local}"
    --mem-fraction-static "${mem_fraction}"
    --context-length "${max_model_len}"
  )
  server_cmd+=("${chat_template_arg[@]}")
  server_cmd+=("${rope_scaling_arg[@]}")

  setsid "${server_cmd[@]}" >"${log_file}" 2>&1 &
  local server_pid="$!"
  # Keep the PID for cleanup, but remove the long launch command from Bash's
  # job table. Otherwise Bash may print a truncated "Killed setsid env ..."
  # job-status line during shutdown, which can look like an unmatched quote
  # syntax error even though this script parses correctly.
  disown "${server_pid}" 2>/dev/null || true
  pids+=("${server_pid}")
  ports+=("${port}")

  local pid="${pids[0]}"
  local base_url="http://127.0.0.1:${port}/v1"
  local t0
  t0="$(date +%s)"
  local ready_timeout="${READY_TIMEOUT:-0}"
  echo "[wait] ${served_name} port=${port} -> ${base_url}/models"
  until curl -sf "${base_url}/models" >/dev/null; do
    sleep 1
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[error] sglang exited early for port ${port}" >&2
      echo "[error] Last 80 lines from ${log_file}:" >&2
      tail -n 80 "${log_file}" >&2 || true
      return 1
    fi
    local now
    now="$(date +%s)"
    if (( ready_timeout > 0 && now - t0 > ready_timeout )); then
      echo "[error] Timeout waiting for port ${port}" >&2
      echo "[error] Last 80 lines from ${log_file}:" >&2
      tail -n 80 "${log_file}" >&2 || true
      return 1
    fi
  done
  if [[ "${served_name}" == "doctor" ]]; then
    DOCTOR_PORTS=("${ports[@]}")
    DOCTOR_PIDS=("${pids[@]}")
  else
    PATIENT_PORTS=("${ports[@]}")
    PATIENT_PIDS=("${pids[@]}")
  fi
}

stop_services() {
  if (( SERVICES_STOPPED )); then
    return 0
  fi
  SERVICES_STOPPED=1

  local pid
  for pid in "${DOCTOR_PIDS[@]:-}" "${PATIENT_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    kill -TERM -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  done
  for pid in "${DOCTOR_PIDS[@]:-}" "${PATIENT_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] || continue
    local t0 now stat
    t0="$(date +%s)"
    while kill -0 "${pid}" 2>/dev/null; do
      stat="$(ps -p "${pid}" -o stat= 2>/dev/null || true)"
      if [[ "${stat}" == Z* ]]; then
        break
      fi
      now="$(date +%s)"
      if (( now - t0 >= SGLANG_SHUTDOWN_TIMEOUT )); then
        echo "[warn] sglang pid=${pid} did not exit after ${SGLANG_SHUTDOWN_TIMEOUT}s; sending SIGKILL" >&2
        kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
        sleep 2
        break
      fi
      sleep 1
    done
    stat="$(ps -p "${pid}" -o stat= 2>/dev/null || true)"
    if [[ -z "${stat}" || "${stat}" == Z* ]]; then
      wait "${pid}" 2>/dev/null || true
    else
      echo "[warn] sglang pid=${pid} is still alive; skip blocking wait" >&2
    fi
  done
  DOCTOR_PIDS=()
  PATIENT_PIDS=()
  DOCTOR_PORTS=()
  PATIENT_PORTS=()
  return 0
}

run_stepwise() {
  local args=(
    --doctor-model-name "${DOCTOR_MODEL}"
    --doctor-backend "${RESOLVED_DOCTOR_BACKEND}"
    --patient-model-name "${PATIENT_MODEL}"
    --doctor-port "${DOCTOR_PORTS[0]}"
    --patient-port "${PATIENT_PORTS[0]}"
  )
  if [[ -n "${INPUT_JSONL}" ]]; then
    args+=(--input-jsonl "${INPUT_JSONL}")
  fi
  if [[ -n "${OUTPUT_JSONL}" ]]; then
    local normalized_output_jsonl
    normalized_output_jsonl="$(normalize_output_jsonl "${OUTPUT_JSONL}")"
    args+=(--output-jsonl "${normalized_output_jsonl}")
  fi
  if [[ -n "${GUIDELINES_ROOT}" ]]; then
    args+=(--guidelines-root "${GUIDELINES_ROOT}")
  fi
  if [[ -n "${DECISION_TREES_JSONL}" ]]; then
    args+=(--decision-trees-jsonl "${DECISION_TREES_JSONL}")
  fi
  if [[ -n "${MAX_TOKENS}" ]]; then
    args+=(--max-tokens "${MAX_TOKENS}")
  fi
  args+=(--max-retries "${MAX_RETRIES}")
  args+=(--case-workers "${CASE_WORKERS}")
  args+=(--request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}")
  if [[ -n "${LINE_RANGE_START}" && -n "${LINE_RANGE_END}" ]]; then
    args+=(--line-range "${LINE_RANGE_START}" "${LINE_RANGE_END}")
  fi
  "${PYTHON_BIN:-python}" "${ROOT_DIR}/predict_multiturn_stepwise_sglang.py" "${args[@]}"
}

trap stop_services EXIT

cd "${PROJECT_ROOT}"

if [[ -z "${PATIENT_BASE_PORT}" ]]; then
  PATIENT_BASE_PORT=$((BASE_PORT + 1))
fi

echo "[config] doctor_backend=${RESOLVED_DOCTOR_BACKEND}"
if [[ "${RESOLVED_DOCTOR_BACKEND}" == "sglang" ]]; then
  echo "[config] doctor_model=$(resolve_model_path "${DOCTOR_MODEL}")"
fi
echo "[config] patient_model=$(resolve_model_path "${PATIENT_MODEL}")"
echo "[config] patient_gpu=${PATIENT_GPUS}"
echo "[config] case_workers=${CASE_WORKERS}"
echo "[config] request_timeout_seconds=${REQUEST_TIMEOUT_SECONDS}"
if [[ "${RESOLVED_DOCTOR_BACKEND}" == "sglang" ]]; then
  echo "[config] deployment=2x1gpu doctor_gpu=${DOCTOR_GPUS} patient_gpu=${PATIENT_GPUS}"
  echo "[start] doctor and patient services"
  start_services "${DOCTOR_MODEL}" "doctor" "${MAX_MODEL_LEN_DOCTOR:-65536}" "${BASE_PORT}"
else
  DOCTOR_PORTS=("${BASE_PORT}")
  DOCTOR_PIDS=()
  echo "[config] deployment=${RESOLVED_DOCTOR_BACKEND}_doctor+local_patient"
  echo "[start] patient service; doctor uses a remote API backend"
fi
case "${RUN_PATIENT_SERVER,,}" in
  1|true|yes|on)
    start_services "${PATIENT_MODEL}" "patient" "${MAX_MODEL_LEN_PATIENT:-8192}" "${PATIENT_BASE_PORT}"
    ;;
  0|false|no|off)
    PATIENT_PORTS=("${PATIENT_BASE_PORT}")
    PATIENT_PIDS=()
    echo "[reuse] patient service already running at http://127.0.0.1:${PATIENT_BASE_PORT}/v1"
    ;;
  *)
    echo "[error] RUN_PATIENT_SERVER must be true or false, got: ${RUN_PATIENT_SERVER}" >&2
    exit 1
    ;;
esac
run_stepwise
trap - EXIT
stop_services
