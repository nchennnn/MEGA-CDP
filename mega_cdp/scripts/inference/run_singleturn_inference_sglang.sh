#!/usr/bin/env bash
# Run single-turn MEGA-CDP inference.
set -euo pipefail

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}" 
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SERVED_MODEL_NAME}}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
READY_TIMEOUT="${READY_TIMEOUT:-0}"
RUN_SERVER="${RUN_SERVER:-1}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
INPUT_JSONL="${INPUT_JSONL:-datasets/mega_cdp/test/vignette_cases.jsonl}"
GUIDELINES_ROOT="${GUIDELINES_ROOT:-datasets/mega_cdp/test/guideline_md}"
OUTPUT_JSONL="${OUTPUT_JSONL:-}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-0}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
LINE_RANGE_START="${LINE_RANGE_START:-}"
LINE_RANGE_END="${LINE_RANGE_END:-}"

if [[ -n "${LINE_RANGE_START}" || -n "${LINE_RANGE_END}" ]]; then
  if [[ -z "${LINE_RANGE_START}" || -z "${LINE_RANGE_END}" ]]; then
    echo "[error] LINE_RANGE_START and LINE_RANGE_END must be set together" >&2
    exit 1
  fi
fi

if [[ "${RUN_SERVER}" == "1" && ! -d "${MODEL_PATH}" ]]; then
  echo "[error] Model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ "${RUN_SERVER}" == "1" && -n "${CHAT_TEMPLATE}" && ! -f "${CHAT_TEMPLATE}" ]]; then
  echo "[error] Chat template not found: ${CHAT_TEMPLATE}" >&2
  exit 1
fi

log_dir="${PROJECT_ROOT}/logs/$(date +"%Y%m%d_%H%M%S")_singleturn_inference"
mkdir -p "${log_dir}"
log_file="${log_dir}/sglang_singleturn_${PORT}.log"
base_url="http://127.0.0.1:${PORT}/v1"
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    echo "[stop] stopping inference service pid=${server_pid}"
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${PROJECT_ROOT}"
if [[ "${RUN_SERVER}" == "1" ]]; then
  echo "[start] SGLang model=${MODEL_PATH} port=${PORT}"
  chat_template_args=()
  [[ -n "${CHAT_TEMPLATE}" ]] && chat_template_args+=(--chat-template "${CHAT_TEMPLATE}")
  SGLANG_ENABLE_JIT_DEEPGEMM=0 "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --context-length "${MAX_MODEL_LEN}" \
    "${chat_template_args[@]}" \
    >"${log_file}" 2>&1 &
  server_pid=$!
  start_ts="$(date +%s)"
  until curl -sf "${base_url}/models" >/dev/null; do
    sleep 1
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -n 100 "${log_file}" >&2 || true
      exit 1
    fi
    now="$(date +%s)"
    if (( READY_TIMEOUT > 0 && now - start_ts > READY_TIMEOUT )); then
      tail -n 100 "${log_file}" >&2 || true
      exit 1
    fi
  done
else
  echo "[reuse] inference service at ${base_url}"
  curl -sf "${base_url}/models" >/dev/null
fi


if [[ "${RUN_INFERENCE}" != "1" ]]; then
  wait "${server_pid}"
  exit 0
fi
args=(
  --served-model-name "${SERVED_MODEL_NAME}"
  --backend sglang
  --experiment-name "${EXPERIMENT_NAME}"
  --base-url "${base_url}"
  --input-jsonl "${INPUT_JSONL}"
  --guidelines-root "${GUIDELINES_ROOT}"
  --max-tokens "${MAX_TOKENS}"
  --request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}"
)
[[ -n "${OUTPUT_JSONL}" ]] && args+=(--output-jsonl "${OUTPUT_JSONL}")
[[ -n "${TEMPERATURE}" ]] && args+=(--temperature "${TEMPERATURE}")
[[ -n "${TOP_P}" ]] && args+=(--top-p "${TOP_P}")
[[ -n "${TOP_K}" ]] && args+=(--top-k "${TOP_K}")
if [[ -n "${LINE_RANGE_START}" && -n "${LINE_RANGE_END}" ]]; then
  args+=(--line-range "${LINE_RANGE_START}" "${LINE_RANGE_END}")
fi
case "${ENABLE_THINKING,,}" in
  "") ;;
  1|true|yes|on) args+=(--enable-thinking) ;;
  0|false|no|off) args+=(--no-enable-thinking) ;;
  *) echo "[error] ENABLE_THINKING must be true, false, or empty" >&2; exit 1 ;;
esac
"${PYTHON}" "${SCRIPT_DIR}/predict_singleturn_sglang.py" "${args[@]}"

echo "[done] single-turn inference completed"
