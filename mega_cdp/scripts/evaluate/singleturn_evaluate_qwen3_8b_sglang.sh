#!/usr/bin/env bash
# Run single-turn evaluation.
set -euo pipefail

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${MODEL_PATH:-/path/to/Qwen3-8B}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3_8B}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${PROJECT_ROOT}/mega_cdp/template/qwen3_nonthinking.jinja}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MEM_FRACTION="${MEM_FRACTION:-0.75}"
READY_TIMEOUT="${READY_TIMEOUT:-0}"
RUN_SERVER="${RUN_SERVER:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_TASK="${EVAL_TASK:-both}"
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-}"
EVAL_PARSED_JSONL="${EVAL_PARSED_JSONL:-}"
EVAL_GT_PATHS_JSONL="${EVAL_GT_PATHS_JSONL:-}"
PATHWAY_OUTPUT_JSONL="${PATHWAY_OUTPUT_JSONL:-}"
ACCURACY_OUTPUT_JSONL="${ACCURACY_OUTPUT_JSONL:-}"
QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-4096}"
EVAL_REQUEST_WORKERS="${EVAL_REQUEST_WORKERS:-1}"
QWEN_REQUEST_TIMEOUT_SECONDS="${QWEN_REQUEST_TIMEOUT_SECONDS:-0}"

if [[ "${EVAL_TASK}" != "pathway" && "${EVAL_TASK}" != "accuracy" && "${EVAL_TASK}" != "both" ]]; then
  echo "[error] EVAL_TASK must be pathway, accuracy, or both: ${EVAL_TASK}" >&2
  exit 1
fi
if [[ "${RUN_SERVER}" == "1" && ! -d "${JUDGE_MODEL_PATH}" ]]; then
  echo "[error] Judge model directory not found: ${JUDGE_MODEL_PATH}" >&2
  exit 1
fi
if [[ "${RUN_SERVER}" == "1" && ! -f "${CHAT_TEMPLATE}" ]]; then
  echo "[error] Chat template not found: ${CHAT_TEMPLATE}" >&2
  exit 1
fi

log_dir="${PROJECT_ROOT}/logs/$(date +"%Y%m%d_%H%M%S")_singleturn_evaluate_qwen3_8b"
mkdir -p "${log_dir}"
log_file="${log_dir}/sglang_qwen3_8b_${PORT}.log"
base_url="http://127.0.0.1:${PORT}/v1"
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    echo "[stop] stopping judge service pid=${server_pid}"
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${RUN_SERVER}" == "1" ]]; then
  echo "[start] SGLang judge gpu=${CUDA_VISIBLE_DEVICES} port=${PORT}"
  SGLANG_ENABLE_JIT_DEEPGEMM=0 "${PYTHON}" -m sglang.launch_server \
    --model-path "${JUDGE_MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --context-length "${MAX_MODEL_LEN}" \
    --chat-template "${CHAT_TEMPLATE}" \
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
  echo "[reuse] judge service at ${base_url}"
  curl -sf "${base_url}/models" >/dev/null
fi

if [[ "${RUN_EVAL}" != "1" ]]; then
  wait "${server_pid}"
  exit 0
fi
if [[ -z "${EVAL_MODEL_NAME}" ]]; then
  echo "[error] EVAL_MODEL_NAME is required" >&2
  exit 1
fi
common_args=(
  --model-name "${EVAL_MODEL_NAME}"
  --qwen-base-url "${base_url}"
  --qwen-model-name "${SERVED_MODEL_NAME}"
  --qwen-max-tokens "${QWEN_MAX_TOKENS}"
  --request-workers "${EVAL_REQUEST_WORKERS}"
  --qwen-request-timeout-seconds "${QWEN_REQUEST_TIMEOUT_SECONDS}"
)
[[ -n "${EVAL_PARSED_JSONL}" ]] && common_args+=(--parsed-jsonl "${EVAL_PARSED_JSONL}")
[[ -n "${EVAL_GT_PATHS_JSONL}" ]] && common_args+=(--gt-paths-jsonl "${EVAL_GT_PATHS_JSONL}")

if [[ "${EVAL_TASK}" == "pathway" || "${EVAL_TASK}" == "both" ]]; then
  args=("${common_args[@]}")
  [[ -n "${PATHWAY_OUTPUT_JSONL}" ]] && args+=(--output-jsonl "${PATHWAY_OUTPUT_JSONL}")
  "${PYTHON}" "${SCRIPT_DIR}/pathway_consistency_eval_batch_singleturn.py" "${args[@]}"
fi
if [[ "${EVAL_TASK}" == "accuracy" || "${EVAL_TASK}" == "both" ]]; then
  args=("${common_args[@]}")
  [[ -n "${ACCURACY_OUTPUT_JSONL}" ]] && args+=(--output-jsonl "${ACCURACY_OUTPUT_JSONL}")
  "${PYTHON}" "${SCRIPT_DIR}/accuracy_eval_batch_singleturn.py" "${args[@]}"
fi

echo "[done] single-turn evaluation completed"
