# MEGA-CDP

Code for running inference and evaluation on MEGA-CDP, a benchmark for measuring guideline-adherent clinical decision pathway generation in large language models.

This repository provides the release toolkit for both settings used by MEGA-CDP:

- **Single-turn vignette setting**: the model receives a guideline and a case vignette, then outputs a full clinical decision process and final outcome.
- **Multi-turn interactive setting**: the model acts as a doctor, asks a patient simulator yes/no questions, and stops with a final clinical conclusion.

The dataset is released at [MEGA-CDP](https://huggingface.co/datasets/nchennnn/MEGA-CDP).

## Repository Structure

```text
MEGA-CDP/
  datasets/                         # local dataset
  mega_cdp/
    scripts/
      inference/                    # single-turn and multi-turn local/API inference
      evaluate/                     # parsing, evaluation, summarization
    template/                       # prompt templates
  results/                          # local predictions/evaluations
```

## Environment

```bash
conda create -n megacdp python=3.10
conda activate megacdp
pip install -r requirements.txt
```

## Prepare Dataset

Download [MEGA-CDP](https://huggingface.co/datasets/nchennnn/MEGA-CDP) from the dataset release page and place it under `datasets/mega_cdp`. The directory of the test set used in the project is:

```text
datasets/mega_cdp/test/
```

## Single-Turn Setting

The model receives the guideline and complete vignette in one request. Its output contains a clinical decision process and final conclusion.

### Open-source model inference with SGLang

By default, the model's own chat template and generation configuration are used. For models with shorter context windows, follow the official model card to configure YaRN or reduce `MAX_MODEL_LEN`; we do not repeat this note in later examples.

```bash
conda activate megacdp

EXP_NAME=your_open_model_singleturn
MODEL_PATH=/path/to/model

MODEL_PATH="$MODEL_PATH" \
SERVED_MODEL_NAME="$EXP_NAME" \
EXPERIMENT_NAME="$EXP_NAME" \
INPUT_JSONL=datasets/mega_cdp/test/vignette_cases.jsonl \
GUIDELINES_ROOT=datasets/mega_cdp/test/guideline_md \
OUTPUT_JSONL=results/prediction_singleturn_sglang/${EXP_NAME}/${EXP_NAME}_prediction_output.jsonl \
MAX_MODEL_LEN=131072 \
MAX_TOKENS=32768 \
bash mega_cdp/scripts/inference/run_singleturn_inference_sglang.sh
```

### Closed-source model inference with APIs

API inference supports two explicit backends: `openai` for OpenAI-compatible providers such as GPT and Gemini, and `anthropic` for Claude through the Anthropic Messages API. Set `API_KEY` and the remote model name through environment variables; set `BASE_URL` for OpenAI-compatible providers and optionally `ANTHROPIC_BASE_URL` for a custom Anthropic-compatible endpoint.

```bash
conda activate megacdp

EXP_NAME=api_model_singleturn
API_MODEL_NAME=provider-model-name
export API_KEY=YOUR_API_KEY
export BASE_URL=https://provider.example.com/v1

python mega_cdp/scripts/inference/predict_singleturn_sglang.py \
  --backend openai \
  --model-name "$API_MODEL_NAME" \
  --base-url "$BASE_URL" \
  --experiment-name "$EXP_NAME" \
  --input-jsonl datasets/mega_cdp/test/vignette_cases.jsonl \
  --guidelines-root datasets/mega_cdp/test/guideline_md \
  --output-jsonl results/prediction_singleturn_sglang/${EXP_NAME}/${EXP_NAME}_prediction_output.jsonl \
  --max-tokens 32768
```

### Evaluation

Single-turn evaluation uses reference pathways from `gt_paths.jsonl`.

```bash
conda activate megacdp

EXP_NAME=your_model_singleturn
JUDGE_MODEL_PATH=/path/to/Qwen3-8B
RESULT_DIR=results/prediction_singleturn_sglang/${EXP_NAME}
TEST_ROOT=datasets/mega_cdp/test

python mega_cdp/scripts/evaluate/parse_prediction_output_singleturn.py \
  --experiment-name "$EXP_NAME" \
  --input-jsonl ${RESULT_DIR}/${EXP_NAME}_prediction_output.jsonl \
  --output-jsonl ${RESULT_DIR}/${EXP_NAME}_prediction_output_parsed.jsonl

EVAL_REQUEST_WORKERS=16 \
JUDGE_MODEL_PATH="$JUDGE_MODEL_PATH" \
EVAL_MODEL_NAME="$EXP_NAME" \
EVAL_PARSED_JSONL=${RESULT_DIR}/${EXP_NAME}_prediction_output_parsed.jsonl \
EVAL_GT_PATHS_JSONL=${TEST_ROOT}/gt_paths.jsonl \
PATHWAY_OUTPUT_JSONL=${RESULT_DIR}/${EXP_NAME}_pathway_consistency_eval_qwen.jsonl \
ACCURACY_OUTPUT_JSONL=${RESULT_DIR}/${EXP_NAME}_accuracy_eval_qwen.jsonl \
EVAL_TASK=both \
QWEN_MAX_TOKENS=8192 \
bash mega_cdp/scripts/evaluate/singleturn_evaluate_qwen3_8b_sglang.sh

python mega_cdp/scripts/evaluate/summarize_eval_metrics.py \
  --exp-name "$EXP_NAME" \
  --results-root results/prediction_singleturn_sglang \
  --input-jsonl ${RESULT_DIR}/${EXP_NAME}_pathway_consistency_eval_qwen.jsonl \
  --accuracy-input-jsonl ${RESULT_DIR}/${EXP_NAME}_accuracy_eval_qwen.jsonl \
  --expected-results 5 \
  --accuracy-expected-results 5 \
  --include-skipped-as-worst \
  --include-accuracy-errors-as-zero
```

## Multi-Turn Setting

The evaluated model acts as the doctor. A local Qwen3-8B SGLang service acts as the patient simulator for both open- and closed-source doctor models.

### Open-source doctor inference with SGLang

```bash
conda activate megacdp

EXP_NAME=your_open_model_multiturn
DOCTOR_MODEL_PATH=/path/to/doctor_model
PATIENT_MODEL_PATH=/path/to/Qwen3-8B

CASE_WORKERS=16 \
DOCTOR_MODEL_PATH="$DOCTOR_MODEL_PATH" \
PATIENT_MODEL_PATH="$PATIENT_MODEL_PATH" \
INPUT_JSONL=datasets/mega_cdp/test/vignette_cases.jsonl \
GUIDELINES_ROOT=datasets/mega_cdp/test/guideline_md \
DECISION_TREES_JSONL=datasets/mega_cdp/test/decision_trees.jsonl \
OUTPUT_JSONL=results/prediction_multiturn_sglang/${EXP_NAME}/${EXP_NAME}_prediction_state.jsonl \
MAX_MODEL_LEN_DOCTOR=131072 \
MAX_TOKENS=8192 \
bash mega_cdp/scripts/inference/run_multiturn_inference_sglang.sh
```

### Closed-source doctor inference with APIs

API inference supports two explicit doctor backends: `openai` for OpenAI-compatible providers such as GPT and Gemini, and `anthropic` for Claude through the Anthropic Messages API. Only the patient simulator is deployed locally with SGLang.

```bash
conda activate megacdp

EXP_NAME=api_model_multiturn
API_MODEL_NAME=provider-model-name
export API_KEY=YOUR_API_KEY
export BASE_URL=https://provider.example.com/v1

DOCTOR_MODEL="$API_MODEL_NAME" \
DOCTOR_BACKEND=openai \
PATIENT_MODEL_PATH=/path/to/Qwen3-8B \
CASE_WORKERS=16 \
INPUT_JSONL=datasets/mega_cdp/test/vignette_cases.jsonl \
GUIDELINES_ROOT=datasets/mega_cdp/test/guideline_md \
DECISION_TREES_JSONL=datasets/mega_cdp/test/decision_trees.jsonl \
OUTPUT_JSONL=results/prediction_multiturn_sglang/${EXP_NAME}/${EXP_NAME}_prediction_state.jsonl \
MAX_TOKENS=8192 \
bash mega_cdp/scripts/inference/run_multiturn_inference_sglang.sh
```

### Evaluation

Multi-turn evaluation uses reference pathways from `case_paths.jsonl`.

```bash
conda activate megacdp

EXP_NAME=your_model_multiturn
JUDGE_MODEL_PATH=/path/to/Qwen3-8B
RESULT_DIR=results/prediction_multiturn_sglang/${EXP_NAME}
TEST_ROOT=datasets/mega_cdp/test

python mega_cdp/scripts/evaluate/parse_prediction_output_multiturn.py \
  --experiment-name "$EXP_NAME" \
  --input-jsonl ${RESULT_DIR}/${EXP_NAME}_prediction_state.jsonl \
  --output-jsonl ${RESULT_DIR}/${EXP_NAME}_prediction_state_parsed.jsonl

EVAL_REQUEST_WORKERS=16 \
JUDGE_MODEL_PATH="$JUDGE_MODEL_PATH" \
EVAL_MODEL_NAME="$EXP_NAME" \
EVAL_PARSED_JSONL=${RESULT_DIR}/${EXP_NAME}_prediction_state_parsed.jsonl \
EVAL_CASE_PATHS_JSONL=${TEST_ROOT}/case_paths.jsonl \
PATHWAY_OUTPUT_JSONL=${RESULT_DIR}/${EXP_NAME}_pathway_consistency_eval_qwen.jsonl \
ACCURACY_OUTPUT_JSONL=${RESULT_DIR}/${EXP_NAME}_accuracy_eval_qwen.jsonl \
EVAL_TASK=both \
QWEN_MAX_TOKENS=8192 \
bash mega_cdp/scripts/evaluate/multiturn_evaluate_qwen3_8b_sglang.sh

python mega_cdp/scripts/evaluate/summarize_eval_metrics.py \
  --exp-name "$EXP_NAME" \
  --results-root results/prediction_multiturn_sglang \
  --input-jsonl ${RESULT_DIR}/${EXP_NAME}_pathway_consistency_eval_qwen.jsonl \
  --accuracy-input-jsonl ${RESULT_DIR}/${EXP_NAME}_accuracy_eval_qwen.jsonl \
  --no-require-exact-count \
  --no-require-exact-accuracy-count \
  --include-skipped-as-worst \
  --include-accuracy-errors-as-zero
```

## Citation

```bibtex
@misc{chen2026megacdp,
  title        = {Benchmarking Clinical Decision Pathway Adherence in Large Language Models},
  author       = {Nuo Chen and Xinyang Jiang and Zilong Wang and Zhifei Zhang and Xiaoye Qu and Jiajun Deng and Yulan Guo and Cairong Zhao},
  year         = {2026},
  eprint       = {2608.26592},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2608.26592}
}
```