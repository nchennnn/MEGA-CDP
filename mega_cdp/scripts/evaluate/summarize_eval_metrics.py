#!/usr/bin/env python3
"""Summarize evaluation metrics."""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from pathway_metrics import build_pred_to_gt_mappings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results" / "prediction_multiturn_sglang"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(
        description="Calculate final pathway-consistency DTW and accuracy metrics."
    )
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Pathway consistency input jsonl. Defaults to completed file if present.",
    )
    parser.add_argument(
        "--accuracy-input-jsonl",
        type=Path,
        default=None,
        help="Accuracy eval input jsonl. Defaults to {experiment}_accuracy_eval_qwen.jsonl.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional per-case pathway DTW jsonl. Implies --write-case-jsonl.",
    )
    parser.add_argument("--summary-txt", type=Path, default=None)
    parser.add_argument("--expected-results", type=int, default=5)
    parser.add_argument("--accuracy-expected-results", type=int, default=5)
    parser.add_argument(
        "--eval-method",
        choices=["gpt", "qwen", "bertscore"],
        default="qwen",
        help="Judge score type used for pathway metrics.",
    )
    parser.add_argument("--ot-reg", type=float, default=0.2)
    parser.add_argument("--ot-tau", type=float, default=1.0)
    parser.add_argument(
        "--require-exact-count",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require exactly --expected-results matrices for each non-skipped pathway case.",
    )
    parser.add_argument(
        "--require-exact-accuracy-count",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require exactly --accuracy-expected-results scores for each accuracy case.",
    )
    parser.add_argument(
        "--include-skipped-as-worst",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include skipped pathway cases in the average with dtw_ratio=1.0.",
    )
    parser.add_argument(
        "--include-accuracy-errors-as-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include invalid accuracy cases in the average with accuracy=0.0.",
    )
    parser.add_argument(
        "--write-case-jsonl",
        action="store_true",
        help="Write optional per-case pathway DTW jsonl for debugging.",
    )
    parser.add_argument(
        "--case-metrics-jsonl",
        type=Path,
        default=None,
        help="Combined per-case pathway and accuracy metrics jsonl. Defaults to final_result/{experiment}/{experiment}_case_metrics.jsonl.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve input and output paths."""
    results_root = args.results_root or RESULTS_ROOT
    experiment_dir = results_root / args.exp_name
    pathway_original = (
        experiment_dir / f"{args.exp_name}_pathway_consistency_eval_qwen.jsonl"
    )
    pathway_completed = pathway_original.with_name(f"{pathway_original.stem}_completed.jsonl")
    pathway_input = args.input_jsonl or (
        pathway_completed if pathway_completed.exists() else pathway_original
    )
    accuracy_input = args.accuracy_input_jsonl or (
        experiment_dir / f"{args.exp_name}_accuracy_eval_qwen.jsonl"
    )
    final_result_dir = results_root / "final_result" / args.exp_name
    case_output_jsonl = args.output_jsonl or (
        final_result_dir / f"{args.exp_name}_pathway_dtw_qwen.jsonl"
    )
    summary_txt = args.summary_txt or (
        final_result_dir / f"{args.exp_name}_final_metrics_summary.txt"
    )
    case_metrics_jsonl = args.case_metrics_jsonl or (
        final_result_dir / f"{args.exp_name}_case_metrics.jsonl"
    )
    return pathway_input, accuracy_input, case_output_jsonl, summary_txt, case_metrics_jsonl


def average_and_normalize_matrices(
    matrices: list,
    expected_results: int,
    require_exact_count: bool,
    eval_method: str,
) -> list[list[float]]:
    """Average repeated judge outputs and normalize scores to the expected range."""
    if not matrices:
        raise ValueError("Missing results")
    if require_exact_count and len(matrices) != expected_results:
        raise ValueError(
            f"Expected exactly {expected_results} matrices, found {len(matrices)}"
        )
    matrix_array = np.asarray(matrices, dtype=float)
    if matrix_array.ndim == 2:
        average_matrix = matrix_array
    elif matrix_array.ndim == 3:
        average_matrix = matrix_array.mean(axis=0)
    else:
        raise ValueError(f"Unexpected results shape: {matrix_array.shape}")
    if eval_method in {"gpt", "qwen"}:
        return np.clip((average_matrix - 1.0) / 4.0, 0.0, 1.0).tolist()
    if eval_method == "bertscore":
        return np.clip(average_matrix, 0.0, 1.0).tolist()
    raise ValueError(f"Unsupported eval_method: {eval_method}")


def get_thresholds(eval_method: str) -> list[float]:
    """Return score thresholds used by the pathway mapping metrics."""
    if eval_method == "gpt":
        return [0.7]
    if eval_method == "qwen":
        return [0.9]
    if eval_method == "bertscore":
        return [0.2]
    raise ValueError(f"Unsupported eval_method: {eval_method}")


def build_worst_pathway_metrics(eval_method: str) -> dict:
    """Build metrics for skipped cases when they are counted as worst cases."""
    threshold_keys = [str(threshold) for threshold in get_thresholds(eval_method)]
    return {
        "consistency_score": {key: 0.0 for key in threshold_keys},
        "mapping_sequence": {key: [] for key in threshold_keys},
        "dtw_score": 1.0,
        "dtw_ratio": 1.0,
        "dtw_path": [],
        "ot": {
            "avg_cost": 1.0,
            "similarity": 0.0,
            "missing_rate": 1.0,
            "extra_rate": 1.0,
            "transported_mass": 0.0,
        },
    }


def build_output_record(record: dict) -> dict:
    """Build a per-case metrics record."""
    return {
        "id": record.get("id"),
        "case_path_id": record.get("case_path_id"),
        "path_index": record.get("path_index"),
        "filename": record.get("filename"),
        "language": record.get("language"),
        "eval_method": record.get("eval_method", "qwen"),
    }


def build_case_key(record: dict) -> str:
    """Build a per-case metrics record."""
    return "|".join(
        str(record.get(key, ""))
        for key in ("filename", "case_path_id", "path_index")
    )


def build_case_identity(record: dict) -> dict:
    """Build a per-case metrics record."""
    return {
        "id": record.get("id"),
        "case_path_id": record.get("case_path_id"),
        "path_index": record.get("path_index"),
        "filename": record.get("filename"),
        "language": record.get("language"),
    }


def build_metric_error_record(
    *,
    metric_type: str,
    line_number: int,
    record: dict,
    error: Exception,
) -> dict:
    """Build a per-case metrics record."""
    results = record.get("results")
    result_count = len(results) if isinstance(results, list) else None
    return {
        "metric_type": metric_type,
        "line_number": line_number,
        "id": record.get("id"),
        "case_path_id": record.get("case_path_id"),
        "path_index": record.get("path_index"),
        "filename": record.get("filename"),
        "language": record.get("language"),
        "eval_method": record.get("eval_method", "qwen"),
        "skipped": record.get("skipped"),
        "record_reason": record.get("reason"),
        "error_type": type(error).__name__,
        "error": str(error),
        "results_type": type(results).__name__,
        "results_count": result_count,
        "results": results,
    }


def to_jsonable(value):
    """Convert numpy values to plain Python values for JSON output."""
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    return value


def iter_numbers(value) -> Iterable[float]:
    """Yield normalized values from a nested input structure."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_numbers(item)


def normalize_accuracy_scores(
    results,
    expected_results: int,
    require_exact_count: bool,
) -> list[float]:
    """Normalize raw judge outputs into numeric values used for metrics."""
    scores = list(iter_numbers(results))
    if not scores:
        raise ValueError("No numeric values found in accuracy results")
    if require_exact_count and len(scores) != expected_results:
        raise ValueError(f"Expected exactly {expected_results} accuracy scores, found {len(scores)}")
    for score in scores:
        if not 1.0 <= score <= 5.0:
            raise ValueError(f"Accuracy score out of range 1-5: {score}")
    return [(score - 1.0) / 4.0 for score in scores]


def calculate_pathway_metrics(
    input_jsonl: Path,
    args: argparse.Namespace,
    case_output_jsonl: Path,
    metric_error_jsonl: Path,
    write_case_jsonl: bool,
) -> dict[str, object]:
    """Calculate aggregate metrics and bookkeeping for this result type."""
    if not input_jsonl.exists():
        raise FileNotFoundError(input_jsonl)

    total = 0
    valid = 0
    input_skipped = 0
    worst = 0
    excluded = 0
    metric_errors = 0
    dtw_scores: list[float] = []
    dtw_ratios: list[float] = []
    consistency_values: dict[str, list[float]] = {}
    ot_values: dict[str, list[float]] = {}
    case_metrics: dict[str, dict] = {}

    output_file = None
    temp_output = None
    if write_case_jsonl:
        case_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        temp_output = case_output_jsonl.with_name(f"{case_output_jsonl.name}.tmp")
        output_file = temp_output.open("w", encoding="utf-8")
    metric_error_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temp_metric_errors = metric_error_jsonl.with_name(f"{metric_error_jsonl.name}.tmp")

    try:
        with (
            input_jsonl.open("r", encoding="utf-8") as input_file,
            temp_metric_errors.open("w", encoding="utf-8") as metric_error_file,
        ):
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                total += 1
                record = json.loads(line)
                if record.get("skipped") is True:
                    input_skipped += 1
                output_record = build_output_record(record)
                case_key = build_case_key(record)
                case_entry = case_metrics.setdefault(case_key, build_case_identity(record))
                case_entry["case_key"] = case_key
                case_entry["pathway_line_number"] = line_number
                case_entry["pathway_input_skipped"] = record.get("skipped") is True
                case_entry["pathway_eval_method"] = record.get("eval_method", "qwen")
                try:
                    if record.get("skipped") is True:
                        raise ValueError(record.get("reason", "input record is skipped"))
                    normalized_matrix = average_and_normalize_matrices(
                        record.get("results"),
                        args.expected_results,
                        args.require_exact_count,
                        args.eval_method,
                    )
                    (
                        consistency_score,
                        dtw_score,
                        dtw_ratio,
                        mapping_sequence,
                        dtw_path,
                        _difference_map,
                        _transport_plan,
                        ot_metrics,
                    ) = build_pred_to_gt_mappings(
                        matrix=normalized_matrix,
                        thresholds=get_thresholds(args.eval_method),
                        ot_reg=args.ot_reg,
                        ot_tau=args.ot_tau,
                        eval_method=args.eval_method,
                    )
                    pathway_metrics = {
                        "consistency_score": consistency_score,
                        "mapping_sequence": mapping_sequence,
                        "dtw_score": dtw_score,
                        "dtw_ratio": dtw_ratio,
                        "dtw_path": dtw_path,
                        "ot": ot_metrics,
                    }
                    output_record["skipped"] = False
                    output_record["metrics"] = pathway_metrics
                    case_entry["pathway_skipped"] = False
                    case_entry["pathway_error"] = None
                    case_entry["pathway_assigned_worst_case"] = False
                    case_entry["pathway_metrics"] = pathway_metrics
                    dtw_scores.append(dtw_score)
                    dtw_ratios.append(dtw_ratio)
                    for key, value in consistency_score.items():
                        consistency_values.setdefault(key, []).append(float(value))
                    for key, value in ot_metrics.items():
                        ot_values.setdefault(key, []).append(float(value))
                    valid += 1
                    print(
                        f"[pathway {total}] line={line_number} key="
                        f"({record.get('case_path_id')}, {record.get('path_index')}) "
                        f"dtw_score={dtw_score:.6f} dtw_ratio={dtw_ratio:.6f}",
                        flush=True,
                    )
                except Exception as exc:
                    output_record["skipped"] = True
                    output_record["reason"] = str(exc)
                    case_entry["pathway_skipped"] = True
                    case_entry["pathway_error"] = str(exc)
                    case_entry["pathway_metric_error"] = record.get("skipped") is not True
                    if args.include_skipped_as_worst:
                        worst_metrics = build_worst_pathway_metrics(args.eval_method)
                        output_record["metrics"] = worst_metrics | {"assigned_worst_case": True}
                        case_entry["pathway_assigned_worst_case"] = True
                        case_entry["pathway_metrics"] = worst_metrics
                        dtw_scores.append(1.0)
                        dtw_ratios.append(1.0)
                        for key, value in worst_metrics["consistency_score"].items():
                            consistency_values.setdefault(key, []).append(float(value))
                        for key, value in worst_metrics["ot"].items():
                            ot_values.setdefault(key, []).append(float(value))
                        worst += 1
                        status = "assigned_worst"
                    else:
                        excluded += 1
                        status = "excluded"
                        case_entry["pathway_assigned_worst_case"] = False
                        case_entry["pathway_metrics"] = None
                    print(
                        f"[pathway {total}] line={line_number} key="
                        f"({record.get('case_path_id')}, {record.get('path_index')}) "
                        f"{status}: {exc}",
                        flush=True,
                    )
                    if record.get("skipped") is not True:
                        metric_errors += 1
                        metric_error_file.write(
                            json.dumps(
                                build_metric_error_record(
                                    metric_type="pathway_consistency",
                                    line_number=line_number,
                                    record=record,
                                    error=exc,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        metric_error_file.flush()
                if output_file is not None:
                    output_file.write(json.dumps(to_jsonable(output_record), ensure_ascii=False) + "\n")
                    output_file.flush()
    finally:
        if output_file is not None:
            output_file.close()

    if temp_output is not None:
        temp_output.replace(case_output_jsonl)
    temp_metric_errors.replace(metric_error_jsonl)

    return {
        "input_jsonl": input_jsonl,
        "case_output_jsonl": case_output_jsonl if write_case_jsonl else None,
        "metric_error_jsonl": metric_error_jsonl,
        "total_cases": total,
        "input_skipped_cases": input_skipped,
        "non_skipped_cases": total - input_skipped,
        "non_skipped_case_ratio": ((total - input_skipped) / total) if total else float("nan"),
        "valid_cases": valid,
        "metric_error_cases": metric_errors,
        "worst_case_assigned": worst,
        "excluded_cases": excluded,
        "average_dtw_score": float(np.mean(dtw_scores)) if dtw_scores else float("nan"),
        "average_dtw_ratio": float(np.mean(dtw_ratios)) if dtw_ratios else float("nan"),
        "average_consistency_score": {
            key: float(np.mean(values)) for key, values in sorted(consistency_values.items())
        },
        "average_ot": {
            key: float(np.mean(values)) for key, values in sorted(ot_values.items())
        },
        "case_metrics": case_metrics,
    }


def calculate_accuracy_metrics(
    input_jsonl: Path,
    args: argparse.Namespace,
    metric_error_jsonl: Path,
) -> dict[str, object]:
    """Calculate aggregate metrics and bookkeeping for this result type."""
    if not input_jsonl.exists():
        raise FileNotFoundError(input_jsonl)

    total = 0
    valid = 0
    input_skipped = 0
    error_as_zero = 0
    excluded = 0
    metric_errors = 0
    accuracy_values: list[float] = []
    case_metrics: dict[str, dict] = {}

    metric_error_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temp_metric_errors = metric_error_jsonl.with_name(f"{metric_error_jsonl.name}.tmp")
    with (
        input_jsonl.open("r", encoding="utf-8") as input_file,
        temp_metric_errors.open("w", encoding="utf-8") as metric_error_file,
    ):
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            if record.get("skipped") is True:
                input_skipped += 1
            case_key = build_case_key(record)
            case_entry = case_metrics.setdefault(case_key, build_case_identity(record))
            case_entry["case_key"] = case_key
            case_entry["accuracy_line_number"] = line_number
            case_entry["accuracy_input_skipped"] = record.get("skipped") is True
            case_entry["accuracy_eval_method"] = record.get("eval_method", "qwen")
            try:
                normalized_scores = normalize_accuracy_scores(
                    record.get("results"),
                    args.accuracy_expected_results,
                    args.require_exact_accuracy_count,
                )
                case_accuracy = sum(normalized_scores) / len(normalized_scores)
                accuracy_values.append(case_accuracy)
                valid += 1
                case_entry["accuracy_skipped"] = False
                case_entry["accuracy_error"] = None
                case_entry["accuracy_assigned_zero"] = False
                case_entry["accuracy_metrics"] = {
                    "accuracy": case_accuracy,
                    "normalized_scores": normalized_scores,
                }
                print(
                    f"[accuracy {total}] line={line_number} key="
                    f"({record.get('case_path_id')}, {record.get('path_index')}) "
                    f"accuracy={case_accuracy:.6f}",
                    flush=True,
                )
            except Exception as exc:
                case_entry["accuracy_skipped"] = True
                case_entry["accuracy_error"] = str(exc)
                case_entry["accuracy_metric_error"] = record.get("skipped") is not True
                # Malformed or skipped accuracy rows are counted as zero when requested.
                if args.include_accuracy_errors_as_zero:
                    accuracy_values.append(0.0)
                    error_as_zero += 1
                    status = "assigned_zero"
                    case_entry["accuracy_assigned_zero"] = True
                    case_entry["accuracy_metrics"] = {
                        "accuracy": 0.0,
                        "normalized_scores": [],
                    }
                else:
                    excluded += 1
                    status = "excluded"
                    case_entry["accuracy_assigned_zero"] = False
                    case_entry["accuracy_metrics"] = None
                print(
                    f"[accuracy {total}] line={line_number} key="
                    f"({record.get('case_path_id')}, {record.get('path_index')}) "
                    f"{status}: {exc}",
                    flush=True,
                )
                if record.get("skipped") is not True:
                    metric_errors += 1
                    metric_error_file.write(
                        json.dumps(
                            build_metric_error_record(
                                metric_type="accuracy",
                                line_number=line_number,
                                record=record,
                                error=exc,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    metric_error_file.flush()

    temp_metric_errors.replace(metric_error_jsonl)

    return {
        "input_jsonl": input_jsonl,
        "metric_error_jsonl": metric_error_jsonl,
        "total_cases": total,
        "input_skipped_cases": input_skipped,
        "non_skipped_cases": total - input_skipped,
        "non_skipped_case_ratio": ((total - input_skipped) / total) if total else float("nan"),
        "valid_cases": valid,
        "metric_error_cases": metric_errors,
        "error_cases_assigned_zero": error_as_zero,
        "excluded_cases": excluded,
        "average_accuracy": float(np.mean(accuracy_values)) if accuracy_values else float("nan"),
        "case_metrics": case_metrics,
    }


def write_combined_case_metrics(
    output_jsonl: Path,
    pathway: dict[str, object],
    accuracy: dict[str, object],
) -> None:
    """Write merged per-case pathway and accuracy metrics."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pathway_cases = pathway.get("case_metrics", {}) or {}
    accuracy_cases = accuracy.get("case_metrics", {}) or {}
    all_keys = sorted(set(pathway_cases) | set(accuracy_cases))
    tmp = output_jsonl.with_name(f"{output_jsonl.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for key in all_keys:
            merged = {}
            if key in pathway_cases:
                merged.update(pathway_cases[key])
            if key in accuracy_cases:
                for field, value in accuracy_cases[key].items():
                    if field not in merged or merged.get(field) in (None, ""):
                        merged[field] = value
                    elif field in {"accuracy_metrics", "accuracy_error", "accuracy_skipped", "accuracy_assigned_zero", "accuracy_metric_error", "accuracy_line_number", "accuracy_input_skipped", "accuracy_eval_method"}:
                        merged[field] = value
            merged.setdefault("case_key", key)
            merged.setdefault("pathway_metrics", None)
            merged.setdefault("accuracy_metrics", None)
            f.write(json.dumps(to_jsonable(merged), ensure_ascii=False) + "\n")
    tmp.replace(output_jsonl)


def format_metric_dict(prefix: str, values: dict[str, float]) -> str:
    """Format a dictionary of aggregate metrics for the summary file."""
    if not values:
        return ""
    lines = []
    for key, value in sorted(values.items()):
        safe_key = str(key).replace(".", "_")
        lines.append(f"{prefix}_{safe_key}: {value:.10f}")
    return "\n".join(lines) + "\n"


def build_summary(
    exp_name: str,
    pathway: dict[str, object],
    accuracy: dict[str, object],
) -> str:
    """Build the human-readable final metrics summary text."""
    case_output = pathway.get("case_output_jsonl") or "not_written"
    return (
        f"exp_name: {exp_name}\n"
        "\n"
        "[pathway_consistency]\n"
        f"input_jsonl: {pathway['input_jsonl']}\n"
        f"case_output_jsonl: {case_output}\n"
        f"metric_error_jsonl: {pathway['metric_error_jsonl']}\n"
        f"total_cases: {pathway['total_cases']}\n"
        f"input_skipped_cases: {pathway['input_skipped_cases']}\n"
        f"non_skipped_cases: {pathway['non_skipped_cases']}\n"
        f"non_skipped_case_ratio: {pathway['non_skipped_case_ratio']:.10f}\n"
        f"valid_cases: {pathway['valid_cases']}\n"
        f"metric_error_cases: {pathway['metric_error_cases']}\n"
        f"worst_case_assigned: {pathway['worst_case_assigned']}\n"
        f"excluded_cases: {pathway['excluded_cases']}\n"
        f"average_dtw_score: {pathway['average_dtw_score']:.10f}\n"
        f"average_dtw_ratio: {pathway['average_dtw_ratio']:.10f}\n"
        f"{format_metric_dict('average_consistency_score', pathway.get('average_consistency_score', {}))}"
        f"{format_metric_dict('average_ot', pathway.get('average_ot', {}))}"
        "lower_is_better: true\n"
        "\n"
        "[accuracy]\n"
        f"input_jsonl: {accuracy['input_jsonl']}\n"
        f"metric_error_jsonl: {accuracy['metric_error_jsonl']}\n"
        f"total_cases: {accuracy['total_cases']}\n"
        f"input_skipped_cases: {accuracy['input_skipped_cases']}\n"
        f"non_skipped_cases: {accuracy['non_skipped_cases']}\n"
        f"non_skipped_case_ratio: {accuracy['non_skipped_case_ratio']:.10f}\n"
        f"valid_cases: {accuracy['valid_cases']}\n"
        f"metric_error_cases: {accuracy['metric_error_cases']}\n"
        f"error_cases_assigned_zero: {accuracy['error_cases_assigned_zero']}\n"
        f"excluded_cases: {accuracy['excluded_cases']}\n"
        f"average_accuracy: {accuracy['average_accuracy']:.10f}\n"
        "higher_is_better: true\n"
        "scale: normalized_from_1_5_to_0_1\n"
    )


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    pathway_input, accuracy_input, case_output_jsonl, summary_txt, case_metrics_jsonl = resolve_paths(args)
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    pathway_metric_error_jsonl = summary_txt.with_name(
        f"{args.exp_name}_pathway_metric_errors.jsonl"
    )
    accuracy_metric_error_jsonl = summary_txt.with_name(
        f"{args.exp_name}_accuracy_metric_errors.jsonl"
    )

    write_case_jsonl = args.write_case_jsonl or args.output_jsonl is not None
    pathway_metrics = calculate_pathway_metrics(
        pathway_input,
        args,
        case_output_jsonl,
        pathway_metric_error_jsonl,
        write_case_jsonl,
    )
    accuracy_metrics = calculate_accuracy_metrics(
        accuracy_input,
        args,
        accuracy_metric_error_jsonl,
    )
    write_combined_case_metrics(case_metrics_jsonl, pathway_metrics, accuracy_metrics)
    pathway_metrics.pop("case_metrics", None)
    accuracy_metrics.pop("case_metrics", None)
    summary = build_summary(args.exp_name, pathway_metrics, accuracy_metrics)
    summary_txt.write_text(summary, encoding="utf-8")
    print(
        f"[done]\n{summary}summary_txt: {summary_txt}\n"
        f"case_metrics_jsonl: {case_metrics_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
