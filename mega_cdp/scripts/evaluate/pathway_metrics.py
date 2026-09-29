"""Pathway alignment metrics."""

from collections.abc import Sequence

import numpy as np

try:
    from ot.unbalanced import sinkhorn_knopp_unbalanced
except ImportError:
    sinkhorn_knopp_unbalanced = None

from optimal_transport import ot_consistency_score


def dp_evaluation(mapping_sequence: Sequence[int]) -> float:
    """Return the longest non-decreasing subsequence length."""
    if not mapping_sequence:
        return 0.0
    length = len(mapping_sequence)
    dp = np.ones(length)
    for i in range(length):
        for j in range(i):
            if mapping_sequence[j] <= mapping_sequence[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return float(np.max(dp))


def dynamic_time_warping(
    difference_map: list[list[float]],
) -> tuple[float, float, list[tuple[int, int]]]:
    """Compute DTW cost and average DTW cost over the chosen path."""
    if not difference_map or not difference_map[0]:
        return 0.0, 0.0, []

    row_count = len(difference_map)
    column_count = len(difference_map[0])
    if any(len(row) != column_count for row in difference_map):
        raise ValueError("difference_map must be rectangular")

    dp = [[float("inf")] * column_count for _ in range(row_count)]
    backtrack: list[list[tuple[int, int] | None]] = [
        [None] * column_count for _ in range(row_count)
    ]
    for i in range(row_count):
        for j in range(column_count):
            cost = float(difference_map[i][j])
            if i == 0 and j == 0:
                dp[i][j] = cost
                continue
            candidates = []
            if i > 0:
                candidates.append((dp[i - 1][j], (i - 1, j)))
            if j > 0:
                candidates.append((dp[i][j - 1], (i, j - 1)))
            if i > 0 and j > 0:
                candidates.append((dp[i - 1][j - 1], (i - 1, j - 1)))
            previous_cost, previous_coordinate = min(candidates, key=lambda item: item[0])
            dp[i][j] = cost + previous_cost
            backtrack[i][j] = previous_coordinate

    path: list[tuple[int, int]] = []
    i, j = row_count - 1, column_count - 1
    while True:
        path.append((i, j))
        previous = backtrack[i][j]
        if previous is None:
            break
        i, j = previous
    path.reverse()
    dtw_score = float(dp[-1][-1])
    return dtw_score, float(dtw_score / len(path)), path[::-1]


def build_pred_to_gt_mappings(
    matrix: Sequence[Sequence[float]],
    thresholds: Sequence[float],
    ot_reg: float,
    ot_tau: float,
    eval_method: str,
) -> tuple[
    dict[str, float],
    float,
    float,
    dict[str, list[int]],
    list[tuple[int, int]],
    list[list[float]],
    np.ndarray,
    dict[str, float],
]:
    """Compute threshold mappings, DTW, and OT metrics from a similarity matrix."""
    alpha = 1.0
    if eval_method in {"gpt", "qwen"}:
        beta = 0.2
    elif eval_method == "bertscore":
        beta = 0.1
    else:
        raise ValueError(f"Unsupported eval_method: {eval_method}")

    row_count = len(matrix)
    column_count = len(matrix[0]) if matrix else 0
    row_pos = (np.arange(row_count) + 0.5) / row_count if row_count else np.array([])
    col_pos = (np.arange(column_count) + 0.5) / column_count if column_count else np.array([])
    position_cost = np.abs(row_pos[:, None] - col_pos[None, :]) if row_count and column_count else np.array([])

    similarity_map: list[list[float]] = []
    difference_map: list[list[float]] = []
    ot_cost_map: list[list[float]] = []
    mapping_sequences: dict[str, list[int]] = {}

    for row_index, score_vector in enumerate(matrix):
        if not score_vector:
            continue
        similarity_map.append([float(score) for score in score_vector])
        scores = np.asarray(score_vector, dtype=float)
        semantic_cost = 1.0 - np.clip(scores, 0.0, 1.0)
        ot_cost_row = alpha * semantic_cost + beta * position_cost[row_index, :] / (alpha + beta)
        difference_map.append(semantic_cost.tolist())
        ot_cost_map.append(ot_cost_row.tolist())

        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        for threshold in thresholds:
            key = f"{threshold:.2f}".rstrip("0").rstrip(".")
            sequence = mapping_sequences.setdefault(key, [])
            if best_score > threshold:
                sequence.append(best_index + 1)

    path_length = max(len(matrix), 1)
    consistency_scores = {
        key: dp_evaluation(sequence) / path_length
        for key, sequence in mapping_sequences.items()
    }
    dtw_score, dtw_ratio, dtw_path = dynamic_time_warping(difference_map)

    transport_plan = np.array([])
    ot_metrics: dict[str, float] = {}
    if similarity_map and similarity_map[0]:
        if sinkhorn_knopp_unbalanced is None:
            raise ImportError("Install POT to compute OT metrics: pip install POT")
        cost_matrix = np.asarray(ot_cost_map, dtype=float)
        n_rows, n_columns = cost_matrix.shape
        source_mass = np.full(n_rows, 1.0 / n_rows, dtype=float)
        target_mass = np.full(n_columns, 1.0 / n_columns, dtype=float)
        transport_plan = sinkhorn_knopp_unbalanced(
            source_mass,
            target_mass,
            cost_matrix,
            ot_reg,
            ot_tau,
            numItermax=3000,
            stopThr=1e-9,
        )
        ot_metrics = ot_consistency_score(transport_plan, cost_matrix, source_mass, target_mass)

    return (
        consistency_scores,
        dtw_score,
        dtw_ratio,
        mapping_sequences,
        dtw_path,
        difference_map,
        transport_plan,
        ot_metrics,
    )
