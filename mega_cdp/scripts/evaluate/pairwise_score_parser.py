"""Parse pathway consistency score matrices."""

from __future__ import annotations

import re
from typing import Any


PAIR_PATTERN = re.compile(
    r"^R\s*(\d+)\s*-\s*C\s*(\d+)\s*:\s*([0-9]+(?:\.[0-9]+)?)$",
    re.IGNORECASE,
)


def parse_pairwise_score_matrix(
    response_text: str,
    reference_pathway: list[Any],
    candidate_pathway: list[Any],
) -> list[list[float]]:
    """Parse ``Ri-Cj: score`` pairs into a complete score matrix.

    The output must contain every expected in-range pair exactly once, and every
    score must be within the benchmark's 1--5 scale.
    """
    cleaned = (response_text or "").strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        cleaned = cleaned[first_newline + 1 :] if first_newline >= 0 else ""
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    parts = [
        part.strip()
        for line in cleaned.splitlines()
        for part in re.split(r"\s*,\s*", line)
        if part.strip()
    ]
    if not parts:
        raise ValueError("Empty response after cleaning")

    rows = len(reference_pathway)
    columns = len(candidate_pathway)
    matrix: list[list[float | None]] = [[None] * columns for _ in range(rows)]
    seen: set[tuple[int, int]] = set()
    for part in parts:
        match = PAIR_PATTERN.fullmatch(part)
        if match is None:
            raise ValueError(f"Invalid pair format: {part}")
        ref_index = int(match.group(1))
        cand_index = int(match.group(2))
        score = float(match.group(3))
        if not 1 <= ref_index <= rows:
            raise ValueError(f"Reference index out of range: {part}")
        if not 1 <= cand_index <= columns:
            raise ValueError(f"Candidate index out of range: {part}")
        if not 1.0 <= score <= 5.0:
            raise ValueError(f"Score out of range (1-5): {part}")
        key = (ref_index, cand_index)
        if key in seen:
            raise ValueError(f"Duplicate pair: {part}")
        seen.add(key)
        matrix[ref_index - 1][cand_index - 1] = score

    if len(seen) != rows * columns:
        raise ValueError("Incomplete consistency matrix")
    return [[float(value) for value in row] for row in matrix]
