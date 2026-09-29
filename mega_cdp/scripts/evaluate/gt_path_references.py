#!/usr/bin/env python3
"""Load single-turn evaluation references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ReferenceKey = tuple[Any, Any, str]


def reference_key(record: dict[str, Any]) -> ReferenceKey:
    """Return the stable identity shared by predictions and GT path records."""
    guideline_id = record.get("case_path_id", record.get("guideline_id"))
    path_id = record.get("path_index", record.get("path_id"))
    filename = record.get("filename")
    if guideline_id is None or path_id is None or not isinstance(filename, str):
        raise ValueError(
            "Reference identity requires case_path_id/guideline_id, "
            "path_index/path_id, and filename"
        )
    return guideline_id, path_id, filename


def load_gt_path_references(path: Path) -> dict[ReferenceKey, dict[str, Any]]:
    """Index one gt_paths record per single-turn vignette case."""
    references: dict[ReferenceKey, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            key = reference_key(record)
            if key in references:
                raise ValueError(f"Duplicate gt_paths reference {key!r} in {path}")
            references[key] = record
    return references


def resolve_reference(
    prediction: dict[str, Any],
    references: dict[ReferenceKey, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve one parsed prediction to its case-specific GT reference."""
    key = reference_key(prediction)
    try:
        return references[key]
    except KeyError as exc:
        raise ValueError(f"No matching gt_paths record for {key!r}") from exc


def load_reference_steps(reference: dict[str, Any]) -> list[str]:
    """Decode the complete guideline-defined single-turn reasoning pathway."""
    output_text = reference.get("output_text")
    if not isinstance(output_text, str):
        raise ValueError("gt_paths output_text must be a string")
    try:
        steps = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError("gt_paths output_text is not valid JSON") from exc
    if not isinstance(steps, list) or not steps:
        raise ValueError("gt_paths output_text must be a non-empty JSON list")
    if any(not isinstance(step, str) or not step.strip() for step in steps):
        raise ValueError("gt_paths pathway steps must be non-empty strings")
    return [step.strip() for step in steps]


def load_reference_conclusion(reference: dict[str, Any]) -> str:
    """Return the final item of the case-specific GT reasoning pathway."""
    return load_reference_steps(reference)[-1]
