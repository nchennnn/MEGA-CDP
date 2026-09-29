#!/usr/bin/env python3
"""Load multi-turn evaluation references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ReferenceKey = tuple[Any, Any, str]


def reference_key(record: dict[str, Any]) -> ReferenceKey:
    """Return the stable key used to align predictions with references."""
    case_path_id = record.get("case_path_id", record.get("guideline_id"))
    path_index = record.get("path_index", record.get("path_id"))
    filename = record.get("filename")
    if case_path_id is None or path_index is None or not isinstance(filename, str):
        raise ValueError(
            "Reference identity requires case_path_id/guideline_id, "
            "path_index/path_id, and filename"
        )
    return case_path_id, path_index, filename


def load_case_path_references(path: Path) -> dict[ReferenceKey, dict[str, Any]]:
    """Expand each guideline row into one reference record per decision path."""
    references: dict[ReferenceKey, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                guideline = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc

            guideline_id = guideline.get("guideline_id")
            filename = guideline.get("filename")
            language = guideline.get("language")
            paths = guideline.get("paths")
            if guideline_id is None or not isinstance(filename, str):
                raise ValueError(f"Missing guideline_id or filename at {path}:{line_number}")
            if not isinstance(paths, list):
                raise ValueError(f"paths must be a list at {path}:{line_number}")

            for path_index, decision_path in enumerate(paths, start=1):
                if decision_path is None or decision_path == []:
                    continue
                if not isinstance(decision_path, list):
                    raise ValueError(
                        f"paths[{path_index - 1}] must be a list at "
                        f"{path}:{line_number}"
                    )
                reference = {
                    "case_path_id": guideline_id,
                    "path_index": path_index,
                    "filename": filename,
                    "language": language,
                    "decision_path": decision_path,
                }
                key = reference_key(reference)
                if key in references:
                    raise ValueError(f"Duplicate case-path reference {key!r} in {path}")
                references[key] = reference
    return references


def resolve_reference(
    parsed_record: dict[str, Any],
    references: dict[ReferenceKey, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve one prediction record to its matching case_paths reference."""
    key = reference_key(parsed_record)
    try:
        return references[key]
    except KeyError as exc:
        raise ValueError(f"No matching case path for {key!r}") from exc


def build_aligned_reference_lines(
    parsed_lines: Iterable[str],
    references: dict[ReferenceKey, dict[str, Any]],
) -> list[str]:
    """Format reference pathway steps for prompts."""
    lines: list[str] = []
    for line_number, line in enumerate(parsed_lines, start=1):
        if not line.strip():
            raise ValueError(f"Empty parsed jsonl line at {line_number}")
        parsed_record = json.loads(line)
        reference = resolve_reference(parsed_record, references)
        lines.append(json.dumps(reference, ensure_ascii=False) + "\n")
    return lines


def load_reference_steps(reference: dict[str, Any]) -> list[str]:
    """Extract the reference pathway steps without the final conclusion."""
    decision_path = reference.get("decision_path")
    if not isinstance(decision_path, list):
        raise ValueError("Reference decision_path must be a list")
    steps: list[str] = []
    for entry in decision_path:
        if not isinstance(entry, dict):
            raise ValueError("Reference decision_path entries must be objects")
        condition = entry.get("condition")
        if condition is None and "conclusion" in entry:
            continue
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError("Reference condition must be a non-empty string")
        steps.append(condition.strip())
    if not steps:
        raise ValueError("Reference pathway has no condition steps")
    return steps


def load_reference_conclusion(reference: dict[str, Any]) -> str:
    """Extract the reference final clinical decision conclusion."""
    decision_path = reference.get("decision_path")
    if not isinstance(decision_path, list) or not decision_path:
        raise ValueError("Reference decision_path must be a non-empty list")
    for entry in reversed(decision_path):
        if not isinstance(entry, dict):
            raise ValueError("Reference decision_path entries must be objects")
        conclusion = entry.get("conclusion")
        if isinstance(conclusion, str) and conclusion.strip():
            return conclusion.strip()
    raise ValueError("Reference decision_path has no non-empty conclusion")
