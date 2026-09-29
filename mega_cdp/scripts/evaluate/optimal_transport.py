"""Optimal-transport helper metrics for pathway evaluation."""

import numpy as np


def ot_consistency_score(
    transport_plan: np.ndarray,
    cost_matrix: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
) -> dict[str, float]:
    """Summarize an unbalanced OT plan with cost, similarity, and mass gaps."""
    transport_plan = np.asarray(transport_plan, dtype=float)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    source_mass = np.asarray(source_mass, dtype=float)
    target_mass = np.asarray(target_mass, dtype=float)

    transported_mass = float(transport_plan.sum())
    if transported_mass <= 0.0:
        return {
            "avg_cost": 1.0,
            "similarity": 0.0,
            "missing_rate": 1.0,
            "extra_rate": 1.0,
            "transported_mass": 0.0,
        }

    avg_cost = float((transport_plan * cost_matrix).sum() / transported_mass)
    column_mass = transport_plan.sum(axis=0)
    row_mass = transport_plan.sum(axis=1)
    missing_mass = np.clip(target_mass - column_mass, 0.0, None).sum()
    extra_mass = np.clip(source_mass - row_mass, 0.0, None).sum()

    return {
        "avg_cost": avg_cost,
        "similarity": float(1.0 - avg_cost),
        "missing_rate": float(missing_mass / max(target_mass.sum(), 1e-12)),
        "extra_rate": float(extra_mass / max(source_mass.sum(), 1e-12)),
        "transported_mass": transported_mass,
    }
