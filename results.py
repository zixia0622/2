from __future__ import annotations

from typing import Dict, Iterable, Sequence

import pandas as pd

from .optimization import BASE_TREATMENT, available_actions, solve_budget_allocation


def build_treatment_summary(scoring: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in scoring.columns:
        if not col.startswith("delta_rev_"):
            continue
        action = col.replace("delta_rev_", "")
        cost_col = f"delta_cost_{action}"
        roi_col = f"delta_roi_{action}"
        clipped_cost = scoring[cost_col].clip(lower=0.0)
        rows.append(
            {
                "action": action,
                "avg_delta_rev": float(scoring[col].mean()),
                "median_delta_rev": float(scoring[col].median()),
                "avg_delta_cost": float(scoring[cost_col].mean()),
                "avg_delta_cost_clipped": float(clipped_cost.mean()),
                "avg_delta_roi": float(scoring[roi_col].mean()),
                "positive_uplift_ratio": float((scoring[col] > 0).mean()),
                "positive_cost_ratio": float((scoring[cost_col] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_delta_rev", ascending=False)


def build_budget_action_mix(
    scoring: pd.DataFrame,
    budgets: Sequence[float],
    solver_mode: str = "hard",
    tau: float = 0.75,
    cost_temperature: float = 0.5,
    lambda_init_map: Dict[float, float] | None = None,
) -> pd.DataFrame:
    rows = []
    actions = [BASE_TREATMENT] + available_actions(scoring)
    for budget in budgets:
        lambda_init = None if lambda_init_map is None else lambda_init_map.get(float(budget))
        result = solve_budget_allocation(
            scoring,
            budget,
            solver_mode=solver_mode,
            tau=tau,
            cost_temperature=cost_temperature,
            lambda_init=lambda_init,
        )
        if solver_mode == "soft":
            total_mass = max(len(result.selected_actions), 1)
            for action in actions:
                prob_col = f"prob_{action}"
                mass = float(result.selected_actions[prob_col].sum()) if prob_col in result.selected_actions.columns else 0.0
                rows.append(
                    {
                        "budget": budget,
                        "solver_mode": solver_mode,
                        "action": action,
                        "selected_count": mass,
                        "selected_ratio": float(mass / total_mass),
                    }
                )
            continue

        mix = result.selected_actions["selected_action"].value_counts().to_dict()
        for action, count in mix.items():
            rows.append(
                {
                    "budget": budget,
                    "solver_mode": solver_mode,
                    "action": action,
                    "selected_count": float(count),
                    "selected_ratio": float(count / max(len(result.selected_actions), 1)),
                }
            )
    return pd.DataFrame(rows).sort_values(["budget", "selected_count"], ascending=[True, False])


def build_bu_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    grouped = dataset.groupby("BU").agg(
        sample_count=("User_id", "count"),
        avg_rev_7d=("Y_rev_7d", "mean"),
        avg_cost_7d=("Y_cost_7d", "mean"),
        order_rate_7d=("Y_order_7d", "mean"),
        avg_visit_7d=("Y_visit_7d", "mean"),
    )
    return grouped.reset_index().sort_values("sample_count", ascending=False)
