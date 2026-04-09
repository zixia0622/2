from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class AllocationResult:
    budget: float
    lambda_dual: float
    total_cost: float
    total_value: float
    selected_actions: pd.DataFrame
    solver_mode: str


BASE_TREATMENT = "T0_no_coupon"


def available_actions(scoring: pd.DataFrame) -> List[str]:
    actions = []
    for col in scoring.columns:
        if col.startswith("delta_rev_"):
            actions.append(col.replace("delta_rev_", ""))
    return sorted(actions)


def _softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    safe_tau = max(float(tau), 1e-6)
    shifted = scores / safe_tau
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)


def _softplus(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return np.log1p(np.exp(clipped))


def prepare_action_tensors(
    delta_rev: np.ndarray,
    raw_delta_cost: np.ndarray,
    smooth_cost: bool = False,
    cost_temperature: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if delta_rev.size == 0:
        n_rows = raw_delta_cost.shape[0] if raw_delta_cost.ndim == 2 else len(raw_delta_cost)
        zeros = np.zeros((n_rows, 1), dtype=float)
        feasible = np.ones((n_rows, 1), dtype=bool)
        return zeros, zeros, feasible

    if smooth_cost:
        temperature = max(cost_temperature, 1e-6)
        cost_nonbase = _softplus(raw_delta_cost / temperature) * temperature
    else:
        cost_nonbase = np.clip(raw_delta_cost, 0.0, None)

    feasible_nonbase = delta_rev > 0.0
    if not smooth_cost:
        feasible_nonbase &= cost_nonbase > 0.0

    zeros = np.zeros((delta_rev.shape[0], 1), dtype=float)
    values = np.concatenate([zeros, delta_rev], axis=1)
    costs = np.concatenate([zeros, cost_nonbase], axis=1)
    feasible = np.concatenate([np.ones((delta_rev.shape[0], 1), dtype=bool), feasible_nonbase], axis=1)
    return values, costs, feasible


def gain_matrix(values: np.ndarray, costs: np.ndarray, feasible: np.ndarray, lambda_dual: float) -> np.ndarray:
    gain = values - float(lambda_dual) * costs
    gain = np.where(feasible, gain, -1e12)
    gain[:, 0] = 0.0
    return gain


def soft_action_probabilities(gain: np.ndarray, tau: float) -> np.ndarray:
    return _softmax(gain, tau=tau)


def action_tensors_from_scoring(
    scoring: pd.DataFrame,
    smooth_cost: bool = False,
    cost_temperature: float = 0.5,
) -> tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    actions = available_actions(scoring)
    if not actions:
        n_rows = len(scoring)
        zeros = np.zeros((n_rows, 1), dtype=float)
        feasible = np.ones((n_rows, 1), dtype=bool)
        return [BASE_TREATMENT], zeros, zeros, feasible

    delta_rev = np.column_stack([scoring[f"delta_rev_{action}"].to_numpy(dtype=float) for action in actions])
    raw_delta_cost = np.column_stack([scoring[f"delta_cost_{action}"].to_numpy(dtype=float) for action in actions])
    values, costs, feasible = prepare_action_tensors(
        delta_rev=delta_rev,
        raw_delta_cost=raw_delta_cost,
        smooth_cost=smooth_cost,
        cost_temperature=cost_temperature,
    )
    return [BASE_TREATMENT] + actions, values, costs, feasible


def choose_actions_for_lambda(
    scoring: pd.DataFrame,
    lambda_dual: float,
    solver_mode: str = "hard",
    tau: float = 0.75,
    smooth_cost: bool | None = None,
    cost_temperature: float = 0.5,
) -> pd.DataFrame:
    if solver_mode not in {"hard", "soft"}:
        raise ValueError("solver_mode must be one of: hard, soft")
    if smooth_cost is None:
        smooth_cost = solver_mode == "soft"

    action_names, values, costs, feasible = action_tensors_from_scoring(
        scoring,
        smooth_cost=smooth_cost,
        cost_temperature=cost_temperature,
    )
    gain = gain_matrix(values, costs, feasible, lambda_dual=lambda_dual)

    rows = {
        "index": scoring.index.to_numpy(),
        "User_id": scoring["User_id"].to_numpy(),
        "BU": scoring["BU"].to_numpy(),
        "Decision_date": scoring["Decision_date"].to_numpy(),
    }

    if solver_mode == "hard":
        chosen_idx = gain.argmax(axis=1)
        rows["selected_action"] = np.array(action_names, dtype=object)[chosen_idx]
        rows["selected_prob"] = np.ones(len(scoring), dtype=float)
        rows["selected_value"] = values[np.arange(len(scoring)), chosen_idx]
        rows["selected_cost"] = costs[np.arange(len(scoring)), chosen_idx]
        rows["selected_score"] = gain[np.arange(len(scoring)), chosen_idx]
        return pd.DataFrame(rows)

    probs = soft_action_probabilities(gain, tau=tau)
    top_idx = probs.argmax(axis=1)
    rows["selected_action"] = np.array(action_names, dtype=object)[top_idx]
    rows["selected_prob"] = probs[np.arange(len(scoring)), top_idx]
    rows["selected_value"] = (probs * values).sum(axis=1)
    rows["selected_cost"] = (probs * costs).sum(axis=1)
    rows["selected_score"] = (probs * gain).sum(axis=1)
    for idx, action in enumerate(action_names):
        rows[f"prob_{action}"] = probs[:, idx]
    return pd.DataFrame(rows)


def solve_budget_allocation(
    scoring: pd.DataFrame,
    budget: float,
    max_iter: int = 50,
    solver_mode: str = "hard",
    tau: float = 0.75,
    smooth_cost: bool | None = None,
    cost_temperature: float = 0.5,
    lambda_init: float | None = None,
) -> AllocationResult:
    if smooth_cost is None:
        smooth_cost = solver_mode == "soft"

    lo = 0.0
    hi = max(float(lambda_init) if lambda_init is not None else 10.0, 1e-6)
    selected = choose_actions_for_lambda(
        scoring,
        hi,
        solver_mode=solver_mode,
        tau=tau,
        smooth_cost=smooth_cost,
        cost_temperature=cost_temperature,
    )
    while selected["selected_cost"].sum() > budget and hi < 1e6:
        lo = hi
        hi *= 2.0
        selected = choose_actions_for_lambda(
            scoring,
            hi,
            solver_mode=solver_mode,
            tau=tau,
            smooth_cost=smooth_cost,
            cost_temperature=cost_temperature,
        )

    best_lambda = hi
    best_selected = selected
    for _ in range(max_iter):
        lam = (lo + hi) / 2.0
        candidate = choose_actions_for_lambda(
            scoring,
            lam,
            solver_mode=solver_mode,
            tau=tau,
            smooth_cost=smooth_cost,
            cost_temperature=cost_temperature,
        )
        total_cost = candidate["selected_cost"].sum()
        if total_cost > budget:
            lo = lam
        else:
            hi = lam
            best_lambda = lam
            best_selected = candidate
    return AllocationResult(
        budget=budget,
        lambda_dual=best_lambda,
        total_cost=float(best_selected["selected_cost"].sum()),
        total_value=float(best_selected["selected_value"].sum()),
        selected_actions=best_selected,
        solver_mode=solver_mode,
    )


def summarize_allocations(
    scoring: pd.DataFrame,
    budgets: Sequence[float],
    solver_mode: str = "hard",
    tau: float = 0.75,
    smooth_cost: bool | None = None,
    cost_temperature: float = 0.5,
    lambda_init_map: Dict[float, float] | None = None,
) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        lambda_init = None if lambda_init_map is None else lambda_init_map.get(float(budget))
        result = solve_budget_allocation(
            scoring,
            budget,
            solver_mode=solver_mode,
            tau=tau,
            smooth_cost=smooth_cost,
            cost_temperature=cost_temperature,
            lambda_init=lambda_init,
        )
        if solver_mode == "soft" and f"prob_{BASE_TREATMENT}" in result.selected_actions.columns:
            selected_user_count = float((1.0 - result.selected_actions[f"prob_{BASE_TREATMENT}"]).sum())
        else:
            selected_user_count = float((result.selected_actions["selected_action"] != BASE_TREATMENT).sum())
        rows.append(
            {
                "budget": result.budget,
                "solver_mode": solver_mode,
                "lambda_dual": result.lambda_dual,
                "total_cost": result.total_cost,
                "total_value": result.total_value,
                "selected_user_count": selected_user_count,
                "avg_roi": result.total_value / (result.total_cost + 1e-6),
            }
        )
    return pd.DataFrame(rows)
