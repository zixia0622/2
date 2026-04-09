from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import (
    BUDGET_GRID,
    DFCL_COST_TEMPERATURE,
    DFCL_DECISION_WEIGHT,
    DFCL_DUAL_LR,
    DFCL_EPOCHS,
    DFCL_L2,
    DFCL_LR,
    DFCL_TAU,
    MIN_TREATMENT_SAMPLES,
    RANDOM_STATE,
)
from .optimization import gain_matrix, prepare_action_tensors, soft_action_probabilities

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback when sklearn is unavailable
    SKLEARN_AVAILABLE = False


@dataclass
class DRResult:
    scoring_table: pd.DataFrame
    feature_columns: List[str]
    treatments: List[str]
    model_mode: str
    dual_lambdas: Dict[float, float]


TARGET_COLUMNS = [
    "Y_rev_7d",
    "Y_cost_7d",
    "Y_cnt_7d",
    "Y_order_7d",
    "Y_visit_7d",
]

AUX_DROP_COLUMNS = [
    "User_id",
    "Decision_date",
    "Coupon_id",
    "Coupon_batch_id",
    "Order_id",
]

BASE_TREATMENT = "T0_no_coupon"


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = set(TARGET_COLUMNS + AUX_DROP_COLUMNS + ["treatment"])
    feature_columns = [c for c in df.columns if c not in drop_cols]
    filtered: List[str] = []
    for col in feature_columns:
        # Datetime columns are converted into numeric day offsets elsewhere and are unstable in encoders.
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        filtered.append(col)
    return filtered


def filter_treatments(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["treatment"].value_counts()
    keep = counts[counts >= MIN_TREATMENT_SAMPLES].index.tolist()
    if BASE_TREATMENT not in keep and BASE_TREATMENT in counts.index:
        keep.append(BASE_TREATMENT)
    return df[df["treatment"].isin(keep)].copy()


def _fit_empirical_propensity(df: pd.DataFrame) -> pd.DataFrame:
    by_bu = df.groupby(["BU", "treatment"]).size().rename("cnt").reset_index()
    bu_totals = by_bu.groupby("BU")["cnt"].transform("sum")
    by_bu["propensity"] = by_bu["cnt"] / bu_totals
    return by_bu[["BU", "treatment", "propensity"]]


def _fit_empirical_outcomes(df: pd.DataFrame, target: str) -> pd.DataFrame:
    global_means = df.groupby("treatment")[target].mean().rename("global_mean")
    bu_means = df.groupby(["BU", "treatment"])[target].mean().rename("bu_mean").reset_index()
    bu_counts = df.groupby(["BU", "treatment"])[target].size().rename("bu_cnt").reset_index()
    merged = bu_means.merge(bu_counts, on=["BU", "treatment"], how="left")
    merged["global_mean"] = merged["treatment"].map(global_means)
    alpha = 20.0
    merged["pred_mean"] = (merged["bu_cnt"] * merged["bu_mean"] + alpha * merged["global_mean"]) / (merged["bu_cnt"] + alpha)
    return merged[["BU", "treatment", "pred_mean"]]


def _build_preprocessor(df: pd.DataFrame, feature_columns: List[str]):
    num_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_columns if c not in num_cols]
    cat_cols = [c for c in cat_cols if not pd.api.types.is_datetime64_any_dtype(df[c])]
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN", keep_empty_features=True)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers=transformers)


def _fit_propensity_model(df: pd.DataFrame, feature_columns: List[str]):
    preprocessor = _build_preprocessor(df, feature_columns)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(df[feature_columns], df["treatment"])
    return pipe


def _fit_outcome_models(df: pd.DataFrame, feature_columns: List[str], target: str) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for treatment, group in df.groupby("treatment"):
        preprocessor = _build_preprocessor(group, feature_columns)
        model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(group[feature_columns], group[target])
        models[treatment] = pipe
    return models


def _prepare_dense_features(df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    num_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_columns if c not in num_cols]
    cat_cols = [c for c in cat_cols if not pd.api.types.is_datetime64_any_dtype(df[c])]

    parts: List[np.ndarray] = []
    if num_cols:
        num_df = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        means = num_df.mean(axis=0)
        stds = num_df.std(axis=0).replace(0.0, 1.0).fillna(1.0)
        parts.append(((num_df - means) / stds).to_numpy(dtype=float))
    if cat_cols:
        cat_df = df[cat_cols].fillna("UNKNOWN").astype(str)
        parts.append(pd.get_dummies(cat_df, dummy_na=False).to_numpy(dtype=float))
    if not parts:
        return np.zeros((len(df), 1), dtype=float)
    return np.concatenate(parts, axis=1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _resolve_budget_grid(budget_grid: Sequence[float] | None) -> tuple[float, ...]:
    if budget_grid is None:
        return tuple(float(v) for v in BUDGET_GRID)
    return tuple(float(v) for v in budget_grid)


def _build_scoring_table(
    df: pd.DataFrame,
    treatments: List[str],
    propensity_map: Dict[str, np.ndarray],
    mu_rev_map: Dict[str, np.ndarray],
    mu_cost_map: Dict[str, np.ndarray],
    base_treatment: str,
    model_mode: str,
    dual_lambdas: Dict[float, float],
) -> DRResult:
    scoring = df[["User_id", "BU", "Decision_date", "treatment", "Y_rev_7d", "Y_cost_7d"]].copy()
    y_rev = df["Y_rev_7d"].to_numpy(dtype=float)
    y_cost = df["Y_cost_7d"].to_numpy(dtype=float)

    for treatment in treatments:
        ps = np.clip(propensity_map[treatment], 1e-3, 1.0)
        rev_hat = np.asarray(mu_rev_map[treatment], dtype=float)
        cost_hat = np.asarray(mu_cost_map[treatment], dtype=float)
        obs = (df["treatment"] == treatment).astype(float).to_numpy(dtype=float)

        scoring[f"ps_{treatment}"] = ps
        scoring[f"mu_rev_{treatment}"] = rev_hat
        scoring[f"mu_cost_{treatment}"] = cost_hat
        scoring[f"dr_rev_{treatment}"] = rev_hat + obs * (y_rev - rev_hat) / ps
        scoring[f"dr_cost_{treatment}"] = cost_hat + obs * (y_cost - cost_hat) / ps

    lambda_ref = float(np.mean(list(dual_lambdas.values()))) if dual_lambdas else 1.0
    for treatment in treatments:
        if treatment == base_treatment:
            continue
        scoring[f"delta_rev_{treatment}"] = scoring[f"dr_rev_{treatment}"] - scoring[f"dr_rev_{base_treatment}"]
        scoring[f"delta_cost_{treatment}"] = scoring[f"dr_cost_{treatment}"] - scoring[f"dr_cost_{base_treatment}"]
        scoring[f"delta_roi_{treatment}"] = scoring[f"delta_rev_{treatment}"] / (np.abs(scoring[f"delta_cost_{treatment}"]) + 1e-6)
        scoring[f"dual_score_{treatment}"] = scoring[f"delta_rev_{treatment}"] - lambda_ref * scoring[f"delta_cost_{treatment}"]

    return DRResult(
        scoring_table=scoring,
        feature_columns=[],
        treatments=treatments,
        model_mode=model_mode,
        dual_lambdas=dual_lambdas,
    )


def _estimate_empirical(df: pd.DataFrame, base_treatment: str) -> DRResult:
    treatments = sorted(df["treatment"].unique().tolist())
    propensity_table = _fit_empirical_propensity(df)
    rev_table = _fit_empirical_outcomes(df, "Y_rev_7d")
    cost_table = _fit_empirical_outcomes(df, "Y_cost_7d")

    global_prop = df["treatment"].value_counts(normalize=True).to_dict()
    global_rev = df.groupby("treatment")["Y_rev_7d"].mean().to_dict()
    global_cost = df.groupby("treatment")["Y_cost_7d"].mean().to_dict()

    propensity_map: Dict[str, np.ndarray] = {}
    mu_rev_map: Dict[str, np.ndarray] = {}
    mu_cost_map: Dict[str, np.ndarray] = {}
    for treatment in treatments:
        ps_map = propensity_table.loc[propensity_table["treatment"] == treatment].set_index("BU")["propensity"].to_dict()
        rev_map = rev_table.loc[rev_table["treatment"] == treatment].set_index("BU")["pred_mean"].to_dict()
        cost_map = cost_table.loc[cost_table["treatment"] == treatment].set_index("BU")["pred_mean"].to_dict()

        propensity_map[treatment] = df["BU"].map(ps_map).fillna(global_prop.get(treatment, 1.0 / max(len(treatments), 1))).to_numpy(dtype=float)
        mu_rev_map[treatment] = df["BU"].map(rev_map).fillna(global_rev.get(treatment, 0.0)).to_numpy(dtype=float)
        mu_cost_map[treatment] = df["BU"].map(cost_map).fillna(global_cost.get(treatment, 0.0)).to_numpy(dtype=float)

    return _build_scoring_table(
        df=df,
        treatments=treatments,
        propensity_map=propensity_map,
        mu_rev_map=mu_rev_map,
        mu_cost_map=mu_cost_map,
        base_treatment=base_treatment,
        model_mode="empirical",
        dual_lambdas={},
    )


def _estimate_sklearn(df: pd.DataFrame, feature_columns: List[str], base_treatment: str) -> DRResult:
    propensity_pipe = _fit_propensity_model(df, feature_columns)
    treatments = propensity_pipe.named_steps["model"].classes_.tolist()
    propensity_proba = propensity_pipe.predict_proba(df[feature_columns])
    propensity_df = pd.DataFrame(propensity_proba, columns=[f"ps_{c}" for c in treatments], index=df.index)

    rev_models = _fit_outcome_models(df, feature_columns, "Y_rev_7d")
    cost_models = _fit_outcome_models(df, feature_columns, "Y_cost_7d")

    propensity_map = {treatment: propensity_df[f"ps_{treatment}"].to_numpy(dtype=float) for treatment in treatments}
    mu_rev_map = {treatment: rev_models[treatment].predict(df[feature_columns]).astype(float) for treatment in treatments}
    mu_cost_map = {treatment: cost_models[treatment].predict(df[feature_columns]).astype(float) for treatment in treatments}

    result = _build_scoring_table(
        df=df,
        treatments=treatments,
        propensity_map=propensity_map,
        mu_rev_map=mu_rev_map,
        mu_cost_map=mu_cost_map,
        base_treatment=base_treatment,
        model_mode="sklearn",
        dual_lambdas={},
    )
    result.feature_columns = feature_columns
    return result


def _fit_dfcl_linear_heads(
    df: pd.DataFrame,
    feature_columns: List[str],
    treatments: List[str],
    base_treatment: str,
    budget_grid: Sequence[float],
):
    rng = np.random.default_rng(RANDOM_STATE)
    X = _prepare_dense_features(df, feature_columns)
    n_samples, n_features = X.shape
    n_treatments = len(treatments)
    treatment_to_idx = {name: idx for idx, name in enumerate(treatments)}
    obs_idx = df["treatment"].map(treatment_to_idx).to_numpy(dtype=int)

    y_rev = df["Y_rev_7d"].to_numpy(dtype=float)
    y_cost = df["Y_cost_7d"].to_numpy(dtype=float)

    wr = rng.normal(scale=0.01, size=(n_treatments, n_features))
    wc = rng.normal(scale=0.01, size=(n_treatments, n_features))
    br = np.zeros(n_treatments, dtype=float)
    bc = np.zeros(n_treatments, dtype=float)

    for treatment, idx in treatment_to_idx.items():
        mask = obs_idx == idx
        if mask.any():
            br[idx] = float(y_rev[mask].mean())
            bc[idx] = float(y_cost[mask].mean())

    base_idx = treatment_to_idx[base_treatment]
    nonbase_idx = [idx for idx in range(n_treatments) if idx != base_idx]
    train_budgets = np.asarray(budget_grid, dtype=float)
    lambdas = np.full(len(train_budgets), 0.1, dtype=float)

    for _ in range(DFCL_EPOCHS):
        mu_rev = X @ wr.T + br
        mu_cost = X @ wc.T + bc

        grad_mu_rev = np.zeros_like(mu_rev)
        grad_mu_cost = np.zeros_like(mu_cost)

        factual_rev_error = mu_rev[np.arange(n_samples), obs_idx] - y_rev
        factual_cost_error = mu_cost[np.arange(n_samples), obs_idx] - y_cost
        grad_mu_rev[np.arange(n_samples), obs_idx] += 2.0 * factual_rev_error / max(n_samples, 1)
        grad_mu_cost[np.arange(n_samples), obs_idx] += 2.0 * factual_cost_error / max(n_samples, 1)

        delta_rev = mu_rev[:, nonbase_idx] - mu_rev[:, [base_idx]]
        raw_delta_cost = mu_cost[:, nonbase_idx] - mu_cost[:, [base_idx]]
        values, positive_costs_with_base, feasible = prepare_action_tensors(
            delta_rev=delta_rev,
            raw_delta_cost=raw_delta_cost,
            smooth_cost=True,
            cost_temperature=DFCL_COST_TEMPERATURE,
        )
        positive_cost = positive_costs_with_base[:, 1:]
        positive_cost_grad = _sigmoid(raw_delta_cost / max(DFCL_COST_TEMPERATURE, 1e-6))

        for budget_idx, budget in enumerate(train_budgets):
            lambda_dual = lambdas[budget_idx]
            gain = gain_matrix(values, positive_costs_with_base, feasible, lambda_dual=lambda_dual)
            probs = soft_action_probabilities(gain, tau=DFCL_TAU)
            probs_nonbase = probs[:, 1:]
            gain_nonbase = gain[:, 1:]
            expected_gain = (probs_nonbase * gain_nonbase).sum(axis=1, keepdims=True)
            expected_spend = (probs_nonbase * positive_cost).sum(axis=1)
            total_spend = float(expected_spend.sum())
            budget_gap = (total_spend - budget) / max(n_samples, 1)
            lambdas[budget_idx] = max(0.0, lambda_dual + DFCL_DUAL_LR * budget_gap)

            grad_gain = -(
                DFCL_DECISION_WEIGHT / max(len(train_budgets), 1) / max(n_samples, 1)
            ) * probs_nonbase * (1.0 + (gain_nonbase - expected_gain) / max(DFCL_TAU, 1e-6))

            grad_mu_rev[:, nonbase_idx] += grad_gain
            grad_mu_rev[:, [base_idx]] -= grad_gain.sum(axis=1, keepdims=True)

            grad_raw_cost = grad_gain * (-lambda_dual) * positive_cost_grad
            grad_mu_cost[:, nonbase_idx] += grad_raw_cost
            grad_mu_cost[:, [base_idx]] -= grad_raw_cost.sum(axis=1, keepdims=True)

        grad_wr = grad_mu_rev.T @ X + 2.0 * DFCL_L2 * wr
        grad_wc = grad_mu_cost.T @ X + 2.0 * DFCL_L2 * wc
        grad_br = grad_mu_rev.sum(axis=0)
        grad_bc = grad_mu_cost.sum(axis=0)

        wr -= DFCL_LR * grad_wr
        wc -= DFCL_LR * grad_wc
        br -= DFCL_LR * grad_br
        bc -= DFCL_LR * grad_bc

    final_mu_rev = X @ wr.T + br
    final_mu_cost = X @ wc.T + bc
    dual_lambdas = {float(budget): float(lambdas[idx]) for idx, budget in enumerate(train_budgets)}
    return final_mu_rev, final_mu_cost, dual_lambdas


def _estimate_dfcl(
    df: pd.DataFrame,
    feature_columns: List[str],
    base_treatment: str,
    budget_grid: Sequence[float],
) -> DRResult:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for dfcl mode because the propensity model uses multinomial logistic regression")

    propensity_pipe = _fit_propensity_model(df, feature_columns)
    treatments = propensity_pipe.named_steps["model"].classes_.tolist()
    if base_treatment not in treatments:
        raise ValueError(f"Base treatment {base_treatment} not found after propensity fitting")

    propensity_proba = propensity_pipe.predict_proba(df[feature_columns])
    propensity_df = pd.DataFrame(propensity_proba, columns=[f"ps_{c}" for c in treatments], index=df.index)
    mu_rev_matrix, mu_cost_matrix, dual_lambdas = _fit_dfcl_linear_heads(
        df,
        feature_columns,
        treatments,
        base_treatment,
        budget_grid,
    )

    treatment_to_idx = {name: idx for idx, name in enumerate(treatments)}
    propensity_map = {treatment: propensity_df[f"ps_{treatment}"].to_numpy(dtype=float) for treatment in treatments}
    mu_rev_map = {treatment: mu_rev_matrix[:, treatment_to_idx[treatment]] for treatment in treatments}
    mu_cost_map = {treatment: mu_cost_matrix[:, treatment_to_idx[treatment]] for treatment in treatments}

    result = _build_scoring_table(
        df=df,
        treatments=treatments,
        propensity_map=propensity_map,
        mu_rev_map=mu_rev_map,
        mu_cost_map=mu_cost_map,
        base_treatment=base_treatment,
        model_mode="dfcl",
        dual_lambdas=dual_lambdas,
    )
    result.feature_columns = feature_columns
    return result


def estimate_dr_scores(
    df: pd.DataFrame,
    feature_columns: List[str],
    base_treatment: str = BASE_TREATMENT,
    model_mode: str = "auto",
    budget_grid: Sequence[float] | None = None,
) -> DRResult:
    df = filter_treatments(df)
    if base_treatment not in set(df["treatment"]):
        raise ValueError(f"Base treatment {base_treatment} not found in dataset")

    if model_mode not in {"auto", "dfcl", "sklearn", "empirical"}:
        raise ValueError("model_mode must be one of: auto, dfcl, sklearn, empirical")

    resolved_budgets = _resolve_budget_grid(budget_grid)

    if model_mode == "empirical":
        return _estimate_empirical(df, base_treatment)
    if model_mode == "sklearn":
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is unavailable in the current environment")
        return _estimate_sklearn(df, feature_columns, base_treatment)
    if model_mode == "dfcl":
        return _estimate_dfcl(df, feature_columns, base_treatment, resolved_budgets)
    if SKLEARN_AVAILABLE:
        return _estimate_dfcl(df, feature_columns, base_treatment, resolved_budgets)
    return _estimate_empirical(df, base_treatment)
