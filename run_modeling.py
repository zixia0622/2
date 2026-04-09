from __future__ import annotations

import argparse

from .config import BUDGET_GRID, DFCL_COST_TEMPERATURE, DFCL_TAU, OUTPUT_DIR, PROCESSED_DIR
from .modeling import estimate_dr_scores, get_feature_columns
from .optimization import summarize_allocations
from .pipeline import build_model_dataset
from .results import build_budget_action_mix, build_bu_summary, build_treatment_summary
from .utils import ensure_dir


def _parse_budget_grid(budget_arg: str | None) -> tuple[float, ...]:
    if not budget_arg:
        return tuple(float(v) for v in BUDGET_GRID)
    return tuple(float(part.strip()) for part in budget_arg.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run subsidy uplift + optimization modeling pipeline")
    parser.add_argument("--sample-nrows", type=int, default=None, help="Optional row limit for each raw table for quick experiments")
    parser.add_argument("--output-prefix", type=str, default="baseline", help="Prefix for generated output files")
    parser.add_argument(
        "--model-mode",
        type=str,
        default="auto",
        choices=["auto", "dfcl", "sklearn", "empirical"],
        help="Estimator mode for DR scoring",
    )
    parser.add_argument(
        "--budget-grid",
        type=str,
        default=None,
        help="Comma-separated budgets for both training and allocation summaries, e.g. 5,10,20,40",
    )
    args = parser.parse_args()

    ensure_dir(PROCESSED_DIR)
    ensure_dir(OUTPUT_DIR)

    budget_grid = _parse_budget_grid(args.budget_grid)
    dataset = build_model_dataset(sample_nrows=args.sample_nrows)
    feature_columns = get_feature_columns(dataset)
    dr_result = estimate_dr_scores(dataset, feature_columns, model_mode=args.model_mode, budget_grid=budget_grid)

    hard_summary = summarize_allocations(
        dr_result.scoring_table,
        budget_grid,
        solver_mode="hard",
        lambda_init_map=dr_result.dual_lambdas,
    )
    soft_summary = summarize_allocations(
        dr_result.scoring_table,
        budget_grid,
        solver_mode="soft",
        tau=DFCL_TAU,
        cost_temperature=DFCL_COST_TEMPERATURE,
        lambda_init_map=dr_result.dual_lambdas,
    )
    treatment_summary = build_treatment_summary(dr_result.scoring_table)
    hard_action_mix = build_budget_action_mix(
        dr_result.scoring_table,
        budget_grid,
        solver_mode="hard",
        lambda_init_map=dr_result.dual_lambdas,
    )
    soft_action_mix = build_budget_action_mix(
        dr_result.scoring_table,
        budget_grid,
        solver_mode="soft",
        tau=DFCL_TAU,
        cost_temperature=DFCL_COST_TEMPERATURE,
        lambda_init_map=dr_result.dual_lambdas,
    )
    bu_summary = build_bu_summary(dataset)

    dataset_path = PROCESSED_DIR / f"{args.output_prefix}_model_dataset.csv"
    scoring_path = OUTPUT_DIR / f"{args.output_prefix}_dr_scoring.csv"
    hard_allocation_path = OUTPUT_DIR / f"{args.output_prefix}_allocation_summary.csv"
    soft_allocation_path = OUTPUT_DIR / f"{args.output_prefix}_allocation_summary_soft.csv"
    treatment_path = OUTPUT_DIR / f"{args.output_prefix}_treatment_summary.csv"
    hard_action_mix_path = OUTPUT_DIR / f"{args.output_prefix}_budget_action_mix.csv"
    soft_action_mix_path = OUTPUT_DIR / f"{args.output_prefix}_budget_action_mix_soft.csv"
    bu_summary_path = OUTPUT_DIR / f"{args.output_prefix}_bu_summary.csv"

    dataset.to_csv(dataset_path, index=False)
    dr_result.scoring_table.to_csv(scoring_path, index=False)
    hard_summary.to_csv(hard_allocation_path, index=False)
    soft_summary.to_csv(soft_allocation_path, index=False)
    treatment_summary.to_csv(treatment_path, index=False)
    hard_action_mix.to_csv(hard_action_mix_path, index=False)
    soft_action_mix.to_csv(soft_action_mix_path, index=False)
    bu_summary.to_csv(bu_summary_path, index=False)

    print(f"Model mode: {dr_result.model_mode}")
    print(f"Budget grid: {budget_grid}")
    if dr_result.dual_lambdas:
        print(f"Dual lambdas: {dr_result.dual_lambdas}")
    print(f"Saved dataset to: {dataset_path}")
    print(f"Saved DR scoring to: {scoring_path}")
    print(f"Saved hard allocation summary to: {hard_allocation_path}")
    print(f"Saved soft allocation summary to: {soft_allocation_path}")
    print(f"Saved treatment summary to: {treatment_path}")
    print(f"Saved hard budget action mix to: {hard_action_mix_path}")
    print(f"Saved soft budget action mix to: {soft_action_mix_path}")
    print(f"Saved BU summary to: {bu_summary_path}")
    print(hard_summary.to_string(index=False))


if __name__ == "__main__":
    main()
