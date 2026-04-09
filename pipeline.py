from __future__ import annotations

import pandas as pd

from .data_io import load_raw_data
from .features import (
    add_coupon_history_features,
    add_order_features,
    add_outcomes,
    add_profile_features,
    add_visit_features,
    assign_treatment,
    build_main_coupon_action,
    build_user_bu_day_panel,
    finalize_feature_table,
    infer_treatment_bins,
    merge_coupon_actions,
)
from .preprocess import add_profile_derived_features, clean_all


def build_model_dataset(sample_nrows: int | None = None) -> pd.DataFrame:
    frames = load_raw_data(sample_nrows=sample_nrows)
    frames = clean_all(frames)

    profile = add_profile_derived_features(frames["profile"])
    visits = frames["visit"]
    coupon = frames["coupon"]
    order = frames["order"]

    panel = build_user_bu_day_panel(order, coupon, visits)
    coupon_actions = build_main_coupon_action(coupon)
    panel = merge_coupon_actions(panel, coupon_actions)

    treatment_rules = infer_treatment_bins(coupon_actions)
    panel["treatment"] = panel.apply(assign_treatment, axis=1, bin_rules=treatment_rules)

    panel = add_visit_features(panel, visits)
    panel = add_order_features(panel, order)
    panel = add_coupon_history_features(panel, coupon)
    panel = add_profile_features(panel, profile)
    panel = add_outcomes(panel, order, visits)
    panel = finalize_feature_table(panel)
    return panel
