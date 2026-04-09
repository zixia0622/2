from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_WINDOWS, MAX_TREATMENTS_PER_BU, OUTCOME_WINDOW_DAYS


def build_main_coupon_action(coupon: pd.DataFrame) -> pd.DataFrame:
    df = coupon.copy()
    df = df.rename(columns={"Coupon_bu": "BU", "Receive_date": "Decision_date"})
    df = df.sort_values(["User_id", "BU", "Decision_date", "Coupon_amt", "Price_limit"], ascending=[True, True, True, False, False])
    daily = df.groupby(["User_id", "BU", "Decision_date"], as_index=False).first()
    daily["no_coupon"] = 0
    return daily


def infer_treatment_bins(coupon_actions: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for bu, g in coupon_actions.groupby("BU"):
        amt = g["Coupon_amt"].dropna()
        limit = g["Price_limit"].dropna()
        result[bu] = {
            "amt_p33": float(amt.quantile(0.33)) if not amt.empty else 0.0,
            "amt_p66": float(amt.quantile(0.66)) if not amt.empty else 0.0,
            "limit_p50": float(limit.quantile(0.5)) if not limit.empty else 0.0,
        }
    return result


def assign_treatment(row: pd.Series, bin_rules: dict[str, dict[str, float]]) -> str:
    if int(row.get("no_coupon", 0)) == 1:
        return "T0_no_coupon"
    bu = row["BU"]
    rules = bin_rules.get(bu, {"amt_p33": 0.0, "amt_p66": 0.0, "limit_p50": 0.0})
    coupon_amt = row.get("Coupon_amt")
    price_limit = row.get("Price_limit")
    coupon_type = str(row.get("Coupon_type", "NULL"))

    if pd.isna(coupon_amt):
        return "T4_sparse_or_unknown"
    if coupon_amt == 0:
        return "T4_discount_or_zero_amt"
    if coupon_amt <= rules["amt_p33"]:
        return "T1_low_coupon"
    if coupon_amt <= rules["amt_p66"]:
        return "T2_mid_coupon"
    if price_limit is not None and not pd.isna(price_limit) and price_limit > rules["limit_p50"] and coupon_type != "NULL":
        return "T3_high_threshold_coupon"
    return "T3_high_coupon"


def build_user_bu_day_panel(order: pd.DataFrame, coupon: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    order_pairs = order[["User_id", "BU_name", "Pay_date"]].rename(columns={"BU_name": "BU", "Pay_date": "Decision_date"})
    coupon_pairs = coupon[["User_id", "Coupon_bu", "Receive_date"]].rename(columns={"Coupon_bu": "BU", "Receive_date": "Decision_date"})
    order_pairs = order_pairs.dropna(subset=["BU", "Decision_date"])
    coupon_pairs = coupon_pairs.dropna(subset=["BU", "Decision_date"])
    panel = pd.concat([order_pairs, coupon_pairs], ignore_index=True).drop_duplicates()

    active_users = visits[["User_id"]].drop_duplicates()
    user_bu = pd.concat([
        order[["User_id", "BU_name"]].rename(columns={"BU_name": "BU"}),
        coupon[["User_id", "Coupon_bu"]].rename(columns={"Coupon_bu": "BU"}),
    ], ignore_index=True).dropna().drop_duplicates()
    fallback_dates = visits.rename(columns={"Visit_date": "Decision_date"}).merge(user_bu, on="User_id", how="inner")
    fallback_dates = fallback_dates[["User_id", "BU", "Decision_date"]]
    panel = pd.concat([panel, fallback_dates], ignore_index=True).drop_duplicates()
    panel = panel.sort_values(["User_id", "BU", "Decision_date"]).reset_index(drop=True)
    return panel


def merge_coupon_actions(panel: pd.DataFrame, coupon_actions: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    merged = panel.merge(
        coupon_actions,
        on=["User_id", "BU", "Decision_date"],
        how="left",
        suffixes=("", "_coupon"),
    )
    merged["no_coupon"] = merged["Coupon_id"].isna().astype(int)
    return merged


def add_visit_features(panel: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    visit_daily = visits.groupby(["User_id", "Visit_date"]).size().rename("visit_cnt").reset_index()
    for window in FEATURE_WINDOWS:
        feat_values = []
        for user_id, g in panel.groupby("User_id"):
            user_visits = visit_daily.loc[visit_daily["User_id"] == user_id, ["Visit_date", "visit_cnt"]].sort_values("Visit_date")
            dates = g["Decision_date"]
            counts = []
            for dt in dates:
                mask = (user_visits["Visit_date"] < dt) & (user_visits["Visit_date"] >= dt - pd.Timedelta(days=window))
                counts.append(float(user_visits.loc[mask, "visit_cnt"].sum()))
            feat_values.extend(counts)
        panel[f"visit_cnt_{window}d"] = feat_values
    return panel


def add_order_features(panel: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    order = order.rename(columns={"BU_name": "BU"})
    for window in FEATURE_WINDOWS:
        order_cnts = []
        pay_sums = []
        subsidy_sums = []
        for (user_id, bu), g in panel.groupby(["User_id", "BU"]):
            user_orders = order.loc[(order["User_id"] == user_id) & (order["BU"] == bu), ["Pay_date", "Actual_pay", "Reduce_amount"]].sort_values("Pay_date")
            for dt in g["Decision_date"]:
                mask = (user_orders["Pay_date"] < dt) & (user_orders["Pay_date"] >= dt - pd.Timedelta(days=window))
                hist = user_orders.loc[mask]
                order_cnts.append(float(len(hist)))
                pay_sums.append(float(hist["Actual_pay"].sum()))
                subsidy_sums.append(float(hist["Reduce_amount"].sum()))
        panel[f"order_cnt_{window}d"] = order_cnts
        panel[f"pay_sum_{window}d"] = pay_sums
        panel[f"subsidy_sum_{window}d"] = subsidy_sums
    panel["avg_pay_30d"] = panel["pay_sum_30d"] / panel["order_cnt_30d"].replace(0, np.nan)
    panel["avg_pay_30d"] = panel["avg_pay_30d"].fillna(0.0)
    panel["subsidy_ratio_30d"] = panel["subsidy_sum_30d"] / (panel["pay_sum_30d"] + panel["subsidy_sum_30d"] + 1e-6)
    return panel


def add_coupon_history_features(panel: pd.DataFrame, coupon: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    coupon = coupon.rename(columns={"Coupon_bu": "BU", "Receive_date": "Decision_date_raw"})
    coupon["used_flag"] = (coupon["Coupon_status"] == "2").astype(int)
    coupon["expired_flag"] = (coupon["Coupon_status"] == "1").astype(int)
    for window in FEATURE_WINDOWS:
        receive_cnts = []
        use_cnts = []
        avg_amt = []
        avg_limit = []
        for (user_id, bu), g in panel.groupby(["User_id", "BU"]):
            hist_coupon = coupon.loc[(coupon["User_id"] == user_id) & (coupon["BU"] == bu)].sort_values("Decision_date_raw")
            for dt in g["Decision_date"]:
                mask = (hist_coupon["Decision_date_raw"] < dt) & (hist_coupon["Decision_date_raw"] >= dt - pd.Timedelta(days=window))
                hist = hist_coupon.loc[mask]
                receive_cnts.append(float(len(hist)))
                use_cnts.append(float(hist["used_flag"].sum()))
                avg_amt.append(float(hist["Coupon_amt"].dropna().mean()) if not hist.empty else 0.0)
                avg_limit.append(float(hist["Price_limit"].dropna().mean()) if not hist.empty else 0.0)
        panel[f"coupon_receive_cnt_{window}d"] = receive_cnts
        panel[f"coupon_use_cnt_{window}d"] = use_cnts
        panel[f"avg_coupon_amt_{window}d"] = avg_amt
        panel[f"avg_price_limit_{window}d"] = avg_limit
    panel["coupon_redeem_rate_30d"] = panel["coupon_use_cnt_30d"] / panel["coupon_receive_cnt_30d"].replace(0, np.nan)
    panel["coupon_redeem_rate_30d"] = panel["coupon_redeem_rate_30d"].fillna(0.0)
    panel["coupon_receive_use_gap_30d"] = panel["coupon_receive_cnt_30d"] - panel["coupon_use_cnt_30d"]
    panel["high_freq_low_redeem_flag"] = ((panel["coupon_receive_cnt_30d"] >= 3) & (panel["coupon_redeem_rate_30d"] < 0.3)).astype(int)
    return panel


def add_profile_features(panel: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    merged = panel.merge(profile, on="User_id", how="left")
    bu_list = ["A", "B", "C", "D", "E"]
    for bu in bu_list:
        last_col = f"{bu}_last_date"
        if last_col in merged.columns:
            merged[f"days_since_last_{bu}"] = (merged["Decision_date"] - merged[last_col]).dt.days
            merged[f"days_since_last_{bu}"] = merged[f"days_since_last_{bu}"].fillna(9999)
    return merged


def add_outcomes(panel: pd.DataFrame, order: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    order = order.rename(columns={"BU_name": "BU"})
    future_order_cnt = []
    future_pay_sum = []
    future_subsidy_sum = []
    future_visit_cnt = []
    for _, row in panel[["User_id", "BU", "Decision_date"]].iterrows():
        end_date = row["Decision_date"] + pd.Timedelta(days=OUTCOME_WINDOW_DAYS)
        order_mask = (
            (order["User_id"] == row["User_id"]) &
            (order["BU"] == row["BU"]) &
            (order["Pay_date"] >= row["Decision_date"]) &
            (order["Pay_date"] < end_date)
        )
        visit_mask = (
            (visits["User_id"] == row["User_id"]) &
            (visits["Visit_date"] >= row["Decision_date"]) &
            (visits["Visit_date"] < end_date)
        )
        future_orders = order.loc[order_mask]
        future_visits = visits.loc[visit_mask]
        future_order_cnt.append(float(len(future_orders)))
        future_pay_sum.append(float(future_orders["Actual_pay"].sum()))
        future_subsidy_sum.append(float(future_orders["Reduce_amount"].sum()))
        future_visit_cnt.append(float(future_visits.shape[0]))
    panel["Y_cnt_7d"] = future_order_cnt
    panel["Y_rev_7d"] = future_pay_sum
    panel["Y_cost_7d"] = future_subsidy_sum
    panel["Y_visit_7d"] = future_visit_cnt
    panel["Y_order_7d"] = (panel["Y_cnt_7d"] > 0).astype(int)
    panel["coupon_only_buyer_flag"] = ((panel["order_cnt_30d"] > 0) & (panel["subsidy_ratio_30d"] > 0.7)).astype(int)
    return panel


def finalize_feature_table(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0.0)
    cat_cols = [c for c in ["BU", "treatment"] if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].fillna("UNKNOWN")
    return df
