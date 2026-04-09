from __future__ import annotations

import numpy as np
import pandas as pd


def clean_profile(profile: pd.DataFrame) -> pd.DataFrame:
    df = profile.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_visit(visit: pd.DataFrame) -> pd.DataFrame:
    df = visit.copy()
    df["Visit_date"] = pd.to_datetime(df["Visit_date"])
    return df.drop_duplicates()


def clean_coupon(coupon: pd.DataFrame) -> pd.DataFrame:
    df = coupon.copy()
    for col in ["Receive_date", "Start_date", "End_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["Coupon_amt"] = pd.to_numeric(df["Coupon_amt"], errors="coerce")
    df["Price_limit"] = pd.to_numeric(df["Price_limit"], errors="coerce")
    df["Coupon_status"] = df["Coupon_status"].astype(str)
    df["Coupon_bu"] = df["Coupon_bu"].astype(str)
    return df.drop_duplicates()


def clean_order(order: pd.DataFrame) -> pd.DataFrame:
    df = order.copy()
    df["Pay_date"] = pd.to_datetime(df["Pay_date"], errors="coerce")
    df["Actual_pay"] = pd.to_numeric(df["Actual_pay"], errors="coerce").fillna(0.0)
    df["Reduce_amount"] = pd.to_numeric(df["Reduce_amount"], errors="coerce").fillna(0.0)
    df["BU_name"] = df["BU_name"].astype(str)
    return df.drop_duplicates(subset=["Order_id"])


def clean_all(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "profile": clean_profile(frames["profile"]),
        "visit": clean_visit(frames["visit"]),
        "coupon": clean_coupon(frames["coupon"]),
        "order": clean_order(frames["order"]),
    }


def add_profile_derived_features(profile: pd.DataFrame) -> pd.DataFrame:
    df = profile.copy()
    bu_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    level_map = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}

    for bu in bu_map:
        last_date_col = f"{bu}_last_date"
        cnt_col = f"{bu}_cnt_level"
        pay_col = f"{bu}_pay_level"
        if last_date_col in df.columns:
            df[last_date_col] = pd.to_datetime(df[last_date_col], errors="coerce")
        if cnt_col in df.columns:
            df[f"{bu}_cnt_level_num"] = df[cnt_col].map(level_map).fillna(0)
        if pay_col in df.columns:
            df[f"{bu}_pay_level_num"] = df[pay_col].map(level_map).fillna(0)

    city_map = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    df["city_rank_num"] = df["city_rank"].map(city_map).fillna(-1)
    return df
