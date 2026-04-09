from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .config import (
    ARCHIVE_PREFIX,
    COUPON_FILE,
    DATA_ARCHIVE,
    ORDER_FILE,
    PROFILE_FILE,
    RAW_DIR,
    VISIT_FILE,
)
from .utils import ensure_dir, extract_member_from_rar


FILE_MAP: Dict[str, str] = {
    "profile": PROFILE_FILE,
    "visit": VISIT_FILE,
    "coupon": COUPON_FILE,
    "order": ORDER_FILE,
}


def _archive_member(file_name: str) -> str:
    return f"{ARCHIVE_PREFIX}/{file_name}"


def ensure_raw_data() -> Dict[str, Path]:
    ensure_dir(RAW_DIR)
    paths: Dict[str, Path] = {}
    for key, file_name in FILE_MAP.items():
        out_path = RAW_DIR / file_name
        paths[key] = extract_member_from_rar(DATA_ARCHIVE, _archive_member(file_name), out_path)
    return paths


def load_raw_data(sample_nrows: int | None = None) -> Dict[str, pd.DataFrame]:
    paths = ensure_raw_data()
    frames = {}
    for key, path in paths.items():
        frames[key] = pd.read_csv(path, nrows=sample_nrows, low_memory=False)
    return frames
