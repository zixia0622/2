from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_member_from_rar(archive: Path, member_name: str, output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    if output_path.exists():
        return output_path
    cmd = ["bsdtar", "-xOf", str(archive), member_name]
    with output_path.open("wb") as f:
        subprocess.run(cmd, check=True, stdout=f)
    return output_path


def rarity_bucket(counts: Iterable[int]) -> str:
    # Reserved helper for future extensions.
    _ = counts
    return ""
