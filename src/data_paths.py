from pathlib import Path
import os


def raw_data_dir():
    override = os.environ.get("FRAILTY_RAW_DATA")
    if override:
        return Path(override).expanduser()

    root = Path(__file__).resolve().parents[1]
    for parent in (root, *root.parents):
        candidate = parent / "data"
        if any((candidate / cohort).is_dir() for cohort in ("NHATS", "HRS", "SHARE")):
            return candidate
    return root / "data"
