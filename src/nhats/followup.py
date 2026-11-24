"""Vital-status follow-up and consecutive interview windows in NHATS."""

from __future__ import annotations

import glob

import pandas as pd

from src.nhats import config

ALIVE_STATUS = {60, 61, 63, 64, 75, 76, 77, 78, 85, 87}


def known_alive_rounds() -> dict[int, list[int]]:
    path = sorted(
        glob.glob(str(config.RAW_NHATS / "NHATS_R14_*" / "NHATS_Round_14_Tracker_File*.dta"))
    )[0]
    cols = [f"r{r}status" for r in config.ROUNDS]
    t = pd.read_stata(path, columns=["spid", *cols], convert_categoricals=False)
    out = {}
    for pid, row in zip(t["spid"].astype("int64"), t[cols].to_numpy()):
        out[int(pid)] = [r for r, s in zip(config.ROUNDS, row) if s in ALIVE_STATUS]
    return out


def consecutive(rounds: list[int], index: int) -> list[int]:
    out, nxt = [], index + 1
    for r in rounds:
        if r < nxt:
            continue
        if r != nxt:
            break
        out.append(r)
        nxt += 1
    return out
