"""NHATS SP interview phenotype inputs from the original Stata files."""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from src.nhats import config

NEG = {-1, -7, -8, -9}


def clean(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.mask(s.isin(NEG) | (s < 0))


def col(df: pd.DataFrame, name: str) -> pd.Series:
    return (
        pd.to_numeric(df[name], errors="coerce")
        if name in df.columns
        else pd.Series(np.nan, index=df.index)
    )


def sp_file(r: int) -> str:
    return sorted(
        glob.glob(str(config.RAW_NHATS / f"NHATS_R{r}_*" / f"NHATS_Round_{r}_SP_File*.dta"))
    )[0]


def extract_round(r: int, nursing_home: bool = False) -> pd.DataFrame:
    path = sp_file(r)
    have = pd.read_stata(path, iterator=True).variable_labels()
    want = [
        "spid",
        f"r{r}dresid",
        f"is{r}resptype",
        f"ss{r}lowenergy",
        f"ss{r}loenlmtat",
        f"pa{r}evrgowalk",
        f"pa{r}vigoractv",
        f"hw{r}currweigh",
        f"hw{r}howtallft",
        f"hw{r}howtallin",
        f"hw{r}lst10pnds",
        f"hw{r}trytolose",
        f"wa{r}dwlkadm",
        f"wa{r}wlkc1rslt",
        f"wa{r}wlkc1secs",
        f"wa{r}wlk1hndr",
        *[f"wa{r}rsn{k}1wkc" for k in (1, 2, 3)],
        f"gr{r}dgripadm",
        f"gr{r}grp1reslt",
        f"gr{r}grp1rdng",
        f"gr{r}grp2reslt",
        f"gr{r}grp2rdng",
        *[f"gr{r}grp{t}noat{k}" for t in (1, 2) for k in (1, 2, 3)],
    ]
    df = pd.read_stata(path, columns=[c for c in want if c in have], convert_categoricals=False)
    residence = col(df, f"r{r}dresid")
    if nursing_home:
        eligible = residence.eq(4) & col(df, f"is{r}resptype").isin([1, 2])
    else:
        eligible = residence.isin([1, 2])
    df = df[eligible].copy()
    out = pd.DataFrame({"spid": df["spid"].astype("int64").to_numpy(), "round": r})
    idx = df.index
    out.index = idx
    out["resid"] = col(df, f"r{r}dresid")
    out["proxy"] = (col(df, f"is{r}resptype") == 2).astype(float)

    # Exhaustion: low energy that limited activities.
    le, lim = col(df, f"ss{r}lowenergy"), col(df, f"ss{r}loenlmtat")
    exh = pd.Series(np.nan, index=idx)
    exh[(le == 1) & (lim == 1)] = 1.0
    exh[(le == 2) | ((le == 1) & (lim == 2))] = 0.0
    out["exhaustion"] = exh

    # Low physical activity: never walks for exercise and no vigorous activity.
    wk, vg = col(df, f"pa{r}evrgowalk"), col(df, f"pa{r}vigoractv")
    pa = pd.Series(np.nan, index=idx)
    pa[(wk == 2) & (vg == 2)] = 1.0
    pa[(wk == 1) | (vg == 1)] = 0.0
    out["low_pa"] = pa

    # Unintentional loss of >= 10 lb in the last year. BMI < 18.5 is added after imputation.
    lost, trying = col(df, f"hw{r}lst10pnds"), col(df, f"hw{r}trytolose")
    uwl = pd.Series(np.nan, index=idx)
    uwl[(lost == 1) & (trying == 2)] = 1.0
    uwl[(lost == 2) | ((lost == 1) & (trying == 1))] = 0.0
    out["unint_wl"] = uwl

    ft = clean(col(df, f"hw{r}howtallft")).where(lambda s: s.between(3, 7))
    inch = clean(col(df, f"hw{r}howtallin")).where(lambda s: s.between(0, 11))
    h_in = (ft * 12 + inch).where(lambda s: s.between(45, 90))
    out["height_cm_raw"] = h_in * 2.54
    out["weight_lb"] = clean(col(df, f"hw{r}currweigh")).where(lambda s: s.between(50, 700))

    # Walking: first usual-pace trial, 3 m course.
    adm = col(df, f"wa{r}dwlkadm")
    res1 = col(df, f"wa{r}wlkc1rslt")
    secs = clean(col(df, f"wa{r}wlkc1secs")).where(lambda s: s.between(0, 120))
    hndr = clean(col(df, f"wa{r}wlk1hndr")).where(lambda s: s.between(0, 99)).fillna(0)
    t1 = (secs + hndr / 100).where(lambda s: s > 0)
    speed = (config.WALK_COURSE_M / t1).where(res1 == 1)
    unsafe = pd.concat([col(df, f"wa{r}rsn{k}1wkc") == 1 for k in (1, 2, 3)], axis=1).any(axis=1)
    walk_hnc = speed.isna() & ((adm == 4) | (res1 == 2) | ((res1 == 3) & unsafe))
    out["walk_health_nc"] = walk_hnc.astype(float)
    out["speed"] = speed.where(~walk_hnc, 0.0)
    out["walk_adm"] = adm

    # Grip: maximum of completed trials.
    gadm = col(df, f"gr{r}dgripadm")
    g1 = (
        clean(col(df, f"gr{r}grp1rdng"))
        .where(lambda s: s.between(0.1, 120))
        .where(col(df, f"gr{r}grp1reslt") == 1)
    )
    g2 = (
        clean(col(df, f"gr{r}grp2rdng"))
        .where(lambda s: s.between(0.1, 120))
        .where(col(df, f"gr{r}grp2reslt") == 1)
    )
    grip = pd.concat([g1, g2], axis=1).max(axis=1, skipna=True)
    attempted = (col(df, f"gr{r}grp1reslt") == 2) | (col(df, f"gr{r}grp2reslt") == 2)
    gunsafe = pd.concat(
        [col(df, f"gr{r}grp{t}noat{k}") == 1 for t in (1, 2) for k in (1, 2, 3)], axis=1
    ).any(axis=1)
    grip_hnc = grip.isna() & ((gadm == 4) | attempted | gunsafe)
    out["grip_health_nc"] = grip_hnc.astype(float)
    out["grip"] = grip.where(~grip_hnc, 0.0)
    out["grip_adm"] = gadm
    if r in config.NO_PERFORMANCE_ROUNDS:
        out[["speed", "grip"]] = np.nan
        out[["walk_health_nc", "grip_health_nc"]] = 0.0
    return out.reset_index(drop=True)


def carry_height(panel: pd.DataFrame) -> pd.Series:
    """Use the nearest earlier height, then the nearest later height for the same participant."""
    h = panel.sort_values(["spid", "round"]).groupby("spid")["height_cm_raw"]
    return h.ffill().combine_first(h.bfill()).reindex(panel.sort_values(["spid", "round"]).index)
