"""Read demographic, clinical, survey and outcome inputs from NHATS SP/Tracker files."""

from __future__ import annotations

import re
import pandas as pd
import numpy as np
from src.nhats import config


def clean(x):
    return pd.to_numeric(x, errors="coerce").where(lambda v: v >= 0)


def yes_no(x):
    return x.map({1: 1.0, 2: 0.0})


def first_observed(df, cols):
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(clean).bfill(axis=1).iloc[:, 0]


def read_round(path, r):
    with pd.read_stata(path, iterator=True) as src:
        available = list(src.variable_labels())
    race = [c for c in available if re.fullmatch(r"rl\d+d?racehisp", c)]
    education = [c for c in available if re.fullmatch(r"el\d+d?higstschl", c)]
    weights = [f"w{r}anfinwgt0", *[f"w{r}an{year}wgt0" for year in (2011, 2015, 2022)]]
    chronic = [f"hc{r}disescn{j}" for j in (1, 2, 3, 4, 5, 6, 7, 8, 10)]
    vars_ = [
        "spid",
        *race,
        *education,
        *weights,
        *chronic,
        f"r{r}d2intvrage",
        f"r{r}dgender",
        f"w{r}varstrat",
        f"w{r}varunit",
        f"hh{r}dmarstat",
        f"hh{r}martlstat",
        f"sd{r}smokedreg",
        f"sd{r}smokesnow",
        f"hc{r}health",
        f"ip{r}cmedicaid",
        f"hc{r}hosptstay",
        f"hc{r}faleninyr",
        f"hc{r}multifall",
    ]
    df = pd.read_stata(
        path,
        columns=[c for c in dict.fromkeys(vars_) if c in available],
        convert_categoricals=False,
    )

    def col(c):
        return df[c] if c in df else pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({"spid": df.spid.astype("int64"), "round": r})
    out["age_cat_final"] = clean(col(f"r{r}d2intvrage"))
    out["gender_final"] = clean(col(f"r{r}dgender")).where(lambda v: v.isin([1, 2]))
    out["race_ethnicity"] = first_observed(df, race)
    out["education"] = first_observed(df, education)
    weight = pd.concat([clean(col(c)).where(lambda v: v > 0) for c in weights], axis=1)
    out["analytic_weight_final"] = weight.bfill(axis=1).iloc[:, 0].fillna(0.0)
    out["survey_var_stratum"] = clean(col(f"w{r}varstrat"))
    out["survey_var_unit"] = clean(col(f"w{r}varunit"))
    out["self_rated_health"] = clean(col(f"hc{r}health"))
    out["medicaid"] = yes_no(col(f"ip{r}cmedicaid"))
    mar = clean(col(f"hh{r}dmarstat")).combine_first(clean(col(f"hh{r}martlstat")))
    out["married_partnered"] = mar.map({1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0})
    smoke, ever = col(f"sd{r}smokesnow"), col(f"sd{r}smokedreg")
    out["current_smoker"] = np.select(
        [smoke.eq(1), smoke.isin([2, -1]) | ever.eq(2)], [1.0, 0.0], np.nan
    )
    conditions = pd.concat([col(c).map({1: 1.0, 7: 1.0, 2: 0.0}) for c in chronic], axis=1)
    out["chronic9_count"] = conditions.sum(axis=1, min_count=len(chronic))
    out["hospital_stay"] = yes_no(col(f"hc{r}hosptstay"))
    out["fall_last_year"] = yes_no(col(f"hc{r}faleninyr"))
    out["multiple_falls"] = yes_no(col(f"hc{r}multifall")).mask(out.fall_last_year.eq(0), 0.0)
    return out


def read_tracker():
    files = sorted(config.RAW_NHATS.glob("NHATS_R14_*/*Round_14_Tracker_File*.dta"))
    if len(files) != 1:
        raise ValueError(
            f"Expected one NHATS round-14 tracker in {config.RAW_NHATS}; found {len(files)}"
        )
    statuses = [f"r{r}status" for r in config.ROUNDS]
    tracker = pd.read_stata(
        files[0], columns=["spid", *statuses], convert_categoricals=False
    ).set_index("spid")
    dead = tracker.isin([62, 81, 86])
    onset = pd.Series(
        np.where(dead.any(axis=1), dead.to_numpy().argmax(axis=1) + 1, np.nan), index=tracker.index
    )
    return tracker, onset
