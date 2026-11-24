"""Round-specific predictive mean matching and Fried scoring with the Bandeen-Roche cutpoints."""

from __future__ import annotations


import numpy as np
import pandas as pd
from statsmodels.imputation.mice import MICEData

from src.nhats import config

IMPUTE = [
    "speed",
    "grip",
    "height_cm",
    "bmi",
    "exhaustion",
    "low_pa",
    "unint_wl",
    "education_c",
    "medicaid",
    "chronic9_count",
    "self_rated_health",
    "hospital_stay",
    "fall_last_year",
]
PREDICT = [
    "age_cat_final",
    "female",
    "race_black",
    "race_hisp",
    "race_other",
    "rc",
    "proxy",
    "walk_health_nc",
    "grip_health_nc",
    "married_partnered",
    "current_smoker",
]


def design(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    race = pd.to_numeric(df["race_ethnicity"], errors="coerce")
    x["female"] = (df["gender_final"] == 2).astype(float)
    x["race_black"] = (race == 2).astype(float)
    x["race_hisp"] = (race == 4).astype(float)
    x["race_other"] = (~race.isin([1, 2, 4])).astype(float)
    x["rc"] = (df["resid"] == 2).astype(float)
    for c in [
        "age_cat_final",
        "proxy",
        "walk_health_nc",
        "grip_health_nc",
        "married_partnered",
        "current_smoker",
    ]:
        x[c] = pd.to_numeric(df[c], errors="coerce")
    for c in IMPUTE:
        x[c] = pd.to_numeric(df[c], errors="coerce")
    return x


def score(df: pd.DataFrame) -> pd.DataFrame:
    female = df["gender_final"] == 2
    h = df["height_cm"]
    short = np.where(female, h <= config.HEIGHT_SPLIT_CM[2], h <= config.HEIGHT_SPLIT_CM[1])
    cut_s = np.select(
        [female & short, female & ~short, ~female & short, ~female & ~short],
        [
            config.SLOW_CUTS[(2, "short")],
            config.SLOW_CUTS[(2, "tall")],
            config.SLOW_CUTS[(1, "short")],
            config.SLOW_CUTS[(1, "tall")],
        ],
    )
    b = df["bmi"]
    bcat = np.select([b < 18.5, b < 25, b < 30], ["under", "normal", "over"], "obese")
    cut_g = np.array([config.GRIP_CUTS[(2 if f else 1, c)] for f, c in zip(female, bcat)])
    out = pd.DataFrame(index=df.index)
    out["slow"] = (df["speed"] <= cut_s).astype(float)
    out["weak"] = (df["grip"] <= cut_g).astype(float)
    out["exhaustion"] = df["exhaustion"]
    out["low_pa"] = df["low_pa"]
    out["weight_loss"] = ((df["unint_wl"] == 1) | (b < 18.5)).astype(float)
    out["score"] = out[config.COMPONENTS].sum(axis=1)
    out["bmi_category"] = np.select(
        [b < 18.5, b < 25, b < 30], ["underweight", "normal", "overweight"], "obese"
    )
    return out


def impute_round(df: pd.DataFrame, m: int, r: int) -> pd.DataFrame:
    x = design(df)
    cols = [c for c in PREDICT + IMPUTE if x[c].notna().any()]
    x = x[cols].reset_index(drop=True)
    np.random.seed(config.SEED + 1000 * m + r)
    imp = MICEData(x, perturbation_method="gaussian", k_pmm=10)
    imp.update_all(config.MICE_ITER)
    done = imp.data.copy()
    done.index = df.index
    return done
