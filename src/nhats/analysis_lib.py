"""Discrete-time competing-risk models and standardised cumulative incidence."""

from __future__ import annotations

import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


def prepare(land: pd.DataFrame) -> pd.DataFrame:
    d = land.copy()
    race = pd.to_numeric(d["race_ethnicity"], errors="coerce")
    d["race_collapsed"] = np.select(
        [race.eq(1), race.eq(2), race.eq(4)], ["white", "black", "hispanic"], "other"
    )
    d["index_age_cat"] = d["age_cat_final"].astype(int).astype(str)
    d["gender"] = d["gender_final"].astype(int).astype(str)
    d["education_collapsed"] = d["education_c"].astype(int).astype(str)
    # Index rounds cluster by NHATS sample (2011, 2015 and 2022 entry); single
    # rounds such as 3 or 5 hold a handful of people and destabilise the fits.
    d["index_round_num"] = d["index_round"].astype(int)
    d["index_round"] = np.select(
        [d["index_round_num"] <= 5, d["index_round_num"] <= 11], ["2-5", "6-11"], "12-14"
    )
    d["married_partnered"] = d["married_partnered"].astype(int).astype(str)
    d["current_smoker"] = d["current_smoker"].astype(int).astype(str)
    d["medicaid"] = d["medicaid"].astype(int).astype(str)
    cc = d["chronic9_count"].round().astype(int)
    d["chronic9_cat"] = np.where(cc >= 3, "3plus", cc.astype(str))
    d["bmi_category"] = d["bmi_category"].astype(str)
    ps = d["prior_score"].astype(int)
    d["prior_group"] = np.select([ps.eq(0), ps.eq(1)], ["prior0", "prior1"], "prior2plus")
    d["index_score"] = d["index_score"].astype(int)
    return d


def levels_for(df: pd.DataFrame, covs: list[str]) -> dict[str, list[str]]:
    out = {}
    for c in covs:
        vals = sorted(df[c].astype(str).unique().tolist())
        # most frequent level as reference keeps the design well conditioned
        ref = df[c].astype(str).value_counts().idxmax()
        out[c] = [ref] + [v for v in vals if v != ref]
    return out


def cov_matrix(df: pd.DataFrame, covs: list[str], levels: dict[str, list[str]]) -> np.ndarray:
    cols = []
    for c in covs:
        v = df[c].astype(str).to_numpy()
        for lv in levels[c][1:]:
            cols.append((v == lv).astype(float))
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))


def episodes(df: pd.DataFrame, kind: str, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Last interval in the death model (endD), last interval in the outcome
    model (endX) and event type (0 censored, 1 outcome X, 2 death).

    X follow-up runs over consecutive observed rounds (W_Fc or W_Rc); death
    follow-up runs while the person is known to be alive (W_V). t_frail,
    t_falls and t_hosp are already restricted to the consecutive window.
    """
    tD = df["t_death"].to_numpy(float)
    WV = df["W_V"].to_numpy(float)
    if kind == "fried":
        WX, tX = df["W_Fc"].to_numpy(float), df["t_frail"].to_numpy(float)
    elif kind == "falls":
        WX, tX = df["W_Rc"].to_numpy(float), df["t_falls"].to_numpy(float)
    elif kind == "hosp":
        WX, tX = df["W_Rc"].to_numpy(float), df["t_hosp"].to_numpy(float)
    elif kind == "death":
        WX, tX = np.zeros(len(df)), np.full(len(df), np.nan)
    else:
        raise ValueError(kind)
    x_in = ~np.isnan(tX) & (tX <= H)
    d_in = ~np.isnan(tD) & (tD <= WV + 1) & (tD <= H) & ~x_in
    ev = np.zeros(len(df), dtype=int)
    ev[x_in] = 1
    ev[d_in] = 2
    capD = np.minimum(WV, H)
    capX = np.minimum(np.minimum(WX, WV), H)
    endD = np.where(ev == 1, tX, np.where(ev == 2, tD, capD))
    endX = np.where(ev == 1, tX, np.where(ev == 2, np.minimum(tD - 1, capX), capX))
    return endD.astype(int), endX.astype(int), ev


def rows_until(end: np.ndarray, event_at_end: np.ndarray):
    reps = np.clip(end, 0, None)
    pid = np.repeat(np.arange(len(end)), reps)
    start = np.repeat(np.cumsum(reps) - reps, reps)
    j = np.arange(len(pid)) - start + 1
    y = ((j == end[pid]) & event_at_end[pid]).astype(float)
    return pid, j, y


def make_X(j: np.ndarray, g: np.ndarray, cov: np.ndarray, H: int, G: int) -> np.ndarray:
    n = len(j)
    I = (j[:, None] == np.arange(1, H + 1)[None, :]).astype(float)
    Gd = (g[:, None] == np.arange(1, G)[None, :]).astype(float)
    inter = (I[:, 1:, None] * Gd[:, None, :]).reshape(n, -1)
    return np.hstack([I, Gd, inter, cov])


def fit_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit(maxiter=200)
    if not res.converged:
        raise RuntimeError("NHATS pooled logistic model did not converge within 200 iterations")
    p = np.asarray(res.params, dtype=float)
    if not np.all(np.isfinite(p)):
        raise FloatingPointError("non-finite coefficients")
    return p


def fit_hazards(df, kind, gcode, cov, H, G, w=None):
    endD, endX, ev = episodes(df, kind, H)
    pid, j, y = rows_until(endD, ev == 2)
    pD = fit_logit(make_X(j, gcode[pid], cov[pid], H, G), y, None if w is None else w[pid])
    pX = None
    if kind != "death":
        pid, j, y = rows_until(endX, ev == 1)
        pX = fit_logit(make_X(j, gcode[pid], cov[pid], H, G), y, None if w is None else w[pid])
    return pD, pX
