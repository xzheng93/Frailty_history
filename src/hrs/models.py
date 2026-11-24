"""HRS wave-specific imputations and standardised four-year risks."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.imputation.mice import MICEData
from src.hrs.cohort import FFP, grip_cutoff, speed_cutoff

SEED = 20261120
OUTCOMES = ("frailty_or_death", "falls", "hosp")


def imputation_frame(data):
    x = (
        data[
            [
                "speed",
                "grip",
                "bmi_use",
                "height_use",
                "exhaustion",
                "low_pa",
                "loss10",
                "agey_e",
                "raeduc",
                "hosp",
                "falls",
                "shlt",
                "adl5a",
            ]
        ]
        .astype(float)
        .copy()
    )
    x["female"] = data.ragender.eq(2).astype(float)
    x["race"] = np.select(
        [
            data.rahispan.eq(1),
            data.rahispan.eq(0) & data.raracem.eq(1),
            data.rahispan.eq(0) & data.raracem.eq(2),
            data.rahispan.eq(0) & data.raracem.eq(3),
        ],
        [4.0, 1.0, 2.0, 3.0],
        default=np.nan,
    )
    return x


def impute_scores(p, m):
    target = p.agey_e.ge(65) & p.PMELIG.eq(1) & p.assessable.ge(3)
    outputs = []
    for year, idx in p[target].groupby("year").groups.items():
        raw = p.loc[idx]
        x = imputation_frame(raw).reset_index(drop=True)
        active = x.columns[x.nunique(dropna=True) > 1].tolist()
        np.random.seed(SEED + 1000 * m + int(year))
        mice = MICEData(x[active], perturbation_method="gaussian", k_pmm=10)
        mice.update_all(10)
        f = x.copy()
        f[active] = mice.data[active]
        if (
            f[
                [
                    "speed",
                    "grip",
                    "bmi_use",
                    "height_use",
                    "exhaustion",
                    "low_pa",
                    "loss10",
                    "raeduc",
                    "race",
                ]
            ]
            .isna()
            .any()
            .any()
        ):
            raise ValueError("Imputation left required inputs unresolved")
        f.index = raw.index
        sex = raw.ragender
        components = raw[FFP].copy()
        proposed = pd.DataFrame(
            {
                "weakness": f.grip.le(grip_cutoff(sex, f.bmi_use)),
                "slow_walk": f.speed.le(speed_cutoff(sex, f.height_use)),
                "low_pa": f.low_pa.eq(1),
                "exhaustion": f.exhaustion.eq(1),
                "weight_loss": f.loss10.eq(1) | f.bmi_use.lt(18.5),
            },
            index=f.index,
        ).astype(float)
        # If a disjunctive FFP item is already resolved, retain it independently of missing inputs.
        components = components.fillna(proposed)
        if not components.isin([0.0, 1.0]).all().all():
            raise ValueError("Non-binary FFP component after scoring")
        score = components.sum(axis=1)
        known = raw.score.notna()
        np.testing.assert_array_equal(raw.loc[known, "score"], score[known])
        outputs.append(
            pd.DataFrame(
                dict(
                    id=raw.id,
                    year=year,
                    completed=score,
                    education=f.raeduc,
                    race=f.race,
                )
            )
        )
    result = pd.concat(outputs, ignore_index=True).set_index(["id", "year"], verify_integrity=True)
    assert len(result) == int(target.sum()) and 2020 not in result.index.get_level_values("year")
    return result


def followup_base(p, pairs):
    d = pairs[
        [
            "id",
            "year",
            "visitmonth",
            "radyear",
            "radmonth",
            "ragender",
            "agey_e",
            "score",
            "score_prior",
        ]
    ].copy()
    d["last_alive"] = d.id.map(p.groupby("id").visitmonth.max())
    for gap in [2, 4]:
        cols = [
            "id",
            "year",
            "visitmonth",
            "falls",
            "hosp",
            "PMELIG",
            "assessable",
            "score",
        ]
        fut = p[cols].copy()
        fut.year -= gap
        fut = fut.rename(columns={n: f"{n}_{gap}" for n in cols[2:]})
        d = d.merge(fut, on=["id", "year"], how="left", validate="one_to_one")
    return d


def map_visit(frame, d, offset, column):
    keys = pd.MultiIndex.from_arrays([d.id, d.year + offset], names=["id", "year"])
    return frame[column].reindex(keys).to_numpy()


def build_analysis(fixed, imp, m):
    d = fixed.copy()
    d["base_score"] = map_visit(imp, d, 0, "completed")
    d["prior_score"] = map_visit(imp, d, -4, "completed")
    d["four_score"] = map_visit(imp, d, 4, "completed")
    for c in ("race", "education"):
        d[c] = map_visit(imp, d, 0, c)
    if d[["base_score", "prior_score", "race", "education"]].isna().any().any():
        raise ValueError("An eligible fixed pair lacks its completed scores or covariates")
    assert not (d.four_score.notna() & (~d.PMELIG_4.eq(1) | ~d.assessable_4.ge(3))).any()
    # Missing death month, known year: sample only months consistent with survival to last interview.
    rng = np.random.default_rng(SEED + 40000 + m)
    dm = d.radyear * 12 + d.radmonth
    unknown_month = d.radyear.notna() & dm.isna()
    for i in d.index[unknown_month]:
        low = max(int(d.at[i, "radyear"] * 12 + 1), int(d.at[i, "last_alive"]))
        high = int(d.at[i, "radyear"] * 12 + 12)
        if low <= high:
            dm.at[i] = rng.integers(low, high + 1)
    # A death date before a documented living interview is unreliable. Use known-alive follow-up.
    d["death_conflict"] = dm.notna() & dm.lt(d.last_alive)
    d["death_time"] = (dm - d.visitmonth).mask(d.death_conflict)
    d["alive_time"] = d.last_alive - d.visitmonth
    d["history"] = d.prior_score.clip(upper=2).astype(int)
    return d


def adjustment(d):
    # Five-year age groups, with ages 85+ combined because few HRS participants at index are older.
    age = np.digitize(d.agey_e, [70, 75, 80, 85])
    educ = np.select(
        [d.education.eq(1), d.education.isin([2, 3]), d.education.ge(4)],
        [0, 1, 2],
        default=-1,
    )
    if np.any(educ < 0):
        raise ValueError("Unrecognised education")
    categories = [
        age,
        d.ragender.to_numpy(),
        d.race.to_numpy(),
        educ,
        d.year.to_numpy(),
    ]
    arrays = [(a[:, None] == np.unique(a)[None, 1:]).astype(float) for a in categories]
    return np.column_stack(arrays)


def first_events(d, kind):
    n = len(d)
    death = d.death_time.to_numpy(float)
    horizon_death = np.where((death >= 0) & (death <= 48), death, np.inf)
    first = np.full(n, np.inf)
    if kind == "mortality":
        return first, [], []
    open_window = np.ones(n, dtype=bool)
    observations = []
    responses = []
    for gap in (2, 4):
        when = (d[f"visitmonth_{gap}"] - d.visitmonth).to_numpy(float)
        q = d[f"{kind}_{gap}"]
        outcome = q.eq(1).to_numpy()
        valid = (
            open_window
            & q.notna().to_numpy()
            & (when > 0)
            & (when <= horizon_death)
            & np.isinf(first)
        )
        within = valid & outcome
        first[within] = when[within]
        observations.append(valid)
        responses.append(outcome)
        # A break in ascertainment ends nonfatal follow-up for the repeated recall items.
        open_window = valid & ~outcome
    return first, observations, responses


def make_design(period, g, cov, periods):
    gc = np.column_stack([((period == t) & (g == v)) for t in periods for v in range(3)])
    return np.column_stack([gc.astype(float), cov])


class Regression:
    """Group identical design rows. Logit for interviews, piecewise exponential for mortality."""

    def __init__(self, periods, j, g, cov, pid, y, exposure, family):
        self.periods = periods
        self.pid = pid
        self.y = y.astype(float)
        self.exposure = exposure
        self.family = family
        X = make_design(j, g[pid], cov[pid], periods)
        self.X, self.index = np.unique(X, axis=0, return_inverse=True)
        self.start = np.zeros(self.X.shape[1])
        base = self.y.sum() / max(exposure.sum(), 1)
        self.start[: 3 * len(periods)] = (
            np.log(base / (1 - base)) if family == "binomial" else np.log(base)
        )

    def fit(self, weights, point=False):
        w = weights[self.pid]
        size = len(self.X)
        count = np.bincount(self.index, weights=w * self.exposure, minlength=size)
        event = np.bincount(self.index, weights=w * self.y, minlength=size)
        b = self.start.copy()
        # A vanishing ridge supplies finite estimates in bootstrap cells with no events.
        ridge = 1.0e-7
        for _ in range(75):
            eta = self.X @ b
            if self.family == "binomial":
                mean = expit(eta)
                deriv = mean * (1 - mean)
            else:
                mean = np.exp(np.clip(eta, -36, 15))
                deriv = mean
            gradient = self.X.T @ (event - count * mean) - ridge * b
            if np.abs(gradient).max() < 2.0e-6:
                if point:
                    self.start = b.copy()
                return b
            info = self.X.T @ ((count * deriv)[:, None] * self.X)
            info.flat[:: info.shape[0] + 1] += ridge
            delta = np.linalg.solve(info, gradient)
            b += delta / max(1, np.max(np.abs(self.X @ delta)) / 3.0)
        raise ArithmeticError("Regression did not converge")

    def predict(self, b, g, cov, period):
        intercept = b[3 * self.periods.index(period) + g]
        eta = intercept + cov @ b[3 * len(self.periods) :]
        if self.family == "binomial":
            return expit(eta)
        return -np.expm1(-np.exp(np.clip(eta, -36, 15)))  # full two-year interval


def outcome_models(d, cov, kind):
    g = d.history.to_numpy()
    dt = d.death_time.to_numpy()
    alive = d.alive_time.to_numpy()
    first, seen, out = first_events(d, kind)
    observed_death = (dt >= 0) & (dt <= 48) & (dt < first)
    end = np.minimum(48, np.minimum(np.where(np.isfinite(dt), dt, alive), first))
    # Month-level resolution: retain deaths in index month with half a month of exposure.
    end = np.where(observed_death & (end == 0), 0.5, end)
    if (end < 0).any():
        raise ValueError("Invalid mortality risk time")
    pids = []
    js = []
    ys = []
    es = []
    for j in [1, 2]:
        left = (j - 1) * 24
        span = np.maximum(0, np.minimum(24, end - left))
        keep = np.flatnonzero(span > 0)
        y = observed_death & (dt <= 24 * j) & ((dt > left) | ((j == 1) & (dt == 0)))
        assert not np.any(y & (span == 0))
        pids.append(keep)
        js.append(np.full(len(keep), j))
        ys.append(y[keep])
        es.append(span[keep] / 24.0)
    deathmodel = Regression(
        (1, 2),
        np.concatenate(js),
        g,
        cov,
        np.concatenate(pids),
        np.concatenate(ys),
        np.concatenate(es),
        "poisson",
    )
    other = None
    if kind != "mortality":
        periods = (1, 2)
        pids = [np.flatnonzero(v) for v in seen]
        js = [np.full(len(a), j) for a, j in zip(pids, periods)]
        ys = np.concatenate([y[i] for i, y in zip(pids, out)])
        other = Regression(
            periods,
            np.concatenate(js),
            g,
            cov,
            np.concatenate(pids),
            ys,
            np.ones(len(ys)),
            "binomial",
        )
    return deathmodel, other


def frailty_at_year_four(d, cov):
    """One scheduled frailty examination and a separate, fixed 48-month death window."""
    g = d.history.to_numpy()
    dt = d.death_time.to_numpy()
    died = (dt >= 0) & (dt <= 48)
    alive = (dt > 48) | (~np.isfinite(dt) & (d.alive_time.to_numpy() >= 48))
    assert not (died & alive).any()
    scored = d.four_score.notna().to_numpy()
    frail = d.four_score.ge(3).to_numpy()
    visit = (d.visitmonth_4 - d.visitmonth).to_numpy()
    valid_visit = visit > 0
    eligible = np.flatnonzero(alive & scored & valid_visit)
    conditional = Regression(
        (2,),
        np.full(len(eligible), 2),
        g,
        cov,
        eligible,
        frail[eligible],
        np.ones(len(eligible)),
        "binomial",
    )
    return conditional


def estimate(models, cov, weights, point):
    results = {}
    marginal_death = []
    for kind in ("mortality", "falls", "hosp"):
        dm, xm = models[kind]
        bd = dm.fit(weights, point)
        bx = xm.fit(weights, point) if xm else None
        inc_x = []
        inc_d = []
        for group in range(3):
            surv = np.ones(len(cov))
            cx = np.zeros(len(cov))
            cd = np.zeros(len(cov))
            for t in (1, 2):
                hd = dm.predict(bd, group, cov, t)
                hx = xm.predict(bx, group, cov, t) if xm and t in xm.periods else 0.0
                cd += surv * hd
                cx += surv * (1 - hd) * hx
                surv *= (1 - hd) * (1 - hx)
            inc_x.append(np.average(cx, weights=weights))
            inc_d.append(np.average(cd, weights=weights))
            if kind == "mortality":
                marginal_death.append(cd)
            if np.max(np.abs(surv + cx + cd - 1)) > 1.0e-10:
                raise ArithmeticError("CIF identity")
        results[kind] = inc_d if kind == "mortality" else inc_x
    model = models["frailty"]
    beta = model.fit(weights, point)
    composite = []
    for group, death in enumerate(marginal_death):
        conditional = model.predict(beta, group, cov, 2)
        composite.append(np.average(death + (1 - death) * conditional, weights=weights))
    return {
        "frailty_or_death": composite,
        "falls": results["falls"],
        "hosp": results["hosp"],
    }
