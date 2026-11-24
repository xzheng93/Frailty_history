"""Reproduce SHARE risk estimates across eligible scheduled assessments (release 9.0.0)."""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
import fcntl
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from statsmodels.imputation.mice import MICEData
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_paths import raw_data_dir
from src.share.share_panel import (
    CONTACTS,
    ITEMS,
    DESIGNS,
    WAVES,
    death_dates,
    eligible_pairs,
    grip_cutoff,
    visit,
    vital,
)
from src.hrs.models import Regression, SEED, frailty_at_year_four, outcome_models
from src.hrs.trends import pooled_trends

PRIMARY = "frailty_or_death"
REPORTS = ("falling_symptom", "overnight_hospital_12m")
INPUTS = ["grip", "bmi", *ITEMS[1:], "education", "sphus", "adl", "gender", "age_int"]
KEYS = ["analysis", "imp", "bootstrap", "endpoint", "group"]


def prepare(root, designs):
    visits = {w: visit(root, w) for w in WAVES}
    tracker = vital(root)
    pairs, flows = {}, []
    for label, schedules in designs.items():
        fixed, flow = eligible_pairs(visits, tracker, schedules)
        pairs[label] = fixed.drop(columns=[f"dead_w{w}" for w in CONTACTS])
        flows.append(flow.assign(analysis=label))
    all_pairs = pd.concat(pairs.values())
    ids = all_pairs.index.unique().sort_values()
    tracker = tracker.loc[ids]
    # Fit a wave once per imputation. The same person's visit has the same score
    # even when the two scheduling rules assign it different roles.
    aux = pd.DataFrame(
        {f"items_w{w}": v[ITEMS].mean(axis=1).reindex(ids).fillna(-1) for w, v in visits.items()}
    )
    datasets = {}
    for w, v in visits.items():
        needed = all_pairs.index[
            all_pairs[["prior_wave", "index_wave", "target_wave"]].eq(w).any(axis=1)
        ]
        raw = v.loc[v.index.isin(needed) & v.regular & v.assessed.ge(3)].sort_index()
        if not len(raw):
            continue
        raw = raw[["date", "country", *INPUTS, "weak", "observed_score"]]
        x = raw[INPUTS].join(
            pd.get_dummies(raw.country.astype(int), prefix="country", drop_first=True)
        )
        x = x.join(aux.drop(columns=f"items_w{w}")).astype(float)
        datasets[w] = (raw, x)
    fixed_by_wave = pd.concat(
        [
            d.groupby(["prior_wave", "index_wave", "target_wave"])
            .agg(
                fixed_people=("date", "size"),
                target_interview=("date_4", "count"),
                target_ffp_assessable=("assessed_4", lambda x: int(x.ge(3).sum())),
            )
            .reset_index()
            .assign(analysis=label)
            for label, d in pairs.items()
        ],
        ignore_index=True,
    )
    return (datasets, pairs, tracker), pd.concat(flows, ignore_index=True), fixed_by_wave


def scores(raw, x, tracker, death, w, imp):
    rel = death.reindex(raw.index) - raw.date
    life = tracker.last_alive.reindex(raw.index) - raw.date
    x = x.assign(
        death48=rel.between(0, 48).astype(float),
        alive48=(rel.gt(48) | (rel.isna() & life.ge(48))).astype(float),
    ).reset_index(drop=True)
    active = x.columns[x.nunique().gt(1)]
    np.random.seed(SEED + w * 1000 + imp)
    mice = MICEData(x[active], k_pmm=10)
    mice.update_all(10)
    x[active] = mice.data[active]
    x.index = raw.index
    if x[["grip", "bmi", *ITEMS[1:], "education"]].isna().any().any():
        raise ValueError(f"Unresolved FFP inputs or education in W{w}")
    weak = x.grip.le(grip_cutoff(raw.gender, x.bmi)).astype(float)
    components = raw[ITEMS].fillna(x[ITEMS[1:]].assign(weak=weak))
    if not components.isin([0, 1]).all().all():
        raise ValueError("Invalid FFP component")
    result = pd.DataFrame({"score": components.sum(axis=1), "education": x.education})
    known = raw.observed_score.dropna()
    np.testing.assert_array_equal(result.loc[known.index, "score"], known)
    return result


def adjustment(d):
    groups = pd.DataFrame(
        dict(
            age=np.digitize(d.age_int, [70, 75, 80, 85]),
            sex=d.gender,
            education=d.education,
            country=d.country,
            index_year=d.int_year,
        )
    )
    if groups.isna().any().any():
        raise ValueError("Missing adjustment data")
    return pd.get_dummies(groups.astype(int).astype(str), drop_first=True, dtype=float).to_numpy()


def robust(fixed, filled, death):
    d = fixed.copy()
    for role, column in [
        ("prior_wave", "history"),
        ("index_wave", "index_score"),
        ("target_wave", "four_score"),
    ]:
        for w, ids in d.groupby(role).groups.items():
            f = filled[w].reindex(ids)
            d.loc[ids, column] = f.score
            if role == "index_wave":
                d.loc[ids, "education"] = f.education
    if d[["history", "index_score", "education"]].isna().any().any():
        raise ValueError("Unscored eligible pair")
    d = d[d.index_score.eq(0)].copy()
    d["history"] = d.history.clip(upper=2).astype(int)
    d["visitmonth"], d["visitmonth_4"] = d.date, d.date_4
    d["death_time"] = death.reindex(d.index) - d.date
    d["alive_time"] = d.last_alive - d.date
    if not d.alive_time.ge(0).all() or d.death_time.dropna().lt(0).any():
        raise ValueError("Impossible follow-up")
    d["death48"] = d.death_time.between(0, 48)
    d["alive48"] = d.death_time.gt(48) | (d.death_time.isna() & d.alive_time.ge(48))
    d["target_interview"] = d.date_4.gt(d.date)
    d["scored_survivor"] = d.alive48 & d.target_interview & d.four_score.notna()
    d["frail_in_survivors"] = d.scored_survivor & d.four_score.ge(3)
    d["survivor_without_score"] = d.alive48 & ~d.scored_survivor
    d["censored_before_48"] = ~(d.alive48 | d.death48)
    d["dead_without_exact_month"] = d.death48 & (d.deathyear.isna() | d.deathmonth.isna())
    d["undatable_recorded_death"] = d.recorded_dead & d.death_time.isna()
    if (
        not d.index.is_unique
        or not d[["death48", "scored_survivor", "survivor_without_score", "censored_before_48"]]
        .sum(axis=1)
        .eq(1)
        .all()
    ):
        raise ValueError("Inconsistent four-year accounting")
    if (d.four_score.notna() & (~d.assessed_4.ge(3) | ~d.regular_4.eq(True))).any():
        raise ValueError("Scored an ineligible target visit")
    return d


def describe(d):
    flags = [
        "death48",
        "scored_survivor",
        "frail_in_survivors",
        "survivor_without_score",
        "censored_before_48",
        "dead_without_exact_month",
        "undatable_recorded_death",
        "target_interview",
    ]
    by = ["prior_wave", "index_wave", "target_wave", "history"]
    g = d.groupby(by)
    counts = g.size().rename("n").to_frame().join(g[flags].sum())
    for endpoint in REPORTS:
        y = d[endpoint].where(d.target_interview).groupby([d[c] for c in by])
        counts[endpoint + "_respondents"], counts[endpoint + "_events"] = (
            y.count(),
            y.sum().astype(int),
        )
    intervals = []
    for w, group in [("all", d), *d.groupby("index_wave")]:
        for name, months in {
            "prior_to_index": group.date - group.date_prior,
            "index_to_scored_followup": (group.date_4 - group.date)[group.scored_survivor],
        }.items():
            intervals.append([w, name, len(months), *months.quantile([0, 0.25, 0.5, 0.75, 1])])
    intervals = pd.DataFrame(
        intervals, columns=["index_wave", "interval", "n", "min", "q25", "median", "q75", "max"]
    )
    return counts.reset_index().rename(columns={"history": "group"}), intervals


def estimate(d, imp):
    cov = adjustment(d)
    mortality, _ = outcome_models(d, cov, "mortality")
    frailty = frailty_at_year_four(d, cov)
    models = {}
    for endpoint in REPORTS:
        sub = d[d.target_interview & d[endpoint].notna()]
        c = adjustment(sub)
        ids = np.arange(len(sub))
        fit = Regression(
            (1,),
            np.ones(len(sub)),
            sub.history.to_numpy(),
            c,
            ids,
            sub[endpoint].to_numpy(),
            np.ones(len(sub)),
            "binomial",
        )
        models[endpoint] = (fit, c, d.index.get_indexer(sub.index))
    rng = np.random.default_rng(SEED + 700 * imp)
    draws = []
    for b in range(51):
        n = len(d)
        weights = np.ones(n) if b == 0 else np.bincount(rng.integers(n, size=n), minlength=n)
        bd, bf = mortality.fit(weights, point=b == 0), frailty.fit(weights, point=b == 0)
        for g in range(3):
            survive = (1 - mortality.predict(bd, g, cov, 1)) * (
                1 - mortality.predict(bd, g, cov, 2)
            )
            risk = 1 - survive + survive * frailty.predict(bf, g, cov, 2)
            draws.append((b, PRIMARY, g, np.average(risk, weights=weights)))
        for endpoint, (fit, c, ids) in models.items():
            w = weights[ids]
            beta = fit.fit(w, point=b == 0)
            for g in range(3):
                draws.append((b, endpoint, g, np.average(fit.predict(beta, g, c, 1), weights=w)))
    return pd.DataFrame(draws, columns=["bootstrap", "endpoint", "group", "risk"])


def initialize(data):
    global DATA
    DATA = data


def imputed_samples(imp, data):
    datasets, pairs, tracker = data
    death = death_dates(tracker, np.random.default_rng(SEED + 40000 + imp))
    filled = {w: scores(raw, x, tracker, death, w, imp) for w, (raw, x) in datasets.items()}
    for label, fixed in pairs.items():
        yield label, robust(fixed, filled, death)


def one_imputation(imp):
    with threadpool_limits(limits=1):
        output = []
        for label, d in imputed_samples(imp, DATA):
            c, intervals = describe(d)
            output.append(
                [
                    frame.assign(analysis=label, imp=imp)
                    for frame in (estimate(d, imp), c, intervals)
                ]
            )
            print(
                f"SHARE {label}, imputation {imp}: {len(d)} index-robust participants", flush=True
            )
    return [pd.concat(frames, ignore_index=True) for frames in zip(*output)]


def export(staged, results, flow, by_wave):
    draws, counts, intervals = [pd.concat(frames, ignore_index=True) for frames in zip(*results)]
    draws = draws.sort_values(KEYS)
    refkeys = KEYS[:-1]
    ref = draws.loc[draws.group.eq(0), refkeys + ["risk"]].rename(columns={"risk": "reference"})
    draws = draws.merge(ref, on=refkeys, validate="many_to_one")
    draws["rd"] = draws.risk - draws.pop("reference")
    if draws.duplicated(KEYS).any() or not draws.risk.between(0, 1).all():
        raise ValueError("Invalid resampling grid")
    trend = []
    for label, sub in draws.groupby("analysis"):
        if set(sub.endpoint) != {PRIMARY, *REPORTS}:
            raise ValueError("Missing endpoint")
        trend.append(pooled_trends(sub).assign(analysis=label))
    by = ["analysis", "endpoint", "group"]
    metrics = ["risk", "rd"]
    points = draws[draws.bootstrap.eq(0)].groupby(by)[metrics].mean()
    boot = draws[draws.bootstrap.gt(0)].groupby(by)[metrics]
    for suffix, p in [("_lo", 0.025), ("_hi", 0.975)]:
        points = points.join(boot.quantile(p).add_suffix(suffix))
    summary = points.mul(100).reset_index()
    for endpoint in REPORTS:
        y, n = counts[endpoint + "_events"], counts[endpoint + "_respondents"]
        if not (
            y.le(n) & n.le(counts.target_interview) & counts.target_interview.le(counts.n)
        ).all():
            raise ValueError("Invalid report denominator")
    for name, frame in {
        "flow": flow,
        "fixed_pairs_by_wave": by_wave,
        "counts_by_imputation": counts,
        "assessment_intervals": intervals,
        "four_year_estimates": summary[summary.endpoint.eq(PRIMARY)],
        "followup_report_estimates": summary[summary.endpoint.ne(PRIMARY)],
        "trends": pd.concat(trend),
        "aggregate_resamples": draws[KEYS + metrics],
    }.items():
        filename = name + (".csv.gz" if name == "aggregate_resamples" else ".csv")
        frame.to_csv(staged / filename, index=False)


def run(root, out, workers=3):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data, flow, by_wave = prepare(root, DESIGNS)
        print(flow.to_string(index=False), flush=True)
        if workers == 1:
            initialize(data)
            results = list(map(one_imputation, range(1, 11)))
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=initialize,
                initargs=(data,),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                results = list(pool.map(one_imputation, range(1, 11)))
        with TemporaryDirectory(dir=out) as temp:
            staged = Path(temp)
            export(staged, results, flow, by_wave)
            for file in staged.iterdir():
                file.replace(out / file.name)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir() / "SHARE")
    parser.add_argument("--output", type=Path, default=base / "results/share")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run(args.raw_dir, args.output, args.workers)
