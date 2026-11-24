"""Fit NHATS risk and Cox models by imputation. Bootstrap 0 is the original sample."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from time import monotonic

# Each worker fits a separate model. Bound BLAS threads before importing NumPy.
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.duration.hazard_regression import PHReg

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nhats import analysis_lib as models
from src.nhats import config


HERE = Path(__file__).resolve().parent
PRIOR = config.GROUPS
JOINT = [f"index{i}_{prior}" for i in range(3) for prior in PRIOR]
FOUR = ["prior0", "prior1", "prior2", "prior3plus"]
ENDPOINTS = {"fried": "frailty_or_death", "falls": "falls", "hosp": "hosp"}
AGE_STRATA = {"age65_74": ("1", "2"), "age75_84": ("3", "4"), "age85plus": ("5", "6")}
COLUMNS = ["imp", "b", "analysis", "outcome", "group", "year", "value"]
INPUT_FILES = {"main": "landmarks_imputed.csv.gz", "complete": "complete_landmarks.csv.gz"}
SAMPLES = {}


def load_samples(path):
    data = models.prepare(pd.read_csv(path))
    if data.duplicated(["imp", "spid"]).any() or set(data.imp.unique()) != set(range(1, 11)):
        raise ValueError("Landmarks must contain one record per person and imputation.")
    robust = data[data.index_score.eq(0)]
    nonfrail = data[data.index_score.le(2)]
    robust_levels = models.levels_for(robust, config.MODEL1)
    joint_levels = models.levels_for(nonfrail, config.MODEL0)
    noage = [c for c in config.MODEL0 if c != "index_age_cat"]
    SAMPLES.clear()
    for m, frame in robust.groupby("imp"):
        d = frame.reset_index(drop=True)
        j = nonfrail[nonfrail.imp.eq(m)].reset_index(drop=True)
        SAMPLES[m] = {
            "robust": d,
            "cov": models.cov_matrix(d, config.MODEL0, robust_levels),
            "extended": models.cov_matrix(d, config.MODEL1, robust_levels),
            "noage": models.cov_matrix(d, noage, robust_levels),
            "groups": d.prior_score.clip(upper=2).to_numpy(int),
            "joint": j,
            "joint_cov": models.cov_matrix(j, config.MODEL0, joint_levels),
            "joint_groups": (3 * j.index_score + j.prior_score.clip(upper=2)).to_numpy(int),
        }


def sampled_people(length, imp, replicate, salt):
    if replicate == 0:
        return np.arange(length)
    rng = np.random.default_rng(config.SEED + 7919 * imp + 104729 * salt + replicate)
    return rng.integers(0, length, length)


def standardise(death_coef, outcome_coef, target_cov, horizon, group_count):
    """Average group-specific cumulative incidences over the target population."""
    n_intercepts = horizon * group_count
    death_offset = target_cov @ death_coef[n_intercepts:]
    event_offset = target_cov @ outcome_coef[n_intercepts:]
    years = np.arange(1, horizon + 1)
    comp, event, death = np.empty((3, group_count, horizon))
    for g in range(group_count):
        design = models.make_X(
            years, np.full(horizon, g), np.empty((horizon, 0)), horizon, group_count
        )
        hd = expit(death_offset[:, None] + design @ death_coef[:n_intercepts])
        hx = expit(event_offset[:, None] + design @ outcome_coef[:n_intercepts])
        surv = np.cumprod((1 - hd) * (1 - hx), axis=1)
        start = np.column_stack((np.ones(len(target_cov)), surv[:, :-1]))
        comp[g] = (1 - surv).mean(axis=0)
        event[g] = np.cumsum(start * (1 - hd) * hx, axis=1).mean(axis=0)
        death[g] = np.cumsum(start * hd, axis=1).mean(axis=0)
    if not np.allclose(comp, event + death, rtol=0, atol=1e-11):
        raise ArithmeticError("The two first-event incidences must sum to the composite.")
    return comp, event, death


def fit_replicate(version, imp, replicate):
    sample = SAMPLES[imp]
    base = sample["robust"]
    ids = sampled_people(len(base), imp, replicate, 1)
    d = base.iloc[ids]
    cov = sample["cov"][ids]
    groups = sample["groups"][ids]
    rows = []

    def fit(analysis, kind, people, group, x, labels=PRIOR, horizon=4):
        params = models.fit_hazards(people, kind, group, x, horizon, len(labels))
        comp, event, death = standardise(*params, x, horizon, len(labels))
        arrays = [(ENDPOINTS[kind], comp if kind == "fried" else event)]
        if kind == "fried" and analysis == "main" and version == "main":
            arrays += [("frailty", event), ("death_before_frailty", death)]
        years = range(1, horizon + 1) if analysis == "joint" else (horizon,)
        for outcome, array in arrays:
            for g, label in enumerate(labels):
                for year in years:
                    rows.append(
                        (imp, replicate, analysis, outcome, label, year, array[g, year - 1])
                    )
        if analysis == "joint" and replicate == 0:
            restricted, _, _ = standardise(
                *params, x[people.index_score.eq(0)], horizon, len(labels)
            )
            rows.append(
                (
                    imp,
                    0,
                    "joint_index0_check",
                    "frailty_or_death",
                    "index0_prior2plus",
                    horizon,
                    restricted[2, -1],
                )
            )

    for kind in ENDPOINTS:
        fit("main", kind, d, groups, cov)
    if version == "complete":
        return rows

    fit("model1", "fried", d, groups, sample["extended"][ids])
    if replicate == 0:  # only the six-year point estimates are requested
        fit("six_year", "fried", d, groups, cov, horizon=6)
    keep = d.index_round_num.ne(13).to_numpy()
    fit("excl_index_round13", "fried", d[keep], groups[keep], cov[keep])
    for label, ages in AGE_STRATA.items():
        keep = d.index_age_cat.isin(ages).to_numpy()
        fit(label, "fried", d[keep], groups[keep], sample["noage"][ids[keep]])

    ids = sampled_people(len(base), imp, replicate, 4)
    d = base.iloc[ids]
    fit(
        "four_level",
        "fried",
        d,
        d.prior_score.clip(upper=3).to_numpy(int),
        sample["cov"][ids],
        FOUR,
    )
    joint = sample["joint"]
    ids = sampled_people(len(joint), imp, replicate, 2)
    fit(
        "joint",
        "fried",
        joint.iloc[ids],
        sample["joint_groups"][ids],
        sample["joint_cov"][ids],
        JOINT,
    )
    return rows


def cox_estimates(imp):
    d = SAMPLES[imp]["robust"]
    g = SAMPLES[imp]["groups"]
    cov = np.column_stack([g == 1, g == 2, SAMPLES[imp]["cov"]])
    rows = []
    for kind, outcome in ENDPOINTS.items():
        end_d, end_x, event_type = models.episodes(d, kind, 4)
        if kind == "fried":
            # Frailty is observable only in consecutive assessable rounds.
            # The tracker still ascertains deaths after this window closes.
            time = end_d
            event = event_type != 0
        else:
            time = end_x
            event = event_type == 1
        eligible = time > 0
        x = cov[eligible]
        cols = np.r_[True, True, x[:, 2:].std(axis=0) > 0]
        model = PHReg(
            time[eligible], x[:, cols], status=event[eligible].astype(float), ties="breslow"
        ).fit()
        for col, label in enumerate(PRIOR[1:]):
            rows.append(
                (
                    imp,
                    outcome,
                    label,
                    model.params[col],
                    model.bse[col],
                    int(eligible.sum()),
                    int(event[eligible].sum()),
                )
            )
    return rows


def save_csv(data, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        stage = Path(f.name)
    try:
        data.to_csv(stage, index=False, compression="gzip" if path.suffix == ".gz" else None)
        stage.replace(path)
    finally:
        stage.unlink(missing_ok=True)


def fit_batch(path, version, imp, first, stop):
    if path.is_file() and os.environ.get("FRAILTY_REFIT") != "1":
        return
    rows = []
    for b in range(first, stop):
        rows.extend(fit_replicate(version, imp, b))
    frame = pd.DataFrame.from_records(rows, columns=COLUMNS)
    if not np.isfinite(frame.value).all() or not frame.value.between(-1e-12, 1 + 1e-12).all():
        raise ArithmeticError(
            f"Invalid risks for {version}, imputation {imp}, starting replicate {first}."
        )
    save_csv(frame, path)


def fingerprint(path):
    digest = sha256()
    # Exclude gzip header metadata such as a fresh run's output timestamp.
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    for name in ("config.py", "fit_models.py", "analysis_lib.py"):
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()


def fit_version(version, workers):
    path = config.WORK / INPUT_FILES[version]
    checkpoint = config.WORK / "checkpoints" / version / fingerprint(path)
    checkpoint.mkdir(parents=True, exist_ok=True)
    ranges = [(0, 1)] + [(b, min(b + 5, 51)) for b in range(1, 51, 5)]
    tasks = [
        (checkpoint / f"imp{m:02}_b{first:02}-{stop - 1:02}.csv.gz", version, m, first, stop)
        for first, stop in ranges
        for m in range(1, 11)
    ]
    start = monotonic()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=load_samples, initargs=(path,)
    ) as executor:
        jobs = {executor.submit(fit_batch, *task): task for task in tasks}
        for n, job in enumerate(as_completed(jobs), 1):
            job.result()
            if n % 10 == 0:
                print(
                    f"NHATS {version}: {n}/{len(tasks)} batches ({monotonic() - start:.1f}s)",
                    flush=True,
                )
        if version == "main":
            coeff = [row for block in executor.map(cox_estimates, range(1, 11)) for row in block]
            cox = pd.DataFrame.from_records(
                coeff, columns=["imp", "outcome", "group", "logHR", "se", "n", "events"]
            )
            if (
                len(cox) != 60
                or not np.isfinite(cox[["logHR", "se"]]).all().all()
                or not cox.se.gt(0).all()
            ):
                raise ArithmeticError("Incomplete Cox estimates or invalid standard errors.")
            save_csv(cox, config.WORK / "cox_by_imputation.csv")

    combined = pd.concat([pd.read_csv(task[0]) for task in tasks], ignore_index=True)
    if (
        combined.duplicated(COLUMNS[:-1]).any()
        or len(combined[["imp", "b"]].drop_duplicates()) != 510
    ):
        raise ValueError("Invalid participant-resampling grid.")
    if version == "main":
        check = combined[combined.analysis.eq("joint_index0_check")].copy()
        assert len(check) == 10 and check.b.eq(0).all()
        save_csv(
            check.rename(columns={"value": "risk"})[["imp", "group", "year", "risk"]],
            config.WORK / "joint_index0_standardisation.csv",
        )
        combined = combined[combined.analysis.ne("joint_index0_check")]
    keys = ["analysis", "outcome", "group", "year"]
    expected = combined.groupby(keys).size()
    expected_n = np.where(expected.index.get_level_values("analysis") == "six_year", 10, 510)
    if not np.array_equal(expected.to_numpy(), expected_n):
        raise ValueError("At least one estimate lacks a complete set of replicates.")
    save_csv(combined, config.WORK / f"{version}_draws.csv.gz")
    print(
        f"NHATS {version}: finished {len(combined)} saved estimates in {monotonic() - start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--version", choices=("both", "main", "complete"), default="both")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for version in INPUT_FILES if args.version == "both" else [args.version]:
        fit_version(version, args.workers)
