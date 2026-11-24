"""Reproduce HRS selection and the three adjusted four-year outcomes directly from RAND and Core."""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.hrs.cohort import panel, participant_flow
from src.hrs.models import (
    OUTCOMES,
    SEED,
    adjustment,
    build_analysis,
    estimate,
    followup_base,
    frailty_at_year_four,
    impute_scores,
    outcome_models,
)


SCORING_INPUTS = (
    "speed",
    "grip",
    "bmi_use",
    "height_use",
    "exhaustion",
    "low_pa",
    "loss10",
    "ragender",
)


def scoring_input_flags(eligible):
    """Raw scoring inputs before imputation, including observed values for qualifying test failures."""
    flags = pd.DataFrame(index=eligible.id)
    for visit, suffix in [("index", ""), ("prior", "_prior")]:
        columns = [name + suffix for name in SCORING_INPUTS]
        flags[visit + "_missing_scoring_input"] = eligible[columns].isna().any(axis=1).to_numpy()
    flags["either_missing_scoring_input"] = flags.any(axis=1)
    return flags


def missingness_counts(robust, flags, m):
    linked = robust.set_index("id").join(flags, validate="one_to_one")
    if linked[flags.columns].isna().any().any():
        raise ValueError("Missing scoring-input flags after linking fixed pairs")
    rows = []
    for group, subset in linked.groupby("history"):
        row = {"imp": m, "group": group, "n": len(subset)}
        row.update({key: int(subset[key].sum()) for key in flags})
        row["either_unresolved_component_score"] = int(
            (subset.score.isna() | subset.score_prior.isna()).sum()
        )
        rows.append(row)
    return rows


def fit(raw_dir, work):
    work.mkdir(parents=True, exist_ok=True)
    visits, _, _ = panel(raw_dir)
    eligible, flow, exclusions, flow_year = participant_flow(visits)
    flow.to_csv(work / "flow.csv", index=False)
    exclusions.to_csv(work / "pair_exam_exclusions.csv", index=False)
    flow_year.to_csv(work / "flow_by_index_year.csv", index=False)
    fixed = followup_base(visits, eligible)
    flags = scoring_input_flags(eligible)
    draws, counts, by_year, all_index, missing = [], [], [], [], []
    for m in range(1, 11):
        completed = impute_scores(visits, m)
        paired = build_analysis(fixed, completed, m)
        categories = paired.base_score.clip(upper=3).astype(int)
        for score, n in categories.value_counts().sort_index().items():
            all_index.append((m, score, n))
        robust = paired.loc[paired.base_score.eq(0)].reset_index(drop=True)
        assert robust.id.is_unique
        missing.extend(missingness_counts(robust, flags, m))
        robust["prior_incomplete"] = robust.score_prior.isna()
        robust["index_incomplete"] = robust.score.isna()
        robust["either_incomplete"] = robust.prior_incomplete | robust.index_incomplete
        for group, subset in robust.groupby("history"):
            counts.append(
                dict(
                    imp=m,
                    group=group,
                    n=len(subset),
                    prior_incomplete=int(subset.prior_incomplete.sum()),
                    index_incomplete=int(subset.index_incomplete.sum()),
                    either_incomplete=int(subset.either_incomplete.sum()),
                )
            )
        for (year, group), subset in robust.groupby(["year", "history"]):
            by_year.append((m, year, group, len(subset), int(subset.either_incomplete.sum())))
        cov = adjustment(robust)
        models = {key: outcome_models(robust, cov, key) for key in ("mortality", "falls", "hosp")}
        models["frailty"] = frailty_at_year_four(robust, cov)
        rng = np.random.default_rng(SEED + 700 * m)
        for b in range(51):
            weights = (
                np.ones(len(robust))
                if b == 0
                else np.bincount(rng.integers(0, len(robust), len(robust)), minlength=len(robust))
            )
            risks = estimate(models, cov, weights, b == 0)
            for endpoint in OUTCOMES:
                values = risks[endpoint]
                for group, value in enumerate(values):
                    draws.append((m, b, endpoint, group, value, value - values[0]))
        print(
            f"HRS imputation {m}/10 complete, 50 participant bootstrap resamples.",
            flush=True,
        )
    frame = pd.DataFrame(draws, columns=["imp", "bootstrap", "endpoint", "group", "risk", "rd"])
    keys = ["imp", "bootstrap", "endpoint", "group"]
    if (
        len(frame) != 10 * 51 * 3 * 3
        or frame.duplicated(keys).any()
        or not frame.risk.between(0, 1).all()
        or not np.isfinite(frame.rd).all()
    ):
        raise ValueError("Incomplete, duplicated, or invalid HRS bootstrap estimates")
    risk_wide = frame.pivot(index=keys[:3], columns="group", values="risk")
    rd_wide = frame.pivot(index=keys[:3], columns="group", values="rd")
    np.testing.assert_allclose(rd_wide, risk_wide.sub(risk_wide[0], axis=0))
    frame.to_csv(work / "aggregate_resamples.csv.gz", index=False)
    pd.DataFrame(counts).to_csv(work / "imputation_counts.csv", index=False)
    pd.DataFrame(missing).to_csv(work / "missing_scoring_inputs_by_imputation.csv", index=False)
    pd.DataFrame(by_year, columns=["imp", "index_year", "group", "n", "either_incomplete"]).to_csv(
        work / "imputation_counts_by_index_year.csv", index=False
    )
    pd.DataFrame(all_index, columns=["imp", "index_score_3plus_topcoded", "n"]).to_csv(
        work / "imputed_index_groups.csv", index=False
    )
    groups = ["endpoint", "group"]
    point = frame[frame.bootstrap.eq(0)].groupby(groups)[["risk", "rd"]].mean() * 100
    bootstrap = frame[frame.bootstrap.gt(0)].groupby(groups)[["risk", "rd"]]
    for suffix, pct in [("lo", 0.025), ("hi", 0.975)]:
        bounds = bootstrap.quantile(pct).mul(100).add_suffix("_" + suffix)
        point = point.join(bounds)
    point.reset_index().to_csv(work / "four_year_estimates.csv", index=False)
    from src.hrs.trends import pooled_trends

    pooled_trends(frame).to_csv(work / "trends.csv", index=False)


def locate_raw_data(package, explicit_dir=None):
    """Use the supplied HRS folder, shared raw-data root, or the project's data/HRS folder."""
    if explicit_dir is not None:
        return Path(explicit_dir).expanduser()
    shared_root = os.environ.get("FRAILTY_RAW_DATA")
    if shared_root:
        return Path(shared_root).expanduser() / "HRS"
    return next(
        (
            parent / "data/HRS"
            for parent in (package, *package.parents)
            if (parent / "data/HRS/randhrs1992_2022v1.dta").is_file()
        ),
        None,
    )


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Folder containing the RAND .dta and HRS Core/ archives",
    )
    parser.add_argument("--work-dir", type=Path, default=repo / "work/hrs")
    args = parser.parse_args()
    raw_dir = locate_raw_data(repo, args.raw_dir)
    if (
        raw_dir is None
        or not (raw_dir / "randhrs1992_2022v1.dta").is_file()
        or not (raw_dir / "Core").is_dir()
    ):
        parser.error(
            "HRS data missing. Provide --raw-dir /path/to/HRS or set FRAILTY_RAW_DATA to its parent."
        )
    with threadpool_limits(limits=1):
        fit(raw_dir, args.work_dir.expanduser())
