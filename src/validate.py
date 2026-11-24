"""Cross-check reported results against fixed pairs and paired participant resamples."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = ("frailty_or_death", "falls", "hosp")
GROUPS = ("prior0", "prior1", "prior2plus")


def equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)


def point_ci(values):
    return [values[:, 0].mean(), *np.percentile(values[:, 1:], [2.5, 97.5])]


def resample_array(frame, groups):
    """Require precisely the same 50 resample IDs in every imputation and group."""
    wide = frame.pivot(index=["imp", "bootstrap"], columns="group", values="risk")
    expected = pd.MultiIndex.from_product([range(1, 11), range(51)], names=["imp", "bootstrap"])
    if not wide.index.equals(expected) or set(wide.columns) != set(groups):
        raise ValueError("Missing resample IDs or groups")
    values = 100 * wide[list(groups)].to_numpy().reshape(10, 51, len(groups))
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise ValueError("Invalid resample probability")
    return values


def trend_interval(risks):
    """Find the minimum-variance unbiased linear slope using its single free coefficient."""
    points = risks[:, 0, :]
    within = np.mean([np.cov(batch.T) for batch in risks[:, 1:, :]], axis=0)
    between = 1.1 * np.cov(points.T)
    total = within + between
    # All three-group slope contrasts have the form [-1, 1, 0] + a * [1, -2, 1].
    first = np.array([-1, 1, 0])
    curvature = np.array([1, -2, 1])
    a = -(curvature @ total @ first) / (curvature @ total @ curvature)
    contrast = first + a * curvature
    slope = (points @ contrast).mean()
    variance = contrast @ total @ contrast
    between_variance = contrast @ between @ contrast
    df = 9 * (variance / between_variance) ** 2 if between_variance > 0 else np.inf
    se = np.sqrt(variance)
    radius = t.isf(0.025, df) * se
    return [slope, slope - radius, slope + radius, 2 * t.sf(abs(slope) / se, df)]


def check_main(work, cohort):
    if cohort == "nhats":
        frame = pd.read_csv(work / "main_draws.csv.gz")
        draws = frame[frame.analysis.eq("main")].rename(
            columns={"b": "bootstrap", "outcome": "endpoint", "value": "risk"}
        )
        point = pd.read_csv(work / "tables/table2.csv").set_index(["outcome", "measure", "group"])
        trend = pd.read_csv(work / "tables/trends.csv").set_index("outcome")
        groups = GROUPS
    else:
        draws = pd.read_csv(work / "aggregate_resamples.csv.gz")
        point = pd.read_csv(work / "four_year_estimates.csv").set_index(["endpoint", "group"])
        trend = (
            pd.read_csv(work / "trends.csv")
            .rename(columns={"lo": "ci_low", "hi": "ci_high"})
            .set_index("endpoint")
        )
        groups = (0, 1, 2)
    for endpoint in OUTCOMES:
        data = resample_array(draws[draws.endpoint.eq(endpoint)], groups)
        equal(
            trend.loc[endpoint, ["trend_pp", "ci_low", "ci_high", "p_trend"]], trend_interval(data)
        )
        for j, group in enumerate(groups):
            for measure in ("risk", "rd") if j else ("risk",):
                values = data[:, :, j]
                if measure == "rd":
                    values = values - data[:, :, 0]
                if cohort == "nhats":
                    row = point.loc[(endpoint, measure, group), ["estimate", "ci_low", "ci_high"]]
                else:
                    row = point.loc[(endpoint, group), [measure, f"{measure}_lo", f"{measure}_hi"]]
                equal(row, point_ci(values))
    if cohort == "nhats":
        components = [
            resample_array(draws[draws.endpoint.eq(endpoint)], groups)
            for endpoint in ("frailty", "death_before_frailty", "frailty_or_death")
        ]
        equal(components[0] + components[1], components[2])
        first_event = pd.read_csv(work / "tables/s1_components.csv")
        if len(first_event) != 8:
            raise ValueError("Expected 6 first-event risks and 2 paired differences")
        for endpoint, component in zip(("frailty", "death_before_frailty"), components[:2]):
            for row in first_event[first_event.outcome.eq(endpoint)].itertuples():
                values = component[:, :, groups.index(row.group)]
                if row.measure == "rd":
                    values = values - component[:, :, 0]
                equal([row.estimate, row.ci_low, row.ci_high], point_ci(values))
        return frame


def check_nhats(work, frame):
    cols = [
        "spid",
        "imp",
        "index_score",
        "prior_score",
        "index_round",
        "W_V",
        "W_Fc",
        "t_frail",
        "t_death",
        "gender_final",
        "age_cat_final",
    ]
    land = pd.read_csv(work / "landmarks_imputed.csv.gz", usecols=cols)
    counts = land.groupby("spid").agg(
        n=("imp", "size"), imps=("imp", "nunique"), rounds=("index_round", "nunique")
    )
    if not (counts.n.eq(10) & counts.imps.eq(10) & counts.rounds.eq(1)).all():
        raise ValueError("Index pair or imputed participation changed between imputations")
    robust = land[land.index_score.eq(0)]
    table1 = pd.read_csv(work / "tables/table1.csv").set_index(["characteristic", "group"])
    for j, group in enumerate([*GROUPS, "overall"]):
        strata = robust if group == "overall" else robust[robust.prior_score.clip(upper=2).eq(j)]
        for label, mask in [
            ("Female sex", strata.gender_final.eq(2)),
            ("Age 65–69 years", strata.age_cat_final.eq(1)),
        ]:
            equal(
                table1.loc[(label, group), ["mean_n", "mean_denominator", "pct"]],
                [mask.sum() / 10, len(strata) / 10, mask.mean() * 100],
            )
    joint = pd.read_csv(work / "tables/joint_curves.csv")
    contrasts = pd.read_csv(work / "tables/s3_contrasts.csv").set_index("comparison")
    all_groups = [f"index{i}_{g}" for i in range(3) for g in GROUPS]
    event_draws = frame[frame.analysis.eq("joint")].rename(
        columns={"b": "bootstrap", "value": "risk"}
    )
    for year in range(1, 5):
        subset = event_draws[event_draws.year.eq(year)]
        data = resample_array(subset, all_groups)
        for i, row in (
            joint[joint.year.eq(year)].set_index(["index_score", "prior_group"]).iterrows()
        ):
            index_score, group = i
            position = all_groups.index(f"index{index_score}_{group}")
            equal(row[["risk_pct", "ci_low", "ci_high"]], point_ci(data[:, :, position]))
            d = land[
                land.index_score.eq(index_score)
                & land.prior_score.clip(upper=2).eq(GROUPS.index(group))
            ]
            entered = (d.t_frail.isna() | d.t_frail.ge(year)) & (
                d.t_death.isna() | d.t_death.ge(year)
            )
            observed = d.W_V.ge(year) | d.t_death.eq(year)
            equal(row.n_at_risk_mean, (entered & observed).sum() / 10)
    for ref in ("index1_prior0", "index1_prior1"):
        expected = point_ci(
            data[:, :, all_groups.index("index0_prior2plus")] - data[:, :, all_groups.index(ref)]
        )
        equal(
            contrasts.loc[f"index0_prior2plus vs {ref}", ["rd_pp", "ci_low", "ci_high"]], expected
        )
    fits = pd.read_csv(work / "cox_by_imputation.csv")
    cox = pd.read_csv(work / "tables/s4_cox.csv").set_index(["outcome", "group"])
    for key, batch in fits.groupby(["outcome", "group"]):
        if len(batch) != 10 or batch.imp.nunique() != 10:
            raise ValueError("Expected ten independent Cox estimates per comparison")
        radius = 1.96 * np.sqrt(np.mean(batch.se**2) + 1.1 * np.var(batch.logHR, ddof=1))
        central = batch.logHR.mean()
        equal(
            cox.loc[key, ["hr", "ci_low", "ci_high"]],
            np.exp([central, central - radius, central + radius]),
        )


def check_cohort_flow(work):
    import json

    nhats, hrs = work / "nhats", work / "hrs"
    flows = [
        (nhats / "cohort_flow.csv", "n_remaining", "n_excluded"),
        (hrs / "flow.csv", "remaining_people", "excluded_people"),
    ]
    for filename, remaining, excluded in flows:
        flow = pd.read_csv(filename)
        equal(-flow[remaining].diff().dropna(), flow[excluded].iloc[1:])
    final_hrs = flow[remaining].iat[-1]
    no_exam = pd.read_csv(hrs / "pair_exam_exclusions.csv")
    equal(no_exam.n.sum(), flow[excluded].iat[-1])
    if not no_exam.waves_without_exam.eq(
        no_exam.wave_without_exam.eq("both").astype(int) + 1
    ).all():
        raise ValueError("Inconsistent one-wave and two-wave exam exclusions")
    index = pd.read_csv(hrs / "imputed_index_groups.csv")
    totals = index.groupby("imp").n.sum()
    equal(totals, np.repeat(final_hrs, 10))
    missed = pd.read_csv(hrs / "missing_scoring_inputs_by_imputation.csv")
    by_imp = missed.groupby("imp").sum(numeric_only=True)
    equal(by_imp.n, index[index.index_score_3plus_topcoded.eq(0)].sort_values("imp").n)
    if (
        missed.either_missing_scoring_input
        < missed[["prior_missing_scoring_input", "index_missing_scoring_input"]].max(axis=1)
    ).any():
        raise ValueError("A missing HRS input at either visit was dropped")
    if (missed.either_unresolved_component_score > missed.either_missing_scoring_input).any():
        raise ValueError("Unresolved HRS components despite complete scoring inputs")
    panel = pd.read_pickle(nhats / "all_interviews.pkl.gz").set_index(["spid", "round"])
    inputs = [
        "speed",
        "grip",
        "low_pa",
        "exhaustion",
        "unint_wl",
        "bmi",
        "height_cm",
        "gender_final",
    ]
    pairs = pd.read_csv(nhats / "fixed_pairs.csv.gz")
    included = set(pairs.spid)
    for window in ("prior_round", "index_round"):
        idx = list(zip(pairs.spid, pairs[window]))
        records = panel.loc[idx, inputs].copy()
        included &= set(records.dropna().index.get_level_values("spid"))
    complete = pd.read_csv(nhats / "complete_pair_scores.csv.gz")
    if included != set(complete.spid) or len(complete) != len(included):
        raise ValueError("Incorrect two-round complete-case population")
    reported = json.loads((nhats / "input_summary.json").read_text())
    robust = complete[complete.index_score.eq(0)]
    equal(
        [len(complete), len(robust)], [reported["n_complete_pairs"], reported["n_complete_robust"]]
    )
    for score, n in robust.prior_score.clip(upper=2).value_counts().items():
        equal(n, reported["n_complete_robust_by_prior"][str(int(score))])
    land = pd.read_csv(nhats / "landmarks_imputed.csv.gz", usecols=["imp", "spid", "index_score"])
    rows = land[land.index_score.eq(0)]
    missing = rows[~rows.spid.isin(included)].groupby("imp").size()
    counts = pd.read_csv(nhats / "imputation_counts.csv").set_index("imp")
    equal(missing, counts.incomplete_among_robust)
    equal(100 * missing.sum() / len(rows), reported["ratio_of_mean_counts_percent"])
    nursing_home = panel[panel.resid.eq(4)].reset_index()
    index_rounds = pairs.set_index("spid").index_round
    after_index = nursing_home[nursing_home["round"].gt(nursing_home.spid.map(index_rounds))]
    equal(
        [
            len(nursing_home),
            nursing_home.assessed.sum(),
            len(after_index),
            after_index.assessed.sum(),
        ],
        [
            reported[k]
            for k in (
                "nursing_home_interviews",
                "nursing_home_assessable",
                "post_index_nursing_home_interviews",
                "post_index_nursing_home_assessable",
            )
        ],
    )


def check_sensitivities(work, main):
    complete = pd.read_csv(work / "complete_draws.csv.gz")
    s2 = pd.read_csv(work / "tables/s2_complete.csv")
    if len(s2) != 13:
        raise ValueError(
            "Expected 9 risks and 4 differences in the complete-case sensitivity analysis"
        )
    for outcome, sub in s2.groupby("outcome"):
        batch = complete[
            complete.analysis.eq("main") & complete.outcome.eq(outcome) & complete.year.eq(4)
        ]
        risks = resample_array(batch.rename(columns={"b": "bootstrap", "value": "risk"}), GROUPS)
        for row in sub.itertuples():
            values = risks[:, :, GROUPS.index(row.group)]
            if row.measure == "rd":
                values = values - risks[:, :, 0]
            equal([row.estimate, row.ci_low, row.ci_high], point_ci(values))
    s5 = pd.read_csv(work / "tables/s5_rd.csv")
    if len(s5) != 10:
        raise ValueError("Expected two paired RDs in each of the five sensitivity analyses")
    main = main[main.outcome.eq("frailty_or_death")].rename(
        columns={"b": "bootstrap", "value": "risk"}
    )
    for analysis, subset in s5.groupby("analysis"):
        risks = resample_array(main[main.analysis.eq(analysis) & main.year.eq(4)], GROUPS)
        for row in subset.itertuples():
            values = risks[:, :, GROUPS.index(row.group)] - risks[:, :, 0]
            equal([row.rd_pp, row.ci_low, row.ci_high], point_ci(values))
    severity = pd.read_csv(work / "tables/s5_prior_severity.csv")
    levels = ("prior0", "prior1", "prior2", "prior3plus")
    risks = resample_array(main[main.analysis.eq("four_level") & main.year.eq(4)], levels)
    for row in severity.itertuples():
        equal(
            [row.risk_pct, row.ci_low, row.ci_high], point_ci(risks[:, :, levels.index(row.group)])
        )
    land = pd.read_csv(work / "landmarks_imputed.csv.gz", usecols=["index_score", "prior_score"])
    equal(
        severity.loc[severity.group.eq("prior3plus"), "mean_n_prior3plus"].item(),
        (land.index_score.eq(0) & land.prior_score.ge(3)).sum() / 10,
    )
    six = main[main.analysis.eq("six_year") & main.year.eq(6) & main.bootstrap.eq(0)]
    reported = pd.read_csv(work / "tables/s5_six_year.csv").set_index("group")
    equal(reported.risk_pct, six.groupby("group").risk.mean().reindex(reported.index) * 100)


def check_share(root):
    draws = pd.read_csv(root / "aggregate_resamples.csv.gz")
    summaries = pd.concat(
        [
            pd.read_csv(root / f)
            for f in ("four_year_estimates.csv", "followup_report_estimates.csv")
        ]
    )
    keys = ["analysis", "endpoint", "group"]
    summaries = summaries.set_index(keys, verify_integrity=True)
    trends = pd.read_csv(root / "trends.csv").set_index(keys[:-1], verify_integrity=True)
    flow = pd.read_csv(root / "flow.csv")
    by_wave = pd.read_csv(root / "fixed_pairs_by_wave.csv")
    counts = pd.read_csv(root / "counts_by_imputation.csv")
    partition = ["death48", "scored_survivor", "survivor_without_score", "censored_before_48"]
    equal(counts[partition].sum(axis=1), counts.n)
    schedules = {"main": {2, 5, 6, 7}, "include_long_gap": {2, 4, 5, 6, 7}}
    for design, waves in schedules.items():
        fixed = by_wave[by_wave.analysis.eq(design)]
        assert (
            set(fixed.index_wave) == waves
            and (fixed.target_ffp_assessable <= fixed.target_interview).all()
        )
        design_flow = flow[flow.analysis.eq(design)]
        assert design_flow.n_people.is_monotonic_decreasing
        equal(design_flow.n_people.iloc[-1], fixed.fixed_people.sum())
        n = counts[counts.analysis.eq(design)].groupby(["imp", "index_wave"]).n.sum().unstack()
        assert n.index.tolist() == list(range(1, 11))
        assert n.le(fixed.set_index("index_wave").fixed_people, axis=1).all().all()
        for endpoint in ("frailty_or_death", "falling_symptom", "overnight_hospital_12m"):
            sub = draws[draws.analysis.eq(design) & draws.endpoint.eq(endpoint)]
            risks = resample_array(sub, (0, 1, 2))
            equal(
                trends.loc[(design, endpoint), ["trend_pp", "lo", "hi", "p_trend"]],
                trend_interval(risks),
            )
            for g in range(3):
                reported = summaries.loc[(design, endpoint, g)]
                for measure, vals in (
                    ("risk", risks[:, :, g]),
                    ("rd", risks[:, :, g] - risks[:, :, 0]),
                ):
                    equal(reported[[measure, measure + "_lo", measure + "_hi"]], point_ci(vals))
    print(
        "Validated SHARE all-wave participant counts, endpoint denominators, paired intervals and six trends."
    )


def main(share_results=None):
    work = ROOT / "work"
    nhats = check_main(work / "nhats", "nhats")
    check_main(work / "hrs", "hrs")
    check_nhats(work / "nhats", nhats)
    check_cohort_flow(work)
    check_sensitivities(work / "nhats", nhats)
    if share_results is not None:
        check_share(share_results)
    print(
        "Validated paired 500-resample intervals, six Rubin-pooled trends, first-event "
        "decomposition, joint curves and risk sets, baseline and missing-input counts, "
        "complete-case and protocol sensitivity analyses, and adjusted Cox HRs."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share-results", type=Path)
    main(parser.parse_args().share_results)
