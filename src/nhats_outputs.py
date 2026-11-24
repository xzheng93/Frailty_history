"""Summarise NHATS participant characteristics, standardised risks and sensitivity analyses."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hrs.trends import pooled_trends
from src.nhats.analysis_lib import episodes, prepare

WORK = Path(__file__).resolve().parents[1] / "work/nhats"
PRIMARY = "frailty_or_death"
OUTCOMES = (PRIMARY, "falls", "hosp")
COMPONENTS = ("frailty", "death_before_frailty")
GROUPS = ("prior0", "prior1", "prior2plus")
JOINT = tuple(f"index{i}_{g}" for i in range(3) for g in GROUPS)
SENSITIVITIES = ("model1", "excl_index_round13", "age65_74", "age75_84", "age85plus")
IMPUTATIONS = list(range(1, 11))
DRAWS = pd.MultiIndex.from_product([IMPUTATIONS, range(51)], names=["imp", "b"])
LANDMARK_COLS = [
    "imp",
    "spid",
    "prior_score",
    "index_score",
    "index_round",
    "age_cat_final",
    "gender_final",
    "race_ethnicity",
    "education_c",
    "married_partnered",
    "current_smoker",
    "medicaid",
    "chronic9_count",
    "bmi_category",
    "W_Fc",
    "W_V",
    "t_death",
    "t_frail",
]


def save(table, path):
    if table.empty:
        raise ValueError(f"No rows for {path.name}")
    stage = path.with_suffix(".pending")
    table.to_csv(stage, index=False)
    stage.replace(path)


def check_imputations(df):
    if sorted(df.imp.unique()) != IMPUTATIONS:
        raise ValueError("Expected imputations 1 through 10")


def table_one(land):
    d = prepare(land.loc[land.index_score.eq(0)])
    d["female"] = d.gender.eq("2")
    characteristics = [
        (f"Age {label} years", d.index_age_cat.eq(str(code)))
        for code, label in enumerate(("65–69", "70–74", "75–79", "80–84", "85–89", "≥90"), 1)
    ]
    characteristics += [
        ("Female sex", d.female),
        ("White non-Hispanic", d.race_collapsed.eq("white")),
        ("Black non-Hispanic", d.race_collapsed.eq("black")),
        ("Hispanic", d.race_collapsed.eq("hispanic")),
        ("Other or unknown race/ethnicity", d.race_collapsed.eq("other")),
        ("Bachelor’s degree or higher", d.education_collapsed.eq("3")),
        ("Married or living with a partner", d.married_partnered.eq("1")),
        ("Current smoker", d.current_smoker.eq("1")),
        ("Medicaid coverage", d.medicaid.eq("1")),
        ("Three or more chronic conditions", d.chronic9_cat.eq("3plus")),
        ("BMI ≥30 kg/m²", d.bmi_category.eq("obese")),
    ]
    # The reported percentages use unrounded mean counts, as specified in Table 1.
    rows = []
    for label, mask in characteristics:
        for group in ("overall", *GROUPS):
            selected = d if group == "overall" else d.loc[d.prior_group.eq(group)]
            counts = selected.groupby("imp").size().reindex(IMPUTATIONS, fill_value=0)
            numerator = selected.loc[mask.loc[selected.index]].groupby("imp").size()
            mean_n = numerator.reindex(IMPUTATIONS, fill_value=0).mean()
            rows.append((label, group, mean_n, counts.mean(), 100 * mean_n / counts.mean()))
    result = pd.DataFrame(
        rows, columns=["characteristic", "group", "mean_n", "mean_denominator", "pct"]
    )
    for labels in ([l for l, _ in characteristics[:6]], [l for l, _ in characteristics[7:11]]):
        by_group = result[result.characteristic.isin(labels)].groupby("group")
        np.testing.assert_allclose(by_group.mean_n.sum(), by_group.mean_denominator.first())
    return result


def risk_sets(land):
    sample = land.loc[land.index_score.between(0, 2)].copy()
    sample["prior_group"] = sample.prior_score.clip(upper=2).map(dict(enumerate(GROUPS)))
    end, _, _ = episodes(sample, "fried", 4)
    index = pd.MultiIndex.from_product(
        [IMPUTATIONS, range(3), GROUPS], names=["imp", "index_score", "prior_group"]
    )
    rows = []
    for year in range(1, 5):
        at_risk = sample.loc[end >= year]
        counts = at_risk.groupby(index.names).size().reindex(index, fill_value=0)
        mean_counts = counts.groupby(["index_score", "prior_group"]).mean()
        for (idx, group), mean_n in mean_counts.items():
            rows.append((idx, group, year, mean_n))
    result = pd.DataFrame(rows, columns=["index_score", "prior_group", "year", "n_at_risk_mean"])
    if (result.groupby(["index_score", "prior_group"]).n_at_risk_mean.diff() > 0).any():
        raise ValueError("A joint risk set increased over follow-up")
    return result


def risk_draws(data, analysis, outcome, year, groups, *, points_only=False):
    subset = data.loc[data.analysis.eq(analysis) & data.outcome.eq(outcome) & data.year.eq(year)]
    if points_only:
        subset = subset.loc[subset.b.eq(0)]
    risks = subset.pivot(index=["imp", "b"], columns="group", values="value").sort_index()
    expected = DRAWS[DRAWS.get_level_values("b") == 0] if points_only else DRAWS
    if not risks.index.equals(expected) or set(risks.columns) != set(groups):
        raise ValueError(f"Incomplete fits for {analysis}, {outcome}, year {year}")
    risks = risks.loc[:, list(groups)]
    if not np.isfinite(risks).all().all() or (
        risks.min().min() < -1e-10 or risks.max().max() > 1 + 1e-10
    ):
        raise ValueError(f"Invalid risk in {analysis}, {outcome}, year {year}")
    return risks


def pooled(draws):
    """Mean across 10 original fits and percentile interval across 500 resamples, in % or pp."""
    points = draws.xs(0, level="b")
    bootstrap = draws.loc[draws.index.get_level_values("b") > 0]
    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
    return tuple(100 * x for x in (points.mean(), lo, hi))


def risk_table(risks_by_outcome, contrasts):
    rows = []
    for outcome, risks in risks_by_outcome.items():
        for group in GROUPS:
            rows.append((outcome, "risk", group, *pooled(risks[group])))
        for group in contrasts[outcome]:
            rows.append((outcome, "rd", group, *pooled(risks[group] - risks["prior0"])))
    return pd.DataFrame(
        rows, columns=["outcome", "measure", "group", "estimate", "ci_low", "ci_high"]
    )


def joint_outputs(main, land):
    results = []
    for year in range(1, 5):
        draws = risk_draws(main, "joint", PRIMARY, year, JOINT)
        for label in JOINT:
            idx, prior = label.split("_")
            results.append((int(idx[-1]), prior, year, *pooled(draws[label])))
    joint = pd.DataFrame(
        results, columns=["index_score", "prior_group", "year", "risk_pct", "ci_low", "ci_high"]
    )
    by_group = joint.groupby(["index_score", "prior_group"])
    if (by_group.risk_pct.diff() < -1e-8).any():
        raise ValueError("Joint cumulative risk decreased over follow-up")
    joint = joint.merge(
        risk_sets(land), validate="one_to_one", on=["index_score", "prior_group", "year"]
    )
    contrasts = []
    for ref in ("index1_prior0", "index1_prior1"):
        contrasts.append(
            (f"index0_prior2plus vs {ref}", *pooled(draws["index0_prior2plus"] - draws[ref]))
        )
    contrasts = pd.DataFrame(contrasts, columns=["comparison", "rd_pp", "ci_low", "ci_high"])
    return joint, contrasts


def stratified_trends(main):
    selected = main.loc[
        main.analysis.eq("main") & main.outcome.isin(OUTCOMES) & main.year.eq(4),
        ["imp", "b", "outcome", "group", "value"],
    ].copy()
    selected["group"] = selected.group.map({g: i for i, g in enumerate(GROUPS)})
    selected = selected.rename(columns={"b": "bootstrap", "outcome": "endpoint", "value": "risk"})
    trends = pooled_trends(selected).rename(
        columns={"endpoint": "outcome", "lo": "ci_low", "hi": "ci_high"}
    )
    return trends[["outcome", "trend_pp", "ci_low", "ci_high", "p_trend"]]


def cox_table(path):
    cox = pd.read_csv(path)
    rows = []
    for outcome in OUTCOMES:
        for group in GROUPS[1:]:
            fits = cox.loc[cox.outcome.eq(outcome) & cox.group.eq(group)].sort_values("imp")
            if (
                fits.imp.tolist() != IMPUTATIONS
                or not np.isfinite(fits[["logHR", "se"]]).all().all()
            ):
                raise ValueError(f"Incomplete Cox fits for {outcome}, {group}")
            if (fits.se <= 0).any():
                raise ValueError(f"Invalid within-imputation SE for {outcome}, {group}")
            mean = fits.logHR.mean()
            variance = fits.se.pow(2).mean() + (1 + 1 / len(fits)) * fits.logHR.var(ddof=1)
            margin = 1.96 * np.sqrt(variance)
            rows.append((outcome, group, *np.exp([mean, mean - margin, mean + margin])))
    return pd.DataFrame(rows, columns=["outcome", "group", "hr", "ci_low", "ci_high"])


def sensitivity(main, land):
    rows = []
    for analysis in SENSITIVITIES:
        draws = risk_draws(main, analysis, PRIMARY, 4, GROUPS)
        for group in GROUPS[1:]:
            rows.append((analysis, group, *pooled(draws[group] - draws["prior0"])))
    sensitivity_rd = pd.DataFrame(rows, columns=["analysis", "group", "rd_pp", "ci_low", "ci_high"])
    four = risk_draws(main, "four_level", PRIMARY, 4, ("prior0", "prior1", "prior2", "prior3plus"))
    n3 = len(land.loc[land.index_score.eq(0) & land.prior_score.ge(3)]) / len(IMPUTATIONS)
    severity = pd.DataFrame(
        [
            (g, *pooled(four[g]), n3 if g == "prior3plus" else np.nan)
            for g in ("prior2", "prior3plus")
        ],
        columns=["group", "risk_pct", "ci_low", "ci_high", "mean_n_prior3plus"],
    )
    six = risk_draws(main, "six_year", PRIMARY, 6, GROUPS, points_only=True)
    six_year = six.xs(0, level="b").mean().mul(100).rename("risk_pct").reset_index()
    return sensitivity_rd, severity, six_year


def model_tables(work, land):
    main = pd.read_csv(work / "main_draws.csv.gz")
    complete = pd.read_csv(work / "complete_draws.csv.gz")
    main_risks = {o: risk_draws(main, "main", o, 4, GROUPS) for o in OUTCOMES}
    first_event = {o: risk_draws(main, "main", o, 4, GROUPS) for o in COMPONENTS}
    np.testing.assert_allclose(
        first_event["frailty"] + first_event["death_before_frailty"], main_risks[PRIMARY], atol=1e-9
    )
    complete_risks = {o: risk_draws(complete, "main", o, 4, GROUPS) for o in OUTCOMES}
    curves, differences = joint_outputs(main, land)
    std = pd.read_csv(work / "joint_index0_standardisation.csv")
    std = std.loc[std.group.eq("index0_prior2plus") & std.year.eq(4)].sort_values("imp")
    if std.imp.tolist() != IMPUTATIONS or not std.risk.between(0, 1).all():
        raise ValueError("Expected one index-0-standardised fourth-year risk per imputation")
    std_row = pd.DataFrame(
        [(0, "prior2plus", 4, 100 * std.risk.mean())],
        columns=["index_score", "prior_group", "year", "risk_pct"],
    )
    rd, severity, six_year = sensitivity(main, land)
    return {
        "table2": risk_table(main_risks, dict.fromkeys(OUTCOMES, GROUPS[1:])),
        "trends": stratified_trends(main),
        "joint_curves": curves,
        "s3_contrasts": differences,
        "s3_standardisation": std_row,
        "s1_components": risk_table(first_event, dict.fromkeys(COMPONENTS, ("prior2plus",))),
        "s2_complete": risk_table(
            complete_risks, {o: GROUPS[1:] if o == PRIMARY else ("prior2plus",) for o in OUTCOMES}
        ),
        "s4_cox": cox_table(work / "cox_by_imputation.csv"),
        "s5_rd": rd,
        "s5_prior_severity": severity,
        "s5_six_year": six_year,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work", type=Path, default=WORK, help="Directory of newly generated NHATS files"
    )
    parser.add_argument(
        "--descriptive-only", action="store_true", help="Export Table 1 before model fits finish"
    )
    args = parser.parse_args()
    work = args.work
    land = pd.read_csv(work / "landmarks_imputed.csv.gz", usecols=LANDMARK_COLS)
    check_imputations(land)
    if land.duplicated(["imp", "spid"]).any() or land.groupby("spid").imp.nunique().ne(10).any():
        raise ValueError("The fixed eligible pair must give one row per person and imputation")
    out = work / "tables"
    out.mkdir(exist_ok=True)
    save(table_one(land), out / "table1.csv")
    if not args.descriptive_only:
        for name, table in model_tables(work, land).items():
            save(table, out / f"{name}.csv")


if __name__ == "__main__":
    main()
