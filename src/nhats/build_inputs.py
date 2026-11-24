"""Build NHATS analytic inputs directly from SP interviews and the Round 14 tracker."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
import numpy as np
import pandas as pd
from statsmodels.imputation.mice import MICEData
from src.nhats import config
from src.nhats.covariates import read_round, read_tracker
from src.nhats import followup, imputation
from src.nhats.phenotype import carry_height, extract_round, sp_file

KEY = ["spid", "round"]
INPUTS = ["speed", "grip", "exhaustion", "low_pa", "unint_wl", "height_cm", "bmi", "gender_final"]
COV = [
    "age_cat_final",
    "gender_final",
    "race_ethnicity",
    "married_partnered",
    "current_smoker",
    "analytic_weight_final",
    "survey_var_stratum",
    "survey_var_unit",
]


def prepare_raw():
    config.WORK.mkdir(parents=True, exist_ok=True)
    config.DATA.mkdir(exist_ok=True)
    config.RESULTS.mkdir(exist_ok=True)
    phenotype = []
    covariates = []
    for r in config.ROUNDS:
        phenotype.append(extract_round(r))
        if r > 1:
            phenotype.append(extract_round(r, nursing_home=True))
        covariates.append(read_round(sp_file(r), r))
        print("Read raw NHATS round", r, flush=True)
    pv = pd.concat(phenotype, ignore_index=True)
    cv = pd.concat(covariates, ignore_index=True).sort_values(KEY)
    static = cv.groupby("spid")[["gender_final", "race_ethnicity", "education"]].first()
    cv["gender_final"] = cv.gender_final.fillna(cv.spid.map(static.gender_final))
    for col in ("race_ethnicity", "education"):
        cv[col] = cv.spid.map(static[col])
    full = (
        pv.merge(cv, on=KEY, how="left", validate="one_to_one")
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    edu = full.education
    full["education_c"] = np.select(
        [edu.le(4), edu.isin([5, 6, 7]), edu.ge(8)], [1.0, 2.0, 3.0], np.nan
    )
    tracker, death = read_tracker()
    full["death_round"] = full.spid.map(death)
    community = full.resid.isin([1, 2])
    for rows, scope in [(community, full[community]), (~community, full)]:
        heights = carry_height(scope)
        full.loc[rows, "height_cm"] = heights.reindex(full.index[rows])
    full["bmi"] = (703 * full.weight_lb / (full.height_cm / 2.54) ** 2).where(
        lambda s: s.between(10, 80)
    )
    full["n_observed"] = full[INPUTS[:5]].notna().sum(axis=1)
    full["assessed"] = ~full["round"].isin(config.NO_PERFORMANCE_ROUNDS) & full.n_observed.ge(
        config.MIN_OBSERVED_COMPONENTS
    )
    assert not full.duplicated(KEY).any() and not (full["round"] >= full.death_round).any()
    assert full.gender_final.isin([1, 2]).all()
    # Read back the CSV so community PMM donors use the same numeric precision on reruns.
    path = config.DATA / "br_panel_preimputation.csv.gz"
    full.loc[community].to_csv(path, index=False)
    outpatient = pd.read_csv(path)
    full = (
        pd.concat([outpatient, full.loc[~community]], ignore_index=True)
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    # Heights at post-index nursing-home visits use the serialized community history.
    nh = full.resid.eq(4)
    full.loc[nh, "height_cm"] = carry_height(full).loc[nh]
    full.loc[nh, "bmi"] = (
        (703 * full.weight_lb / (full.height_cm / 2.54) ** 2)
        .where(lambda v: v.between(10, 80))
        .loc[nh]
    )
    full.to_pickle(config.WORK / "all_interviews.pkl.gz")
    return full


def impute_one_round(r):
    full = pd.read_pickle(config.WORK / "all_interviews.pkl.gz")
    one = full.loc[full["round"].eq(r)].reset_index(drop=True)
    out = one.loc[one.resid.isin([1, 2])].reset_index(drop=True)
    nursing = one.resid.eq(4)
    nh_eligible = nursing & one.assessed
    if r in config.NO_PERFORMANCE_ROUNDS:
        return r
    pieces = []
    for m in range(1, config.N_IMP + 1):
        done = imputation.impute_round(out, m, r)
        filled = out.copy()
        # Carry the PMM draws into modeling when clinical predictor values were originally missing.
        for c in [*imputation.IMPUTE, "married_partnered", "current_smoker"]:
            filled[c] = done[c]
        selected = filled.loc[filled.assessed]
        keep = selected[
            [
                *KEY,
                "education_c",
                "medicaid",
                "chronic9_count",
                "married_partnered",
                "current_smoker",
            ]
        ].copy()
        keep = pd.concat([keep, imputation.score(selected)], axis=1)
        keep["imp"] = m
        if nh_eligible.any():
            x = imputation.design(one)
            x["nursing_home"] = nursing.astype(float)
            x = x.loc[:, x.notna().any()]
            np.random.seed(config.SEED + 1000 * m + r)
            sampler = MICEData(x.copy(), perturbation_method="gaussian", k_pmm=10)
            sampler.update_all(config.MICE_ITER)
            added = one.loc[nh_eligible].copy()
            for c in imputation.IMPUTE:
                added[c] = sampler.data.loc[nh_eligible, c].to_numpy()
            ns = added[KEY].copy()
            ns["score"] = imputation.score(added).score
            ns["imp"] = m
            keep = pd.concat([keep, ns], ignore_index=True)
        pieces.append(keep)
    frame = pd.concat(pieces, ignore_index=True)
    path = config.WORK / f"scores_r{r}.pkl"
    tmp = path.with_suffix(".partial")
    frame.to_pickle(tmp)
    tmp.replace(path)
    return r


def impute_all(workers):
    rounds = [r for r in config.ROUNDS if r not in config.NO_PERFORMANCE_ROUNDS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(impute_one_round, r) for r in rounds]
        for fut in as_completed(jobs):
            print(
                "Imputed community and eligible nursing-home interviews in round",
                fut.result(),
                flush=True,
            )


def first_event(flag, window):
    positive = flag & window
    return np.where(positive.any(axis=1), positive.argmax(axis=1) + 1, np.nan)


def fixed_pairs():
    """Fix earliest eligible scheduled annual pair using pre-imputation information."""
    p = pd.read_pickle(config.WORK / "all_interviews.pkl.gz")
    ids_source = set()
    for r in config.ROUNDS:
        ids_source.update(
            pd.read_stata(sp_file(r), columns=["spid"], convert_categoricals=False).spid
        )
    scheduled = p.loc[~p["round"].isin(config.NO_PERFORMANCE_ROUNDS)]
    prior = scheduled[KEY + ["resid", "assessed", "age_cat_final"]].rename(
        columns={
            "round": "prior_round",
            "resid": "prior_resid",
            "assessed": "prior_assessed",
            "age_cat_final": "prior_age_cat",
        }
    )
    prior["round"] = prior.prior_round + 1
    pairs = scheduled.merge(prior, on=KEY, validate="one_to_one")
    end = p.groupby("spid")["round"].max()
    stages = {
        "consecutive": pairs["round"].notna(),
        "outside": pairs.resid.isin([1, 2]) & pairs.prior_resid.isin([1, 2]),
        "three_components": pairs.assessed & pairs.prior_assessed,
        "subsequent_interview_or_death": pairs.spid.map(end).gt(pairs["round"])
        | pairs.death_round.gt(pairs["round"]),
        "age65": pairs.prior_age_cat.ge(1),
    }
    rows = [("source", len(ids_source), 0)]
    selected = pairs
    changes = {}
    for label, condition in stages.items():
        selected = selected.loc[condition.reindex(selected.index)]
        left = selected.spid.nunique()
        changes[label] = int(rows[-1][1] - left)
        rows.append((label, left, changes[label]))
    pd.DataFrame(rows, columns=["step", "n_remaining", "n_excluded"]).to_csv(
        config.WORK / "cohort_flow.csv", index=False
    )
    b = (
        selected.sort_values(KEY)
        .drop_duplicates("spid")
        .reset_index(drop=True)
        .rename(columns={"round": "index_round"})
    )
    ids = b.spid.to_numpy()
    ix = b.index_round.to_numpy()
    offsets = np.arange(1, len(config.ROUNDS) + 1)
    rr = ix[:, None] + offsets
    keys = pd.MultiIndex.from_arrays([np.repeat(ids, len(offsets)), rr.ravel()], names=KEY)
    tab = p.set_index(KEY)

    def matrix(col):
        return tab[col].reindex(keys).to_numpy(dtype=float, na_value=np.nan).reshape(rr.shape)

    assess = matrix("assessed")
    person = np.logical_and.accumulate(np.isfinite(assess), axis=1)
    phenotype = np.logical_and.accumulate(assess == 1, axis=1)
    b["W_Fc"] = phenotype.sum(axis=1)
    b["W_Rc"] = person.sum(axis=1)
    alive = followup.known_alive_rounds()
    b["W_V"] = [max(0, max(alive.get(i, []) + [int(end[i])]) - r) for i, r in zip(ids, ix)]
    b["t_death"] = b.death_round - b.index_round
    assert b.t_death.dropna().gt(0).all() and (b.W_V >= b.W_Rc).all() and (b.W_Rc >= b.W_Fc).all()
    for field, event in [("multiple_falls", "t_falls"), ("hospital_stay", "t_hosp")]:
        b[event] = first_event(matrix(field) == 1, person)
    index_key = pd.MultiIndex.from_arrays([ids, ix], names=KEY)
    prior_key = pd.MultiIndex.from_arrays([ids, b.prior_round], names=KEY)
    pa = tab.loc[prior_key, INPUTS].set_axis(ids)
    ia = tab.loc[index_key, INPUTS].set_axis(ids)
    complete = ia.notna().all(axis=1) & pa.notna().all(axis=1)
    cc = pd.DataFrame(
        {
            "prior_score": imputation.score(pa[complete]).score,
            "index_score": imputation.score(ia[complete]).score,
        }
    )
    cc0 = cc.loc[cc.index_score.eq(0)].prior_score.clip(upper=2).value_counts().sort_index()
    nh = p.resid.eq(4)
    ixmap = p.spid.map(b.set_index("spid").index_round)
    post = nh & p["round"].gt(ixmap)
    n = nh & p.assessed
    counts = {
        "source": len(ids_source),
        "excluded_by_step": changes,
        "n_pairs": len(b),
        "all_settings_interviews": len(p),
        "nursing_home_interviews": int(nh.sum()),
        "nursing_home_assessable": int(n.sum()),
        "nursing_home_R10": int((nh & p["round"].eq(10)).sum()),
        "nursing_home_other_insufficient": int((nh & ~p.assessed & ~p["round"].eq(10)).sum()),
        "nursing_home_eligible_complete_score_inputs": int(
            p.loc[n, INPUTS].notna().all(axis=1).sum()
        ),
        "nursing_home_eligible_partial_score_inputs": int(
            p.loc[n, INPUTS].isna().any(axis=1).sum()
        ),
        "post_index_nursing_home_interviews": int(post.sum()),
        "post_index_nursing_home_assessable": int((post & p.assessed).sum()),
        "n_complete_pairs": len(cc),
        "n_complete_robust": int(cc.index_score.eq(0).sum()),
        "n_complete_robust_by_prior": {str(int(k)): int(v) for k, v in cc0.items()},
        "n_incomplete_pairs": int((~complete).sum()),
        "fixed_index_observed_missing": {
            c: int(b[c].isna().sum())
            for c in COV + ["education_c", "medicaid", "chronic9_count", "bmi"]
        },
    }
    fixed = b[
        [
            "spid",
            "prior_round",
            "index_round",
            "prior_age_cat",
            "W_Fc",
            "W_Rc",
            "W_V",
            "t_death",
            "t_falls",
            "t_hosp",
            *COV,
        ]
    ].copy()
    fixed.to_csv(config.WORK / "fixed_pairs.csv.gz", index=False)
    cc.rename_axis("spid").to_csv(config.WORK / "complete_pair_scores.csv.gz")
    (config.WORK / "eligibility_summary.json").write_text(
        json.dumps(counts, indent=2, allow_nan=False) + "\n"
    )
    return fixed, index_key, prior_key, keys, phenotype, cc, counts


def finish():
    base = pd.read_csv(config.WORK / "fixed_pairs.csv.gz")
    cc = pd.read_csv(config.WORK / "complete_pair_scores.csv.gz").set_index("spid")
    counts = json.loads((config.WORK / "eligibility_summary.json").read_text())
    ik = pd.MultiIndex.from_arrays([base.spid, base.index_round], names=KEY)
    pk = pd.MultiIndex.from_arrays([base.spid, base.prior_round], names=KEY)
    offsets = np.arange(1, len(config.ROUNDS) + 1)
    rounds = base.index_round.to_numpy()[:, None] + offsets
    keys = pd.MultiIndex.from_arrays(
        [np.repeat(base.spid.to_numpy(), len(offsets)), rounds.ravel()], names=KEY
    )
    window = offsets <= base.W_Fc.to_numpy()[:, None]
    chunks = pd.concat(
        [
            pd.read_pickle(config.WORK / f"scores_r{r}.pkl")
            for r in config.ROUNDS
            if r not in config.NO_PERFORMANCE_ROUNDS
        ],
        ignore_index=True,
    )
    assert not chunks.duplicated(["imp", *KEY]).any()
    full = pd.read_pickle(config.WORK / "all_interviews.pkl.gz")
    r1 = full.loc[full["round"].eq(1) & full.assessed, ["spid", "analytic_weight_final"]]
    first = chunks.loc[chunks["round"].eq(1)].merge(r1, on="spid", validate="many_to_one")
    weighted = []
    for m, d in first.groupby("imp"):
        w = d.analytic_weight_final
        row = {"imp": int(m)}
        for state, indicator in [
            ("robust", d.score.eq(0)),
            ("prefrail", d.score.between(1, 2)),
            ("frail", d.score.ge(3)),
        ]:
            row[state] = 100 * np.average(indicator, weights=w)
        weighted.append(row)
    val = pd.DataFrame(weighted).set_index("imp")
    val.to_csv(config.RESULTS / "round1_weighted_prevalence.csv")
    counts["round1_assessable"] = len(r1)
    counts["round1_weighted_phenotype_percent"] = val.mean().to_dict()
    frames = []
    summaries = []
    for m, part in chunks.groupby("imp"):
        score = part.set_index(KEY)
        current = score.loc[ik]
        prior = score.loc[pk]
        out = base.copy()
        out["index_score"] = current.score.to_numpy()
        out["prior_score"] = prior.score.to_numpy()
        assert out[["index_score", "prior_score"]].notna().all().all()
        for c in config.COMPONENTS:
            out["prior_" + c] = prior[c].to_numpy()
        for c in [
            "education_c",
            "medicaid",
            "chronic9_count",
            "bmi_category",
            "married_partnered",
            "current_smoker",
        ]:
            out[c] = current[c].to_numpy()
        sequence = score.score.reindex(keys).to_numpy().reshape(window.shape)
        assert np.isfinite(sequence[window]).all()
        out["t_frail"] = first_event(sequence >= 3, window)
        out.insert(0, "imp", m)
        c0 = out.loc[out.index_score.eq(0)]
        pgroup = c0.prior_score.clip(upper=2)
        missing = int((~c0.spid.isin(cc.index)).sum())
        summaries.append(
            {
                "imp": int(m),
                "robust": len(c0),
                "incomplete_among_robust": missing,
                "incomplete_percent_among_robust": 100 * missing / len(c0),
                **{f"robust_prior{g}": int(pgroup.eq(g).sum()) for g in range(3)},
                **{f"index{g}": int(out.index_score.clip(upper=3).eq(g).sum()) for g in range(4)},
                **{
                    f"index0_raw_missing_{c}": int(base.loc[c0.index, c].isna().sum())
                    for c in ["married_partnered", "current_smoker", "race_ethnicity"]
                },
            }
        )
        frames.append(out)
    pooled = pd.concat(frames, ignore_index=True)
    stats = pd.DataFrame(summaries).set_index("imp")
    stats.to_csv(config.WORK / "imputation_counts.csv")
    counts["imputation_counts"] = stats.agg(["mean", "min", "max"]).to_dict()
    counts["ratio_of_mean_counts_percent"] = (
        100 * stats.incomplete_among_robust.mean() / stats.robust.mean()
    )
    restricted = pooled[pooled.spid.isin(cc.index)].copy()
    for c in ("index_score", "prior_score"):
        assert restricted[c].eq(restricted.spid.map(cc[c])).all()
    for name, df in [("landmarks_imputed", pooled), ("complete_landmarks", restricted)]:
        target = config.WORK / (name + ".csv.gz")
        stage = target.with_name(target.name + ".partial.gz")
        df.to_csv(stage, index=False)
        stage.replace(target)
    stage = config.WORK / "input_summary.partial.json"
    stage.write_text(json.dumps(counts, indent=2, allow_nan=False) + "\n")
    stage.replace(config.WORK / "input_summary.json")
    print("Final NHATS landmarks", len(pooled), len(restricted), flush=True)
