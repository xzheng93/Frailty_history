"""Build SHARE interviews, track survival, and fix the earliest eligible assessment pair."""

import numpy as np
import pandas as pd

from src.hrs.cohort import grip_cutoff, logical_or, threshold

ITEMS = ["weak", "slow", "exhaustion", "low_pa", "shrinking"]
WAVES = (1, 2, 4, 5, 6, 7, 8, 9)
CONTACTS = ("1", "2", "3", "4", "5", "6", "7", "8", "8ca", "9ca", "9")
# Prior, index, scheduled visit approximately four years after index.
SCHEDULE = ((1, 2, 4), (4, 5, 7), (5, 6, 8), (6, 7, 9))
LONG_GAP = (2, 4, 6)
DESIGNS = {"main": SCHEDULE, "include_long_gap": tuple(sorted((*SCHEDULE, LONG_GAP)))}


def module(root, wave, name, fields):
    folder = str(wave).replace("ca", "_covid")
    path = root / f"wave_{folder}" / f"sharew{wave}_rel9-0-0_{name}.dta"
    frame = pd.read_stata(path, columns=["mergeid", *fields], convert_categoricals=False)
    frame = frame.set_index("mergeid", verify_integrity=True).astype(float)
    return frame.where(frame.ge(0))


def visit(root, wave):
    d = module(
        root, wave, "cv_r", ["country", "gender", "age_int", "int_year", "int_month", "interview"]
    )
    d = d.loc[d.interview.eq(1)].copy()
    d["date"] = (d.int_year * 12 + d.int_month).where(
        d.int_month.between(1, 12) & d.int_year.between(2003, 2025)
    )
    fall = "ph010d7" if wave <= 4 else "ph089d1"
    mods = {
        "gv_health": ["bmi", "maxgrip", "sphus", "adl", "euro8", "euro9"],
        "gv_isced": ["isced1997_r"],
        "mh": ["mh011_", "mh012_", "mh013_"],
        "br": ["br016_"],
        "ph": ["ph048d1", "ph048d5", fall],
        "hc": ["hc012_"] + (["hc029_"] if wave == 1 else []),
        "gs": ["gs002_"] + (["gs010d4"] if wave != 1 else []),
    }
    if wave != 1:
        mods["technical_variables"] = ["mn024_"] + (["mn103_"] if wave == 7 else [])
    for key, fields in mods.items():
        d = d.join(module(root, wave, key, fields))
    d["community"] = d.hc029_.isin([1, 5]) if wave == 1 else d.mn024_.eq(1)
    d["regular"] = d.mn103_.eq(0) if wave == 7 else True
    unable = d.gs002_.eq(4)
    if wave != 1:
        unable |= d.gs010d4.eq(1)
    unable &= d.maxgrip.isna()
    d["grip"] = d.maxgrip.mask(unable, 0)
    d["weak"] = threshold(d.grip, grip_cutoff(d.gender, d.bmi)).mask(unable, 1)
    d["slow"] = logical_or(d.ph048d1, d.ph048d5)
    d["exhaustion"] = d.mh013_.map({1: 1.0, 5: 0.0})
    d["low_pa"] = d.br016_.map({1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0})
    followup = d.mh012_.map({1: 1.0, 2: 0.0, 3: 0.0})
    # W5 alone codes no diminution as 5 and the follow-up prompt as 9.
    no_change, probe = (5, 9) if wave == 5 else (2, 3)
    appetite = d.mh011_.dropna()
    if not appetite.isin([1, no_change, probe]).all():
        raise ValueError(f"Unexpected appetite response code in W{wave}")
    d["shrinking"] = d.mh011_.map({1: 1.0, no_change: 0.0}).where(~d.mh011_.eq(probe), followup)
    for item, official in [("shrinking", "euro8"), ("exhaustion", "euro9")]:
        check = d[[item, official]].dropna()
        if check.empty or not check[item].eq(check[official]).all():
            raise ValueError(f"W{wave} {item} disagrees with SHARE {official}")
    d["education"] = d.isced1997_r.map({0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2})
    d["falling_symptom"] = d[fall].map({0: 0.0, 1: 1.0})
    d["overnight_hospital_12m"] = d.hc012_.map({1: 1.0, 5: 0.0})
    d["assessed"] = d[ITEMS].notna().sum(axis=1)
    d["observed_score"] = d[ITEMS].sum(axis=1, min_count=5)
    return d


def vital(root):
    path = root / "all_waves/sharewX_rel9-0-0_gv_allwaves_cv_r.dta"
    stems = ["interview", "int_year", "int_month", "deadoralive"]
    cols = [f"{s}_w{w}" for w in CONTACTS for s in stems]
    t = pd.read_stata(
        path,
        columns=["mergeid", "country", "deceased_year", "deceased_month", *cols],
        convert_categoricals=False,
    ).set_index("mergeid", verify_integrity=True)
    living, exact_living, dead_by, exits = [], [], [], []
    v = pd.DataFrame(index=t.index)
    for w in CONTACTS:
        kind, status = t[f"interview_w{w}"], t[f"deadoralive_w{w}"]
        yr, mo = t[f"int_year_w{w}"].astype(float), t[f"int_month_w{w}"]
        date = (yr * 12 + mo).where(yr.between(2003, 2025) & mo.between(1, 12))
        fieldwork = date.where(kind.eq(1)).groupby(t.country)
        exact_living.append(date.where(kind.eq(1)))
        living.append(date.where(kind.eq(1)).fillna(fieldwork.transform("min").where(status.eq(1))))
        dead_by.append(
            date.where(kind.eq(2)).fillna(fieldwork.transform("max").where(status.eq(2)))
        )
        v[f"dead_w{w}"] = status.eq(2) | kind.eq(2)
        if w != "1":
            prefix = "sl_" if w == "3" else ""
            cols = [prefix + "xt009_", prefix + "xt008_"]
            proxy = module(root, w, "xt", cols).set_axis(["year", "month"], axis=1)
            v[f"dead_w{w}"] |= v.index.isin(proxy.index)
            exits.append(proxy)
    v["last_interview"] = pd.concat(exact_living, axis=1).max(axis=1)
    v["last_alive"] = pd.concat(living, axis=1).max(axis=1)
    v["first_dead"] = pd.concat(dead_by, axis=1).min(axis=1)
    year = t.deceased_year.where(t.deceased_year.between(2003, 2025))
    month = t.deceased_month.where(t.deceased_month.between(1, 12))
    proxy = pd.concat(exits)
    proxy = proxy[proxy.year.between(2003, 2025)].copy()
    proxy["month"] = proxy.month.where(proxy.month.between(1, 12))
    grouped = proxy.groupby(level=0).year
    unique_year = grouped.min().where(grouped.nunique().eq(1))
    v["deathyear"] = year.fillna(unique_year)
    compatible = proxy[proxy.year.eq(v.deathyear.reindex(proxy.index).to_numpy())]
    months = compatible.groupby(level=0).month
    v["deathmonth"] = month.fillna(months.min().where(months.nunique().eq(1)))
    v["recorded_dead"] = v.deathyear.notna() | v[[f"dead_w{w}" for w in CONTACTS]].any(axis=1)
    return v


def eligible_pairs(visits, tracker, schedules):
    candidates = []
    for prior, index, target in schedules:
        p, i, f = visits[prior], visits[index], visits[target]
        # Require fieldwork offering a regular questionnaire in the country,
        # not attendance or an observed phenotype at the individual's future visit.
        offered = f.loc[f.regular, "country"].unique()
        d = i[
            ["date", "age_int", "gender", "int_year", "country", "community", "regular", "assessed"]
        ].join(
            p[["date", "age_int", "country", "community", "regular", "assessed"]].add_suffix(
                "_prior"
            ),
            how="inner",
        )
        d = d.join(tracker, validate="one_to_one")
        later = CONTACTS[CONTACTS.index(str(index)) + 1 :]
        last_possible_death = d.deathyear * 12 + d.deathmonth.fillna(12)
        later_death = (
            d[[f"dead_w{w}" for w in later]].any(axis=1) | last_possible_death.ge(d.date)
        ) & (last_possible_death.isna() | last_possible_death.ge(d.date))
        d["followup"] = d.last_interview.gt(d.date) | later_death
        d["country_eligible"] = d.country.eq(d.country_prior) & d.country.isin(offered)
        d[["prior_wave", "index_wave", "target_wave"]] = prior, index, target
        d = d.join(
            f[["date", "regular", "assessed", "falling_symptom", "overnight_hospital_12m"]].rename(
                columns={"date": "date_4", "regular": "regular_4", "assessed": "assessed_4"}
            )
        )
        candidates.append(d)
    d = pd.concat(candidates)
    active = np.ones(len(d), dtype=bool)
    rows = [("Main interview at both scheduled waves", d.index.nunique(), len(d))]
    for label, mask in (
        ("Country offered scheduled follow-up FFP", d.country_eligible),
        ("Prior age >=65", d.age_int_prior.ge(65)),
        (
            "Community at both visits and valid ordered dates",
            d.community & d.community_prior & d.date.gt(d.date_prior),
        ),
        (
            "Regular interviews and >=3 FFP items at both visits",
            d.regular & d.regular_prior & d.assessed.ge(3) & d.assessed_prior.ge(3),
        ),
        ("Subsequent interview or recorded death", d.followup),
    ):
        active &= mask.to_numpy()
        rows.append((label, d.index[active].nunique(), int(active.sum())))
    d = d.iloc[np.flatnonzero(active)]
    d = d.sort_values(["date", "index_wave"], kind="stable")
    d = d[~d.index.duplicated()].sort_index().copy()
    rows.append(("Earliest eligible pair fixed", len(d), len(d)))
    if not d.gender.isin([1, 2]).all() or d.age_int.isna().any() or not d.index.is_unique:
        raise ValueError("Invalid index identifiers or demographics")
    return d, pd.DataFrame(rows, columns=["step", "n_people", "n_pairs"])


def death_dates(d, rng):
    # Dated deaths take precedence over approximate national fieldwork bounds.
    # For undated deaths, draw within the last-known-alive/first-known-dead interval.
    lower = d.last_alive.copy()
    upper = d.first_dead.copy()
    year = d.deathyear
    known_year = year.notna()
    lower.loc[known_year] = np.maximum(d.last_interview[known_year], year[known_year] * 12 + 1)
    upper.loc[known_year] = year[known_year] * 12 + 12
    compatible = known_year & d.first_dead.ge(lower) & d.last_alive.le(upper)
    lower.loc[compatible] = np.maximum(lower[compatible], d.last_alive[compatible])
    upper.loc[compatible] = np.minimum(upper[compatible], d.first_dead[compatible])
    dm = (year * 12 + d.deathmonth).where(year.notna())
    for key in d.index[d.recorded_dead & dm.isna() & lower.le(upper)]:
        candidates = np.arange(int(lower[key]), int(upper[key]) + 1)
        if pd.notna(d.at[key, "deathmonth"]):
            candidates = candidates[(candidates - 1) % 12 + 1 == int(d.at[key, "deathmonth"])]
        if len(candidates):
            dm.loc[key] = rng.choice(candidates)
    return dm.where(dm.ge(d.last_interview))
