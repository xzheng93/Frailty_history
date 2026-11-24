"""HRS frailty phenotype and prior–index pairs."""

from __future__ import annotations
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import re
import numpy as np
import pandas as pd

WAVES = dict(zip(range(2006, 2024, 2), "KLMNOPQRS"))
INDEX = (2010, 2012, 2014, 2018)
FFP = ["weakness", "slow_walk", "low_pa", "exhaustion", "weight_loss"]
COMMON = ["hhidpn", "ragender", "raracem", "rahispan", "raeduc", "radyear", "radmonth"]
STEMS = (
    "iwstat agey_e iwendy iwendm nhmliv proxy vgactx mdactx ltactx effort going "
    "pmbmi bmi pmhght height weight timwlk hosp shlt adl5a"
).split()


def member(z, filename):
    matches = [n for n in z.namelist() if Path(n).name.lower() == filename.lower()]
    if len(matches) != 1:
        raise ValueError(f"Missing or duplicated archive member: {filename}")
    return matches[0]


def read_section(core_dir, year, section, requested):
    """Use the distributed dictionary to read only the requested fields."""
    yy = str(year)[-2:]
    filename = "h" + yy + section + "_r"
    outer = ZipFile(core_dir / ("h" + yy + ("core" if year == 2022 else "sta") + ".zip"))
    z = ZipFile(BytesIO(outer.read(member(outer, "H22sta.zip")))) if year == 2022 else outer
    try:
        dct = z.read(member(z, filename + ".dct")).decode("latin1")
        schema = re.findall(r"_column\((\d+)\)\s+\w+\s+(\w+)\s+%[-]?(\d+)", dct, re.I)
        prefix = WAVES[year]
        names = ["HHID", "PN"] + [prefix + s for s in requested]
        columns = [
            (int(start) - 1, int(start) - 1 + int(width), key.upper())
            for start, key, width in schema
            if key.upper() in names
        ]
        actual = [c[2] for c in columns]
        if not {"HHID", "PN"}.issubset(actual):
            raise ValueError(f"{year} {section}: no respondent key")
        if year <= 2016:
            with ZipFile(core_dir / ("h" + yy + "da.zip")) as raw:
                with raw.open(member(raw, filename + ".da")) as f:
                    df = pd.read_fwf(f, colspecs=[c[:2] for c in columns], names=actual)
        else:
            with z.open(member(z, filename + ".dta")) as f:
                with pd.io.stata.StataReader(
                    BytesIO(f.read()), convert_categoricals=False
                ) as reader:
                    headers = reader.variable_labels()
                    df = reader.read(columns=[c for c in headers if c.upper() in actual])
                df.columns = df.columns.str.upper()
        df = df.apply(pd.to_numeric, errors="coerce")
        ident = df.pop("HHID") * 1000 + df.pop("PN")
        if ident.isna().any() or ident.duplicated().any():
            raise ValueError(f"{year} {section}: nonunique respondent key")
        df.index = pd.Index(ident.astype("int64"), name="id")
        df.columns = [n[1:] for n in df]
        missing = [
            s for s in requested if s not in df and not re.fullmatch("I(?:813|821)M[2-9]", s)
        ]
        if missing:
            raise ValueError(f"{year} {section}: missing dictionary fields {missing}")
        return df
    finally:
        if z is not outer:
            z.close()
        outer.close()


def logical_or(a, b):
    return pd.Series(
        np.select([a.eq(1) | b.eq(1), a.eq(0) & b.eq(0)], [1.0, 0.0], default=np.nan),
        index=a.index,
    )


def threshold(x, cutoff):
    return x.le(cutoff).astype(float).where(x.notna() & pd.notna(cutoff))


def documented_failure(raw, stem):
    reasons = raw.filter(regex=f"^{stem}M[0-9]+$")
    return reasons.eq(4).any(axis=1)  # explicitly tried, unable to complete


def grip_cutoff(sex, bmi):
    return pd.Series(
        np.select(
            [
                sex.eq(1) & bmi.le(24),
                sex.eq(1) & bmi.le(28),
                sex.eq(1) & bmi.gt(28),
                sex.eq(2) & bmi.le(23),
                sex.eq(2) & bmi.le(26),
                sex.eq(2) & bmi.le(29),
                sex.eq(2) & bmi.gt(29),
            ],
            [29, 30, 32, 17, 17.3, 18, 21],
            default=np.nan,
        ),
        index=sex.index,
    )


def speed_cutoff(sex, ht):
    return pd.Series(
        np.where((sex.eq(1) & ht.le(1.73)) | (sex.eq(2) & ht.le(1.59)), 0.65, 0.76),
        index=sex.index,
    ).where(sex.isin([1, 2]) & ht.notna())


def add_performance(df):
    sex, bmi, ht = df.ragender, df.bmi_use, df.height_use
    glimit = grip_cutoff(sex, bmi)
    valid = df[["I816", "I852", "I851", "I853"]].copy()
    valid = valid.where(valid.ge(0) & valid.le(100))
    left = (valid.I816 + valid.I852) / 2
    right = (valid.I851 + valid.I853) / 2
    hand = df.I815
    # Both hands declared dominant: use the stronger fully measured hand mean.
    both = pd.concat([left, right], axis=1).max(axis=1)
    grip = (
        right.where(hand.eq(1))
        .combine_first(left.where(hand.eq(2)))
        .combine_first(both.where(hand.eq(3)))
    )
    left_valid = valid[["I816", "I852"]].notna().any(axis=1)
    right_valid = valid[["I851", "I853"]].notna().any(axis=1)
    left_failed = df[["I816", "I852"]].eq(993).any(axis=1)
    right_failed = df[["I851", "I853"]].eq(993).any(axis=1)
    no_valid = ~valid.notna().any(axis=1)
    unable = (
        (hand.eq(1) & ~right_valid & right_failed)
        | (hand.eq(2) & ~left_valid & left_failed)
        | (no_valid & (documented_failure(df, "I813") | (left_failed & right_failed)))
    )
    df["grip"] = grip.mask(unable, 0.0)
    df["weakness"] = threshold(grip, glimit).mask(unable, 1.0)
    df["weak_unable"] = unable
    cutoff = speed_cutoff(sex, ht)
    # Harmonized fastest valid usual-pace trial, 2.5 metres.
    speed = 2.5 / df.timwlk.where(df.timwlk.gt(0) & df.timwlk.lt(993))
    unable_walk = speed.isna() & (
        documented_failure(df, "I821") | df[["I823", "I824"]].eq(993).any(axis=1)
    )
    df["speed"] = speed.mask(unable_walk, 0.0)
    df["slow_walk"] = threshold(speed, cutoff).mask(unable_walk, 1.0)
    df["walk_unable"] = unable_walk


def panel(raw_dir):
    path = raw_dir / "randhrs1992_2022v1.dta"
    with pd.io.stata.StataReader(path, convert_categoricals=False) as reader:
        available = set(reader.variable_labels())
    cols = (
        COMMON + ["r7weight", "r7iwstat"] + [f"r{wave}{s}" for wave in range(8, 17) for s in STEMS]
    )
    use = [c for c in cols if c in available]
    with pd.read_stata(
        path,
        columns=use,
        chunksize=1500,
        convert_categoricals=False,
        convert_dates=False,
    ) as chunks:
        rand = pd.concat(chunks, ignore_index=True).set_index("hhidpn", verify_integrity=True)
    rand.index = pd.Index(rand.index.astype("int64"), name="id")
    frames = []
    physical = [
        "PMELIG",
        "NURSHM",
        "I815",
        "I816",
        "I851",
        "I852",
        "I853",
        "I823",
        "I824",
    ] + [f"{s}M{i}" for s in ("I813", "I821") for i in range(1, 7)]
    for wave, (year, letter) in enumerate(WAVES.items(), 8):
        names = [f"r{wave}{s}" for s in STEMS]
        df = rand.reindex(columns=names).copy()
        df.columns = STEMS
        df = df.join(rand[COMMON[1:]])
        # Core interview with living respondent at the time, including death after interview.
        df = df.loc[df.iwstat.isin([1, 2])]
        df["year"] = year
        df["bmi_use"] = df.pmbmi.combine_first(df.bmi)
        df["height_use"] = df.pmhght.combine_first(df.height)
        prev = rand[f"r{wave - 1}weight"].where(rand[f"r{wave - 1}iwstat"].isin([1, 2]))
        prev = prev.reindex(df.index)
        valid_w = df.weight.gt(0) & prev.gt(0)
        df["loss10"] = df.weight.le(0.9 * prev + 1.0e-7).astype(float).where(valid_w)
        low_bmi = df.bmi_use.lt(18.5).astype(float).where(df.bmi_use.notna())
        df["weight_loss"] = logical_or(df.loss10, low_bmi)
        df["exhaustion"] = logical_or(df.effort, df.going)
        freq = {1: 7.0, 2: 4.0, 3: 1.0, 4: 0.5, 5: 0.0}
        df["activity"] = 8 * df.vgactx.map(freq) + 4 * df.mdactx.map(freq) + 2 * df.ltactx.map(freq)
        health = read_section(raw_dir / "Core", year, "c", ["C079", "C080"])
        df = df.join(health)
        cnt = df.C080.where(df.C080.between(0, 97))
        df["falls"] = (
            cnt.ge(2).astype(float).where(df.C079.eq(1) & cnt.notna()).mask(df.C079.eq(5), 0.0)
        )
        if year != 2020:
            raw = read_section(raw_dir / "Core", year, "i", physical)
            df = df.join(raw)
            add_performance(df)
            df["scheduled"] = df.PMELIG.isin([1, 2, 5, 6, 7])
            df["community"] = df.NURSHM.isin([2, 5])
        else:
            df["scheduled"] = False
            df["community"] = df.nhmliv.eq(0)
        frames.append(df.reset_index())
    allrows = pd.concat(frames, ignore_index=True)
    ref = allrows[allrows.agey_e.ge(65) & allrows.activity.notna()].drop_duplicates("id")
    cut = ref.groupby("ragender").activity.quantile(0.2)
    allrows["low_pa"] = threshold(allrows.activity, allrows.ragender.map(cut))
    allrows["assessable"] = allrows[FFP].notna().sum(axis=1)
    allrows["score"] = allrows[FFP].sum(axis=1, min_count=5)
    allrows["visitmonth"] = allrows.iwendy * 12 + allrows.iwendm
    allrows["deathmonth"] = allrows.radyear * 12 + allrows.radmonth
    assert not allrows.duplicated(["id", "year"]).any()
    assert not (allrows.score.eq(0) & ~allrows[FFP].eq(0).all(axis=1)).any()
    return allrows, ref, cut


def participant_flow(visits):
    """Apply the Figure 1 participant exclusions in their displayed order."""
    index = visits[visits.year.isin(INDEX)]
    prior = visits.assign(year=visits.year + 4)
    pairs = index.merge(prior, on=["id", "year"], suffixes=("", "_prior"), validate="one_to_one")
    total = visits.id.nunique()
    rows = [
        {
            "step": "interviewed_in_2006_2022",
            "remaining_people": total,
            "excluded_people": 0,
            "remaining_pairs": None,
        }
    ]
    yearly = []

    def record_years(label, subset):
        for year, group in subset.groupby("year", sort=True):
            yearly.append((label, year, len(group), group.id.nunique()))

    record_years("interview_at_candidate_index_wave", index)
    last_year = visits.groupby("id").year.max()
    death_after_index = (
        pairs.radyear.gt(pairs.iwendy)
        | pairs.deathmonth.ge(pairs.visitmonth)
        | (pairs.iwstat.eq(2) & pairs.radyear.eq(pairs.iwendy))
    )
    stages = [
        ("consecutive_scheduled_assessments", pairs.scheduled & pairs.scheduled_prior),
        ("outside_nursing_home_both", pairs.community & pairs.community_prior),
        (
            "at_least_three_at_both",
            pairs.assessable.ge(3) & pairs.assessable_prior.ge(3),
        ),
        (
            "subsequent_interview_or_death",
            pairs.id.map(last_year).gt(pairs.year) | death_after_index,
        ),
        ("aged_at_least_65_at_prior", pairs.agey_e_prior.ge(65)),
    ]
    included = pd.Series(True, index=pairs.index)
    for label, criterion in stages:
        included &= criterion
        record_years(label, pairs.loc[included])
        remaining = pairs.loc[included, "id"].nunique()
        rows.append(
            {
                "step": label,
                "remaining_people": remaining,
                "excluded_people": rows[-1]["remaining_people"] - remaining,
                "remaining_pairs": int(included.sum()),
            }
        )
    fixed = pairs.loc[included].sort_values(["id", "year"]).drop_duplicates("id")
    rows.append(
        {
            "step": "fixed_earliest_pair",
            "remaining_people": len(fixed),
            "excluded_people": 0,
            "remaining_pairs": len(fixed),
        }
    )
    record_years("fixed_earliest_pair", fixed)
    prior_exam = fixed.PMELIG_prior.eq(1)
    index_exam = fixed.PMELIG.eq(1)
    eligible = fixed.loc[prior_exam & index_exam].copy()
    rows.append(
        {
            "step": "in_person_physical_exam_at_both",
            "remaining_people": len(eligible),
            "excluded_people": len(fixed) - len(eligible),
            "remaining_pairs": len(eligible),
        }
    )
    record_years("in_person_physical_exam_at_both", eligible)
    excluded = fixed.loc[~(prior_exam & index_exam)].copy()
    excluded["waves_without_exam"] = (~prior_exam[excluded.index]).astype(int) + (
        ~index_exam[excluded.index]
    ).astype(int)
    excluded["wave_without_exam"] = np.select(
        [excluded.waves_without_exam.eq(2), excluded.PMELIG_prior.ne(1)],
        ["both", "prior_only"],
        default="index_only",
    )
    excluded["reason"] = np.where(
        excluded.PMELIG.eq(7) | excluded.PMELIG_prior.eq(7),
        "at_least_one_telephone_interview",
        "battery_not_administered_without_telephone",
    )
    lack = excluded.groupby(
        ["year", "waves_without_exam", "wave_without_exam", "reason"], as_index=False
    ).size()
    lack = lack.rename(columns={"year": "index_year", "size": "n"})
    assert eligible.id.is_unique and lack.n.sum() == rows[-1]["excluded_people"]
    years = pd.DataFrame(yearly, columns=["step", "index_year", "records", "people"])
    return eligible, pd.DataFrame(rows), lack, years
