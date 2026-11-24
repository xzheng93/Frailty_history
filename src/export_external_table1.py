"""Baseline characteristics of index-robust HRS and SHARE participants."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_paths import raw_data_dir

SRC = Path(__file__).resolve().parent
BASE = SRC.parent
PRIOR_GROUPS = ("prior0", "prior1", "prior2plus")


def prior_groups(frame):
    yield "overall", frame
    for score, name in enumerate(PRIOR_GROUPS):
        yield name, frame.loc[frame.history.eq(score)]


def summarize_categorical(
    frames: list[pd.DataFrame], rows_spec: list[tuple[str, Callable | None]]
) -> pd.DataFrame:
    rows = []
    for frame in frames:
        groups = list(prior_groups(frame))
        for label, mask in rows_spec:
            row = {"characteristic": label}
            for name, group in groups:
                row[f"{name}_n"] = len(group) if mask is None else mask(group).sum()
            rows.append(row)
    summary = pd.DataFrame(rows).groupby("characteristic", sort=False).mean()
    columns = []
    for name in ("overall", *PRIOR_GROUPS):
        n, pct = f"{name}_n", f"{name}_pct"
        # Percentages use the unrounded mean count across imputations.
        summary[pct] = 100 * summary[n] / summary.at["Participants", n]
        columns.extend([n, pct])
    return summary[columns].reset_index()


def summarize_continuous(frames: list[pd.DataFrame], col: str, label: str) -> pd.DataFrame:
    rows = []
    for frame in frames:
        row = {"characteristic": label}
        for name, group in prior_groups(frame):
            row[f"{name}_mean"] = group[col].mean()
            row[f"{name}_sd"] = group[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).groupby("characteristic", sort=False).mean().reset_index()


def export_hrs(raw_hrs_dir: Path, out_dir: Path) -> None:
    from src.hrs.cohort import panel, participant_flow
    from src.hrs.models import build_analysis, first_events, followup_base, impute_scores

    visits, _, _ = panel(raw_hrs_dir)
    eligible, _, _, _ = participant_flow(visits)
    fixed = followup_base(visits, eligible)

    robust_frames = []
    event_rows = []
    for m in range(1, 11):
        completed = impute_scores(visits, m)
        paired = build_analysis(fixed, completed, m)
        rob = paired.loc[paired.base_score.eq(0)].reset_index(drop=True)
        robust_frames.append(rob)

        dt = rob.death_time.to_numpy(float)
        alive_t = rob.alive_time.to_numpy(float)
        died48 = (dt >= 0) & (dt <= 48)
        alive48 = (dt > 48) | (~np.isfinite(dt) & (alive_t >= 48))
        scored4 = (
            alive48
            & rob.four_score.notna().to_numpy()
            & ((rob.visitmonth_4 - rob.visitmonth).to_numpy() > 0)
        )
        frail4 = scored4 & rob.four_score.ge(3).to_numpy()
        falls_first, _, _ = first_events(rob, "falls")
        hosp_first, _, _ = first_events(rob, "hosp")
        event_rows.append(
            dict(
                imp=m,
                n=len(rob),
                death48=int(died48.sum()),
                scored_survivor=int(scored4.sum()),
                frail_in_survivors=int(frail4.sum()),
                multiple_falls_events=int(np.isfinite(falls_first).sum()),
                hospitalization_events=int(np.isfinite(hosp_first).sum()),
            )
        )

    rows_spec = [
        ("Participants", None),
        ("Age 65-69 years", lambda d: d["agey_e"].lt(70)),
        ("Age 70-74 years", lambda d: d["agey_e"].ge(70) & d["agey_e"].lt(75)),
        ("Age 75-79 years", lambda d: d["agey_e"].ge(75) & d["agey_e"].lt(80)),
        ("Age 80-84 years", lambda d: d["agey_e"].ge(80) & d["agey_e"].lt(85)),
        ("Age >=85 years", lambda d: d["agey_e"].ge(85)),
        ("Female sex", lambda d: d["ragender"].eq(2)),
        ("White non-Hispanic", lambda d: d["race"].eq(1)),
        ("Black non-Hispanic", lambda d: d["race"].eq(2)),
        ("Hispanic", lambda d: d["race"].eq(4)),
        ("Other race/ethnicity", lambda d: d["race"].eq(3)),
        ("Less than high school", lambda d: d["education"].eq(1)),
        ("High school graduate or GED", lambda d: d["education"].isin([2, 3])),
        ("Some college or higher", lambda d: d["education"].ge(4)),
        ("Index wave 2010", lambda d: d["year"].eq(2010)),
        ("Index wave 2012", lambda d: d["year"].eq(2012)),
        ("Index wave 2014", lambda d: d["year"].eq(2014)),
        ("Index wave 2018", lambda d: d["year"].eq(2018)),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    cat = summarize_categorical(robust_frames, rows_spec)
    cont = summarize_continuous(robust_frames, "agey_e", "Age, mean (SD), years")
    events = (
        pd.DataFrame(event_rows)
        .mean(numeric_only=True)
        .drop(index=["imp"])
        .to_frame("mean_across_10_imputations")
    )

    cat.to_csv(out_dir / "table1_hrs.csv", index=False)
    cont.to_csv(out_dir / "table1_hrs_age_mean_sd.csv", index=False)
    events.to_csv(out_dir / "hrs_observed_events.csv")
    print(f"Saved HRS baseline table and observed events to {out_dir}")


def export_share(raw_share_dir: Path, out_dir: Path) -> None:
    from src.share.share_panel import DESIGNS
    from src.share.run import imputed_samples, prepare

    # The fitted main analysis and sensitivity analysis shared the same PMM donors.
    data, _, _ = prepare(raw_share_dir, DESIGNS)
    robust_frames = []
    missing_rows = []
    for imp in range(1, 11):
        for label, d in imputed_samples(imp, data):
            if label == "main":
                robust_frames.append(d.reset_index(drop=True))
                inc = d["assessed"].lt(5) | d["assessed_prior"].lt(5)
                missing_rows.append(
                    dict(imp=imp, n=len(d), incomplete=int(inc.sum()), pct=100.0 * inc.mean())
                )

    n_countries = int(robust_frames[0]["country"].nunique())
    rows_spec = [
        ("Participants", None),
        ("Age 65-69 years", lambda d: d["age_int"].lt(70)),
        ("Age 70-74 years", lambda d: d["age_int"].ge(70) & d["age_int"].lt(75)),
        ("Age 75-79 years", lambda d: d["age_int"].ge(75) & d["age_int"].lt(80)),
        ("Age 80-84 years", lambda d: d["age_int"].ge(80) & d["age_int"].lt(85)),
        ("Age >=85 years", lambda d: d["age_int"].ge(85)),
        ("Female sex", lambda d: d["gender"].eq(2)),
        ("ISCED-97 low (0-2)", lambda d: d["education"].eq(0)),
        ("ISCED-97 medium (3-4)", lambda d: d["education"].eq(1)),
        ("ISCED-97 high (5-6)", lambda d: d["education"].eq(2)),
        ("Index Wave 2", lambda d: d["index_wave"].eq(2)),
        ("Index Wave 5", lambda d: d["index_wave"].eq(5)),
        ("Index Wave 6", lambda d: d["index_wave"].eq(6)),
        ("Index Wave 7", lambda d: d["index_wave"].eq(7)),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    cat = summarize_categorical(robust_frames, rows_spec)
    cont = summarize_continuous(robust_frames, "age_int", "Age, mean (SD), years")
    meta = pd.DataFrame([{"metric": "n_countries", "value": n_countries}])

    cat.to_csv(out_dir / "table1_share.csv", index=False)
    cont.to_csv(out_dir / "table1_share_age_mean_sd.csv", index=False)
    meta.to_csv(out_dir / "table1_share_meta.csv", index=False)
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(out_dir / "share_missing_scoring_inputs.csv", index=False)
    print(f"Saved SHARE baseline table ({n_countries} countries) to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=raw_data_dir(),
        help="Parent folder containing HRS/ and SHARE/",
    )
    parser.add_argument("--hrs-only", action="store_true")
    parser.add_argument("--share-only", action="store_true")
    args = parser.parse_args()

    raw = args.raw_data.expanduser().resolve()
    with threadpool_limits(limits=1):
        if not args.share_only:
            hrs_dir = raw / "HRS"
            if (hrs_dir / "randhrs1992_2022v1.dta").is_file():
                export_hrs(hrs_dir, BASE / "results/hrs_baseline")
            else:
                print(f"Skipping HRS: raw data not found at {hrs_dir}")
        if not args.hrs_only:
            share_dir = raw / "SHARE"
            if share_dir.is_dir():
                export_share(share_dir, BASE / "results/share")
            else:
                print(f"Skipping SHARE: raw data not found at {share_dir}")


if __name__ == "__main__":
    main()
