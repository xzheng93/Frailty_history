# Frailty history and four-year outcomes

Analysis code for *Prior Frailty History and 4-Year Risk of Frailty or Death Among Robust Older Adults: A Multicohort Study*. The primary analysis compares older adults with a current Fried frailty phenotype (FFP) score of 0, grouped by their previous score (0, 1 or ≥2).

## Data

Obtain the original files from the respective cohort data portals. Pass the directory containing the cohort folders to `--raw-data`.

| Cohort | Files under the data directory |
| --- | --- |
| NHATS | `NHATS/data/` with the round-specific Stata SP interview files (rounds 1–14) and the round 14 tracker, in their release folders |
| HRS | `HRS/randhrs1992_2022v1.dta` and `HRS/Core/` with the original Core archives for 2006–2022, waves 8–16 |
| SHARE | Release 9.0.0 Stata files in `SHARE/wave_1/` through `SHARE/wave_9/`, plus `SHARE/wave_8_covid/`, `SHARE/wave_9_covid/` and `SHARE/all_waves/` |

The cohort-specific scripts select the earliest eligible pair of visits before imputation. The main SHARE designs use W2, W5, W6 or W7 as the index visit. The SHARE secondary outcomes measure reported falling symptoms over six months and overnight hospitalisation over twelve months at the scheduled follow-up. NHATS and HRS estimate cumulative four-year risks of multiple falls and hospitalisation.

## Run the analysis

Use Python 3.11 on macOS or Linux. Run these commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -B run.py --raw-data /path/to/data --workers 4
```

The main script runs NHATS and HRS and includes SHARE when its data folder is present. Use `--core-only` to run NHATS and HRS alone.

| Output directory | CSV results |
| --- | --- |
| `work/nhats/tables/` | Baseline characteristics, four-year risks, risk differences, trends, joint risk curves and sensitivity analyses |
| `work/hrs/` | Cohort flow, four-year estimates, trends and resampling results |
| `results/share/` | Cohort flow, four-year frailty or death estimates, follow-up reports, trends and resampling results |

To regenerate the NHATS summaries and check all available cohort estimates against their saved model draws, run:

```bash
python -B run.py --export-only
```

To generate baseline characteristics for HRS and SHARE, run:

```bash
python -B src/export_external_table1.py --raw-data /path/to/data
```

This writes baseline CSVs to `results/hrs_baseline/` and `results/share/`. Counts are averaged across ten imputations.
