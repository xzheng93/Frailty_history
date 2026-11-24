"""NHATS data locations, phenotype thresholds, imputation and model settings."""

from __future__ import annotations

from pathlib import Path

from src.data_paths import raw_data_dir

PACKAGE = Path(__file__).resolve().parents[2]
WORK = PACKAGE / "work/nhats"
DATA = WORK / "community"
RESULTS = WORK / "qa"
RAW_NHATS = raw_data_dir() / "NHATS/data"

ROUNDS = list(range(1, 15))
NO_PERFORMANCE_ROUNDS = {10}  # 2020, telephone interviews, no grip or walking test
WALK_COURSE_M = 3.0

N_IMP = 10
MICE_ITER = 10
SEED = 20261005
MIN_OBSERVED_COMPONENTS = 3

# Bandeen-Roche 2015, Supplementary Table 1a: first usual-pace walk, m/s.
# Slowness = speed at or below the value. Sex coded 1 male, 2 female.
SLOW_CUTS = {
    (2, "short"): 0.35,  # female, height <= 159 cm
    (2, "tall"): 0.45,  # female, height > 159 cm
    (1, "short"): 0.50,  # male, height <= 173 cm
    (1, "tall"): 0.58,  # male, height > 173 cm
}
HEIGHT_SPLIT_CM = {2: 159.0, 1: 173.0}

# Bandeen-Roche 2015, Supplementary Table 2a: max grip over two trials, kg.
GRIP_CUTS = {
    (2, "under"): 9.21,
    (2, "normal"): 12.66,
    (2, "over"): 12.92,
    (2, "obese"): 12.23,
    (1, "under"): 5.52,
    (1, "normal"): 22.75,
    (1, "over"): 25.59,
    (1, "obese"): 25.96,
}

COMPONENTS = ["slow", "weak", "exhaustion", "low_pa", "weight_loss"]

HORIZON = 4
MODEL0 = ["index_age_cat", "gender", "race_collapsed", "education_collapsed", "index_round"]
MODEL1 = MODEL0 + [
    "married_partnered",
    "current_smoker",
    "medicaid",
    "bmi_category",
    "chronic9_cat",
]
GROUPS = ["prior0", "prior1", "prior2plus"]
