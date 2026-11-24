"""Read original NHATS interviews and tracker, fix enrollment, impute scores and build both cohorts."""

from pathlib import Path
import argparse
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nhats import build_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    build_inputs.prepare_raw()
    build_inputs.fixed_pairs()
    build_inputs.impute_all(args.workers)
    build_inputs.finish()
