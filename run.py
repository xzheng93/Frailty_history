"""Reproduce the frailty-history analyses, including SHARE when its raw data are available."""

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from src.data_paths import raw_data_dir

BASE = Path(__file__).resolve().parent
SRC = BASE / "src"
WORK = BASE / "work"
RESULTS = BASE / "results"


@contextmanager
def run_lock():
    WORK.mkdir(exist_ok=True)
    with (WORK / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("Another reproduction run is using this working directory.") from exc
        yield


def run(script, *args, env):
    print(f"Running {script}", flush=True)
    subprocess.run(
        [sys.executable, "-B", str(SRC / script), *map(str, args)],
        cwd=BASE,
        env=env,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=raw_data_dir(),
        help="Directory containing NHATS, HRS and (optionally) SHARE",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Workers for NHATS and SHARE (default 4)"
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Regenerate NHATS CSV summaries and validate the saved cohort estimates",
    )
    parser.add_argument(
        "--core-only", action="store_true", help="Limit reproduction to NHATS and HRS."
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    raw = args.raw_data.expanduser().resolve()
    share_results = RESULTS / "share"
    share_available = (
        (share_results / "fixed_pairs_by_wave.csv").is_file()
        if args.export_only
        else (raw / "SHARE").is_dir()
    )
    include_share = share_available and not args.core_only
    if not args.export_only and not all(
        (raw / cohort).is_dir() for cohort in ("NHATS/data", "HRS")
    ):
        parser.error(f"Missing HRS or NHATS/data under {raw}. Supply --raw-data.")
    env = os.environ.copy()
    env.update(
        FRAILTY_RAW_DATA=str(raw),
        PYTHONDONTWRITEBYTECODE="1",
    )
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    with run_lock():
        incomplete = WORK / ".incomplete_model_run"
        if args.export_only and incomplete.exists():
            parser.error(
                "The preceding model run did not finish. Run the command without --export-only to resume."
            )
        if not args.export_only:
            incomplete.touch()
            run("nhats/run_inputs.py", "--workers", args.workers, env=env)
            run("nhats/fit_models.py", "--workers", args.workers, env=env)
            run("hrs/run.py", "--raw-dir", raw / "HRS", env=env)
            if include_share:
                run(
                    "share/run.py",
                    "--raw-dir",
                    raw / "SHARE",
                    "--output",
                    share_results,
                    "--workers",
                    args.workers,
                    env=env,
                )
            incomplete.unlink()
        run("nhats_outputs.py", env=env)
        share_args = ("--share-results", share_results) if include_share else ()
        run("validate.py", *share_args, env=env)
    print("Validated NHATS and HRS results" + (" and SHARE results." if include_share else "."))


if __name__ == "__main__":
    main()
