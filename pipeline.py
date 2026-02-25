"""
pipeline.py — End-to-end pipeline runner.
Run this to execute the full analysis from raw data to results.

Usage:
    python pipeline.py                  # Full pipeline
    python pipeline.py --step download  # Single step
    python pipeline.py --sample         # Run on sample only (fast)
"""

import logging
import argparse
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
from src.config import DATA_PROCESSED, TLC_MONTHS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)


def run_step(name: str, fn, *args, **kwargs):
    """Wrapper to time and log each pipeline step."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {name}")
    logger.info(f"{'='*60}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    logger.info(f"✓ {name} completed in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run causal inference pipeline")
    parser.add_argument("--step", choices=["download", "clean", "join", "baseline", "iv", "hte", "plots", "all"],
                        default="all", help="Which step to run")
    parser.add_argument("--sample", action="store_true",
                        help="Use 1-month sample instead of full dataset")
    args = parser.parse_args()

    months = [TLC_MONTHS[0]] if args.sample else TLC_MONTHS
    logger.info(f"Running pipeline | step={args.step} | months={months} | sample={args.sample}")

    # ── Step 1: Download ───────────────────────────────────────────────────
    if args.step in ("download", "all"):
        from src.data.download import main as download_main
        run_step("Download", download_main)

    # ── Step 2: Clean ──────────────────────────────────────────────────────
    if args.step in ("clean", "all"):
        from src.data.clean import run_cleaning
        df = run_step("Clean", run_cleaning, months=months, save=True)

    # ── Step 3: Join ───────────────────────────────────────────────────────
    if args.step in ("join", "all"):
        from src.data.join import run_join
        df = run_step("Join", run_join, months=months, save=True)

    # ── Step 4: OLS Baseline ───────────────────────────────────────────────
    if args.step in ("baseline", "all"):
        from src.models.ols_baseline import run_baselines
        ols_results = run_step("OLS Baseline", run_baselines, save=True)

    # ── Step 5: IV Analysis ────────────────────────────────────────────────
    if args.step in ("iv", "all"):
        from src.models.iv_2sls import run_iv_analysis
        iv_results = run_step("IV 2SLS", run_iv_analysis, save=True)

    # ── Step 6: Causal Forest HTE ──────────────────────────────────────────
    if args.step in ("hte", "all"):
        from src.models.causal_forest import run_causal_forest
        hte_results = run_step("Causal Forest HTE", run_causal_forest, save=True)

    # ── Step 7: Plots ──────────────────────────────────────────────────────
    if args.step in ("plots", "all"):
        from src.utils.plots import generate_all_plots
        run_step("Generate Plots", generate_all_plots, save=True)

    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Results in: outputs/figures/ and outputs/tables/")
    logger.info(f"Launch dashboard: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
