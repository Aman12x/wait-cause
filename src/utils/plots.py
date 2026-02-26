"""
plots.py — All project visualizations.
Produces publication-ready figures for portfolio and Streamlit dashboard.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import OUTPUTS_FIGURES, OUTPUTS_TABLES, DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
PALETTE = {"naive": "#e74c3c", "controls": "#e67e22", "iv": "#2ecc71"}


def plot_ols_vs_iv(ols_results: pd.DataFrame, iv_coef: float, iv_ci: tuple, save: bool = True):
    """
    Killer chart: coefficient comparison across OLS (naive), OLS+controls, and IV.
    Shows direction and magnitude of endogeneity bias clearly.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(ols_results["model"]) + ["IV 2SLS (LATE)"]
    coefs = list(ols_results["coef"]) + [iv_coef]
    ci_lows = list(ols_results["ci_low"]) + [iv_ci[0]]
    ci_highs = list(ols_results["ci_high"]) + [iv_ci[1]]
    colors = [PALETTE["naive"], PALETTE["controls"], PALETTE["iv"]]

    y_pos = range(len(models))
    for i, (model, coef, lo, hi, color) in enumerate(zip(models, coefs, ci_lows, ci_highs, colors)):
        ax.barh(i, coef, color=color, alpha=0.85, height=0.5)
        ax.plot([lo, hi], [i, i], color=color, linewidth=2.5)
        ax.plot([lo, lo], [i-0.15, i+0.15], color=color, linewidth=2)
        ax.plot([hi, hi], [i-0.15, i+0.15], color=color, linewidth=2)
        ax.text(max(hi, coef) + 0.0002, i, f"β = {coef:.4f}", va="center", fontsize=10)

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel("Effect of 1-min wait time increase on P(cancellation)", fontsize=11)
    ax.set_title(
        "OLS Understates Wait Time Effect\nIV Corrects for Demand-Side Confounding",
        fontsize=13, fontweight="bold", pad=15
    )

    # Annotation arrow showing bias
    ax.annotate(
        "Endogeneity\nbias ↓",
        xy=(coefs[0], 0), xytext=(coefs[0] - 0.003, 0.7),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="gray"
    )

    plt.tight_layout()
    if save:
        dest = OUTPUTS_FIGURES / "ols_vs_iv_coef.png"
        fig.savefig(dest, bbox_inches="tight")
        logger.info(f"Saved: {dest}")
    return fig


def plot_first_stage(df: pd.DataFrame, save: bool = True):
    """
    Scatter + binned means: rain intensity vs wait time.
    Demonstrates instrument relevance visually.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: binned means
    ax = axes[0]
    df_plot = df.copy()
    df_plot["rain_bin"] = pd.cut(df_plot["rain_intensity_mm"], bins=10)
    binned = df_plot.groupby("rain_bin", observed=True)["wait_time_mins"].mean().reset_index()
    binned["rain_mid"] = binned["rain_bin"].apply(lambda x: x.mid).astype(float)

    ax.bar(range(len(binned)), binned["wait_time_mins"], color="#3498db", alpha=0.8, width=0.7)
    ax.set_xticks(range(len(binned)))
    ax.set_xticklabels([f"{x:.1f}" for x in binned["rain_mid"]], rotation=45, fontsize=8)
    ax.set_xlabel("Rain Intensity (mm/hr)")
    ax.set_ylabel("Mean Wait Time (mins)")
    ax.set_title("First Stage: Rain → Wait Time", fontweight="bold")

    # Right: distribution of wait time by rain/no rain
    ax = axes[1]
    no_rain = df[df["rain_intensity_mm"] <= 0.1]["wait_time_mins"]
    rain = df[df["rain_intensity_mm"] > 0.5]["wait_time_mins"]

    ax.hist(no_rain.clip(0, 25), bins=50, alpha=0.6, color="#3498db", label=f"No Rain (n={len(no_rain):,})", density=True)
    ax.hist(rain.clip(0, 25), bins=50, alpha=0.6, color="#e74c3c", label=f"Rain > 0.5mm (n={len(rain):,})", density=True)
    ax.axvline(no_rain.mean(), color="#3498db", linestyle="--", linewidth=1.5)
    ax.axvline(rain.mean(), color="#e74c3c", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Wait Time (mins)")
    ax.set_ylabel("Density")
    ax.set_title("Wait Time Distribution by Rain Status", fontweight="bold")
    ax.legend()

    plt.suptitle("Instrument Validity: Rain Shifts Wait Time", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        dest = OUTPUTS_FIGURES / "first_stage.png"
        fig.savefig(dest, bbox_inches="tight")
        logger.info(f"Saved: {dest}")
    return fig


def plot_hte_by_borough(borough_hte: pd.DataFrame, save: bool = True):
    """
    Bar chart of mean CATE by borough with confidence intervals.
    Key finding: outer boroughs are more sensitive.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2ecc71" if b in ["Queens", "Bronx", "Staten Island", "Brooklyn"] else "#3498db"
              for b in borough_hte["borough"]]

    bars = ax.barh(
        borough_hte["borough"], borough_hte["mean_cate"],
        color=colors, alpha=0.85, height=0.6
    )

    # Error bars (using std/sqrt(n) as SE)
    se = borough_hte["std_cate"] / np.sqrt(borough_hte["n"])
    ax.errorbar(
        borough_hte["mean_cate"], borough_hte["borough"],
        xerr=1.96 * se, fmt="none", color="black", capsize=4, linewidth=1.5
    )

    for i, (_, row) in enumerate(borough_hte.iterrows()):
        ax.text(
            row["mean_cate"] + 0.001, i,
            f"{row['mean_cate']:+.3f}  (n={row['n']:,})",
            va="center", fontsize=9
        )

    ax.axvline(borough_hte["mean_cate"].mean(), color="black", linestyle="--",
               linewidth=1, label="Overall mean CATE", alpha=0.6)
    ax.set_xlabel("CATE: Effect of +1 min wait time on P(cancellation)")
    ax.set_title(
        "Outer Borough Riders More Sensitive to Wait Time\n(Causal Forest HTE)",
        fontweight="bold"
    )
    ax.legend(fontsize=9)

    outer_patch = mpatches.Patch(color="#2ecc71", alpha=0.85, label="Outer boroughs")
    manhattan_patch = mpatches.Patch(color="#3498db", alpha=0.85, label="Manhattan")
    ax.legend(handles=[outer_patch, manhattan_patch], fontsize=9)

    plt.tight_layout()
    if save:
        dest = OUTPUTS_FIGURES / "hte_by_borough.png"
        fig.savefig(dest, bbox_inches="tight")
        logger.info(f"Saved: {dest}")
    return fig


def plot_cate_distribution(df_test: pd.DataFrame, save: bool = True):
    """Distribution of individual-level CATE estimates."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(df_test["cate"], bins=60, color="#9b59b6", alpha=0.8, edgecolor="white")
    ax.axvline(df_test["cate"].mean(), color="black", linestyle="--",
               linewidth=2, label=f"Mean CATE = {df_test['cate'].mean():.4f}")
    ax.axvline(0, color="#e74c3c", linestyle="-", linewidth=1.5, alpha=0.7, label="Zero effect")

    ax.set_xlabel("Estimated Individual Treatment Effect (CATE)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of CATE Estimates\n(Effect of +1 min wait time on cancellation probability)", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    if save:
        dest = OUTPUTS_FIGURES / "cate_distribution.png"
        fig.savefig(dest, bbox_inches="tight")
        logger.info(f"Saved: {dest}")
    return fig


def generate_all_plots(save: bool = True):
    """Load results and generate all figures."""
    # Load data
    master_path = DATA_PROCESSED / "master.parquet"
    ols_path = OUTPUTS_TABLES / "ols_baseline.csv"
    iv_path = OUTPUTS_TABLES / "iv_results.csv"
    borough_hte_path = OUTPUTS_TABLES / "hte_by_borough.csv"
    cate_path = DATA_PROCESSED / "cate_estimates.parquet"

    figs = {}

    if master_path.exists() and ols_path.exists() and iv_path.exists():
        df = pd.read_parquet(master_path)
        ols = pd.read_csv(ols_path)
        iv = pd.read_csv(iv_path)

        figs["ols_vs_iv"] = plot_ols_vs_iv(
            ols, iv["iv_2sls_coef"].iloc[0],
            (iv["iv_2sls_ci_low"].iloc[0], iv["iv_2sls_ci_high"].iloc[0]),
            save=save
        )
        figs["first_stage"] = plot_first_stage(df, save=save)

    if borough_hte_path.exists():
        borough_hte = pd.read_csv(borough_hte_path)
        figs["hte_borough"] = plot_hte_by_borough(borough_hte, save=save)

    if cate_path.exists():
        df_cate = pd.read_parquet(cate_path)
        figs["cate_dist"] = plot_cate_distribution(df_cate, save=save)

    logger.info(f"Generated {len(figs)} figures.")
    return figs


if __name__ == "__main__":
    generate_all_plots()
