"""
causal_dag.py — The causal graph behind the wait-time → cancellation study.

The IV analysis rests on assumptions that iv_2sls.py states in prose. This module
writes them down as a directed acyclic graph and asks the graph four questions:

  1. Is there any set of observed controls that closes every backdoor path from
     wait time to cancellation? (If not, OLS cannot identify the effect.)
  2. For a given control set, is rainfall a valid instrument by the graphical
     criterion (Brito & Pearl 2002)? If not, which path breaks it?
  3. Which conditional independences does the graph imply among observed
     variables, and do they hold in the data?
  4. How much does the 2SLS estimate move across the control sets the graph
     does and does not license?

Three graphs are compared:
  - "assumed":       rain reaches cancellation only through driver supply → wait time
  - "rain_demand":   threat 1, rain also raises rider demand
  - "price_channel": threat 2, the price signal affects cancellation directly, which
                     gives rain a second route to the outcome (supply → price → cancel)

Rider demand and driver supply are unobserved. surge_proxy is a noisy child of both.

Every row in the data is a recorded trip request, so the sample is selected on rider
demand. The graph carries that as a `trip_recorded` node that every d-separation
query conditions on. Without it the graph wrongly implies that hour, weekend,
holiday and borough are independent of each other among recorded trips.
"""

import logging
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    DATA_PROCESSED, OUTPUTS_TABLES, OUTPUTS_FIGURES,
    TREATMENT_COL, OUTCOME_COL, RANDOM_STATE,
    DAG_LATENT_NODES, DAG_INSTRUMENTS, DAG_CONTROL_SETS,
    DAG_CI_EFFECT_THRESHOLD, DAG_MAX_CONDITIONING_SIZE, DAG_SAMPLE_N,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DAG_VARIANTS = ("assumed", "rain_demand", "price_channel")
THREAT_EDGES = {
    "rain_demand": [("rain_intensity_mm", "rider_demand")],
    "price_channel": [("surge_proxy", OUTCOME_COL)],
}
SELECTION_NODE = "trip_recorded"
CATEGORICAL = {"hour_of_day", "borough"}
CALENDAR = ["hour_of_day", "is_weekend", "is_holiday", "borough"]


# ── Graph definition ────────────────────────────────────────────────────────
def build_dag(variant: str = "assumed") -> nx.DiGraph:
    """Return the study's causal graph, or the assumed graph plus one threat edge."""
    if variant not in DAG_VARIANTS:
        raise ValueError(f"Unknown DAG variant: {variant}")

    G = nx.DiGraph()
    edges = [
        # Summer rain has a diurnal cycle, and the three stations differ by borough
        ("hour_of_day", "rain_intensity_mm"),
        ("borough", "rain_intensity_mm"),
        ("hour_of_day", "wind_speed_ms"),
        ("borough", "wind_speed_ms"),
        # One weather system drives both rain and wind
        ("weather_system", "rain_intensity_mm"),
        ("weather_system", "wind_speed_ms"),
        # Weather slows traffic and thins driver supply
        ("rain_intensity_mm", "driver_supply"),
        ("wind_speed_ms", "driver_supply"),
        # Demand and supply set the price signal and the wait
        ("rider_demand", "surge_proxy"),
        ("driver_supply", "surge_proxy"),
        ("rider_demand", TREATMENT_COL),
        ("driver_supply", TREATMENT_COL),
        # The confounding story: busy periods produce more cancellations directly
        ("rider_demand", OUTCOME_COL),
        # The effect under study
        (TREATMENT_COL, OUTCOME_COL),
        # Selection: a row exists only because a rider asked for a trip
        ("rider_demand", SELECTION_NODE),
    ]
    for c in CALENDAR:
        edges += [(c, "rider_demand"), (c, "driver_supply"), (c, OUTCOME_COL)]
    edges += THREAT_EDGES.get(variant, [])

    G.add_edges_from(edges)
    for n in G.nodes:
        G.nodes[n]["latent"] = n in DAG_LATENT_NODES
    assert nx.is_directed_acyclic_graph(G), "Causal graph contains a cycle"
    return G


def observed_nodes(G: nx.DiGraph) -> list[str]:
    return sorted(n for n in G.nodes if not G.nodes[n]["latent"] and n != SELECTION_NODE)


def d_separated(G: nx.DiGraph, x: str, y: str, given) -> bool:
    """d-separation among recorded trips: the selection node is always conditioned on."""
    return nx.is_d_separator(G, {x}, {y}, set(given) | {SELECTION_NODE})


# ── Identification ──────────────────────────────────────────────────────────
def _open_path(G: nx.DiGraph, x: str, y: str, given: set[str]) -> str:
    """Return one d-connecting path between x and y given `given`, or ''."""
    und = G.to_undirected()
    anc = set(given)
    for g in given:
        anc |= nx.ancestors(G, g)
    for path in nx.all_simple_paths(und, x, y, cutoff=6):
        blocked = False
        for a, b, c in zip(path, path[1:], path[2:]):
            collider = G.has_edge(a, b) and G.has_edge(c, b)
            if collider and b not in anc:
                blocked = True
            if not collider and b in given:
                blocked = True
            if blocked:
                break
        if not blocked:
            return " - ".join(path)
    return ""


def backdoor_sets(G: nx.DiGraph, treatment: str, outcome: str) -> list[tuple[str, ...]]:
    """Every set of observed non-descendants of the treatment that satisfies the backdoor criterion."""
    candidates = [n for n in observed_nodes(G)
                  if n not in (treatment, outcome) and n not in nx.descendants(G, treatment)]
    G_back = G.copy()
    G_back.remove_edges_from(list(G.out_edges(treatment)))
    valid = []
    for k in range(len(candidates) + 1):
        for s in combinations(candidates, k):
            if d_separated(G_back, treatment, outcome, s):
                valid.append(s)
    return valid


def check_instrument(G: nx.DiGraph, instrument: str, treatment: str, outcome: str,
                     controls: list[str]) -> dict:
    """
    Graphical instrument criterion. `instrument` is valid given `controls` when
      (a) no control is a descendant of the outcome,
      (b) controls d-separate instrument and outcome once the treatment → outcome edge is cut,
      (c) controls do NOT d-separate instrument and treatment in that same graph.
    """
    W = set(controls)
    G_cut = G.copy()
    G_cut.remove_edge(treatment, outcome)

    no_descendants = not (W & nx.descendants(G, outcome))
    exclusion = d_separated(G_cut, instrument, outcome, W)
    relevance = not d_separated(G_cut, instrument, treatment, W)
    return {
        "instrument": instrument,
        "controls": ", ".join(controls) if controls else "(none)",
        "controls_not_descendants_of_outcome": no_descendants,
        "exclusion_holds": exclusion,
        "relevance_holds": relevance,
        "valid": no_descendants and exclusion and relevance,
        "violating_path": "" if exclusion else _open_path(G_cut, instrument, outcome, W | {SELECTION_NODE}),
    }


def identification_report(variants=DAG_VARIANTS) -> pd.DataFrame:
    rows = []
    for v in variants:
        G = build_dag(v)
        n_backdoor = len(backdoor_sets(G, TREATMENT_COL, OUTCOME_COL))
        for set_name, controls in DAG_CONTROL_SETS.items():
            for z in DAG_INSTRUMENTS:
                r = check_instrument(G, z, TREATMENT_COL, OUTCOME_COL, controls)
                rows.append({"dag": v, "control_set": set_name,
                             "observed_backdoor_sets": n_backdoor, **r})
    return pd.DataFrame(rows)


# ── Testable implications ───────────────────────────────────────────────────
def implied_independences(G: nx.DiGraph, max_size: int = DAG_MAX_CONDITIONING_SIZE) -> list[dict]:
    """
    For each non-adjacent pair of observed variables, the smallest observed
    conditioning set that d-separates them. Pairs with no such set carry no
    testable implication and are skipped.
    """
    obs = observed_nodes(G)
    out = []
    for x, y in combinations(obs, 2):
        if G.has_edge(x, y) or G.has_edge(y, x):
            continue
        others = [n for n in obs if n not in (x, y)]
        found = None
        for k in range(max_size + 1):
            for s in combinations(others, k):
                if d_separated(G, x, y, s):
                    found = s
                    break
            if found is not None:
                break
        if found is not None:
            out.append({"x": x, "y": y, "given": found})
    return out


def _design(df: pd.DataFrame, cols) -> np.ndarray:
    parts = [np.ones((len(df), 1))]
    for c in cols:
        if c in CATEGORICAL:
            parts.append(pd.get_dummies(df[c], drop_first=True).to_numpy(dtype=float))
        else:
            parts.append(df[[c]].to_numpy(dtype=float))
    return np.hstack(parts)


def _residualize(df: pd.DataFrame, col: str, given) -> np.ndarray:
    """Residuals of `col` on `given`. A categorical column is residualized dummy by dummy."""
    Y = (pd.get_dummies(df[col], drop_first=True).to_numpy(dtype=float)
         if col in CATEGORICAL else df[[col]].to_numpy(dtype=float))
    X = _design(df, given)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta


def partial_association(df: pd.DataFrame, x: str, y: str, given) -> dict:
    """
    Largest absolute partial correlation between x and y after regressing both on
    `given`. With hundreds of thousands of rows every p-value is tiny, so the
    verdict rests on the effect size, not on significance.
    """
    rx, ry = _residualize(df, x, given), _residualize(df, y, given)
    best_r, best_p = 0.0, 1.0
    for i in range(rx.shape[1]):
        for j in range(ry.shape[1]):
            if rx[:, i].std() == 0 or ry[:, j].std() == 0:
                continue
            r, p = stats.pearsonr(rx[:, i], ry[:, j])
            if abs(r) > abs(best_r):
                best_r, best_p = float(r), float(p)
    return {"max_abs_partial_corr": abs(best_r), "pval": best_p}


def test_implications(df: pd.DataFrame, variant: str = "assumed") -> pd.DataFrame:
    G = build_dag(variant)
    rows = []
    for imp in implied_independences(G):
        res = partial_association(df, imp["x"], imp["y"], imp["given"])
        rows.append({
            "dag": variant,
            "x": imp["x"], "y": imp["y"],
            "given": ", ".join(imp["given"]) if imp["given"] else "(none)",
            **res,
            "threshold": DAG_CI_EFFECT_THRESHOLD,
            "holds": res["max_abs_partial_corr"] < DAG_CI_EFFECT_THRESHOLD,
        })
    return pd.DataFrame(rows)


# ── IV sensitivity across control sets ──────────────────────────────────────
def iv_by_control_set(df: pd.DataFrame) -> pd.DataFrame:
    """2SLS estimate of wait time on cancellation under each named control set."""
    import statsmodels.api as sm
    from linearmodels.iv import IV2SLS

    rows = []
    for set_name, controls in DAG_CONTROL_SETS.items():
        cols = [TREATMENT_COL, OUTCOME_COL, *DAG_INSTRUMENTS, *controls]
        d = df.dropna(subset=cols)
        exog = sm.add_constant(pd.DataFrame(_design(d, controls)[:, 1:], index=d.index))
        exog.columns = [str(c) for c in exog.columns]
        model = IV2SLS(d[OUTCOME_COL].astype(float), exog,
                       d[[TREATMENT_COL]].astype(float),
                       d[DAG_INSTRUMENTS].astype(float)).fit(cov_type="robust")
        ci = model.conf_int().loc[TREATMENT_COL]
        fs = model.first_stage.diagnostics.loc[TREATMENT_COL]
        rows.append({
            "control_set": set_name,
            "controls": ", ".join(controls) if controls else "(none)",
            "iv_2sls_coef": float(model.params[TREATMENT_COL]),
            "se": float(model.std_errors[TREATMENT_COL]),
            "ci_low": float(ci.iloc[0]), "ci_high": float(ci.iloc[1]),
            "first_stage_partial_f": float(fs["f.stat"]),
            "n": int(model.nobs),
        })
        logger.info(f"2SLS [{set_name}]: β={rows[-1]['iv_2sls_coef']:.5f} "
                    f"(SE {rows[-1]['se']:.5f}), partial F={rows[-1]['first_stage_partial_f']:.1f}")
    return pd.DataFrame(rows)


# ── Figure ──────────────────────────────────────────────────────────────────
def plot_dag(save: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = build_dag("assumed")
    threat = [e for edges in THREAT_EDGES.values() for e in edges]
    G.add_edges_from(threat)
    pos = {
        "weather_system": (0, 3), "rain_intensity_mm": (1.5, 3.6), "wind_speed_ms": (1.5, 2.4),
        "driver_supply": (3.2, 3.0), "rider_demand": (3.2, 1.0),
        "surge_proxy": (4.8, 2.0), TREATMENT_COL: (4.8, 3.6), OUTCOME_COL: (6.6, 2.4),
        "hour_of_day": (0.3, 1.2), "borough": (0.3, -0.3),
        "is_weekend": (1.9, 0.0), "is_holiday": (3.4, -0.3), SELECTION_NODE: (4.8, 0.3),
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    latent = [n for n in G if G.nodes[n]["latent"]]
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G if n not in latent],
                           node_color="#dbe9f6", edgecolors="#2b5c8a", node_size=2600, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=latent, node_color="white",
                           edgecolors="#888888", linewidths=1.5, node_size=2600, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=[e for e in G.edges if e not in threat],
                           edge_color="#555555", arrowsize=14, node_size=2600, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=threat, edge_color="#c0392b", style="dashed",
                           arrowsize=14, node_size=2600, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n.replace("_", "\n") for n in G}, font_size=7, ax=ax)
    ax.set_title("Assumed causal graph (white = unobserved, dashed red = the two threats to exclusion)")
    ax.axis("off")
    if save:
        dest = OUTPUTS_FIGURES / "causal_dag.png"
        fig.savefig(dest, dpi=150, bbox_inches="tight")
        logger.info(f"Saved DAG figure: {dest}")
    plt.close(fig)


# ── Runner ──────────────────────────────────────────────────────────────────
def run_dag_analysis(df: pd.DataFrame = None, save: bool = True) -> dict:
    ident = identification_report()
    for _, r in ident.iterrows():
        logger.info(f"[{r['dag']}] {r['instrument']} | controls={r['control_set']} → "
                    f"{'VALID' if r['valid'] else 'INVALID via ' + r['violating_path']}")

    if df is None:
        master_path = DATA_PROCESSED / "master.parquet"
        if not master_path.exists():
            raise FileNotFoundError("master.parquet not found. Run join.py first.")
        df = pd.read_parquet(master_path)
    if len(df) > DAG_SAMPLE_N:
        logger.info(f"Sampling {DAG_SAMPLE_N:,} rows from {len(df):,} for DAG tests")
        df = df.sample(DAG_SAMPLE_N, random_state=RANDOM_STATE)

    implications = pd.concat([test_implications(df, v) for v in DAG_VARIANTS],
                             ignore_index=True)
    held = implications.groupby("dag")["holds"].agg(["sum", "count"])
    for v, row in held.iterrows():
        logger.info(f"[{v}] implied independences holding: {int(row['sum'])} of {int(row['count'])}")

    sensitivity = iv_by_control_set(df)

    if save:
        ident.to_csv(OUTPUTS_TABLES / "dag_identification.csv", index=False)
        implications.to_csv(OUTPUTS_TABLES / "dag_implied_independences.csv", index=False)
        sensitivity.to_csv(OUTPUTS_TABLES / "dag_iv_sensitivity.csv", index=False)
        plot_dag(save=True)
        logger.info(f"Saved DAG tables to {OUTPUTS_TABLES}")

    return {"identification": ident, "implications": implications, "sensitivity": sensitivity}


if __name__ == "__main__":
    run_dag_analysis()
