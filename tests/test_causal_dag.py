"""Graph-only tests for the causal DAG. No trip data needed."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.config import TREATMENT_COL, OUTCOME_COL, DAG_CONTROL_SETS
from src.models.causal_dag import (
    build_dag, backdoor_sets, check_instrument, implied_independences,
    observed_nodes, partial_association, d_separated, SELECTION_NODE,
)

RAIN = "rain_intensity_mm"


def test_each_threat_graph_is_acyclic_and_adds_exactly_one_edge():
    a = build_dag("assumed")
    assert nx.is_directed_acyclic_graph(a)
    expected = {"rain_demand": {(RAIN, "rider_demand")},
                "price_channel": {("surge_proxy", OUTCOME_COL)}}
    for v, edge in expected.items():
        g = build_dag(v)
        assert nx.is_directed_acyclic_graph(g)
        assert set(g.edges) - set(a.edges) == edge


def test_latent_nodes_are_never_reported_as_observed():
    assert not {"rider_demand", "driver_supply", "weather_system"} & set(observed_nodes(build_dag()))


def test_no_observed_backdoor_set_exists():
    # Demand is unobserved and confounds wait time with cancellation, so OLS cannot identify the effect
    for v in ("assumed", "rain_demand", "price_channel"):
        assert backdoor_sets(build_dag(v), TREATMENT_COL, OUTCOME_COL) == []


def test_rain_is_valid_with_calendar_controls_under_the_assumed_graph():
    r = check_instrument(build_dag("assumed"), RAIN, TREATMENT_COL, OUTCOME_COL,
                         DAG_CONTROL_SETS["calendar_only"])
    assert r["valid"] and r["violating_path"] == ""


def test_conditioning_on_surge_proxy_opens_a_collider_path():
    # surge_proxy is a common child of supply and demand. Conditioning on it links rain to demand.
    r = check_instrument(build_dag("assumed"), RAIN, TREATMENT_COL, OUTCOME_COL,
                         DAG_CONTROL_SETS["published"])
    assert not r["exclusion_holds"]
    assert "surge_proxy" in r["violating_path"] and "rider_demand" in r["violating_path"]


def test_rain_is_never_valid_under_either_threat():
    for v in ("rain_demand", "price_channel"):
        G = build_dag(v)
        for controls in DAG_CONTROL_SETS.values():
            assert not check_instrument(G, RAIN, TREATMENT_COL, OUTCOME_COL, controls)["valid"]


def test_unconditioned_rain_is_invalid_because_hour_confounds_it():
    r = check_instrument(build_dag("assumed"), RAIN, TREATMENT_COL, OUTCOME_COL, [])
    assert not r["valid"]


def test_implied_independences_only_use_observed_variables():
    G = build_dag("assumed")
    obs = set(observed_nodes(G))
    imps = implied_independences(G)
    assert imps, "graph should imply at least one testable independence"
    for i in imps:
        assert {i["x"], i["y"], *i["given"]} <= obs
        assert d_separated(G, i["x"], i["y"], i["given"])


def test_selection_on_demand_makes_calendar_variables_dependent():
    # Rows are recorded trips. Conditioning on that collider links hour and weekend through demand.
    G = build_dag("assumed")
    assert SELECTION_NODE not in observed_nodes(G)
    assert nx.is_d_separator(G, {"hour_of_day"}, {"is_weekend"}, set())
    assert not d_separated(G, "hour_of_day", "is_weekend", [])


def test_rain_weekend_independence_distinguishes_the_assumed_graph_from_rain_demand():
    given = ["borough", "hour_of_day"]
    assert d_separated(build_dag("assumed"), RAIN, "is_weekend", given)
    assert not d_separated(build_dag("rain_demand"), RAIN, "is_weekend", given)


def test_partial_association_separates_a_chain_from_a_direct_link():
    rng = np.random.default_rng(0)
    n = 20_000
    z = rng.normal(size=n)
    x = z + rng.normal(size=n)
    y = z + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    assert partial_association(df, "x", "y", [])["max_abs_partial_corr"] > 0.3
    assert partial_association(df, "x", "y", ["z"])["max_abs_partial_corr"] < 0.02
