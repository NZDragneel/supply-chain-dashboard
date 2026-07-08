"""
=============================================================================
PRESCRIPTIVE ANALYTICS ENGINE v2 — MILP ROUTE OPTIMISATION + COST OF INACTION
Group 14 | IS6611 | Cork University Business School | 2025-2026

WHAT CHANGED FROM v1 (LP) → v2 (MILP):
  The original LP returned fractional allocations (e.g. "send 37.4% via Cape").
  Real logistics decisions are binary: you either contract a shipping lane or
  you don't. This MILP adds binary route-selection variables (y_i ∈ {0,1}),
  so the model first decides WHICH routes to activate, then allocates volume
  only across active routes.

  This is academically standard (Chopra & Meindl 2016, Ch.5) and matches
  how Irish pharma procurement teams actually operate — they sign framework
  agreements with specific carriers, not fractional contracts.

WHY NO ERP DATA IS NEEDED:
  We use BDI-derived cost indices as proxy variables for ERP-level unit costs.
  Chopra & Meindl (2016) establish that relative cost ratios are sufficient
  for route selection optimality when absolute values are unavailable.
  Our BDI signals (bdi_freight_rate_asia_eu, bdi_suez_premium) provide
  these relative ratios directly from market data.

SOLVER: CBC (via PuLP) — open-source, audit-compliant, EU AI Act compatible.
=============================================================================
"""

import numpy as np
import pulp

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE NETWORK
# Nodes: Mumbai/Hyderabad (source), Shanghai (source), Dublin (destination)
# Edges: 5 routes with cost indices derived from BDI market data
# ─────────────────────────────────────────────────────────────────────────────
ROUTES = {
    "Suez Canal — India": {
        "source": "Mumbai / Hyderabad",
        "hub":    "Suez Canal",
        "dest":   "Dublin Port",
        "base_cost_index":  100,      # BDI baseline = 100
        "transit_days":     22,
        "choke_exposure":   "HIGH",   # Red Sea + Suez choke risk
        "co2_index":        1.00,
        "fixed_setup_cost": 8,        # Cost of activating this lane (MILP integer cost)
        "max_share":        1.0,
    },
    "Cape of Good Hope — India": {
        "source": "Mumbai / Hyderabad",
        "hub":    "Cape of Good Hope",
        "dest":   "Dublin Port",
        "base_cost_index":  145,
        "transit_days":     34,
        "choke_exposure":   "LOW",
        "co2_index":        1.40,
        "fixed_setup_cost": 10,
        "max_share":        1.0,
    },
    "Air Freight — India": {
        "source": "Mumbai / Hyderabad",
        "hub":    "Frankfurt Hub",
        "dest":   "Dublin Airport",
        "base_cost_index":  620,
        "transit_days":     2,
        "choke_exposure":   "NONE",
        "co2_index":        8.50,
        "fixed_setup_cost": 25,
        "max_share":        0.5,      # Air capacity capped at 50% of volume
    },
    "Suez Canal — China": {
        "source": "Shanghai",
        "hub":    "Suez Canal",
        "dest":   "Dublin Port",
        "base_cost_index":  110,
        "transit_days":     28,
        "choke_exposure":   "HIGH",
        "co2_index":        1.10,
        "fixed_setup_cost": 9,
        "max_share":        1.0,
    },
    "Cape of Good Hope — China": {
        "source": "Shanghai",
        "hub":    "Cape of Good Hope",
        "dest":   "Dublin Port",
        "base_cost_index":  160,
        "transit_days":     38,
        "choke_exposure":   "LOW",
        "co2_index":        1.50,
        "fixed_setup_cost": 11,
        "max_share":        1.0,
    },
}

ROUTE_NAMES = list(ROUTES.keys())
N_ROUTES    = len(ROUTE_NAMES)

# ─────────────────────────────────────────────────────────────────────────────
# RISK-CLASS CONSTRAINTS
# These constraints convert the ML risk class into LP/MILP parameters.
# Source: adapted from Chopra & Meindl (2016) risk-pooling framework.
# ─────────────────────────────────────────────────────────────────────────────
CLASS_CONSTRAINTS = {
    0: {
        "label":                "Stable — Cost-Optimised Routing",
        "max_choke_fraction":   1.00,   # No restriction on choke routes
        "risk_cost_multiplier": 1.0,    # No risk penalty
        "force_air_min_frac":   0.00,
        "max_active_routes":    2,      # MILP: activate at most 2 routes (cost efficiency)
    },
    1: {
        "label":                "Minor Stress — Begin Diversification",
        "max_choke_fraction":   0.70,
        "risk_cost_multiplier": 1.30,
        "force_air_min_frac":   0.00,
        "max_active_routes":    3,
    },
    2: {
        "label":                "Medium Disruption — Major Rerouting",
        "max_choke_fraction":   0.40,
        "risk_cost_multiplier": 1.80,
        "force_air_min_frac":   0.05,
        "max_active_routes":    4,
    },
    3: {
        "label":                "Major Crisis — BCP Emergency Routing",
        "max_choke_fraction":   0.10,
        "risk_cost_multiplier": 5.00,
        "force_air_min_frac":   0.30,
        "max_active_routes":    5,      # All routes may be needed
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# BDI-DERIVED COST SCALING
# Adjusts route cost indices using live BDI signal from the dataset.
# When BDI is high (shipping stress), sea routes become relatively more
# expensive vs. air, shifting the MILP towards diversification.
# ─────────────────────────────────────────────────────────────────────────────
def get_bdi_adjusted_costs(bdi_value: float, suez_premium: float) -> dict:
    """
    Scales route costs using actual BDI data from the dataset.
    BDI baseline = 1500. Suez premium baseline = 80.
    Returns a dict of {route_name: adjusted_cost_index}.
    """
    bdi_scale    = bdi_value / 1500.0     # >1 = shipping expensive, <1 = cheap
    suez_scale   = suez_premium / 80.0    # >1 = Suez premium elevated

    adjusted = {}
    for name, r in ROUTES.items():
        base = r["base_cost_index"]
        if r["choke_exposure"] == "HIGH":
            # Suez/Red Sea routes scale with both BDI and Suez premium
            adjusted[name] = base * bdi_scale * suez_scale
        elif r["choke_exposure"] == "LOW":
            # Cape routes scale only with general BDI (not Suez-specific)
            adjusted[name] = base * bdi_scale
        else:
            # Air freight has fixed cost (independent of sea BDI)
            adjusted[name] = base
    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# MILP SOLVER
# ─────────────────────────────────────────────────────────────────────────────
def solve_milp(
    risk_class:   int,
    total_volume: float = 1000.0,
    bdi_value:    float = 1500.0,
    suez_premium: float = 80.0,
) -> dict:
    """
    Solves the Mixed-Integer Linear Programme:

    Decision variables:
      x_i ∈ [0, total_volume]  — continuous volume on route i
      y_i ∈ {0, 1}             — binary: is route i activated?

    Objective (minimise):
      Σ (effective_cost_i × x_i)  +  Σ (fixed_setup_cost_i × y_i)

    Subject to:
      Σ x_i = total_volume                                [demand satisfaction]
      x_i ≤ max_share_i × total_volume × y_i             [volume ≤ capacity if active]
      x_i ≥ 0                                            [non-negativity]
      Σ y_i ≤ max_active_routes                          [MILP: lane activation limit]
      Σ x_i [HIGH choke] ≤ max_choke_fraction × total   [risk exposure cap]
      x_air ≥ force_air_min × total                      [emergency air minimum]
    """
    constraints = CLASS_CONSTRAINTS[risk_class]
    adj_costs   = get_bdi_adjusted_costs(bdi_value, suez_premium)
    risk_mult   = constraints["risk_cost_multiplier"]

    # Effective cost = BDI-adjusted base × risk class multiplier for HIGH routes
    eff_costs = {}
    for name in ROUTE_NAMES:
        r = ROUTES[name]
        c = adj_costs[name]
        if r["choke_exposure"] == "HIGH":
            eff_costs[name] = c * risk_mult
        else:
            eff_costs[name] = c

    prob = pulp.LpProblem("SupplyChain_MILP", pulp.LpMinimize)

    # Continuous volume variables
    x = {name: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=total_volume)
         for i, name in enumerate(ROUTE_NAMES)}

    # Binary activation variables (the MILP part)
    y = {name: pulp.LpVariable(f"y_{i}", cat="Binary")
         for i, name in enumerate(ROUTE_NAMES)}

    # Objective
    prob += (
        pulp.lpSum(eff_costs[n] * x[n] for n in ROUTE_NAMES) +
        pulp.lpSum(ROUTES[n]["fixed_setup_cost"] * 1000 * y[n] for n in ROUTE_NAMES)
    )

    # Constraint 1: all demand must be met
    prob += pulp.lpSum(x[n] for n in ROUTE_NAMES) == total_volume

    # Constraint 2: volume only on active routes (big-M linking)
    for name in ROUTE_NAMES:
        max_cap = ROUTES[name]["max_share"] * total_volume
        prob += x[name] <= max_cap * y[name]

    # Constraint 3: max active routes (MILP integer constraint)
    prob += pulp.lpSum(y[n] for n in ROUTE_NAMES) <= constraints["max_active_routes"]

    # Constraint 4: choke-point exposure cap
    choke_routes = [n for n in ROUTE_NAMES if ROUTES[n]["choke_exposure"] == "HIGH"]
    prob += (pulp.lpSum(x[n] for n in choke_routes) <=
             constraints["max_choke_fraction"] * total_volume)

    # Constraint 5: minimum air freight (emergency)
    air_routes = [n for n in ROUTE_NAMES if "Air" in n]
    air_min    = constraints["force_air_min_frac"] * total_volume
    if air_routes and air_min > 0:
        prob += pulp.lpSum(x[n] for n in air_routes) >= air_min

    # Solve (CBC solver, silent)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[status] == "Optimal":
        allocation     = {}
        active_routes  = []
        for name in ROUTE_NAMES:
            vol = pulp.value(x[name]) or 0.0
            act = pulp.value(y[name]) or 0
            if vol > total_volume * 0.005:
                allocation[name] = round(vol, 1)
            if act > 0.5:
                active_routes.append(name)

        # Compute output metrics
        base_cost = sum(
            ROUTES[n]["base_cost_index"] * v
            for n, v in allocation.items()
        )
        avg_transit = sum(
            ROUTES[n]["transit_days"] * (v / total_volume)
            for n, v in allocation.items()
        )
        choke_pct = sum(
            v for n, v in allocation.items()
            if ROUTES[n]["choke_exposure"] == "HIGH"
        ) / total_volume * 100

        co2 = sum(
            ROUTES[n]["co2_index"] * (v / total_volume)
            for n, v in allocation.items()
        )

        return {
            "success":            True,
            "allocation":         allocation,
            "active_routes":      active_routes,
            "n_active_routes":    len(active_routes),
            "total_volume":       total_volume,
            "base_cost_index":    round(base_cost, 0),
            "avg_transit_days":   round(avg_transit, 1),
            "choke_exposure_pct": round(choke_pct, 1),
            "co2_index":          round(co2, 2),
            "constraint_label":   constraints["label"],
            "bdi_adjusted":       True,
        }
    else:
        return {
            "success":          False,
            "error":            f"Solver status: {pulp.LpStatus[status]}",
            "constraint_label": constraints["label"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# INVENTORY RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────
def get_inventory_recommendation(risk_class: int, baseline_days: int = 30) -> dict:
    multipliers = {0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0}
    supplier_split = {
        0: {"India (Primary)": 70,  "China (Secondary)": 25, "EU Near-shore": 5},
        1: {"India (Primary)": 60,  "China (Secondary)": 30, "EU Near-shore": 10},
        2: {"India (Primary)": 45,  "China (Secondary)": 30, "EU Near-shore": 25},
        3: {"India (Primary)": 30,  "China (Secondary)": 20, "EU Near-shore": 50},
    }
    m = multipliers[risk_class]
    return {
        "recommended_stock_days":      round(baseline_days * m),
        "increase_vs_baseline_pct":    round((m - 1) * 100),
        "supplier_split":              supplier_split[risk_class],
        "holding_cost_rate_per_day":   0.0025,   # 0.25%/day — Chopra & Meindl 2016
    }


# ─────────────────────────────────────────────────────────────────────────────
# COST OF INACTION CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

# Cost parameters derived from BDI market data and industry literature
# Sources: Chopra & Sodhi (2004), BDI averages, Drewry Container Rate Index
COST_PARAMS = {
    # Air vs sea freight premium per tonne — derived from route cost indices
    # Air index 620 vs Suez index 100 → premium ratio ~5.2x
    # At average spot rate $800/tonne sea freight (Drewry 2023): air adds ~$3,200/tonne
    "air_freight_premium_per_tonne_usd":   3200,

    # Cape reroute adds ~12 extra days and ~45% cost vs Suez
    # At $800/tonne base rate: extra cost = $360/tonne per voyage
    "cape_reroute_premium_per_tonne_usd":  360,

    # Safety stock holding cost: 0.25%/day of inventory value
    # Based on Chopra & Meindl (2016) standard pharma holding rate
    # Assumes €150/kg API average value, 1 tonne = 1000 kg
    "holding_cost_per_tonne_per_day_eur":  375,

    # BDI premium uplift: for each 100-point BDI increase above baseline (1500),
    # sea freight costs rise approx $12/tonne (Drewry Container Rate Index 2023)
    "bdi_uplift_per_100pts_per_tonne_usd": 12,

    # EUR/USD rate for conversion
    "eur_usd_rate":                        1.08,
}

# What happens if you delay action by N days at each class transition
CLASS_TRANSITION_COSTS = {
    # Delay acting on Class 1 warning: you miss the early booking window
    # Extra cost = being forced into spot market at Class 2 rates
    "0_to_1": {
        "label":            "Ignoring Class 0→1 signal",
        "description":      "Failing to pre-book alternative routes during stable period forces spot-market procurement at stress prices.",
        "daily_cost_basis": "cape_reroute_premium_per_tonne_usd",
        "volume_fraction":  0.30,   # ~30% of volume affected
        "extra_stock_days": 15,     # Safety stock increase needed
    },
    "1_to_2": {
        "label":            "Ignoring Class 1→2 warning",
        "description":      "Waiting until Medium Disruption forces emergency Cape rerouting at premium. Spot market rates 40–80% above contracted.",
        "daily_cost_basis": "cape_reroute_premium_per_tonne_usd",
        "volume_fraction":  0.50,
        "extra_stock_days": 20,
    },
    "2_to_3": {
        "label":            "Ignoring Class 2→3 alert",
        "description":      "Crisis activation requires emergency air freight for time-critical APIs. Air premium is 5–6× sea cost.",
        "daily_cost_basis": "air_freight_premium_per_tonne_usd",
        "volume_fraction":  0.30,
        "extra_stock_days": 30,
    },
}


def calculate_cost_of_inaction(
    current_class:    int,
    delay_days:       int,
    monthly_volume_t: float,
    bdi_value:        float = 1500.0,
    suez_premium:     float = 80.0,
) -> dict:
    """
    Calculates the estimated financial cost of delaying response
    to the current risk class signal for `delay_days` days.

    Returns itemised cost breakdown in EUR.

    Formula:
      1. Route premium cost  = (affected_volume × premium_per_tonne)
         adjusted by BDI uplift above baseline
      2. Holding cost        = (extra_stock_days × volume × holding_rate)
      3. Total avoidable cost = route premium + holding cost
      4. Cost per day        = total / delay_days

    Academic basis: Chopra & Sodhi (2004) disruption cost model;
    BDI scaling per Drewry Container Rate Index methodology.
    """
    eur_usd = COST_PARAMS["eur_usd_rate"]
    daily_volume = monthly_volume_t / 30.0

    # BDI uplift: how much has freight already increased above baseline?
    bdi_above_baseline   = max(0, bdi_value - 1500) / 100
    suez_above_baseline  = max(0, suez_premium - 80) / 80
    bdi_uplift_per_tonne = bdi_above_baseline * COST_PARAMS["bdi_uplift_per_100pts_per_tonne_usd"]

    results = {}

    # Transitions that apply at or above the current class
    transitions_to_show = []
    if current_class >= 0:
        transitions_to_show.append(("0_to_1", max(0, current_class - 0)))
    if current_class >= 1:
        transitions_to_show.append(("1_to_2", max(0, current_class - 1)))
    if current_class >= 2:
        transitions_to_show.append(("2_to_3", max(0, current_class - 2)))

    for key, _ in transitions_to_show:
        t = CLASS_TRANSITION_COSTS[key]
        base_premium    = COST_PARAMS[t["daily_cost_basis"]]
        adjusted_premium = base_premium + bdi_uplift_per_tonne

        affected_volume  = monthly_volume_t * t["volume_fraction"]
        route_cost_usd   = affected_volume * adjusted_premium
        route_cost_eur   = route_cost_usd / eur_usd

        holding_cost_eur = (
            t["extra_stock_days"] *
            daily_volume *
            COST_PARAMS["holding_cost_per_tonne_per_day_eur"]
        )

        total_avoidable_eur  = route_cost_eur + holding_cost_eur
        # Scale by fraction of delay period (costs accumulate over delay)
        delay_multiplier     = min(delay_days / 14.0, 3.0)  # cap at 3x (6-week max)
        total_with_delay_eur = total_avoidable_eur * delay_multiplier

        results[key] = {
            "label":                  t["label"],
            "description":            t["description"],
            "route_freight_cost_eur": round(route_cost_eur),
            "holding_cost_eur":       round(holding_cost_eur),
            "total_avoidable_eur":    round(total_avoidable_eur),
            "cost_with_delay_eur":    round(total_with_delay_eur),
            "cost_per_day_eur":       round(total_avoidable_eur * (1/14)),
            "bdi_uplift_applied_usd": round(bdi_uplift_per_tonne, 1),
        }

    # Summary for current class
    if results:
        most_relevant_key = list(results.keys())[-1]
        mr = results[most_relevant_key]
        headline = mr["cost_with_delay_eur"]
        per_day  = mr["cost_per_day_eur"]
    else:
        headline = 0
        per_day  = 0

    return {
        "breakdown":               results,
        "headline_cost_eur":       headline,
        "cost_per_day_eur":        per_day,
        "delay_days":              delay_days,
        "monthly_volume_t":        monthly_volume_t,
        "current_class":           current_class,
        "bdi_value":               bdi_value,
        "bdi_uplift_per_tonne":    round(bdi_uplift_per_tonne, 1),
        "note": (
            "Costs are estimated using BDI-derived freight indices as proxy for ERP-level "
            "unit costs (Chopra & Meindl 2016). Holding cost at 0.25%/day of API inventory "
            "value (industry standard). All figures in EUR at 1.08 EUR/USD."
        ),
    }
