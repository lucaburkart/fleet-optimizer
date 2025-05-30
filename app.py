# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatusOptimal, value

# 1) Page configuration
st.set_page_config(page_title="Fleet Optimization", layout="wide")

# 2) Read input data
BASE_PATH = Path(".")
fleet     = pd.read_csv(BASE_PATH / "fleet_data2.1.csv",     delimiter=";")
fuel      = pd.read_csv(BASE_PATH / "tech_fuel_data2.1.csv", delimiter=";")
turbo     = pd.read_csv(BASE_PATH / "turbo_retrofit.1.csv",  delimiter=";")
new_cost  = pd.read_csv(BASE_PATH / "new_ship_cost.1.csv",   delimiter=";")
new_specs = pd.read_csv(BASE_PATH / "new_fleet_data2.1.csv", delimiter=";")

# Normalize string columns only if they exist
for df in (fleet, fuel, turbo, new_cost, new_specs):
    for col in ("Ship_Type", "Fuel", "Fuel_Type"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

# Build lookups
fuel_lu = fuel.set_index(["Year", "Fuel_Type"]).to_dict("index")
T_COST  = turbo.set_index(["Ship_Type", "Year"])["Retrofit_Cost_USD"].to_dict()
T_SAVE  = turbo.set_index(["Ship_Type", "Year"])["Energy_Saving_%"].to_dict()
N_COST  = new_cost.set_index(["Ship_Type", "Fuel", "Year"])["Capex_USD"].to_dict()

# New ship specs lookup
new_specs_idx = new_specs.set_index("Ship_Type")
new_lu = {
    ship: {
        "Energy_per_km_new": row.get("Energy_per_km (MJ/km)_new", row.get("Energy_per_km_new")),
        "Power_new": row.get("Power_kw_new", row.get("Power"))
    }
    for ship, row in new_specs_idx.iterrows()
}

# Constants
years_dec  = list(range(2025, 2051, 5))
years_full = list(range(2025, 2051))
others     = ["Lpg", "Green Methanol", "Green Ammonia"]

def run_fleet_optimization(co2_prices, fuel_price_overrides):
    dfac = lambda y: 1 / ((1 + 0.07) ** (y - 2025))
    ships = fleet["Ship_Type"].unique()

    # Precompute costs
    baseline_cost, retro_cost, new_op_cost = {}, {}, {}
    for s in ships:
        row = fleet[fleet.Ship_Type == s].iloc[0]
        dist, voy = row["Distance"], row["Voyages"]
        mj_base   = row.get("Energy_per_km (MJ/km)", row.get("Energy_per_km"))
        pw        = row["Power"]
        for y in years_full:
            # Diesel baseline
            diesel_lookup = fuel_lu[(y, "Diesel")]
            fc = dist * voy * mj_base / diesel_lookup["Energy_MJ_per_kg"] * diesel_lookup["Price_USD_per_kg"]
            co2t = dist * voy * mj_base * diesel_lookup["CO2_g_per_MJ"] / 1e6
            cc = co2t * co2_prices.get(y, 0)
            ma = pw * diesel_lookup["Maintenance_USD_per_kW"]
            baseline_cost[(s, y)] = (fc + cc + ma) * dfac(y)

            # Retrofit
            save = T_SAVE.get((s, y), 0) / 100
            retro_cost[(s, y)] = ((fc + cc) * (1 - save) + ma) * dfac(y)

            # New options
            for f in others:
                fuel_data  = fuel_lu[(y, f)]
                price_kg   = fuel_price_overrides.get(y, {}).get(f, fuel_data["Price_USD_per_kg"])
                fc_new     = dist * voy * new_lu[s]["Energy_per_km_new"] / fuel_data["Energy_MJ_per_kg"] * price_kg
                co2n       = dist * voy * new_lu[s]["Energy_per_km_new"] * fuel_data["CO2_g_per_MJ"] / 1e6
                cc_new     = co2n * co2_prices.get(y, 0)
                ma_new     = new_lu[s]["Power_new"] * fuel_data["Maintenance_USD_per_kW"]
                new_op_cost[(s, y, f)] = (fc_new + cc_new + ma_new) * dfac(y)

    # Build optimization model
    mdl = LpProblem("Fleet_Optimization", LpMinimize)
    t = LpVariable.dicts("Turbo", [(s, y) for s in ships for y in years_dec], 0, 1, LpBinary)
    n = LpVariable.dicts("New",   [(s, y, f) for s in ships for y in years_dec for f in others], 0, 1, LpBinary)

    # Constraints
    for s in ships:
        mdl += lpSum(t[(s, y)] for y in years_dec) <= 1
        mdl += lpSum(n[(s, y, f)] for y in years_dec for f in others) <= 1
        for y in years_dec:
            mdl += t[(s, y)] <= 1 - lpSum(n[(s, yy, f)] for yy in years_dec if yy <= y for f in others)

    # Objective
    obj = []
    for s in ships:
        for y in years_full:
            # baseline and retrofit
            obj.append(baseline_cost[(s, y)])
            obj.append(retro_cost[(s, y)] * lpSum(t[(s, yy)] for yy in years_dec if yy <= y))
            # new by fuel
            for f in others:
                cum_new_f = lpSum(n[(s, yy, f)] for yy in years_dec if yy <= y)
                obj.append(new_op_cost[(s, y, f)] * cum_new_f)
        # Capex costs
        for y in years_dec:
            if (s, y) in T_COST:
                obj.append(T_COST[(s, y)] * t[(s, y)] * dfac(y))
            for f in others:
                if (s, f, y) in N_COST:
                    obj.append(N_COST[(s, f, y)] * n[(s, y, f)] * dfac(y))

    mdl += lpSum(obj)
    mdl.solve()

    # Build result tables
    comp_val = sum(baseline_cost.values())
    opt_val  = value(mdl.objective)
    comp_df = pd.DataFrame({"Variante": ["Optimiert", "Diesel-only"], "Kosten PV (USD)": [opt_val, comp_val]})
    savings_df = pd.DataFrame({"Messgröße": ["Absolut (USD)", "Relativ (%)"], "Wert": [comp_val - opt_val, (comp_val - opt_val) / comp_val * 100]})
    summary = []
    for s in ships:
        ty = next((y for y in years_dec if t[(s, y)].varValue > 0.5), None)
        chosen = [(yy, ff) for yy in years_dec for ff in others if n[(s, yy, ff)].varValue > 0.5]
        ny, fc = chosen[0] if chosen else (None, None)
        summary.append({"Ship": s, "Turbo_Year": ty, "New_Year": ny, "Fuel": fc})
    summary_df = pd.DataFrame(summary)
    return comp_df, savings_df, summary_df

# 3) UI Sliders
st.sidebar.header("CO₂ Price Settings (€/t)")
co2_prices = {y: st.sidebar.slider(f"CO₂ Price in {y}", 0, 1000, 100, step=50) for y in years_dec}

st.sidebar.header("Fuel Price Overrides (USD/kg)")
fuel_price_overrides = {y: {} for y in years_full}
for f in others:
    st.sidebar.subheader(f)
    for y in years_dec:
        default = fuel_lu[(y, f)]["Price_USD_per_kg"]
        fuel_price_overrides[y][f] = st.sidebar.slider(f"{f} @ {y}", 0.0, 10.0, float(default), step=0.1)

# 4) Run optimization and display
if st.sidebar.button("Run Optimization"):
    with st.spinner("Optimizing..."):
        comp_df, savings_df, summary_df = run_fleet_optimization(co2_prices, fuel_price_overrides)
    st.subheader("Cost Comparison")
    st.dataframe(comp_df)
    st.subheader("Savings")
    st.dataframe(savings_df)
    st.subheader("Decisions")
    st.dataframe(summary_df)


