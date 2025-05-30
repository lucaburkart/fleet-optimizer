Du:
# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatusOptimal, value

# ————————————————
# Page config must come first
st.set_page_config(page_title="Fleet Optimization", layout="wide")
# ————————————————

# Smoke-Test
st.write("✅ App lädt – UI ist aktiv")

# 1) Daten einlesen
BASE_PATH = Path(".")
fleet     = pd.read_csv(BASE_PATH / "fleet_data2.1.csv",     delimiter=";")
fuel      = pd.read_csv(BASE_PATH / "tech_fuel_data2.1.csv", delimiter=";")
turbo     = pd.read_csv(BASE_PATH / "turbo_retrofit.1.csv",  delimiter=";")
new_cost  = pd.read_csv(BASE_PATH / "new_ship_cost.1.csv",   delimiter=";")
new_specs = pd.read_csv(BASE_PATH / "new_fleet_data2.1.csv", delimiter=";")

for df in (fleet, fuel, turbo, new_cost, new_specs):
    for col in ("Ship_Type", "Fuel", "Fuel_Type"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

fuel_lu = fuel.set_index(["Year", "Fuel_Type"]).to_dict("index")
T_COST  = turbo.set_index(["Ship_Type", "Year"])["Retrofit_Cost_USD"].to_dict()
T_SAVE  = turbo.set_index(["Ship_Type", "Year"])["Energy_Saving_%"].to_dict()
N_COST  = new_cost.set_index(["Ship_Type", "Fuel", "Year"])["Capex_USD"].to_dict()

fleet_new_df = new_specs.set_index("Ship_Type")
new_lu = {
    ship: {
        "Energy_per_km_new": row["Energy_per_km (MJ/km)_new"],
        "Capacity_TEU_new":  row["Capacity_TEU_new"],
        "Power_new":         row.get("Power_kw_new", row.get("Power"))
    }
    for ship, row in fleet_new_df.iterrows()
}

# 2) Modell-Funktion mit korrigierter Zielfunktion für Neubauten
def run_fleet_optimization(co2_prices):
    ships      = fleet["Ship_Type"].unique()
    YEARS_DEC  = list(range(2025, 2051, 5))
    YEARS_FULL = list(range(2025, 2051))
    BASIC      = "Diesel"
    OTHERS     = ["Lpg", "Green Methanol", "Green Ammonia"]
    dfac = lambda y: 1 / ((1 + 0.07) ** (y - 2025))

    # Betriebskosten vorberechnen
    baseline_cost = {}
    retro_cost    = {}
    new_op_cost   = {}
    for s in ships:
        row  = fleet[fleet.Ship_Type == s].iloc[0]
        dist = row["Distance"]
        voy  = row["Voyages"]
        mj   = row.get("Energy_per_km (MJ/km)", row.get("Energy_per_km"))
        pw   = row["Power"]
        for y in YEARS_FULL:
            fc_base = dist * voy * mj / fuel_lu[(y, BASIC)]["Energy_MJ_per_kg"] * fuel_lu[(y, BASIC)]["Price_USD_per_kg"]
            co2t    = dist * voy * mj * fuel_lu[(y, BASIC)]["CO2_g_per_MJ"] / 1e6
            cc_base = co2t * co2_prices.get(y, 0)
            ma_base = pw * fuel_lu[(y, BASIC)]["Maintenance_USD_per_kW"]
            baseline_cost[(s, y)] = fc_base + cc_base + ma_base

            save = T_SAVE.get((s, y), 0) / 100
            retro_cost[(s, y)] = (fc_base + cc_base) * (1 - save) + ma_base

            for f in OTHERS:
                mj_new = new_lu[s]["Energy_per_km_new"]
                pw_new = new_lu[s]["Power_new"]
                fc_new = dist * voy * mj_new / fuel_lu[(y, f)]["Energy_MJ_per_kg"] * fuel_lu[(y, f)]["Price_USD_per_kg"]
                co2n   = dist * voy * mj_new * fuel_lu[(y, f)]["CO2_g_per_MJ"] / 1e6
                cc_new = co2n * co2_prices.get(y, 0)
                ma_new = pw_new * fuel_lu[(y, f)]["Maintenance_USD_per_kW"]
                new_op_cost[(s, y, f)] = fc_new + cc_new + ma_new

    mdl = LpProblem("Fleet_Optimization", LpMinimize)
    t = LpVariable.dicts("Turbo", [(s, y) for s in ships for y in YEARS_DEC], 0, 1, LpBinary)
    n = LpVariable.dicts("New",   [(s, y, f) for s in ships for y in YEARS_DEC for f in OTHERS], 0, 1, LpBinary)

    # Nebenbedingungen
    for s in ships:
        mdl += lpSum(t[(s, y)] for y in YEARS_DEC) <= 1
        mdl += lpSum(n[(s, y, f)] for y in YEARS_DEC for f in OTHERS) <= 1
        for y in YEARS_DEC:
            mdl += t[(s, y)] <= 1 - lpSum(n[(s, yy, f)] for yy in YEARS_DEC if yy <= y for f in OTHERS)

    # Zielfunktion
    obj = []
    for s in ships:
        for y in YEARS_FULL:
            cum_retro = lpSum(t[(s, yy)] for yy in YEARS_DEC if yy <= y)
            # pro Fuel-Typ separaten Indikator verwenden
            for f in OTHERS:
                cum_new_f = lpSum(n[(s, yy, f)] for yy in YEARS_DEC if yy <= y)
                obj.append(new_op_cost[(s, y, f)] * cum_new_f * dfac(y))
            obj.append(baseline_cost[(s, y)] * (1 - lpSum(t[(s, yy)] for yy in YEARS_DEC if yy <= y)) * dfac(y))
            obj.append(retro_cost[(s, y)] * (lpSum(t[(s, yy)] for yy in YEARS_DEC if yy <= y)
                                             - lpSum(n[(s, yy2, f2)] for yy2 in YEARS_DEC if yy2 <= y for f2 in OTHERS)
                                            ) * dfac(y))
        # Investitionskosten
        for y in YEARS_DEC:
            if (s, y) in T_COST:
                obj.append(T_COST[(s, y)] * t[(s, y)] * dfac(y))
            for f in OTHERS:
                if (s, f, y) in N_COST:
                    obj.append(N_COST[(s, f, y)] * n[(s, y, f)] * dfac(y))

    mdl += lpSum(obj)
    mdl.solve()

    if mdl.status == LpStatusOptimal:
        optimized_cost = value(mdl.objective)
        diesel_cost    = sum(baseline_cost[(s, y)] * dfac(y) for s in ships for y in YEARS_FULL)
        saving_abs     = diesel_cost - optimized_cost
        saving_pct     = saving_abs / diesel_cost * 100

        comp_df = pd.DataFrame({
            "Variante": ["Optimiert", "Diesel-only"],
            "Kosten PV (USD)": [optimized_cost, diesel_cost]
        })
        savings_df = pd.DataFrame({
            "Messgröße": ["Ersparnis absolut (USD)", "Ersparnis relativ (%)"],
            "Wert":      [saving_abs, saving_pct]
        })
        summary = []
        for s in ships:
            ty = next((y for y in YEARS_DEC if value(t[(s, y)]) > 0.5), None)
            chosen = [(yy, ff) for yy in YEARS_DEC for ff in OTHERS if value(n[(s, yy, ff)]) > 0.5]
            ny, fuel_choice = chosen[0] if chosen else (None, None)
            summary.append({
                "Ship":       s,
                "Turbo_Year": ty,
                "New_Year":   ny,
                "Fuel":       fuel_choice
            })
        summary_df = pd.DataFrame(summary)
        return comp_df, savings_df, summary_df
    else:
        raise RuntimeError(f"Optimierung nicht erfolgreich, Status: {mdl.status}")

# 3) Streamlit UI
st.title("🚢 Fleet Optimization Web App")
st.sidebar.header("CO₂ Price Settings (€/t)")
co2_prices = {}
for year in [2025, 2030, 2035, 2040, 2045, 2050]:
    co2_prices[year] = st.sidebar.slider(f"CO₂ Price in {year}", 0, 1000, 100, step=50)

if st.sidebar.button("🔍 Run Optimization"):
    with st.spinner("Running optimization..."):
        comp_df, savings_df, summary_df = run_fleet_optimization(co2_prices)
    st.success("Done!")
    st.subheader("📊 Cost Comparison")
    st.dataframe(comp_df.style.format({"Kosten PV (USD)": "{:,.0f}"}))
    st.subheader("💰 Savings")
    st.dataframe(savings_df.style.format({"Wert": "{:.2f}"}))
    st.subheader("🚢 Fleet Decisions")
    st.dataframe(summary_df)
