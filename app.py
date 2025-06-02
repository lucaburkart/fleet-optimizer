# app.py  –  Streamlit + PuLP, reine LP-Version
# --------------------------------------------------------------------
# Benötigte Dateien (alle im selben Ordner wie app.py):
#   fleet_data2.1.csv      tech_fuel_data2.csv    co2_price2.1.csv
#   turbo_retrofit.1.csv   new_ship_cost.1.csv    new_fleet_data2.1.csv
#   shipping_routes.xlsx
#
# requirements.txt enthält mindestens:
#   streamlit
#   pandas
#   pulp
#   openpyxl
# --------------------------------------------------------------------

import streamlit as st
import pandas as pd
from pathlib import Path
from pulp import (
    LpProblem, LpMinimize, LpVariable,
    lpSum, LpBinary, LpStatusOptimal, value,
)

st.set_page_config(page_title="Fleet Optimization", layout="wide")
st.write("✅ App lädt – UI ist aktiv")

# --------------------------------------------------------------------
# Optimierungsfunktion (linear)
# --------------------------------------------------------------------
def run_fleet_optimization(co2_prices: dict[int, float],
                           diesel_prices: dict[int, float]):
    """liefert (comp_df, savings_df, summary_df)"""

    BASE = Path(".")

    # 1) Daten einlesen ------------------------------------------------
    fleet     = pd.read_csv(BASE / "fleet_data2.1.csv",     delimiter=";")
    fuel      = pd.read_csv(BASE / "tech_fuel_data2.csv",   delimiter=";")
    co2_df    = pd.read_csv(BASE / "co2_price2.1.csv",      delimiter=";")
    turbo     = pd.read_csv(BASE / "turbo_retrofit.1.csv",  delimiter=";")
    new_cost  = pd.read_csv(BASE / "new_ship_cost.1.csv",   delimiter=";")
    new_specs = pd.read_csv(BASE / "new_fleet_data2.1.csv", delimiter=";")
    routes_df = pd.read_excel(BASE / "shipping_routes.xlsx")

    # 2) Strings normalisieren ----------------------------------------
    for df in (fleet, fuel, turbo, new_cost, new_specs):
        for col in ("Ship_Type", "Fuel", "Fuel_Type"):
            if col in df.columns:
                df[col] = (df[col].astype(str)
                                     .str.strip()
                                     .str.title())
    routes_df["Ship"] = routes_df["Ship"].astype(str).str.strip().str.title()

    # 3) Lookup-Dictionaries ------------------------------------------
    fuel_lu = fuel.set_index(["Year", "Fuel_Type"]).to_dict("index")
    co2_lu  = co2_df.set_index("Year")["CO2_Price_EUR_per_ton"].to_dict()
    T_COST  = turbo.set_index(["Ship_Type", "Year"])["Retrofit_Cost_USD"].to_dict()
    T_SAVE  = turbo.set_index(["Ship_Type", "Year"])["Energy_Saving_%"].to_dict()
    N_COST  = new_cost.set_index(["Ship_Type", "Fuel", "Year"])["Capex_USD"].to_dict()

    new_specs = new_specs.set_index("Ship_Type")
    new_lookup = {
        s: {
            "MJ_new": new_specs.at[s, "Energy_per_km (MJ/km)_new"],
            "P_new":  new_specs.at[s, "Power_kw_new"]
                      if "Power_kw_new" in new_specs.columns else new_specs.at[s, "Power"]
        }
        for s in new_specs.index
    }

    # 4) ERA / ECA-Anteile pro Schiff ---------------------------------
    routes_df = routes_df[
        ["Ship", "Nautical Miles", "Share of ERA", "Energy Consumption [MJ] WtW"]
    ].dropna(subset=["Ship"])
    era = {}
    for s, grp in routes_df.groupby("Ship"):
        tot   = grp["Energy Consumption [MJ] WtW"].sum()
        tot_e = (grp["Energy Consumption [MJ] WtW"] * grp["Share of ERA"]).sum()
        era[s] = {
            "MJ_voy":    tot,
            "MJ_eca":    tot_e,
            "MJ_noeca":  tot - tot_e,
            "share":     (tot_e / tot) if tot else 0.0,
        }

    # 5) Sets & Diskontfaktor -----------------------------------------
    ships      = fleet["Ship_Type"].unique()
    YEARS_DEC  = list(range(2025, 2051, 5))
    YEARS_FULL = list(range(2025, 2051))
    BASIC      = "Diesel"
    OTHERS     = ["Lpg", "Green Methanol", "Green Ammonia"]
    dfac       = lambda y: 1 / (1.07 ** (y - 2025))

    # 6) Barwerte vorberechnen ----------------------------------------
    pv_base, pv_delta_retro, pv_delta_new = {}, {}, {}
    for s in ships:
        row = fleet.loc[fleet.Ship_Type == s].iloc[0]
        voy, P = row["Voyages"], row["Power"]
        mj_old = row.get("Energy_per_km (MJ/km)", row.get("Energy_per_km"))
        e = era.get(s, {"MJ_voy":0,"MJ_eca":0,"MJ_noeca":0,"share":0})
        mj_v, mj_e, mj_n, share = e["MJ_voy"], e["MJ_eca"], e["MJ_noeca"], e["share"]
        mj_new = new_lookup[s]["MJ_new"]
        P_new  = new_lookup[s]["P_new"]
        factor = mj_new / mj_old if mj_old else 1.0

        def baseline_ann(y):
            cost_e = mj_e*voy / fuel_lu[(y,BASIC)]["Energy_MJ_per_kg"] * diesel_prices[y]
            cost_n = mj_n*voy / fuel_lu[(y,"Hfo")]["Energy_MJ_per_kg"] * fuel_lu[(y,"Hfo")]["Price_USD_per_kg"]
            co2    = mj_v*voy*(share*fuel_lu[(y,BASIC)]["CO2_g_per_MJ"]
                               +(1-share)*fuel_lu[(y,"Hfo")]["CO2_g_per_MJ"])/1e6*co2_prices[y]
            maint  = P*(share*fuel_lu[(y,BASIC)]["Maintenance_USD_per_kW"]
                        +(1-share)*fuel_lu[(y,"Hfo")]["Maintenance_USD_per_kW"])
            return (cost_e+cost_n+co2+maint)*dfac(y)

        base_stream = [baseline_ann(y) for y in YEARS_FULL]
        pv_base[s]  = sum(base_stream)

        # Retro-Streams
        for y0 in YEARS_DEC:
            save = T_SAVE.get((s,y0),0)/100
            def retro_ann(y):
                if y < y0:                                 # vor Einbau -> baseline
                    return base_stream[YEARS_FULL.index(y)]
                # nach Einbau
                mj_e_r   = mj_e*voy*(1-save)
                mj_n_r   = mj_n*voy*(1-save)
                cost_e_r = mj_e_r / fuel_lu[(y,BASIC)]["Energy_MJ_per_kg"] * diesel_prices[y]
                cost_n_r = mj_n_r / fuel_lu[(y,"Hfo")]["Energy_MJ_per_kg"] * fuel_lu[(y,"Hfo")]["Price_USD_per_kg"]
                co2_r    = (mj_e_r*fuel_lu[(y,BASIC)]["CO2_g_per_MJ"]
                            + mj_n_r*fuel_lu[(y,"Hfo")]["CO2_g_per_MJ"])/1e6*co2_prices[y]
                maint_r  = maint  # Wartung unverändert
                return (cost_e_r+cost_n_r+co2_r+maint_r)*dfac(y)
            stream_r = [retro_ann(y) for y in YEARS_FULL]
            capex_r  = T_COST.get((s,y0),0)*dfac(y0)
            pv_delta_retro[s,y0] = sum(stream_r) + capex_r - pv_base[s]

        # Neubau-Streams
        for y0 in YEARS_DEC:
            for f in OTHERS:
                def new_ann(y):
                    if y < y0:
                        return base_stream[YEARS_FULL.index(y)]
                    mj_e_new   = mj_e*factor*voy
                    mj_n_new   = mj_n*factor*voy
                    cost_e_new = mj_e_new / fuel_lu[(y,f)]["Energy_MJ_per_kg"] * fuel_lu[(y,f)]["Price_USD_per_kg"]
                    cost_n_new = mj_n_new / fuel_lu[(y,f)]["Energy_MJ_per_kg"] * fuel_lu[(y,f)]["Price_USD_per_kg"]
                    co2_new    = (mj_e_new+mj_n_new)*fuel_lu[(y,f)]["CO2_g_per_MJ"]/1e6*co2_prices[y]
                    maint_new  = P_new*fuel_lu[(y,f)]["Maintenance_USD_per_kW"]
                    return (cost_e_new+cost_n_new+co2_new+maint_new)*dfac(y)
                stream_n = [new_ann(y) for y in YEARS_FULL]
                capex_n  = N_COST.get((s,f,y0),0)*dfac(y0)
                pv_delta_new[s,y0,f] = sum(stream_n) + capex_n - pv_base[s]

    # 7) Modell --------------------------------------------------------
    mdl = LpProblem("FleetOpt", LpMinimize)
    t = LpVariable.dicts("Turbo", [(s,y) for s in ships for y in YEARS_DEC],
                         lowBound=0, upBound=1, cat=LpBinary)
    n = LpVariable.dicts("New", [(s,y,f) for s in ships for y in YEARS_DEC for f in OTHERS],
                         lowBound=0, upBound=1, cat=LpBinary)

    # 7.1) Constraints
    for s in ships:
        mdl += lpSum(t[(s,y)] for y in YEARS_DEC) <= 1
        mdl += lpSum(n[(s,y,f)] for y in YEARS_DEC for f in OTHERS) <= 1
        for y in YEARS_DEC:
            mdl += t[(s,y)] <= 1 - lpSum(n[(s,yy,f)] for yy in YEARS_DEC if yy<=y for f in OTHERS)

    # 7.2) Zielfunktion (rein linear)
    mdl += (
        lpSum(pv_base[s]                                  for s in ships) +
        lpSum(pv_delta_retro[s,y] * t[(s,y)]              for s in ships for y in YEARS_DEC) +
        lpSum(pv_delta_new[s,y,f] * n[(s,y,f)]            for s in ships for y in YEARS_DEC for f in OTHERS)
    )

    mdl.solve()

    if mdl.status != LpStatusOptimal:
        raise RuntimeError(f"Status {mdl.status}: Modell nicht optimal.")

    # 8) Reporting -----------------------------------------------------
    opt_cost   = value(mdl.objective)
    base_cost  = sum(pv_base[s] for s in ships)
    comp_df    = pd.DataFrame({
        "Variante": ["Optimiert", "Diesel-only"],
        "Kosten NPV (USD)": [opt_cost, base_cost]
    })
    savings_df = pd.DataFrame({
        "Messgröße": ["Ersparnis absolut (USD)", "Ersparnis relativ (%)"],
        "Wert":      [base_cost - opt_cost, (base_cost - opt_cost)/base_cost*100]
    })
    summary = []
    for s in ships:
        ty = next((y for y in YEARS_DEC if value(t[(s,y)]) > 0.5), None)
        new_sel = [(y,f) for y in YEARS_DEC for f in OTHERS if value(n[(s,y,f)]) > 0.5]
        ny, fuel = new_sel[0] if new_sel else (None, None)
        summary.append({"Ship": s, "Turbo_Year": ty, "New_Year": ny, "Fuel": fuel})
    summary_df = pd.DataFrame(summary)
    return comp_df, savings_df, summary_df

# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------
st.sidebar.header("CO₂-Preis (USD/t)")
YE_REF = [2025, 2030, 2035, 2040, 2045, 2050]
co2_ref = {y: st.sidebar.slider(str(y), 0, 1000, 100, 50) for y in YE_REF}
co2_prices = {y: co2_ref[max(k for k in YE_REF if k<=y)] for y in range(2025,2051)}

st.sidebar.header("Diesel-Preis (USD/kg)")
diesel_ref = {y: st.sidebar.slider(str(y), 0.0, 10.0, 1.0, 0.5) for y in YE_REF}
diesel_prices = {y: diesel_ref[max(k for k in YE_REF if k<=y)] for y in range(2025,2051)}

if st.sidebar.button("🔍 Run Optimization"):
    with st.spinner("Berechne optimale Flotte …"):
        comp, sav, summ = run_fleet_optimization(co2_prices, diesel_prices)
    st.success("Optimierung abgeschlossen!")

    st.subheader("📊 Kostenvergleich (NPV)")
    st.dataframe(comp.style.format({"Kosten NPV (USD)": "{:,.0f}"}))

    st.subheader("💰 Ersparnis")
    st.dataframe(sav.style.format({"Wert": "{:,.2f}"}))

    st.subheader("🚢 Flotten-Entscheidungen")
    st.dataframe(summ)
