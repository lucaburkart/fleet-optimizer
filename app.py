# app.py  –  Streamlit + PuLP (lineare Version)   Stand 02-Jun-2025
# -----------------------------------------------------------------
# Erwartete Dateien IM SELBEN ORDNER:
#   fleet_data2.1.csv      tech_fuel_data2.csv    co2_price2.1.csv
#   turbo_retrofit.1.csv   new_ship_cost.1.csv    new_fleet_data2.1.csv
#   shipping_routes.xlsx
#
# requirements.txt (mindestens):
#   streamlit
#   pandas
#   pulp
#   openpyxl
# -----------------------------------------------------------------

import streamlit as st
import pandas as pd
from pathlib import Path
from pulp import (
    LpProblem, LpMinimize, LpVariable,
    lpSum, LpBinary, LpStatusOptimal, value,
)

# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fleet Optimization", layout="wide")
st.write("✅ App geladen – bereit zur Optimierung !")

# ─────────────────────────────────────────────────────────────────
# 1) Optimierungs-Funktion (OHNE bilineare Terme)
# ─────────────────────────────────────────────────────────────────
def run_fleet_optimization(co2_prices: dict[int, float],
                           diesel_prices: dict[int, float]):
    BASE = Path(".")

    # ---- 1. Daten einlesen --------------------------------------
    fleet     = pd.read_csv(BASE / "fleet_data2.1.csv",     delimiter=";")
    fuel      = pd.read_csv(BASE / "tech_fuel_data2.csv",   delimiter=";")
    co2_df    = pd.read_csv(BASE / "co2_price2.1.csv",      delimiter=";")
    turbo     = pd.read_csv(BASE / "turbo_retrofit.1.csv",  delimiter=";")
    new_cost  = pd.read_csv(BASE / "new_ship_cost.1.csv",   delimiter=";")
    new_specs = pd.read_csv(BASE / "new_fleet_data2.1.csv", delimiter=";")
    routes_df = pd.read_excel(BASE / "shipping_routes.xlsx")

    # ---- 2. Strings normalisieren ------------------------------
    for df in (fleet, fuel, turbo, new_cost, new_specs):
        for col in ("Ship_Type", "Fuel", "Fuel_Type"):
            if col in df.columns:
                df[col] = (df[col].astype(str)
                                     .str.strip()
                                     .str.title())
    routes_df["Ship"] = routes_df["Ship"].astype(str).str.strip().str.title()

    # ---- 3. Look-ups -------------------------------------------
    fuel_lu = fuel.set_index(["Year", "Fuel_Type"]).to_dict("index")
    co2_lu  = co2_df.set_index("Year")["CO2_Price_EUR_per_ton"].to_dict()
    T_COST  = turbo.set_index(["Ship_Type", "Year"])["Retrofit_Cost_USD"].to_dict()
    T_SAVE  = turbo.set_index(["Ship_Type", "Year"])["Energy_Saving_%"].to_dict()
    N_COST  = new_cost.set_index(["Ship_Type", "Fuel", "Year"])["Capex_USD"].to_dict()

    new_specs = new_specs.set_index("Ship_Type")
    new_lu = {
        s: {
            "MJ_new": new_specs.at[s, "Energy_per_km (MJ/km)_new"],
            "P_new":  new_specs.at[s, "Power_kw_new"]
                      if "Power_kw_new" in new_specs.columns else new_specs.at[s, "Power"]
        }
        for s in new_specs.index
    }

    # ---- 4. ERA-/ECA-Anteile pro Schiff ------------------------
    routes_df = routes_df[
        ["Ship", "Nautical Miles", "Share of ERA", "Energy Consumption [MJ] WtW"]
    ].dropna(subset=["Ship"])
    era = {}
    for s, g in routes_df.groupby("Ship"):
        tot   = g["Energy Consumption [MJ] WtW"].sum()
        tot_e = (g["Energy Consumption [MJ] WtW"] * g["Share of ERA"]).sum()
        era[s] = {
            "MJ_v":    tot,
            "MJ_e":    tot_e,
            "MJ_n":    tot - tot_e,
            "share":   (tot_e / tot) if tot else 0.0,
        }

    # ---- 5. Sets & Diskontierung -------------------------------
    ships      = fleet["Ship_Type"].unique()
    Y_DEC      = list(range(2025, 2051, 5))
    Y_FULL     = list(range(2025, 2051))
    BASIC      = "Diesel"
    OTHERS     = ["Lpg", "Green Methanol", "Green Ammonia"]
    dfac       = lambda y: 1 / (1.07 ** (y - 2025))

    # ---- 6. Barwerte vorberechnen ------------------------------
    pv_base, d_retro, d_new = {}, {}, {}
    for s in ships:
        row = fleet.loc[fleet.Ship_Type == s].iloc[0]
        voy, P = row["Voyages"], row["Power"]
        MJ_old = row.get("Energy_per_km (MJ/km)", row.get("Energy_per_km"))
        e      = era.get(s, {"MJ_v":0,"MJ_e":0,"MJ_n":0,"share":0})
        MJ_v, MJ_e, MJ_n, share = e["MJ_v"], e["MJ_e"], e["MJ_n"], e["share"]
        MJ_new = new_lu[s]["MJ_new"]
        P_new  = new_lu[s]["P_new"]
        factor = MJ_new / MJ_old if MJ_old else 1.0

        # Baseline-Barwert
        pv_base[s] = sum(
            (
                # Diesel im ECA
                MJ_e*voy / fuel_lu[(y,BASIC)]["Energy_MJ_per_kg"] * diesel_prices[y]
                # HFO ausserhalb ECA
              + MJ_n*voy / fuel_lu[(y,"Hfo")]["Energy_MJ_per_kg"] * fuel_lu[(y,"Hfo")]["Price_USD_per_kg"]
                # CO2
              + MJ_v*voy*(share*fuel_lu[(y,BASIC)]["CO2_g_per_MJ"]
                        +(1-share)*fuel_lu[(y,"Hfo")]["CO2_g_per_MJ"]) / 1e6 * co2_prices[y]
                # Wartung
              + P*(share*fuel_lu[(y,BASIC)]["Maintenance_USD_per_kW"]
                  +(1-share)*fuel_lu[(y,"Hfo")]["Maintenance_USD_per_kW"])
            ) * dfac(y)
            for y in Y_FULL
        )

        # Retro-Deltas
        for y0 in Y_DEC:
            save   = T_SAVE.get((s,y0),0)/100
            delta  = (
                sum(
                    (
                        (MJ_e*voy*(1-save)) / fuel_lu[(y,BASIC)]["Energy_MJ_per_kg"] * diesel_prices[y]
                      + (MJ_n*voy*(1-save)) / fuel_lu[(y,"Hfo")]["Energy_MJ_per_kg"] * fuel_lu[(y,"Hfo")]["Price_USD_per_kg"]
                      + (MJ_e*voy*(1-save))*fuel_lu[(y,BASIC)]["CO2_g_per_MJ"]
                      + (MJ_n*voy*(1-save))*fuel_lu[(y,"Hfo")]["CO2_g_per_MJ"]
                    ) / 1e6 * co2_prices[y]   # CO2
                  + P*(share*fuel_lu[(y,BASIC)]["Maintenance_USD_per_kW"]
                      +(1-share)*fuel_lu[(y,"Hfo")]["Maintenance_USD_per_kW"])  # Wartung
                ) * dfac(y)
                for y in Y_FULL if y >= y0
            ) - sum(  # Baseline ab y0
                (
                    MJ_e*voy / fuel_lu[(y,BASIC)]["Energy_MJ_per_kg"] * diesel_prices[y]
                  + MJ_n*voy / fuel_lu[(y,"Hfo")]["Energy_MJ_per_kg"] * fuel_lu[(y,"Hfo")]["Price_USD_per_kg"]
                  + MJ_v*voy*(share*fuel_lu[(y,BASIC)]["CO2_g_per_MJ"]
                              +(1-share)*fuel_lu[(y,"Hfo")]["CO2_g_per_MJ"]) / 1e6 * co2_prices[y]
                  + P*(share*fuel_lu[(y,BASIC)]["Maintenance_USD_per_kW"]
                      +(1-share)*fuel_lu[(y,"Hfo")]["Maintenance_USD_per_kW"])
                ) * dfac(y)
                for y in Y_FULL if y >= y0
            )
            d_retro[s,y0] = delta + T_COST.get((s,y0),0)*dfac(y0)

        # Neubau-Deltas
        for y0 in Y_DEC:
            for f in OTHERS:
                delta = (
                    sum(
                        (
                            (MJ_e*factor*voy) / fuel_lu[(y,f)]["Energy_MJ_per_kg"] * fuel_lu[(y,f)]["Price_USD_per_kg"]
                          + (MJ_n*factor*voy) / fuel_lu[(y,f)]["Energy_MJ_per_kg"] * fuel_lu[(y,f)]["Price_USD_per_kg"]
                          + (MJ_v*factor*voy) * fuel_lu[(y,f)]["CO2_g_per_MJ"] / 1e6 * co2_prices[y]
                          + P_new * fuel_lu[(y,f)]["Maintenance_USD_per_kW"]
                        ) * dfac(y) for y in Y_FULL if y >= y0
                    )
                    - sum(
                        (
                            MJ_e*voy / fuel_lu[(y,BASIC)]["Energy_MJ_per_kg"] * diesel_prices[y]
                          + MJ_n*voy / fuel_lu[(y,"Hfo")]["Energy_MJ_per_kg"] * fuel_lu[(y,"Hfo")]["Price_USD_per_kg"]
                          + MJ_v*voy*(share*fuel_lu[(y,BASIC)]["CO2_g_per_MJ"]
                                     +(1-share)*fuel_lu[(y,"Hfo")]["CO2_g_per_MJ"]) / 1e6 * co2_prices[y]
                          + P*(share*fuel_lu[(y,BASIC)]["Maintenance_USD_per_kW"]
                              +(1-share)*fuel_lu[(y,"Hfo")]["Maintenance_USD_per_kW"])
                        ) * dfac(y) for y in Y_FULL if y >= y0
                    )
                )
                d_new[s,y0,f] = delta + N_COST.get((s,f,y0),0)*dfac(y0)

    # ---- 7. MIP-Modell -------------------------------------------
    mdl = LpProblem("FleetOpt", LpMinimize)
    t = LpVariable.dicts("Turbo", [(s,y) for s in ships for y in Y_DEC],
                         0,1,LpBinary)
    n = LpVariable.dicts("New",   [(s,y,f) for s in ships for y in Y_DEC for f in OTHERS],
                         0,1,LpBinary)

    for s in ships:
        mdl += lpSum(t[(s,y)] for y in Y_DEC) <= 1
        mdl += lpSum(n[(s,y,f)] for y in Y_DEC for f in OTHERS) <= 1
        for y in Y_DEC:
            mdl += t[(s,y)] <= 1 - lpSum(n[(s,yy,f)] for yy in Y_DEC if yy<=y for f in OTHERS)

    mdl += (
        lpSum(pv_base[s]                          for s in ships) +
        lpSum(d_retro[s,y]*t[(s,y)]               for s in ships for y in Y_DEC) +
        lpSum(d_new[s,y,f]*n[(s,y,f)]             for s in ships for y in Y_DEC for f in OTHERS)
    )

    mdl.solve()
    if mdl.status != LpStatusOptimal:
        raise RuntimeError(f"Status {mdl.status}")

    # ---- 8. Berichte ---------------------------------------------
    obj_opt  = value(mdl.objective)
    obj_base = sum(pv_base[s] for s in ships)

    comp_df = pd.DataFrame({
        "Variante": ["Optimiert", "Diesel-only"],
        "Kosten NPV (USD)": [obj_opt, obj_base]
    })
    savings_df = pd.DataFrame({
        "Messgröße": ["Ersparnis absolut (USD)", "Ersparnis relativ (%)"],
        "Wert":      [obj_base - obj_opt, (obj_base - obj_opt)/obj_base*100]
    })
    summary=[]
    for s in ships:
        ty = next((y for y in Y_DEC if value(t[(s,y)])>0.5), None)
        new_sel = [(y,f) for y in Y_DEC for f in OTHERS if value(n[(s,y,f)])>0.5]
        ny, fuel = new_sel[0] if new_sel else (None,None)
        summary.append({"Ship":s,"Turbo_Year":ty,"New_Year":ny,"Fuel":fuel})
    summary_df = pd.DataFrame(summary)
    return comp_df, savings_df, summary_df

# ─────────────────────────────────────────────────────────────────
# 2) Streamlit-UI
# ─────────────────────────────────────────────────────────────────
YE_REF = [2025,2030,2035,2040,2045,2050]

st.sidebar.header("CO₂-Preis (USD/t)")
co2_ref = {y: st.sidebar.slider(str(y),0,1000,100,50) for y in YE_REF}
co2_prices = {y: co2_ref[max(k for k in YE_REF if k<=y)] for y in range(2025,2051)}

st.sidebar.header("Diesel-Preis (USD/kg)")
diesel_ref = {y: st.sidebar.slider(str(y),0.0,10.0,1.0,0.5) for y in YE_REF}
diesel_prices = {y: diesel_ref[max(k for k in YE_REF if k<=y)] for y in range(2025,2051)}

if st.sidebar.button("🔍 Run Optimization"):
    with st.spinner("Berechne optimale Flotte …"):
        comp, sav, summ = run_fleet_optimization(co2_prices, diesel_prices)
    st.success("Optimierung abgeschlossen!")

    st.subheader("📊 Kostenvergleich (NPV)")
    st.dataframe(comp.style.format({"Kosten NPV (USD)":"{:,.0f}"}))

    st.subheader("💰 Ersparnis")
    st.dataframe(sav.style.format({"Wert":"{:,.2f}"}))

    st.subheader("🚢 Flotten-Entscheidungen")
    st.dataframe(summ)
