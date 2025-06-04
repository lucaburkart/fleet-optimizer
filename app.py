# app.py  –  Streamlit + PuLP (inkl. Debug für MJ_old vs. MJ_new)
# Stand: 04-Jun-2025

import streamlit as st
import pandas as pd
from pathlib import Path
from pulp import (
    LpProblem, LpMinimize, LpVariable,
    lpSum, LpBinary, LpStatusOptimal, value,
)

# ───────────────────────────────────────────────────────────────────────────────
# 1) Streamlit-Grundeinstellungen
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fleet Optimization", layout="wide")
st.write("✅ App geladen – UI ist aktiv")

# ───────────────────────────────────────────────────────────────────────────────
# 2) Daten einlesen
# ───────────────────────────────────────────────────────────────────────────────
BASE = Path(".")
fleet     = pd.read_csv(BASE / "fleet_data2.1.csv",     delimiter=";")
fuel      = pd.read_csv(BASE / "tech_fuel_data2.csv",   delimiter=";")
co2_df    = pd.read_csv(BASE / "co2_price2.1.csv",      delimiter=";")
turbo     = pd.read_csv(BASE / "turbo_retrofit.1.csv",  delimiter=";")
new_cost  = pd.read_csv(BASE / "new_ship_cost.1.csv",   delimiter=";")
new_specs = pd.read_csv(BASE / "new_fleet_data2.1.csv", delimiter=";")
routes_df = pd.read_excel(BASE / "shipping_routes.xlsx")

# ───────────────────────────────────────────────────────────────────────────────
# 3) Debug: Vergleiche MJ_old vs. MJ_new direkt nach Einlesen
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("🔍 Debug: Energieverbrauch Alt vs. Neu (MJ/km)")

vergleich = []
for s in fleet["Ship_Type"].unique():
    # MJ_old aus Bestandsdaten (fleet_data2.1.csv)
    row_old = fleet.loc[fleet.Ship_Type == s].iloc[0]
    MJold = row_old.get("Energy_per_km (MJ/km)", row_old.get("Energy_per_km"))
    
    # MJ_new aus Neubau-Daten (new_fleet_data2.1.csv)
    if s in new_specs["Ship_Type"].values:
        row_new = new_specs.loc[new_specs.Ship_Type == s].iloc[0]
        MJnew = row_new["Energy_per_km (MJ/km)_new"]
    else:
        MJnew = None
    
    vergleich.append({
        "Ship_Type":        s,
        "MJ_old (MJ/km)":   MJold,
        "MJ_new (MJ/km)":   MJnew,
        "MJ_new < MJ_old?": (MJnew < MJold) if (MJnew is not None) else "n/a"
    })

df_vergleich = pd.DataFrame(vergleich)
st.dataframe(df_vergleich.style.format({
    "MJ_old (MJ/km)": "{:.0f}",
    "MJ_new (MJ/km)": "{:.0f}"
}))

# ───────────────────────────────────────────────────────────────────────────────
# 4) Funktion: run_fleet_optimization
# ───────────────────────────────────────────────────────────────────────────────
def run_fleet_optimization(co2_prices: dict[int, float],
                           diesel_prices: dict[int, float],
                           hfo_prices: dict[int, float]):
    """
    co2_prices:    Dict Jahr → CO₂‐Preis (USD/t)
    diesel_prices: Dict Jahr → Diesel‐Preis (USD/kg)
    hfo_prices:    Dict Jahr → HFO‐Preis (USD/kg)

    Rückgabe:
      comp_df       → DataFrame: ["Optimiert", "Diesel-only"] vs. NPV-Kosten
      savings_df    → DataFrame: absolute und relative Ersparnis (NPV)
      summary_df    → DataFrame: pro Schiff – gewähltes Retrofit‐Jahr und Neubau‐Jahr/Fuel
      emissions_df  → DataFrame: ["Optimiert", "Diesel-only"] vs. CO₂-Ausstoß (t) 
    """

    # ───────────────────────────────────────────────────────────────────────────
    # 4.1 Look-Up-Dictionaries aufbauen
    # ───────────────────────────────────────────────────────────────────────────
    fuel_lu = fuel.set_index(["Year", "Fuel_Type"]).to_dict("index")
    co2_lu  = co2_df.set_index("Year")["CO2_Price_EUR_per_ton"].to_dict()
    T_COST  = turbo.set_index(["Ship_Type", "Year"])["Retrofit_Cost_USD"].to_dict()
    T_SAVE  = turbo.set_index(["Ship_Type", "Year"])["Energy_Saving_%"].to_dict()
    N_COST  = new_cost.set_index(["Ship_Type", "Fuel", "Year"])["Capex_USD"].to_dict()

    new_specs_indexed = new_specs.set_index("Ship_Type")
    new_lu = {
        s: {
            "MJ_new": new_specs_indexed.at[s, "Energy_per_km (MJ/km)_new"],
            "P_new":  new_specs_indexed.at[s, "Power_kw_new"]
                      if "Power_kw_new" in new_specs_indexed.columns else new_specs_indexed.at[s, "Power"]
        }
        for s in new_specs_indexed.index
    }

    # ───────────────────────────────────────────────────────────────────────────
    # 5) ERA-/ECA-Anteile pro Schiff berechnen
    # ───────────────────────────────────────────────────────────────────────────
    routes_df = routes_df[[
        "Ship", "Nautical Miles", "Share of ERA", "Energy Consumption [MJ] WtW"
    ]].dropna(subset=["Ship"])

    energy_groups = {}
    for ship, grp in routes_df.groupby("Ship"):
        tot_mj     = grp["Energy Consumption [MJ] WtW"].sum()
        tot_mj_eca = (grp["Energy Consumption [MJ] WtW"] * grp["Share of ERA"]).sum()
        energy_groups[ship] = {
            "MJ_v":    tot_mj,
            "MJ_e":    tot_mj_eca,
            "MJ_n":    tot_mj - tot_mj_eca,
            "share":   (tot_mj_eca / tot_mj) if tot_mj > 0 else 0.0
        }

    # ───────────────────────────────────────────────────────────────────────────
    # 6) Parameter-Sets
    # ───────────────────────────────────────────────────────────────────────────
    ships      = fleet["Ship_Type"].unique()
    YEARS_DEC  = list(range(2025, 2051, 5))
    YEARS_FULL = list(range(2025, 2051))
    BASIC      = "Diesel"
    OTHERS     = ["Lpg", "Green Methanol", "Green Ammonia"]
    discount   = 0.07
    dfac       = lambda y: 1 / ((1 + discount) ** (y - 2025))

    # ───────────────────────────────────────────────────────────────────────────
    # 7) Diskontierte Jahreskosten + CO₂-Emissions-Daten berechnen
    # ───────────────────────────────────────────────────────────────────────────
    baseline_cost   = {}
    retro_cost      = {}
    new_cost_op     = {}
    baseline_co2_g  = {}
    retro_co2_g     = {}
    new_co2_g       = {}

    for s in ships:
        row     = fleet.loc[fleet.Ship_Type == s].iloc[0]
        voy     = row["Voyages"]
        P       = row["Power"]
        MJold   = row.get("Energy_per_km (MJ/km)", row.get("Energy_per_km"))

        grp     = energy_groups.get(s, {"MJ_v":0.0, "MJ_e":0.0, "MJ_n":0.0, "share":0.0})
        MJv, MJe, MJn, share = grp["MJ_v"], grp["MJ_e"], grp["MJ_n"], grp["share"]

        MJnew   = new_lu[s]["MJ_new"]
        Pnew    = new_lu[s]["P_new"]
        factor  = (MJnew / MJold) if MJold else 1.0

        for y in YEARS_FULL:
            # ───────────────────────────────────────────────────────────
            # 7.1 Baseline-Jahreskosten & CO₂ (Diesel/HFO-Mix)
            # ───────────────────────────────────────────────────────────
            cost_eca = 0.0
            if MJe > 0:
                kg_eca = (MJe * voy) / fuel_lu[(y, BASIC)]["Energy_MJ_per_kg"]
                cost_eca = kg_eca * diesel_prices[y]

            cost_noeca = 0.0
            if MJn > 0:
                kg_no = (MJn * voy) / fuel_lu[(y, "Hfo")]["Energy_MJ_per_kg"]
                cost_noeca = kg_no * hfo_prices[y]

            co2g = 0.0
            if MJv > 0:
                ef_diesel = fuel_lu[(y, BASIC)]["CO2_g_per_MJ"]
                ef_hfo    = fuel_lu[(y, "Hfo")]["CO2_g_per_MJ"]
                co2g = MJv * voy * (share * ef_diesel + (1 - share) * ef_hfo)
            baseline_co2_g[(s, y)] = co2g

            co2_amt = (co2g / 1_000_000) * co2_prices[y]

            if MJv > 0:
                ma = P * (
                    share * fuel_lu[(y, BASIC)]["Maintenance_USD_per_kW"]
                    + (1 - share) * fuel_lu[(y, "Hfo")]["Maintenance_USD_per_kW"]
                )
            else:
                ma = P * fuel_lu[(y, BASIC)]["Maintenance_USD_per_kW"]

            baseline_cost[(s, y)] = (cost_eca + cost_noeca + co2_amt + ma) * dfac(y)

            # ───────────────────────────────────────────────────────────
            # 7.2 Retrofit-Kosten ab Jahr y (operativ + Capex) & CO₂
            # ───────────────────────────────────────────────────────────
            save_pct = T_SAVE.get((s, y), 0) / 100
            MJe_r = MJe * voy * (1 - save_pct)
            MJn_r = MJn * voy * (1 - save_pct)

            cost_eca_r = 0.0
            if MJe_r > 0:
                cost_eca_r = (MJe_r / fuel_lu[(y, BASIC)]["Energy_MJ_per_kg"]) * diesel_prices[y]
            cost_noeca_r = 0.0
            if MJn_r > 0:
                cost_noeca_r = (MJn_r / fuel_lu[(y, "Hfo")]["Energy_MJ_per_kg"]) * hfo_prices[y]

            co2g_r = 0.0
            if MJv > 0:
                ef_diesel = fuel_lu[(y, BASIC)]["CO2_g_per_MJ"]
                ef_hfo    = fuel_lu[(y, "Hfo")]["CO2_g_per_MJ"]
                co2g_r = MJe_r * ef_diesel + MJn_r * ef_hfo
            retro_co2_g[(s, y)] = co2g_r

            co2_r = (co2g_r / 1_000_000) * co2_prices[y]
            ma_r = ma
            capex_r = T_COST.get((s, y), 0) * dfac(y)
            retro_cost[(s, y)] = ((cost_eca_r + cost_noeca_r + co2_r + ma_r) * dfac(y)) + capex_r

            # ───────────────────────────────────────────────────────────
            # 7.3 Neubau-Kosten ab Jahr y (operativ + Capex) & CO₂
            # ───────────────────────────────────────────────────────────
            MJv_n = MJv * factor
            MJn_n = MJn * factor  # dieser Term wird in der Regel nicht separat gebraucht, bleibt für Konsistenz

            for f in OTHERS:
                cost_eca_n = 0.0
                if MJv_n > 0:
                    cost_eca_n = (MJv_n / fuel_lu[(y, f)]["Energy_MJ_per_kg"]) * fuel_lu[(y, f)]["Price_USD_per_kg"]

                cost_noeca_n = 0.0
                if MJn_n > 0:
                    cost_noeca_n = (MJn_n / fuel_lu[(y, f)]["Energy_MJ_per_kg"]) * fuel_lu[(y, f)]["Price_USD_per_kg"]

                ef_f = fuel_lu[(y, f)]["CO2_g_per_MJ"]
                co2g_n = (MJv_n + MJn_n) * ef_f
                new_co2_g[(s, y, f)] = co2g_n

                co2_n = (co2g_n / 1_000_000) * co2_prices[y]
                ma_n = new_lu[s]["P_new"] * fuel_lu[(y, f)]["Maintenance_USD_per_kW"]
                capex_n = N_COST.get((s, f, y), 0) * dfac(y)

                new_cost_op[(s, y, f)] = ((cost_eca_n + cost_noeca_n + co2_n + ma_n) * dfac(y)) + capex_n

    # ───────────────────────────────────────────────────────────────────────────
    # 8) Barwerte (NPV) berechnen: pv_base, delta_retro, delta_new
    # ───────────────────────────────────────────────────────────────────────────
    pv_base = {s: sum(baseline_cost[(s, y)] for y in YEARS_FULL) for s in ships}

    delta_retro = {}
    for s in ships:
        for y0 in YEARS_DEC:
            diff_sum = sum(
                (retro_cost[(s, y)] - baseline_cost[(s, y)])
                for y in YEARS_FULL if y >= y0
            )
            delta_retro[(s, y0)] = diff_sum

    delta_new = {}
    for s in ships:
        for y0 in YEARS_DEC:
            for f in OTHERS:
                diff_sum = sum(
                    (new_cost_op[(s, y, f)] - baseline_cost[(s, y)])
                    for y in YEARS_FULL if y >= y0
                )
                delta_new[(s, y0, f)] = diff_sum

    # ───────────────────────────────────────────────────────────────────────────
    # 9) MILP-Modell aufsetzen (rein linear)
    # ───────────────────────────────────────────────────────────────────────────
    mdl = LpProblem("Fleet_Optimization", LpMinimize)

    t = LpVariable.dicts("Turbo", [(s, y) for s in ships for y in YEARS_DEC], cat=LpBinary)
    n = LpVariable.dicts("New",   [(s, y, f) for s in ships for y in YEARS_DEC for f in OTHERS], cat=LpBinary)

    for s in ships:
        mdl += lpSum(t[(s, y)] for y in YEARS_DEC) <= 1
        mdl += lpSum(n[(s, y, f)] for y in YEARS_DEC for f in OTHERS) <= 1
        for y in YEARS_DEC:
            mdl += (
                t[(s, y)]
                <= 1
                - lpSum(n[(s, yy, f)] for yy in YEARS_DEC if yy <= y for f in OTHERS)
            )

    mdl += (
        lpSum(pv_base[s] for s in ships)
        + lpSum(delta_retro[(s, y)] * t[(s, y)] for s in ships for y in YEARS_DEC)
        + lpSum(delta_new[(s, y, f)] * n[(s, y, f)]
                for s in ships for y in YEARS_DEC for f in OTHERS)
    )

    mdl.solve()

    if mdl.status != LpStatusOptimal:
        raise RuntimeError(f"Modell nicht optimal (Status {mdl.status})")

    # ───────────────────────────────────────────────────────────────────────────
    # 10) Ergebnis-Reporting (NPV & Flottenentscheidungen)
    # ───────────────────────────────────────────────────────────────────────────
    obj_opt  = value(mdl.objective)
    obj_base = sum(pv_base[s] for s in ships)

    comp_df = pd.DataFrame({
        "Variante": ["Optimiert", "Diesel-only"],
        "Kosten NPV (USD)": [obj_opt, obj_base]
    })
    savings_df = pd.DataFrame({
        "Messgröße": ["Ersparnis absolut (USD)", "Ersparnis relativ (%)"],
        "Wert":      [obj_base - obj_opt, (obj_base - obj_opt) / obj_base * 100]
    })

    summary = []
    for s in ships:
        ty = next((y for y in YEARS_DEC if value(t[(s, y)]) > 0.5), None)
        chosen = [(y, f) for y in YEARS_DEC for f in OTHERS if value(n[(s, y, f)]) > 0.5]
        if chosen:
            ny, fuel_choice = chosen[0]
        else:
            ny, fuel_choice = (None, None)
        summary.append({
            "Ship":       s,
            "Turbo_Year": ty,
            "New_Year":   ny,
            "Fuel":       fuel_choice
        })
    summary_df = pd.DataFrame(summary)

    # ───────────────────────────────────────────────────────────────────────────
    # 11) CO₂-Ausstoß-Vergleich berechnen (ohne Discounting)
    # ───────────────────────────────────────────────────────────────────────────
    # 11.1 Gesamt-Baseline-Emission (t CO₂)
    total_baseline_co2_t = sum(
        baseline_co2_g[(s, y)] / 1_000_000
        for s in ships for y in YEARS_FULL
    )

    # 11.2 Gesamt-Optimierte-Emission (t CO₂)
    total_opt_co2_t = 0.0
    for s in ships:
        row_s  = summary_df.loc[summary_df.Ship == s].iloc[0]
        y_retro= row_s["Turbo_Year"]
        y_new  = row_s["New_Year"]
        fuel_new= row_s["Fuel"]

        for y in YEARS_FULL:
            if y_new is not None and y >= y_new:
                total_opt_co2_t += new_co2_g.get((s, y, fuel_new), 0.0) / 1_000_000
            elif y_retro is not None and y >= y_retro:
                total_opt_co2_t += retro_co2_g.get((s, y), 0.0) / 1_000_000
            else:
                total_opt_co2_t += baseline_co2_g[(s, y)] / 1_000_000

    emissions_df = pd.DataFrame({
        "Variante": ["Optimiert", "Diesel-only"],
        "CO₂-Ausstoß (t)": [total_opt_co2_t, total_baseline_co2_t]
    })

    return comp_df, savings_df, summary_df, emissions_df

# ───────────────────────────────────────────────────────────────────────────────
# 5) Streamlit-UI: Sidebar + Button + Ausgabe
# ───────────────────────────────────────────────────────────────────────────────
YE_REF = [2025, 2030, 2035, 2040, 2045, 2050]

# CO₂‐Preis Slider
st.sidebar.header("CO₂‐Preis (USD/t)")
co2_ref = {y: st.sidebar.slider(f"CO₂ Price in {y}", 0, 1000, 100, 50) for y in YE_REF}
co2_prices = {
    y: co2_ref[max(k for k in YE_REF if k <= y)]
    for y in range(2025, 2051)
}

# Diesel‐Preis Slider
st.sidebar.header("Diesel‐Preis (USD/kg)")
diesel_ref = {y: st.sidebar.slider(f"Diesel Price in {y}", 0.0, 10.0, 1.0, 0.5) for y in YE_REF}
diesel_prices = {
    y: diesel_ref[max(k for k in YE_REF if k <= y)]
    for y in range(2025, 2051)
}

# HFO‐Preis Slider
st.sidebar.header("HFO‐Preis (USD/kg)")
hfo_ref = {y: st.sidebar.slider(f"HFO Price in {y}", 0.0, 1.5, 1.0, 0.1) for y in YE_REF}
hfo_prices = {
    y: hfo_ref[max(k for k in YE_REF if k <= y)]
    for y in range(2025, 2051)
}

if st.sidebar.button("🔍 Run Optimization"):
    with st.spinner("Berechne optimale Flotte…"):
        comp_df, savings_df, summary_df, emissions_df = run_fleet_optimization(
            co2_prices, diesel_prices, hfo_prices
        )
    st.success("Fertig!")

    # ─────────────────────────────────────────────────────────────────────────
    # Ergebnisse anzeigen
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📊 Kostenvergleich (NPV)")
    st.dataframe(comp_df.style.format({"Kosten NPV (USD)": "{:,.0f}"}))

    st.subheader("💰 Ersparnis")
    st.dataframe(savings_df.style.format({"Wert": "{:.2f}"}))

    st.subheader("🚢 Flotten-Entscheidungen")
    st.dataframe(summary_df)

    st.subheader("📉 CO₂-Ausstoßvergleich")
    st.dataframe(emissions_df.style.format({"CO₂-Ausstoß (t)": "{:,.0f}"}))
