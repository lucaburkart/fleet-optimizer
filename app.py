import streamlit as st
import pandas as pd
from pathlib import Path
from pulp import (
    LpProblem, LpMinimize, LpVariable,
    lpSum, LpBinary, LpStatusOptimal, value,
)

# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fleet Optimization", layout="wide")
st.write("✅ App geladen – UI ist aktiv")
# ───────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────────────
# 1) Daten einlesen
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
# Debug: Vergleiche MJ_old vs. MJ_new direkt nach Einlesen
# ───────────────────────────────────────────────────────────────────────────────
st.subheader("🔍 Debug: Energieverbrauch Alt vs. Neu (MJ/km)")

vergleich = []
for s in fleet["Ship_Type"].unique():
    # MJ_old aus Bestandsdaten
    row_old = fleet.loc[fleet.Ship_Type == s].iloc[0]
    MJold = row_old.get("Energy_per_km (MJ/km)", row_old.get("Energy_per_km"))
    
    # MJ_new aus Neubau-Daten
    if s in new_specs["Ship_Type"].values:
        row_new = new_specs.loc[new_specs.Ship_Type == s].iloc[0]
        MJnew = row_new["Energy_per_km (MJ/km)_new"]
    else:
        MJnew = None
    
    vergleich.append({
        "Ship_Type":            s,
        "MJ_old (MJ/km)":       MJold,
        "MJ_new (MJ/km)":       MJnew,
        "MJ_new < MJ_old?":     (MJnew < MJold) if (MJnew is not None) else "n/a"
    })

df_vergleich = pd.DataFrame(vergleich)
st.dataframe(df_vergleich.style.format({
    "MJ_old (MJ/km)": "{:.0f}",
    "MJ_new (MJ/km)": "{:.0f}"
}))

# ───────────────────────────────────────────────────────────────────────────────
# 2) Funktion run_fleet_optimization (unverändert, nur gekürzt gezeigt)
# ───────────────────────────────────────────────────────────────────────────────
def run_fleet_optimization(co2_prices: dict[int, float],
                           diesel_prices: dict[int, float],
                           hfo_prices: dict[int, float]):
    # … hier kommt dein Code aus Punkt 1–9, unverändert …
    return comp_df, savings_df, summary_df, emissions_df

# ───────────────────────────────────────────────────────────────────────────────
# 3) Streamlit-UI: Sidebar + Button
# ───────────────────────────────────────────────────────────────────────────────
YE_REF = [2025, 2030, 2035, 2040, 2045, 2050]

st.sidebar.header("CO₂-Preis (USD/t)")
co2_ref = {y: st.sidebar.slider(f"CO₂ Price in {y}", 0, 1000, 100, 50) for y in YE_REF}
co2_prices = {y: co2_ref[max(k for k in YE_REF if k <= y)] for y in range(2025, 2051)}

st.sidebar.header("Diesel-Preis (USD/kg)")
diesel_ref = {y: st.sidebar.slider(f"Diesel Price in {y}", 0.0, 10.0, 1.0, 0.5) for y in YE_REF}
diesel_prices = {y: diesel_ref[max(k for k in YE_REF if k <= y)] for y in range(2025, 2051)}

st.sidebar.header("HFO-Preis (USD/kg)")
hfo_ref = {y: st.sidebar.slider(f"HFO Price in {y}", 0.0, 1.5, 1.0, 0.1) for y in YE_REF}
hfo_prices = {y: hfo_ref[max(k for k in YE_REF if k <= y)] for y in range(2025, 2051)}

if st.sidebar.button("🔍 Run Optimization"):
    with st.spinner("Berechne optimale Flotte…"):
        comp_df, savings_df, summary_df, emissions_df = run_fleet_optimization(
            co2_prices, diesel_prices, hfo_prices
        )
    st.success("Fertig!")

    # ─────────────────────────────────────────────────────────────────────────
    # Debug: Noch einmal compare MJ_old vs. MJ_new anzeigen
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("🔍 Debug: Energieverbrauch Alt vs. Neu (MJ/km)")
    st.dataframe(df_vergleich.style.format({
        "MJ_old (MJ/km)": "{:.0f}",
        "MJ_new (MJ/km)": "{:.0f}"
    }))

    # ─────────────────────────────────────────────────────────────────────────
    # Ergebnisse anzeigen…
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📊 Kostenvergleich (NPV)")
    st.dataframe(comp_df.style.format({"Kosten NPV (USD)": "{:,.0f}"}))

    st.subheader("💰 Ersparnis")
    st.dataframe(savings_df.style.format({"Wert": "{:.2f}"}))

    st.subheader("🚢 Flotten-Entscheidungen")
    st.dataframe(summary_df)

    st.subheader("📉 CO₂-Ausstoßvergleich")
    st.dataframe(emissions_df.style.format({"CO₂-Ausstoß (t)": "{:,.0f}"}))
