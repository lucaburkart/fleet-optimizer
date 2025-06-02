# app.py

import streamlit as st
import pandas as pd
from pathlib import Path
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpBinary,
    LpStatusOptimal,
    value,
)

# ————————————————
# Page config muss ganz oben stehen
st.set_page_config(page_title="Fleet Optimization", layout="wide")
# ————————————————

st.write("✅ App lädt – UI ist aktiv")

# ---------------------------------------------------------------------------------------------------
# 1) PuLP-basierte Optimierungsfunktion (inkl. Datenimport)
# ---------------------------------------------------------------------------------------------------
def run_fleet_optimization(cf_co2_prices, cf_diesel_prices):
    """
    cf_co2_prices: dict Jahr → CO₂-Preis (USD/t)
    cf_diesel_prices: dict Jahr → Diesel-Preis (USD/kg)
    Rückgabe: (comp_df, savings_df, summary_df)
    """
    BASE_PATH = Path(".")

    # ─── 1) Daten einlesen ───
    # Achte darauf, dass die folgenden Dateien im selben Verzeichnis wie app.py liegen:
    #   ‣ fleet_data2.1.csv
    #   ‣ tech_fuel_data2.csv
    #   ‣ co2_price2.1.csv
    #   ‣ turbo_retrofit.1.csv
    #   ‣ new_ship_cost.1.csv
    #   ‣ new_fleet_data2.1.csv
    #   ‣ shipping_routes.xlsx

    fleet     = pd.read_csv(BASE_PATH / "fleet_data2.1.csv",     delimiter=";")
    # Hier benutzen wir tech_fuel_data2.csv (ohne .1)
    fuel      = pd.read_csv(BASE_PATH / "tech_fuel_data2.csv",   delimiter=";")
    co2_df    = pd.read_csv(BASE_PATH / "co2_price2.1.csv",       delimiter=";")
    turbo     = pd.read_csv(BASE_PATH / "turbo_retrofit.1.csv",   delimiter=";")
    new_cost  = pd.read_csv(BASE_PATH / "new_ship_cost.1.csv",    delimiter=";")
    new_specs = pd.read_csv(BASE_PATH / "new_fleet_data2.1.csv",  delimiter=";")
    routes_df = pd.read_excel(BASE_PATH / "shipping_routes.xlsx")

    # ─── 2) String-Spalten normalisieren ───
    for df in (fleet, fuel, turbo, new_cost, new_specs):
        for col in ("Ship_Type", "Fuel", "Fuel_Type"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()

    routes_df["Ship"] = routes_df["Ship"].astype(str).str.strip().str.title()

    # ─── 3) Lookup-Tabellen aufbauen ───
    fuel_lu = fuel.set_index(["Year", "Fuel_Type"]).to_dict("index")
    co2_lu  = co2_df.set_index("Year")["CO2_Price_EUR_per_ton"].to_dict()
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

    # ─── 4) „Share of ERA“ pro Schiff berechnen ───
    routes_df = routes_df[
        ["Ship", "Nautical Miles", "Share of ERA", "Energy Consumption [MJ] WtW"]
    ].dropna(subset=["Ship"])
    energy_groups = {}
    for ship, grp in routes_df.groupby("Ship"):
        tot_mj       = grp["Energy Consumption [MJ] WtW"].sum()
        tot_mj_eca   = (grp["Energy Consumption [MJ] WtW"] * grp["Share of ERA"]).sum()
        tot_mj_noeca = tot_mj - tot_mj_eca
        share_era    = (tot_mj_eca / tot_mj) if tot_mj > 0 else 0.0

        energy_groups[ship] = {
            "energy_per_voyage_mj":       tot_mj,
            "energy_eca_per_voyage_mj":   tot_mj_eca,
            "energy_noeca_per_voyage_mj": tot_mj_noeca,
            "share_era":                  share_era
        }

    # ─── 5) Modell-Parameter ───
    ships         = fleet["Ship_Type"].unique()
    YEARS_DEC     = list(range(2025, 2051, 5))  # [2025, 2030, 2035, 2040, 2045, 2050]
    YEARS_FULL    = list(range(2025, 2051))     # [2025..2050]
    BASIC         = "Diesel"
    OTHERS        = ["Lpg", "Green Methanol", "Green Ammonia"]
    discount_rate = 0.07
    dfac          = lambda y: 1 / ((1 + discount_rate) ** (y - 2025))

    # ─── 6) Operative Kosten berechnen (Baseline, Retrofit, Neubau) ───
    baseline_cost = {}
    retro_cost    = {}
    new_op_cost   = {}

    for s in ships:
        # a) Fleet-Zeile, Voyages, Power, alte MJ-Rates
        row    = fleet.loc[fleet.Ship_Type == s].iloc[0]
        voy    = row["Voyages"]
        pw     = row["Power"]
        mj_old = row.get("Energy_per_km (MJ/km)", row.get("Energy_per_km"))

        # Falls Schiff nicht in routes_df, setze alles auf 0
        eg = energy_groups.get(s, {
            "energy_per_voyage_mj":       0.0,
            "energy_eca_per_voyage_mj":   0.0,
            "energy_noeca_per_voyage_mj": 0.0,
            "share_era":                  0.0
        })
        energy_voyage       = eg["energy_per_voyage_mj"]
        energy_eca_voyage   = eg["energy_eca_per_voyage_mj"]
        energy_noeca_voyage = eg["energy_noeca_per_voyage_mj"]
        share_era           = eg["share_era"]

        # b) Neue MJ-Rate + Power für Neubau
        mj_new = new_lu[s]["Energy_per_km_new"]
        factor_new_to_old = (mj_new / mj_old) if mj_old != 0 else 1.0

        for y in YEARS_FULL:
            # jährliche Energiemengen in MJ
            ann_mj       = energy_voyage * voy
            ann_mj_eca   = energy_eca_voyage * voy
            ann_mj_noeca = energy_noeca_voyage * voy

            # --- Baseline-Kosten (Fuel ECA, Fuel non-ECA, CO₂, Wartung) ---
            # 1) Fuel ECA (immer Diesel, override mit cf_diesel_prices[y])
            cost_eca = 0.0
            if ann_mj_eca > 1e-9:
                kg_eca   = ann_mj_eca / fuel_lu[(y, BASIC)]["Energy_MJ_per_kg"]
                cost_eca = kg_eca * cf_diesel_prices[y]

            # 2) Fuel non-ECA (immer HFO, Price aus fuel_lu)
            cost_noeca = 0.0
            if ann_mj_noeca > 1e-9:
                kg_noeca   = ann_mj_noeca / fuel_lu[(y, "Hfo")]["Energy_MJ_per_kg"]
                cost_noeca = kg_noeca * fuel_lu[(y, "Hfo")]["Price_USD_per_kg"]

            # 3) CO₂-Kosten Baseline
            co2_base = 0.0
            if ann_mj > 1e-9:
                ef_diesel = fuel_lu[(y, BASIC)]["CO2_g_per_MJ"]
                ef_hfo    = fuel_lu[(y, "Hfo")]["CO2_g_per_MJ"]
                co2t_base = ann_mj * (share_era * ef_diesel + (1 - share_era) * ef_hfo) / 1_000_000
                co2_base  = co2t_base * cf_co2_prices.get(y, 0)

            # 4) Wartungskosten Baseline
            if ann_mj > 1e-9:
                ma_base = pw * (
                    (energy_eca_voyage / energy_voyage) * fuel_lu[(y, BASIC)]["Maintenance_USD_per_kW"] +
                    (energy_noeca_voyage / energy_voyage) * fuel_lu[(y, "Hfo")]["Maintenance_USD_per_kW"]
                )
            else:
                ma_base = pw * fuel_lu[(y, BASIC)]["Maintenance_USD_per_kW"]

            # operative Base-Kosten (diskontiert)
            baseline_cost[(s, y)] = (cost_eca + cost_noeca + co2_base + ma_base) * dfac(y)

            # --- Retrofit-Kosten (mit Turbo-Saving) ---
            save_pct = T_SAVE.get((s, y), 0) / 100
            ann_mj_eca_retro   = ann_mj_eca   * (1 - save_pct)
            ann_mj_noeca_retro = ann_mj_noeca * (1 - save_pct)

            cost_eca_retro = 0.0
            if ann_mj_eca_retro > 1e-9:
                kg_eca_retro   = ann_mj_eca_retro / fuel_lu[(y, BASIC)]["Energy_MJ_per_kg"]
                cost_eca_retro = kg_eca_retro * cf_diesel_prices[y]

            cost_noeca_retro = 0.0
            if ann_mj_noeca_retro > 1e-9:
                kg_noeca_retro   = ann_mj_noeca_retro / fuel_lu[(y, "Hfo")]["Energy_MJ_per_kg"]
                cost_noeca_retro = kg_noeca_retro * fuel_lu[(y, "Hfo")]["Price_USD_per_kg"]

            co2_retro = 0.0
            if ann_mj > 1e-9:
                ef_diesel  = fuel_lu[(y, BASIC)]["CO2_g_per_MJ"]
                ef_hfo     = fuel_lu[(y, "Hfo")]["CO2_g_per_MJ"]
                co2t_retro = (ann_mj_eca_retro * ef_diesel + ann_mj_noeca_retro * ef_hfo) / 1_000_000
                co2_retro  = co2t_retro * cf_co2_prices.get(y, 0)

            if ann_mj > 1e-9:
                ma_retro = pw * (
                    (energy_eca_voyage / energy_voyage) * fuel_lu[(y, BASIC)]["Maintenance_USD_per_kW"] +
                    (energy_noeca_voyage / energy_voyage) * fuel_lu[(y, "Hfo")]["Maintenance_USD_per_kW"]
                )
            else:
                ma_retro = pw * fuel_lu[(y, BASIC)]["Maintenance_USD_per_kW"]

            inv_retro = 0.0
            if (s, y) in T_COST:
                inv_retro = T_COST[(s, y)] * dfac(y)

            # operative Retrofit-Kosten + Capex (diskontiert)
            retro_cost[(s, y)] = (
                (cost_eca_retro + cost_noeca_retro + co2_retro + ma_retro) * dfac(y)
            ) + inv_retro

            # --- Neubau-Kosten (Fuel, CO₂, Wartung, Capex) mit neuen Specs ---
            energy_new_voyage       = energy_voyage       * factor_new_to_old
            energy_new_eca_voyage   = energy_eca_voyage   * factor_new_to_old
            energy_new_noeca_voyage = energy_noeca_voyage * factor_new_to_old

            ann_mj_new       = energy_new_voyage       * voy
            ann_mj_new_eca   = energy_new_eca_voyage   * voy
            ann_mj_new_noeca = energy_new_noeca_voyage * voy

            pw_new = new_lu[s]["Power_new"]
            for f in OTHERS:
                costn_eca = 0.0
                if ann_mj_new_eca > 1e-9:
                    kg_new_eca = ann_mj_new_eca / fuel_lu[(y, f)]["Energy_MJ_per_kg"]
                    costn_eca  = kg_new_eca * fuel_lu[(y, f)]["Price_USD_per_kg"]

                costn_noeca_f = 0.0
                if ann_mj_new_noeca > 1e-9:
                    kg_new_noeca_f = ann_mj_new_noeca / fuel_lu[(y, f)]["Energy_MJ_per_kg"]
                    costn_noeca_f  = kg_new_noeca_f * fuel_lu[(y, f)]["Price_USD_per_kg"]

                ef_fuel = fuel_lu[(y, f)]["CO2_g_per_MJ"]
                co2_new = (ann_mj_new * ef_fuel / 1_000_000) * cf_co2_prices.get(y, 0)

                if ann_mj_new > 1e-9:
                    ma_new_f = pw_new * (
                        (energy_new_eca_voyage / energy_new_voyage) * fuel_lu[(y, f)]["Maintenance_USD_per_kW"] +
                        (energy_new_noeca_voyage / energy_new_voyage) * fuel_lu[(y, f)]["Maintenance_USD_per_kW"]
                    )
                else:
                    ma_new_f = pw_new * fuel_lu[(y, f)]["Maintenance_USD_per_kW"]

                inv_new = 0.0
                if (s, f, y) in N_COST:
                    inv_new = N_COST[(s, f, y)] * dfac(y)

                new_op_cost[(s, y, f)] = (
                    (costn_eca + costn_noeca_f + co2_new + ma_new_f) * dfac(y)
                ) + inv_new

    # ─── 7) MIP-Modell mit PuLP definieren ───
    mdl = LpProblem("Fleet_Optimization_NoDemand", LpMinimize)

    t = LpVariable.dicts(
        "Turbo",
        [(s, y) for s in ships for y in YEARS_DEC],
        lowBound=0,
        upBound=1,
        cat=LpBinary,
    )
    n = LpVariable.dicts(
        "New",
        [(s, y, f) for s in ships for y in YEARS_DEC for f in OTHERS],
        lowBound=0,
        upBound=1,
        cat=LpBinary,
    )

    # 7.1) Constraints: max einmal Retrofit, max einmal Neubau, kein Retrofit nach Neubau
    for s in ships:
        mdl += lpSum(t[(s, y)] for y in YEARS_DEC) <= 1, f"maxRetro_{s}"
        mdl += lpSum(n[(s, y, f)] for y in YEARS_DEC for f in OTHERS) <= 1, f"maxNew_{s}"
        for y in YEARS_DEC:
            mdl += (
                t[(s, y)]
                <= 1
                - lpSum(
                    n[(s, yy, f)]
                    for yy in YEARS_DEC
                    if yy <= y
                    for f in OTHERS
                )
            ), f"noRetroAfterNew_{s}_{y}"

    # 7.2) Zielfunktion
    obj_terms = []
    for s in ships:
        for y in YEARS_FULL:
            cum_retro = lpSum(t[(s, yy)] for yy in YEARS_DEC if yy <= y)
            cum_new   = lpSum(n[(s, yy, f)] for yy in YEARS_DEC if yy <= y for f in OTHERS)

            base_active  = 1 - cum_retro - cum_new
            retro_active = cum_retro * (1 - cum_new)
            new_active   = cum_new

            obj_terms.append(baseline_cost[(s, y)] * base_active)
            obj_terms.append(retro_cost[(s, y)]   * retro_active)
            for f in OTHERS:
                obj_terms.append(new_op_cost[(s, y, f)] * new_active)

    mdl += lpSum(obj_terms)

    # 8) Optimieren
    mdl.solve()

    if mdl.status == LpStatusOptimal:
        optimized_cost   = value(mdl.objective)
        diesel_only_cost = sum(baseline_cost[(s, y)] for s in ships for y in YEARS_FULL)

        saving_abs = diesel_only_cost - optimized_cost
        saving_pct = saving_abs / diesel_only_cost * 100

        comp_df = pd.DataFrame({
            "Variante": ["Optimiert", "Diesel-only"],
            "Kosten NPV (USD)": [optimized_cost, diesel_only_cost]
        })
        savings_df = pd.DataFrame({
            "Messgröße": ["Ersparnis absolut (USD)", "Ersparnis relativ (%)"],
            "Wert":      [saving_abs, saving_pct]
        })

        summary = []
        for s in ships:
            ty = next((yy for yy in YEARS_DEC if value(t[(s, yy)]) > 0.5), None)
            chosen_new = [(yy, f) for yy in YEARS_DEC for f in OTHERS if value(n[(s, yy, f)]) > 0.5]
            if chosen_new:
                ny, fuel_chosen = chosen_new[0]
            else:
                ny, fuel_chosen = (None, None)

            summary.append({
                "Ship":       s,
                "Turbo_Year": ty,
                "New_Year":   ny,
                "Fuel":       fuel_chosen
            })
        summary_df = pd.DataFrame(summary)

        return comp_df, savings_df, summary_df

    else:
        raise RuntimeError(f"Optimierung nicht optimal gelöst (Status {mdl.status})")


# ---------------------------------------------------------------------------------------------------
# 2) Streamlit-UI (Sliders + Button)
# ---------------------------------------------------------------------------------------------------

# CO₂-Preis Slider (blockweise alle 5 Jahre)
st.sidebar.header("CO₂ Price Settings (USD/t)")
slider_years = [2025, 2030, 2035, 2040, 2045, 2050]
base_co2_prices = {
    y: st.sidebar.slider(f"CO₂ Price in {y}", 0, 1000, 100, step=50)
    for y in slider_years
}
co2_prices = {}
for y in range(2025, 2051):
    last_ref = max(year for year in slider_years if year <= y)
    co2_prices[y] = base_co2_prices[last_ref]

# Diesel-Preis Slider (USD/kg, blockweise alle 5 Jahre)
st.sidebar.header("Diesel Price Settings (USD/kg)")
base_diesel_prices = {
    y: st.sidebar.slider(f"Diesel Price in {y}", 0.0, 10.0, 1.0, step=0.5)
    for y in slider_years
}
diesel_prices = {}
for y in range(2025, 2051):
    last_ref = max(year for year in slider_years if year <= y)
    diesel_prices[y] = base_diesel_prices[last_ref]

# Button: Run Optimization
if st.sidebar.button("🔍 Run Optimization"):
    with st.spinner("Running optimization..."):
        comp_df, savings_df, summary_df = run_fleet_optimization(co2_prices, diesel_prices)

    st.success("Done!")
    st.subheader("📊 Cost Comparison")
    st.dataframe(comp_df.style.format({"Kosten NPV (USD)": "{:,.0f}"}))

    st.subheader("💰 Savings")
    st.dataframe(savings_df.style.format({"Wert": "{:.2f}"}))

    st.subheader("🚢 Fleet Decisions")
    st.dataframe(summary_df)
