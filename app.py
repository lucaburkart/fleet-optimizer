# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatusOptimal, value

# 1) page config (must be first)
st.set_page_config(page_title="Fleet Optimization", layout="wide")

# 2) read data
BASE_PATH = Path(".")
fleet     = pd.read_csv(BASE_PATH / "fleet_data2.1.csv",     delimiter=";")
fuel      = pd.read_csv(BASE_PATH / "tech_fuel_data2.1.csv", delimiter=";")
turbo     = pd.read_csv(BASE_PATH / "turbo_retrofit.1.csv",  delimiter=";")
new_cost  = pd.read_csv(BASE_PATH / "new_ship_cost.1.csv",   delimiter=";")
new_specs = pd.read_csv(BASE_PATH / "new_fleet_data2.1.csv", delimiter=";")

# normalize text columns
for df in (fleet, fuel, turbo, new_cost, new_specs):
    for col in ("Ship_Type", "Fuel", "Fuel_Type"):
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

# build lookup tables
fuel_lu = fuel.set_index(["Year", "Fuel_Type"]).to_dict("index")
T_COST  = turbo.set_index(["Ship_Type", "Year"])["Retrofit_Cost_USD"].to_dict()
T_SAVE  = turbo.set_index(["Ship_Type", "Year"])["Energy_Saving_%"].to_dict()
N_COST  = new_cost.set_index(["Ship_Type", "Fuel", "Year"])["Capex_USD"].to_dict()

new_specs_idx = new_specs.set_index("Ship_Type")
new_lu = {
    ship:{
      "Energy_per_km_new":row["Energy_per_km (MJ/km)_new"],
      "Power_new":row.get("Power_kw_new",row.get("Power"))
    }
    for ship,row in new_specs_idx.iterrows()
}

YEARS_DEC  = list(range(2025,2051,5))
YEARS_FULL = list(range(2025,2051))
BASIC      = "Diesel"
OTHERS     = ["Lpg","Green Methanol","Green Ammonia"]

# 3) UI sliders
st.sidebar.header("CO₂ Price Settings (€/t)")
co2_prices = {
    year: st.sidebar.slider(f"CO₂ Price in {year}", 0, 1000, 100, step=50)
    for year in YEARS_DEC
}

st.sidebar.header("Fuel Price Overrides (USD/kg)")
# default = first available price in your fuel CSV
fuel_defaults = { (y,f):fuel_lu[(y,f)]["Price_USD_per_kg"] for y in YEARS_FULL for f in OTHERS }

fuel_price_overrides = {
    year:{}
    for year in YEARS_FULL
}
for f in OTHERS:
    st.sidebar.subheader(f"{f} Price (USD/kg)")
    for year in YEARS_DEC:
        default = fuel_defaults.get((year,f), 1.0)
        fuel_price_overrides[year][f] = st.sidebar.slider(
            f"{f} @ {year}", 0.0, 10.0, float(default), step=0.1
        )

# 4) model function
def run_fleet_optimization(co2_prices, fuel_price_overrides):
    ships = fleet["Ship_Type"].unique()
    dfac = lambda y: 1/((1+0.07)**(y-2025))

    # precompute costs
    baseline_cost = {}
    retro_cost    = {}
    new_op_cost   = {}
    for s in ships:
        row = fleet[fleet.Ship_Type==s].iloc[0]
        dist,voy = row["Distance"],row["Voyages"]
        mj_base  = row.get("Energy_per_km (MJ/km)",row.get("Energy_per_km"))
        pw = row["Power"]

        for y in YEARS_FULL:
            # base diesel
            price_kg = fuel_lu[(y,BASIC)]["Price_USD_per_kg"]
            fc = dist*voy*mj_base/ fuel_lu[(y,BASIC)]["Energy_MJ_per_kg"] * price_kg
            co2t   = dist*voy*mj_base*fuel_lu[(y,BASIC)]["CO2_g_per_MJ"]/1e6
            cc     = co2t*co2_prices.get(y,0)
            ma     = pw*fuel_lu[(y,BASIC)]["Maintenance_USD_per_kW"]
            baseline_cost[(s,y)] = (fc+cc+ma)*dfac(y)

            # retrofit
            save = T_SAVE.get((s,y),0)/100
            retro_cost[(s,y)] = ((fc+cc)*(1-save)+ma)*dfac(y)

            # new options
            for f in OTHERS:
                # override fuel price if set
                p_kg = fuel_price_overrides[y].get(f,
                      fuel_lu[(y,f)]["Price_USD_per_kg"])
                fc_new = dist*voy*new_lu[s]["Energy_per_km_new"]/fuel_lu[(y,f)]["Energy_MJ_per_kg"] * p_kg
                co2n   = dist*voy*new_lu[s]["Energy_per_km_new"]*fuel_lu[(y,f)]["CO2_g_per_MJ"]/1e6
                cc_new = co2n*co2_prices.get(y,0)
                ma_new = new_lu[s]["Power_new"]*fuel_lu[(y,f)]["Maintenance_USD_per_kW"]
                new_op_cost[(s,y,f)] = (fc_new+cc_new+ma_new)*dfac(y)

    # build MIP
    mdl = LpProblem("Fleet_Optimization",LpMinimize)
    t = LpVariable.dicts("Turbo",[(s,y) for s in ships for y in YEARS_DEC],0,1,LpBinary)
    n = LpVariable.dicts("New",  [(s,y,f) for s in ships for y in YEARS_DEC for f in OTHERS],0,1,LpBinary)

    for s in ships:
        mdl += lpSum(t[(s,y)] for y in YEARS_DEC)<=1
        mdl += lpSum(n[(s,y,f)] for y in YEARS_DEC for f in OTHERS)<=1
        for y in YEARS_DEC:
            mdl += t[(s,y)] <= 1-lpSum(n[(s,yy,f)] for yy in YEARS_DEC if yy<=y for f in OTHERS)

    # objective
    obj = []
    for s in ships:
        for y in YEARS_FULL:
            # diesel + retrofit
            obj.append(baseline_cost[(s,y)])
            obj.append(retro_cost[(s,y)] * lpSum(t[(s,yy)] for yy in YEARS_DEC if yy<=y))
            # one fuel at a time
            for f in OTHERS:
                cum_new_f = lpSum(n[(s,yy,f)] for yy in YEARS_DEC if yy<=y)
                obj.append(new_op_cost[(s,y,f)] * cum_new_f)
        # capex
        for y in YEARS_DEC:
            if (s,y) in T_COST:
                obj.append(T_COST[(s,y)] * t[(s,y)]*dfac(y))
            for f in OTHERS:
                if (s,f,y) in N_COST:
                    obj.append(N_COST[(s,f,y)] * n[(s,y,f)]*dfac(y))

    mdl += lpSum(obj)
    mdl.solve()

    # parse results...
    comp = sum(baseline_cost.values())
    opt = value(mdl.objective)
    comp_df = pd.DataFrame({
      "Variante":["Optimiert","Diesel-only"],
      "Kosten PV (USD)":[opt,comp]
    })
    savings_df = pd.DataFrame({
      "Messgröße":["Absolut (USD)","Relativ (%)"],
      "Wert":[comp-opt,(comp-opt)/comp*100]
    })
    summary=[]
    for s in ships:
        ty = next((y for y in YEARS_DEC if t[(s,y)].X>0.5),None)
        sel = [(y,f) for y in YEARS_DEC for f in OTHERS if n[(s,y,f)].X>0.5]
        ny,f = sel[0] if sel else (None,None)
        summary.append({"Ship":s,"Turbo_Year":ty,"New_Year":ny,"Fuel":f})
    summary_df = pd.DataFrame(summary)
    return comp_df,savings_df,summary_df

# 5) UI: run & show
st.title("🚢 Fleet Optimization Web App")
if st.sidebar.button("Run Optimization"):
    with st.spinner():
        c,su,sm = run_fleet_optimization(co2_prices,fuel_price_overrides)
    st.dataframe(c), st.dataframe(su), st.dataframe(sm)
