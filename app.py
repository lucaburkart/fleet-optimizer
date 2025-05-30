# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatusOptimal, value

# 1) Page config
st.set_page_config(page_title="Fleet Optimization", layout="wide")

# 2) Read data
BASE_PATH = Path(".")
fleet     = pd.read_csv(BASE_PATH / "fleet_data2.1.csv",     delimiter=";")
fuel      = pd.read_csv(BASE_PATH / "tech_fuel_data2.1.csv", delimiter=";")
turbo     = pd.read_csv(BASE_PATH / "turbo_retrofit.1.csv",  delimiter=";")
new_cost  = pd.read_csv(BASE_PATH / "new_ship_cost.1.csv",   delimiter=";")
new_specs = pd.read_csv(BASE_PATH / "new_fleet_data2.1.csv", delimiter=";")

# Normalize
for df in (fleet, fuel, turbo, new_cost, new_specs):
    for col in ("Ship_Type","Fuel","Fuel_Type"): df[col] = df[col].astype(str).str.strip().str.title()

fuel_lu = fuel.set_index(["Year","Fuel_Type"]).to_dict("index")
T_COST  = turbo.set_index(["Ship_Type","Year"])["Retrofit_Cost_USD"].to_dict()
T_SAVE  = turbo.set_index(["Ship_Type","Year"])["Energy_Saving_%"].to_dict()
N_COST  = new_cost.set_index(["Ship_Type","Fuel","Year"])["Capex_USD"].to_dict()

new_specs_idx = new_specs.set_index("Ship_Type")
new_lu = {s:{"Energy_per_km_new":r["Energy_per_km (MJ/km)_new"],"Power_new":r.get("Power_kw_new",r.get("Power"))} for s,r in new_specs_idx.iterrows()}

YEARS_DEC  = list(range(2025,2051,5))
YEARS_FULL = list(range(2025,2051))
OTHERS     = ["Lpg","Green Methanol","Green Ammonia"]

def run_fleet_optimization(co2_prices, fuel_price_overrides):
    dfac = lambda y: 1/((1+0.07)**(y-2025))
    ships = fleet["Ship_Type"].unique()
    # precompute costs
    baseline, retro, new_op = {},{},{}
    for s in ships:
        r = fleet[fleet.Ship_Type==s].iloc[0]
        dist,voy,mj,pw = r["Distance"],r["Voyages"],r.get("Energy_per_km (MJ/km)",r.get("Energy_per_km")),r["Power"]
        for y in YEARS_FULL:
            # baseline
            price_kg = fuel_lu[(y,"Diesel")]["Price_USD_per_kg"]
            fc = dist*voy*mj/fuel_lu[(y,"Diesel")]["Energy_MJ_per_kg"]*price_kg
            co2t = dist*voy*mj*fuel_lu[(y,"Diesel")]["CO2_g_per_MJ"]/1e6
            cc = co2t*co2_prices[y]
            ma = pw*fuel_lu[(y,"Diesel")]["Maintenance_USD_per_kW"]
            baseline[(s,y)] = (fc+cc+ma)*dfac(y)
            # retrofit
            save = T_SAVE.get((s,y),0)/100
            retro[(s,y)] = ((fc+cc)*(1-save)+ma)*dfac(y)
            # new
            for f in OTHERS:
                p_kg = fuel_price_overrides[y].get(f,fuel_lu[(y,f)]["Price_USD_per_kg"])
                fc_new = dist*voy*new_lu[s]["Energy_per_km_new"]/fuel_lu[(y,f)]["Energy_MJ_per_kg"]*p_kg
                co2n = dist*voy*new_lu[s]["Energy_per_km_new"]*fuel_lu[(y,f)]["CO2_g_per_MJ"]/1e6
                cc_new=co2n*co2_prices[y]
                ma_new=new_lu[s]["Power_new"]*fuel_lu[(y,f)]["Maintenance_USD_per_kW"]
                new_op[(s,y,f)] = (fc_new+cc_new+ma_new)*dfac(y)
    # model
    mdl=LpProblem("Fleet_Optimization",LpMinimize)
    t=LpVariable.dicts("Turbo",[(s,y) for s in ships for y in YEARS_DEC],0,1,LpBinary)
    n=LpVariable.dicts("New",  [(s,y,f) for s in ships for y in YEARS_DEC for f in OTHERS],0,1,LpBinary)
    # constraints
    for s in ships:
        mdl+=lpSum(t[(s,y)] for y in YEARS_DEC)<=1
        mdl+=lpSum(n[(s,y,f)] for y in YEARS_DEC for f in OTHERS)<=1
        for y in YEARS_DEC:
            mdl+=t[(s,y)]<=1-lpSum(n[(s,yy,f)] for yy in YEARS_DEC if yy<=y for f in OTHERS)
    # objective
    obj=[]
    for s in ships:
        for y in YEARS_FULL:
            # baseline
            obj.append(baseline[(s,y)])
            # retrofit activity
            obj.append(retro[(s,y)]*lpSum(t[(s,yy)] for yy in YEARS_DEC if yy<=y))
            # new by fuel
            for f in OTHERS:
                cum_new_f=lpSum(n[(s,yy,f)] for yy in YEARS_DEC if yy<=y)
                obj.append(new_op[(s,y,f)]*cum_new_f)
        # capex
        for y in YEARS_DEC:
            if (s,y) in T_COST: obj.append(T_COST[(s,y)]*t[(s,y)]*dfac(y))
            for f in OTHERS:
                if (s,f,y) in N_COST: obj.append(N_COST[(s,f,y)]*n[(s,y,f)]*dfac(y))
    mdl+=lpSum(obj)
    mdl.solve()
    # results
    comp=sum(baseline.values())
    opt=value(mdl.objective)
    comp_df=pd.DataFrame({"Variante":["Optimiert","Diesel-only"],"Kosten PV (USD)":[opt,comp]})
    savings_df=pd.DataFrame({"Messgröße":["Absolut (USD)","Relativ (%)"],"Wert":[comp-opt,(comp-opt)/comp*100]})
    summary=[]
    for s in ships:
        ty=next((y for y in YEARS_DEC if t[(s,y)].varValue>0.5),None)
        sel=[(yy,ff) for yy in YEARS_DEC for ff in OTHERS if n[(s,yy,ff)].varValue>0.5]
        ny,fc = sel[0] if sel else (None,None)
        summary.append({"Ship":s,"Turbo_Year":ty,"New_Year":ny,"Fuel":fc})
    summary_df=pd.DataFrame(summary)
    return comp_df,savings_df,summary_df
# UI
st.sidebar.header("CO₂ Price Settings (€/t)")
co2_prices={year:st.sidebar.slider(f"CO₂ in {year}",0,1000,100,50) for year in YEARS_DEC}
st.sidebar.header("Fuel Price Overrides (USD/kg)")
fuel_price_overrides={year:{} for year in YEARS_FULL}
for f in OTHERS:
    st.sidebar.subheader(f)
    for y in YEARS_DEC:
        fuel_price_overrides[y][f]=st.sidebar.slider(f"{f}@{y}",0.0,10.0,fuel_lu[(y,f)]["Price_USD_per_kg"],0.1)
if st.sidebar.button("Run Optimization"):
    with st.spinner():
        comp_df,savings_df,summary_df=run_fleet_optimization(co2_prices,fuel_price_overrides)
    st.subheader("Cost Comparison")
    st.dataframe(comp_df)
    st.subheader("Savings")
    st.dataframe(savings_df)
    st.subheader("Decisions")
    st.dataframe(summary_df)
