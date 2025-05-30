# app.py
import streamlit as st
import pandas as pd
from fleet_optimization import run_fleet_optimization

st.set_page_config(page_title="Fleet Optimization", layout="wide")
st.title("🚢 Fleet Optimization Web App")

st.sidebar.header("CO₂ Price Settings (€/t)")
co2_prices = {}
for year in [2025, 2030, 2035, 2040, 2045, 2050]:
    co2_prices[year] = st.sidebar.slider(f"CO₂ Price in {year}", 0, 5000, 100, step=50)

if st.button("🔍 Run Optimization"):
    try:
        with st.spinner("Running optimization model..."):
            comp_df, savings_df, summary_df = run_fleet_optimization(co2_prices)
        st.success("Optimization complete!")

        st.subheader("📊 Cost Comparison")
        st.dataframe(comp_df.style.format({"Kosten PV (USD)": "{:,.0f}"}))

        st.subheader("💰 Savings")
        st.dataframe(savings_df.style.format({"Wert": "{:,.2f}"}))

        st.subheader("🚢 Fleet Decisions")
        st.dataframe(summary_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
