# 1) Einlesen der Bestandsdaten
fleet = pd.read_csv("fleet_data2.1.csv", delimiter=";")

# 2) Einlesen der Neubau‐Daten
new_specs = pd.read_csv("new_fleet_data2.1.csv", delimiter=";")

# 3) Vergleichstabelle zusammenbauen
vergleich = []
for s in fleet["Ship_Type"].unique():
    # MJ_old aus Bestandsdaten
    row_old = fleet.loc[fleet.Ship_Type == s].iloc[0]
    MJold = row_old["Energy_per_km (MJ/km)"]
    
    # MJ_new aus Neubau-Daten
    row_new = new_specs.loc[new_specs.Ship_Type == s].iloc[0]
    MJnew = row_new["Energy_per_km (MJ/km)_new"]
    
    vergleich.append({
        "Schiff": s,
        "MJ_old (MJ/km)": MJold,
        "MJ_new (MJ/km)": MJnew,
        "MJ_new < MJ_old?": MJnew < MJold
    })

df_vergleich = pd.DataFrame(vergleich)
st.write("🔍 Vergleich MJ_old vs. MJ_new", df_vergleich)
