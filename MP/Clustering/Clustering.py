import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster.cluster import KMeans
import matplotlib.pyplot as plt

# 1. LOAD DATA
temp = pd.read_csv("final_annual_temperature_data.csv")
co2 = pd.read_csv("final_co2_monthly.csv")

# Use temperature trend (last available year) as global baseline
latest_temp = temp['Annual_Mean_Absolute'].iloc[-1]

# Use CO2 trend (last available year)
latest_co2 = co2['trend_monthly'].iloc[-1]

# 2. SIMULATE GEOGRAPHICAL REGIONS
regions = ["North America", "South America", "Europe", "Africa",
           "Asia", "Oceania", "Middle East", "Arctic"]

# Simulate temperature variation around global mean
region_temps = latest_temp + np.random.normal(0, 0.5, len(regions))

# OPTIONAL: simulate CO₂ concentration differences
region_co2 = latest_co2 + np.random.normal(0, 3.0, len(regions))

df = pd.DataFrame({
    "Region": regions,
    "Temp": region_temps,
    "CO2": region_co2
})

print("Simulated Regional Dataset:")
print(df)

# 3. PREPARE FOR CLUSTERING
features = df[['Temp', 'CO2']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 4. K-MEANS CLUSTERING
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

print("\nCluster Assignments:")
print(df)

# 5. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.scatter(df['Temp'], df['CO2'], c=df['Cluster'], s=200)

for i, label in enumerate(df['Region']):
    plt.text(df['Temp'][i] + 0.01, df['CO2'][i] + 0.01, label)

plt.xlabel("Regional Temperature (°C)")
plt.ylabel("Regional CO₂ Concentration (ppm)")
plt.title("K-Means Clustering of Regions by Temperature and CO₂")
plt.grid(True)
plt.show()