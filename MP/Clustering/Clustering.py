import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load data
df = pd.read_csv("../csv_files/GlobalLandTemperaturesByCountry.csv")

# Keep necessary columns
df = df[['dt', 'AverageTemperature', 'Country']].dropna()

# Convert date
df['dt'] = pd.to_datetime(df['dt'])
df['Year'] = df['dt'].dt.year

# 2. Aggregate to country-level means
country_temp = (
    df.groupby('Country')['AverageTemperature']
      .mean()
      .reset_index()
)

# 3. Simulate regional CO2
# Global baseline ~400 ppm with regional variation
np.random.seed(42)
country_temp['CO2'] = 400 + np.random.normal(0, 15, size=len(country_temp))

# 4. Feature matrix
X = country_temp[['AverageTemperature', 'CO2']]

# Standardize for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
country_temp['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. Visualization
plt.figure(figsize=(8,6))
plt.scatter(
    country_temp['AverageTemperature'],
    country_temp['CO2'],
    c=country_temp['Cluster']
)
plt.xlabel("Average Temperature (°C)")
plt.ylabel("Simulated CO₂ (ppm)")
plt.title("Climate Clusters Based on Temperature and CO₂")
plt.grid(True)
plt.show()

# 7. Save output to CSV
output_path = "../csv_files/climate_clusters_by_country.csv"
country_temp.to_csv(output_path, index=False)