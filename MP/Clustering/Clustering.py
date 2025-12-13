import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
df = pd.read_csv("../../../pre_data/csv_file/region/GlobalLandTemperaturesByCountry.csv")

# Keep necessary columns
df = df[['dt', 'AverageTemperature', 'Country']].dropna()

# Convert date
df['dt'] = pd.to_datetime(df['dt'])
df['Year'] = df['dt'].dt.year

# 2. AGGREGATE TO COUNTRY-LEVEL MEANS
country_temp = (
    df.groupby('Country')['AverageTemperature']
      .mean()
      .reset_index()
)

# 3. SIMULATE REGIONAL CO₂ (OPTIONAL)
# Global baseline ~400 ppm with regional variation
np.random.seed(42)
country_temp['CO2'] = 400 + np.random.normal(0, 15, size=len(country_temp))

# 4. FEATURE MATRIX
X = country_temp[['AverageTemperature', 'CO2']]

# Standardize for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. APPLY K-MEANS CLUSTERING
kmeans = KMeans(n_clusters=4, random_state=42)
country_temp['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. VISUALIZATION
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

# 7. DISPLAY SAMPLE OUTPUT
print(country_temp.head())