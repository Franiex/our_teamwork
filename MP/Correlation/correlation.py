import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATASETS
annual_temp = pd.read_csv("../../../pre_data/csv_file/globe/final_annual_temperature_data.csv")
co2 = pd.read_csv("../../../pre_data/csv_file/globe/final_co2_monthly.csv")
global_temp = pd.read_csv("../../../pre_data/csv_file/region/GlobalLandTemperaturesByCountry.csv")

# 2. PREPROCESS GLOBAL TEMPERATURE DATA
global_temp = global_temp[['dt', 'AverageTemperature', 'Country']].dropna()
global_temp['dt'] = pd.to_datetime(global_temp['dt'])
global_temp['Year'] = global_temp['dt'].dt.year

# Country-level annual mean temperature
country_annual_temp = (
    global_temp.groupby(['Country', 'Year'])['AverageTemperature']
    .mean()
    .reset_index()
)

# 3. PREPROCESS CO₂ DATA (ANNUAL MEAN)
co2_annual = (
    co2.groupby('year')['smoothed_monthly']
    .mean()
    .reset_index()
    .rename(columns={'year': 'Year', 'smoothed_monthly': 'CO2'})
)

# 4. MERGE ALL DATASETS
merged = country_annual_temp.merge(co2_annual, on='Year', how='left')

# Merge with final annual temperature data if Year exists
if 'Year' in annual_temp.columns:
    merged = merged.merge(
        annual_temp,
        on='Year',
        how='left',
        suffixes=('_Country', '_Global')
    )

# 5. SELECT NUMERIC FEATURES
numeric_df = merged.select_dtypes(include=['float64', 'int64'])

print("Numeric features included in correlation analysis:")
print(numeric_df.columns)

# 6. COMPUTE CORRELATION MATRIX
correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# 7. VISUALIZE CORRELATION HEATMAP
plt.figure(figsize=(11, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)
plt.title("Correlation Analysis: Temperature, CO₂, and Time")
plt.show()