import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
annual_temp = pd.read_csv("../csv_files/annual_temperature_data.csv")
co2 = pd.read_csv("../csv_files/final_co2_monthly.csv")
global_temp = pd.read_csv("../csv_files/GlobalLandTemperaturesByCountry.csv")

# 2. Preprocess global temperature data
global_temp = global_temp[['dt', 'AverageTemperature', 'Country']].dropna()
global_temp['dt'] = pd.to_datetime(global_temp['dt'])
global_temp['Year'] = global_temp['dt'].dt.year

# Country-level annual mean temperature
country_annual_temp = (
    global_temp.groupby(['Country', 'Year'])['AverageTemperature']
    .mean()
    .reset_index()
)

# 3. Add rolling mean (time feature)
country_annual_temp = country_annual_temp.sort_values('Year')
country_annual_temp['Temp_5yr_RollingMean'] = (
    country_annual_temp
    .groupby('Country')['AverageTemperature']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
)

# 4. Preprocess CO2 data (annual)
co2_annual = (
    co2.groupby('year')['smoothed_monthly']
    .mean()
    .reset_index()
    .rename(columns={'year': 'Year', 'smoothed_monthly': 'CO2'})
)

# 5. Merge datasets
merged = country_annual_temp.merge(co2_annual, on='Year', how='left')

# Merge with final annual temperature data if Year exists
if 'Year' in annual_temp.columns:
    merged = merged.merge(
        annual_temp,
        on='Year',
        how='left',
        suffixes=('_Country', '_Global')
    )

# 6. Select numeric features (now includes Year + rolling mean)
numeric_df = merged.select_dtypes(include=['float64', 'int64'])

print("Numeric features included in correlation analysis:")
print(numeric_df.columns)

# 7. Compute correlation
correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# 8. Visualization
plt.figure(figsize=(12, 9))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)
plt.title("Correlation Analysis: Temperature, CO₂, and Time Features")
plt.show()