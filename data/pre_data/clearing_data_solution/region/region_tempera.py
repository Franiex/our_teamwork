import pandas as pd

# 1. Read CSV file
df = pd.read_csv("../../csv_file/region/GlobalLandTemperaturesByCountry.csv")
country = ""
target_country = country
temperature_col = "AverageTemperature"

# 2. Filter specified country + Remove missing values in temperature column
df_filtered = df[df["Country"] == target_country].dropna(subset=[temperature_col])

# 3. Process date: Extract year
df_filtered["dt"] = pd.to_datetime(df_filtered["dt"])
df_filtered["year"] = df_filtered["dt"].dt.year

# 4. Group by year, calculate statistics, and keep two decimal places
yearly_stats = df_filtered.groupby("year")[temperature_col].agg(
    mean="mean",
    median="median",
    standard_deviation="std"
).reset_index()

# 5. Keep two decimal places
yearly_stats = yearly_stats.round(2)

# Output results
print("Annual temperature statistics for the specified country:")
print(yearly_stats)

# Save as new CSV
yearly_stats.to_csv(f"../../csv_file/region/{country}_tempera.csv", index=False)