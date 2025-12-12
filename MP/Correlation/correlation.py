import pandas as pd
import matplotlib.pyplot as plt

# For heatmap
import seaborn as sns

# 1. LOAD DATA
temp = pd.read_csv("final_annual_temperature_data.csv")
co2 = pd.read_csv("final_co2_monthly.csv")

# Expected columns:
# temp: country | year | average_temperature | (maybe rolling_mean if you added it)
# co2 : year | month | smoothed_monthly | trend_monthly

# 2. AGGREGATE CO₂ TO ANNUAL AVERAGE
co2_annual = (
    co2.groupby("year")["smoothed_monthly"]
       .mean()
       .reset_index()
       .rename(columns={"smoothed_monthly": "co2"})
)

# 3. MERGE DATASETS
df = temp.merge(co2_annual, on="year", how="left")

# 4. SELECT NUMERIC FEATURES
numeric_df = df.select_dtypes(include=['float64', 'int64'])

print("Numeric features used in correlation analysis:")
print(numeric_df.columns)

# 5. COMPUTE CORRELATION MATRIX
corr = numeric_df.corr()

print("\nCorrelation Matrix:")
print(corr)

# 6. VISUALIZE CORRELATION HEATMAP
plt.figure(figsize=(10, 7))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)
plt.title("Correlation Matrix of Climate Features")
plt.show()