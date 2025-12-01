import pandas as pd

# 1. Read CSV file (comma-separated by default, no need to specify sep)
input_file = r"../../csv_file/globe/original_co2.csv"  # Replace with your actual file path
df = pd.read_csv(input_file, encoding="utf-8")  # Replace with "gbk" if an error occurs

# 2. Group by "year+month" and calculate monthly averages of smoothed and trend (keep 2 decimal places)
monthly_df = df.groupby(
    by=["year", "month"],  # Match column names in CSV header
    as_index=False
).agg(
    smoothed_monthly=("smoothed", lambda x: round(x.mean(), 2)),
    trend_monthly=("trend", lambda x: round(x.mean(), 2))
)

# 3. Save processed monthly data to new CSV
output_file = "../../csv_file/globe/final_co2_monthly.csv"
monthly_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Monthly average data saved to: {output_file}")
print("Preview of processed data:")
print(monthly_df.head())