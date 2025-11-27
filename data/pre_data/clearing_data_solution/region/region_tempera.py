import pandas as pd

# 1. 读取CSV文件
df = pd.read_csv("../../csv_file/region/GlobalLandTemperaturesByCountry.csv")
country = ""
target_country = country
temperature_col = "AverageTemperature"

# 2. 筛选指定国家 + 去除温度列的缺失值
df_filtered = df[df["Country"] == target_country].dropna(subset=[temperature_col])

# 3. 处理日期：提取年份
df_filtered["dt"] = pd.to_datetime(df_filtered["dt"])
df_filtered["year"] = df_filtered["dt"].dt.year

# 4. 按年份分组，计算统计量，并保留两位小数
yearly_stats = df_filtered.groupby("year")[temperature_col].agg(
    mean="mean",
    median="median",
    standard_deviation="std"
).reset_index()

# 5. 保留两位小数
yearly_stats = yearly_stats.round(2)

# 输出结果
print("指定国家的年度温度统计：")
print(yearly_stats)

# 保存为新CSV
yearly_stats.to_csv(f"../../csv_file/region/{country}_tempera.csv", index=False)