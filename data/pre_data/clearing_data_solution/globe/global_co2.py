import pandas as pd

# 1. 读取CSV文件（默认逗号分隔，无需额外指定sep）
input_file = "../../csv_file/globe/original_co2.csv"  # 替换为你的实际文件路径
df = pd.read_csv(input_file, encoding="utf-8")  # 若报错可替换为"gbk"

# 2. 按「年+月」分组，计算smoothed和trend的月度均值（保留2位小数）
monthly_df = df.groupby(
    by=["year", "month"],  # 匹配CSV表头的列名
    as_index=False
).agg(
    smoothed_monthly=("smoothed", lambda x: round(x.mean(), 2)),
    trend_monthly=("trend", lambda x: round(x.mean(), 2))
)

# 3. 保存处理后的月度数据到新CSV
output_file = "final_co2_monthly.csv"
monthly_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"月度均值数据已保存至：{output_file}")
print("处理后的数据预览：")
print(monthly_df.head())