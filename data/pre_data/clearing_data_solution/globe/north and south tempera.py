import pandas as pd
import numpy as np

# ---------------------- 1. 读取NASA GISS南北半球月度温度数据（公开链接） ----------------------
# GISS南北半球月度温度异常数据公开链接（若链接失效，可去https://data.giss.nasa.gov/gistemp/tabledata_v3/更新）
nh_url = "https://data.giss.nasa.gov/gistemp/tabledata_v3/NH.Ts+dSST.csv"  # 北半球
sh_url = "https://data.giss.nasa.gov/gistemp/tabledata_v3/SH.Ts+dSST.csv"  # 南半球

# 读取数据（跳过说明行，GISS数据前几行是注释，实际数据从第1行开始）
nh_data = pd.read_csv(nh_url, skiprows=1)
sh_data = pd.read_csv(sh_url, skiprows=1)

# ---------------------- 2. 数据预处理（统一逻辑） ----------------------
month_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def process_hemisphere_data(data):
    """处理单个半球的数据：清洗、计算绝对温度、年度统计量"""
    # 保留年份和月度列
    data = data[['Year'] + month_columns].copy()
    # 处理缺失值（GISS用'***'表示缺失）
    data[month_columns] = data[month_columns].replace('***', np.nan).astype(float)
    # 计算每月绝对温度（异常值 + 13.9，1951-1980基准期平均温度）
    month_abs_cols = [f'{m}_Abs' for m in month_columns]
    data[month_abs_cols] = data[month_columns] + 13.9
    # 计算年度绝对温度统计量
    data['Annual_Mean_Abs'] = data[month_abs_cols].mean(axis=1).round(2)
    data['Annual_Median_Abs'] = data[month_abs_cols].median(axis=1).round(2)
    data['Annual_Std_Abs'] = data[month_abs_cols].std(axis=1).round(2)
    # 保留关键列
    return data[['Year', 'Annual_Mean_Abs', 'Annual_Median_Abs', 'Annual_Std_Abs']]

# 处理南北半球数据
nh_processed = process_hemisphere_data(nh_data)
sh_processed = process_hemisphere_data(sh_data)

# ---------------------- 3. 合并为对比数据 ----------------------
# 添加半球标识
nh_processed['Hemisphere'] = 'Northern'
sh_processed['Hemisphere'] = 'Southern'

# 合并数据（按年份对齐）
comparison_data = pd.concat([nh_processed, sh_processed], axis=0).sort_values('Year')

# ---------------------- 4. 保存为对比CSV ----------------------
comparison_data.to_csv('south_and_north_ hemisphere_comparison.csv', index=False, float_format='%.2f')

# ---------------------- 5. 显示示例结果 ----------------------
print("南北半球温度年度变化对比（前10行）：")
print(comparison_data.head(10))