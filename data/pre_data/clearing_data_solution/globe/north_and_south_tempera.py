import pandas as pd
import numpy as np

# ---------------------- 1. 读取NASA GISS南北半球月度温度数据（公开链接） ----------------------
nh_url = "https://data.giss.nasa.gov/gistemp/tabledata_v3/NH.Ts+dSST.csv"  # 北半球
sh_url = "https://data.giss.nasa.gov/gistemp/tabledata_v3/SH.Ts+dSST.csv"  # 南半球

# 读取数据（跳过说明行）
nh_data = pd.read_csv(nh_url, skiprows=1)
sh_data = pd.read_csv(sh_url, skiprows=1)

# ---------------------- 2. 数据预处理（统一逻辑） ----------------------
month_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def process_hemisphere_data(data):
    """处理单个半球的数据：清洗、计算绝对温度、年度统计量"""
    data = data[['Year'] + month_columns].copy()
    data[month_columns] = data[month_columns].replace('***', np.nan).astype(float)
    # 计算每月绝对温度
    month_abs_cols = [f'{m}_Abs' for m in month_columns]
    data[month_abs_cols] = data[month_columns] + 13.9
    # 计算年度统计量
    data['Annual_Mean_Abs'] = data[month_abs_cols].mean(axis=1).round(2)
    data['Annual_Median_Abs'] = data[month_abs_cols].median(axis=1).round(2)
    data['Annual_Std_Abs'] = data[month_abs_cols].std(axis=1).round(2)
    return data[['Year', 'Annual_Mean_Abs', 'Annual_Median_Abs', 'Annual_Std_Abs']]

# 处理南北半球数据
nh_processed = process_hemisphere_data(nh_data)
sh_processed = process_hemisphere_data(sh_data)

# ---------------------- 3. 合并为对比数据（固定南北顺序） ----------------------
# 添加半球标识
nh_processed['Hemisphere'] = 'Northern'
sh_processed['Hemisphere'] = 'Southern'

# 合并数据：先按年份排序，再按半球固定顺序（Northern在前，Southern在后）
comparison_data = pd.concat([nh_processed, sh_processed], axis=0)
# 将Hemisphere设为分类类型并指定顺序，确保排序时Northern优先
comparison_data['Hemisphere'] = pd.Categorical(comparison_data['Hemisphere'],
                                               categories=['Northern', 'Southern'],
                                               ordered=True)
# 先按Year排序，再按Hemisphere排序（保证同一年份先北后南）
comparison_data = comparison_data.sort_values(['Year', 'Hemisphere']).reset_index(drop=True)

# ---------------------- 4. 保存为对比CSV ----------------------
comparison_data.to_csv('south_and_north_ hemisphere_comparison.csv', index=False, float_format='%.2f')

# ---------------------- 5. 显示示例结果 ----------------------
print("南北半球温度年度变化对比（按年份+半球顺序排列，前10行）：")
print(comparison_data.head(10))