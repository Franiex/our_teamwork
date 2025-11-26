import pandas as pd
import numpy as np

# 1. 读取NASA GISS月度温度异常数据（包含-.xx格式数据，pandas会自动识别为负数）
url = r"data/pre_data/globe/original_tempera_difference.csv"
data = pd.read_csv(url, skiprows=1)  # 跳过标题行

# 2. 定义月度列名（J到D对应1-12月）
month_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 3. 数据预处理：处理可能的缺失值，转换为数值类型
data[month_columns] = data[month_columns].replace('***', np.nan)  # 替换缺失值标记
data[month_columns] = data[month_columns].astype(float)  # 转换为浮点数

# 4. 计算每月的绝对温度（异常值 + 13.9）
month_abs_columns = [f'{month}_Abs' for month in month_columns]  # 定义月度绝对温度列名
data[month_abs_columns] = data[month_columns] + 13.9  # 生成每月绝对温度

# 5. 计算年度绝对温度统计量
data['Annual_Mean_Absolute'] = data[month_abs_columns].mean(axis=1)    # 年度绝对温度均值
data['Annual_Median_Absolute'] = data[month_abs_columns].median(axis=1)# 年度绝对温度中位数
data['Annual_Std_Absolute'] = data[month_abs_columns].std(axis=1)      # 年度绝对温度标准差

# 6. 筛选关键列（仅保留年份和绝对温度统计量）并统一保留两位小数
result = data[['Year', 'Annual_Mean_Absolute', 'Annual_Median_Absolute', 'Annual_Std_Absolute']].copy()
result = result.round(2)  # 所有数值列统一保留两位小数

# 7. 保存结果到CSV文件
result.to_csv('annual_temperature_data.csv', index=False, float_format='%.2f')

# 8. 显示前10行结果
print("处理后的年度绝对温度数据（仅含绝对温度统计量）：")
print(result.head(10))