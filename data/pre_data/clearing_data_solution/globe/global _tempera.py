import pandas as pd
import numpy as np

# 1. Read NASA GISS monthly temperature anomaly data (data with -.xx format will be automatically recognized as negative numbers)
url = r"../../csv_file/globe/original_tempera_difference.csv"
data = pd.read_csv(url, skiprows=1)  # Skip header row

# 2. Define monthly column names (J to D correspond to January-December)
month_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 3. Data preprocessing: Handle possible missing values and convert to numeric type
data[month_columns] = data[month_columns].replace('***', np.nan)  # Replace missing value markers
data[month_columns] = data[month_columns].astype(float)  # Convert to float

# 4. Calculate monthly absolute temperature (anomaly value + 13.9)
month_abs_columns = [f'{month}_Abs' for month in month_columns]  # Define monthly absolute temperature column names
data[month_abs_columns] = data[month_columns] + 13.9  # Generate monthly absolute temperatures

# 5. Calculate annual absolute temperature statistics
data['Annual_Mean_Absolute'] = data[month_abs_columns].mean(axis=1)    # Annual absolute temperature mean
data['Annual_Median_Absolute'] = data[month_abs_columns].median(axis=1)# Annual absolute temperature median
data['Annual_Std_Absolute'] = data[month_abs_columns].std(axis=1)      # Annual absolute temperature standard deviation

# 6. Filter key columns (only keep year and absolute temperature statistics) and retain two decimal places uniformly
result = data[['Year', 'Annual_Mean_Absolute', 'Annual_Median_Absolute', 'Annual_Std_Absolute']].copy()
result = result.round(2)  # Keep two decimal places for all numeric columns

# 7. Save results to CSV file
result.to_csv('../../csv_file/globe/annual_temperature_data.csv', index=False, float_format='%.2f')

# 8. Display first 10 rows of results
print("Processed annual absolute temperature data (absolute temperature statistics only):")
print(result.head(10))