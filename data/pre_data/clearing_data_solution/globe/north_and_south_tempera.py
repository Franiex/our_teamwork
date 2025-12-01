import pandas as pd
import numpy as np

# ---------------------- 1. Read NASA GISS monthly temperature data for Northern and Southern Hemispheres (public link) ----------------------
nh_url = "https://data.giss.nasa.gov/gistemp/tabledata_v3/NH.Ts+dSST.csv"  # Northern Hemisphere
sh_url = "https://data.giss.nasa.gov/gistemp/tabledata_v3/SH.Ts+dSST.csv"  # Southern Hemisphere

# Read data (skip description rows)
nh_data = pd.read_csv(nh_url, skiprows=1)
sh_data = pd.read_csv(sh_url, skiprows=1)

# ---------------------- 2. Data Preprocessing (unified logic) ----------------------
month_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def process_hemisphere_data(data):
    """Process data for a single hemisphere: cleaning, calculate absolute temperature, annual statistics"""
    data = data[['Year'] + month_columns].copy()
    data[month_columns] = data[month_columns].replace('***', np.nan).astype(float)
    # Calculate monthly absolute temperature
    month_abs_cols = [f'{m}_Abs' for m in month_columns]
    data[month_abs_cols] = data[month_columns] + 13.9
    # Calculate annual statistics
    data['Annual_Mean_Abs'] = data[month_abs_cols].mean(axis=1).round(2)
    data['Annual_Median_Abs'] = data[month_abs_cols].median(axis=1).round(2)
    data['Annual_Std_Abs'] = data[month_abs_cols].std(axis=1).round(2)
    return data[['Year', 'Annual_Mean_Abs', 'Annual_Median_Abs', 'Annual_Std_Abs']]

# Process Northern and Southern Hemisphere data
nh_processed = process_hemisphere_data(nh_data)
sh_processed = process_hemisphere_data(sh_data)

# ---------------------- 3. Merge into comparison data (fixed order: Northern first, then Southern) ----------------------
# Add hemisphere identifier
nh_processed['Hemisphere'] = 'Northern'
sh_processed['Hemisphere'] = 'Southern'

# Merge data: sort by year first, then by hemisphere in fixed order (Northern first, Southern second)
comparison_data = pd.concat([nh_processed, sh_processed], axis=0)
# Set Hemisphere as categorical type with specified order to ensure Northern comes first when sorting
comparison_data['Hemisphere'] = pd.Categorical(comparison_data['Hemisphere'],
                                               categories=['Northern', 'Southern'],
                                               ordered=True)
# Sort by Year first, then by Hemisphere (ensuring Northern comes before Southern for the same year)
comparison_data = comparison_data.sort_values(['Year', 'Hemisphere']).reset_index(drop=True)

# ---------------------- 4. Save as comparison CSV ----------------------
comparison_data.to_csv('south_and_north_ hemisphere_comparison.csv', index=False, float_format='%.2f')

# ---------------------- 5. Display sample results ----------------------
print("Annual temperature change comparison between Northern and Southern Hemispheres (sorted by year + hemisphere, first 10 rows):")
print(comparison_data.head(10))