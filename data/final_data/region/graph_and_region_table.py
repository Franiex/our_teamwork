import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

print("Current working directory:", os.getcwd())
print("\nFinding CSV files...")
print("="*80)

possible_paths = [
    '../pre_data/csv_file/region/',
    '../../pre_data/csv_file/region/',
    './pre_data/csv_file/region/',
    'pre_data/csv_file/region/',
]

base_path = None
for path in possible_paths:
    test_file = path + 'Malaysia_tempera.csv'
    if os.path.exists(test_file):
        print(f"✓ FOUND! Path: '{path}'")
        base_path = path
        break

if base_path is None:
    print("⚠ Could not find the CSV files.")
    base_path = input("\nEnter the correct path to the CSV files: ")

csv_files = glob.glob(base_path + '*_tempera.csv')
print(f"\nFound {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")
print("="*80)

countries = {}
for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    country_name = filename.replace('_tempera.csv', '').replace('-', ' ')
    countries[country_name] = csv_file

print(f"\nLoading {len(countries)} countries...")

data_dict = {}
for country, filename in countries.items():
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    data_dict[country] = df
    print(f"  ✓ Loaded {country}")

output_dir = 'region_graph_table'
graph_dir = os.path.join(output_dir, 'graph')
table_dir = os.path.join(output_dir, 'table')

os.makedirs(graph_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)
print(f"\n✓ Created output directories:")
print(f"  - {output_dir}/")
print(f"  - {graph_dir}/")
print(f"  - {table_dir}/")
print("="*80)

plt.figure(figsize=(16, 10))

for country, df in data_dict.items():
    plt.plot(df['year'], df['mean'], label=country, linewidth=1.5, alpha=0.8)

plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Mean Temperature (°C)', fontsize=14, fontweight='bold')
plt.title('Temperature Trends Comparison Across 15 Countries (1743-2013)',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=10, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
graph_file = os.path.join(graph_dir, 'temperature_comparison.png')
plt.savefig(graph_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Chart saved: {graph_file}")
print("="*80)

growth_analysis = []

for country, df in data_dict.items():
    first_30 = df.head(30)['mean'].mean()
    last_30 = df.tail(30)['mean'].mean()
    growth = last_30 - first_30
    growth_pct = (growth / abs(first_30)) * 100
    overall_mean = df['mean'].mean()
    start_year = df['year'].min()
    end_year = df['year'].max()

    growth_analysis.append({
        'Country': country,
        'First_30_Years_Avg': round(first_30, 2),
        'Last_30_Years_Avg': round(last_30, 2),
        'Temperature_Growth_°C': round(growth, 2),
        'Growth_Percentage': round(growth_pct, 2),
        'Overall_Mean': round(overall_mean, 2),
        'Start_Year': start_year,
        'End_Year': end_year
    })

growth_df = pd.DataFrame(growth_analysis)
growth_df = growth_df.sort_values('Temperature_Growth_°C', ascending=False)

print("\nTEMPERATURE GROWTH ANALYSIS (SORTED BY ABSOLUTE GROWTH):")
print("="*80)
print(growth_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(18, 10))
ax.axis('tight')
ax.axis('off')

table_data = []
table_data.append(['Rank', 'Country', 'First 30 Yrs\nAvg (°C)', 'Last 30 Yrs\nAvg (°C)',
                   'Growth\n(°C)', 'Growth\n(%)', 'Overall\nMean (°C)', 'Data Period'])

for idx, (i, row) in enumerate(growth_df.iterrows(), 1):
    table_data.append([
        str(idx),
        row['Country'],
        f"{row['First_30_Years_Avg']:.2f}",
        f"{row['Last_30_Years_Avg']:.2f}",
        f"{row['Temperature_Growth_°C']:.2f}",
        f"{row['Growth_Percentage']:.2f}",
        f"{row['Overall_Mean']:.2f}",
        f"{row['Start_Year']}-{row['End_Year']}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.06, 0.12, 0.12, 0.12, 0.10, 0.10, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(len(table_data[0])):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#FFFFFF')

        if i <= 5:
            if j == 0:
                cell.set_facecolor('#FFD966')
                cell.set_text_props(weight='bold')

plt.title('Temperature Growth Analysis - All 15 Countries',
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
table_file = os.path.join(table_dir, 'temperature_growth_table.png')
plt.savefig(table_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Table saved: {table_file}")

print("\n" + "="*80)
print("TOP 5 COUNTRIES WITH HIGHEST TEMPERATURE GROWTH:")
print("="*80)

top5 = growth_df.head(5)

for idx, (i, row) in enumerate(top5.iterrows(), 1):
    print(f"\n{idx}. {row['Country'].upper()}")
    print(f"   Temperature Growth: +{row['Temperature_Growth_°C']}°C")
    print(f"   From: {row['First_30_Years_Avg']}°C (first 30 years)")
    print(f"   To: {row['Last_30_Years_Avg']}°C (last 30 years)")
    print(f"   Percentage Change: {row['Growth_Percentage']:.2f}%")
    print(f"   Data Period: {row['Start_Year']}-{row['End_Year']}")

csv_file = os.path.join(output_dir, 'temperature_growth_analysis.csv')
growth_df.to_csv(csv_file, index=False)

print("\n" + "="*80)
print("✓ ALL FILES SAVED SUCCESSFULLY!")
print("="*80)
print(f"\nOutput directory: {output_dir}/")
print(f"  ├── graph/")
print(f"  │   └── temperature_comparison.png")
print(f"  ├── table/")
print(f"  │   └── temperature_growth_table.png")
print(f"  └── temperature_growth_analysis.csv")