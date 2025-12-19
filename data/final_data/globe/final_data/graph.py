import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = 'yearly_globe_data'
graphs_dir = os.path.join(base_dir, 'graphs')
tables_dir = os.path.join(base_dir, 'tables')

os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

df = pd.read_csv('final_annual_temperature_data.csv')

# ===== GENERATE TABLES =====
print("=== GENERATING TABLES ===")

# Decade summary
df['Decade'] = (df['Year'] // 10) * 10
decade_summary = df.groupby('Decade').agg({
    'Annual_Mean_Absolute': 'mean',
    'Annual_Median_Absolute': 'mean',
    'Annual_Std_Absolute': 'mean'
}).round(2)
decade_summary.to_csv(os.path.join(tables_dir, 'decade_summary.csv'))

# Warmest years
warmest = df.nlargest(10, 'Annual_Mean_Absolute')[['Year', 'Annual_Mean_Absolute', 'Annual_Median_Absolute']]
warmest.to_csv(os.path.join(tables_dir, 'warmest_years.csv'), index=False)

# Coldest years
coldest = df.nsmallest(10, 'Annual_Mean_Absolute')[['Year', 'Annual_Mean_Absolute', 'Annual_Median_Absolute']]
coldest.to_csv(os.path.join(tables_dir, 'coldest_years.csv'), index=False)

# Recent trends
recent = df.tail(20)[['Year', 'Annual_Mean_Absolute', 'Annual_Median_Absolute', 'Annual_Std_Absolute']]
recent.to_csv(os.path.join(tables_dir, 'recent_trends.csv'), index=False)

# Century summary
century_stats = df.copy()
century_stats['Century'] = century_stats['Year'].apply(lambda x: '19th' if x < 1900 else ('20th' if x < 2000 else '21st'))
century_summary = century_stats.groupby('Century').agg({
    'Annual_Mean_Absolute': ['min', 'max', 'mean'],
    'Annual_Median_Absolute': ['min', 'max', 'mean']
}).round(2)
century_summary.to_csv(os.path.join(tables_dir, 'century_summary.csv'))

# Statistics overview
stats_overview = pd.DataFrame({
    'Metric': ['Total Years', 'Earliest Year', 'Latest Year', 'Coldest Temperature (°C)',
               'Warmest Temperature (°C)', 'Average Temperature (°C)', 'Temperature Range (°C)'],
    'Value': [len(df), df['Year'].min(), df['Year'].max(),
              df['Annual_Mean_Absolute'].min(), df['Annual_Mean_Absolute'].max(),
              round(df['Annual_Mean_Absolute'].mean(), 2),
              round(df['Annual_Mean_Absolute'].max() - df['Annual_Mean_Absolute'].min(), 2)]
})
stats_overview.to_csv(os.path.join(tables_dir, 'statistics_overview.csv'), index=False)

# ===== GENERATE GRAPHS =====
print("\n=== GENERATING GRAPHS ===")

# Graph 1: 4-panel analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].plot(df['Year'], df['Annual_Mean_Absolute'], label='Mean', linewidth=2, color='#e74c3c')
axes[0, 0].plot(df['Year'], df['Annual_Median_Absolute'], label='Median', linewidth=2, linestyle='--', color='#3498db')
axes[0, 0].set_xlabel('Year', fontsize=11)
axes[0, 0].set_ylabel('Temperature (°C)', fontsize=11)
axes[0, 0].set_title('Annual Temperature Trends (1880-2025)', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].bar(decade_summary.index, decade_summary['Annual_Mean_Absolute'],
               color='coral', edgecolor='black', width=8)
axes[0, 1].set_xlabel('Decade', fontsize=11)
axes[0, 1].set_ylabel('Average Temperature (°C)', fontsize=11)
axes[0, 1].set_title('Average Temperature by Decade', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].tick_params(axis='x', rotation=45)

axes[1, 0].plot(df['Year'], df['Annual_Mean_Absolute'], color='blue', linewidth=2)
axes[1, 0].fill_between(df['Year'],
                         df['Annual_Mean_Absolute'] - df['Annual_Std_Absolute'],
                         df['Annual_Mean_Absolute'] + df['Annual_Std_Absolute'],
                         alpha=0.3, color='lightblue')
axes[1, 0].set_xlabel('Year', fontsize=11)
axes[1, 0].set_ylabel('Temperature (°C)', fontsize=11)
axes[1, 0].set_title('Temperature Mean with Standard Deviation', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

recent_df = df[df['Year'] >= 1975]
axes[1, 1].scatter(recent_df['Year'], recent_df['Annual_Mean_Absolute'],
                   s=50, alpha=0.6, color='red', edgecolor='darkred')
z = np.polyfit(recent_df['Year'], recent_df['Annual_Mean_Absolute'], 1)
p = np.poly1d(z)
axes[1, 1].plot(recent_df['Year'], p(recent_df['Year']),
                "b--", linewidth=2, label='Trend line')
axes[1, 1].set_xlabel('Year', fontsize=11)
axes[1, 1].set_ylabel('Temperature (°C)', fontsize=11)
axes[1, 1].set_title('Recent Temperature Trend (1975-2025)', fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'temperature_analysis_4panel.png'), dpi=300, bbox_inches='tight')
plt.close()


fig2, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['Year'], df['Annual_Mean_Absolute'], linewidth=2.5, color='#e74c3c', label='Annual Mean Temperature')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.set_title('Global Temperature Trend (1880-2025)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'temperature_timeline.png'), dpi=300, bbox_inches='tight')
plt.close()

fig3, ax3 = plt.subplots(figsize=(12, 7))
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(decade_summary)))
bars = ax3.bar(decade_summary.index, decade_summary['Annual_Mean_Absolute'],
               color=colors, edgecolor='black', width=8)
ax3.set_xlabel('Decade', fontsize=12)
ax3.set_ylabel('Average Temperature (°C)', fontsize=12)
ax3.set_title('Decade-wise Temperature Evolution', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.tick_params(axis='x', rotation=45)
for i, (decade, temp) in enumerate(zip(decade_summary.index, decade_summary['Annual_Mean_Absolute'])):
    ax3.text(decade, temp + 0.05, f'{temp:.2f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'decade_evolution.png'), dpi=300, bbox_inches='tight')
plt.close()

print("=== FILES ORGANIZED SUCCESSFULLY ===")
