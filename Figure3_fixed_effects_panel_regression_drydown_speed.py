import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# ==================== 1. Load results ====================
csv_path = ".../warm_season/LFE_drivers/duration_10_50_panel_coefficients_cross_source_summary.csv"
df = pd.read_csv(csv_path)

regions = df['region'].values
region_names = ['Tropical', 'Arid', 'Temperate', 'Cold']
variables = ['pulse_sm', 'rs', 'E', 'tas']

# ==================== 2. Plot settings ====================
bar_width = 0.12
spacing = 0.04
n_vars = len(variables)

total_width = n_vars * bar_width + (n_vars - 1) * spacing
start_offset = -total_width / 2
offsets = [start_offset + i * (bar_width + spacing) for i in range(n_vars)]

colors = {
    'pulse_sm': '#2c7bb6',
    'tas': '#d7191c',
    'rs': '#fdae61',
    'E': '#abd9e9'
}
labels = {
    'pulse_sm': r'Pulse$_{SM}$',
    'tas': r'Drydown$_{Tas}$',
    'rs': r'Drydown$_{SW}$',
    'E': r'Drydown$_{ET}$'
}

fig, ax = plt.subplots(figsize=(13, 4))
x = np.arange(len(regions))

first_x = []
last_x = []

for i, var in enumerate(variables):
    means = df[f'{var}_mean'].values.copy().astype(float)
    stds = df[f'{var}_std'].values.copy().astype(float)

    means[1] *= -1

    offset = offsets[i]
    rects = ax.bar(
        x + offset,
        means,
        bar_width,
        yerr=stds,
        label=labels[var],
        color=colors[var],
        capsize=3,
        error_kw={'elinewidth': 1, 'capthick': 1}
    )

    if i == 0:
        first_x = [x[j] + offset for j in range(len(x))]

    if i == n_vars - 1:
        last_x = [x[j] + offset for j in range(len(x))]

group_centers = [(first_x[j] + last_x[j]) / 2 for j in range(len(x))]

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

ax.set_xticks(group_centers)
ax.set_xticklabels(region_names, fontsize=14)

# ax.set_xlabel('Climate region', fontsize=14, labelpad=10)
ax.set_ylabel('Standardized coefficient', fontsize=14, labelpad=10)
ax.tick_params(axis='y', labelsize=14)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, pos: '0' if abs(x) < 1e-10 else f'{x:g}')
)
ax.legend(loc='lower right', frameon=False, ncol=2, fontsize=14)

ax.text(
    -0.075,
    1.02,
    'E',
    transform=ax.transAxes,
    fontsize=15,
    fontweight='bold',
    va='top',
    ha='left'
)

ax.set_ylim(-0.6, 0.5)

save_path = '.../drivers/'
plt.savefig(
    save_path + "pr_variability_drivers_LFE_1980_2023.jpg",
    dpi=500,
    bbox_inches='tight'
)