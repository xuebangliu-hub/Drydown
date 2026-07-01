import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FuncFormatter
from matplotlib import cm

# -------------------- Font settings ---------------------
font_size = 14  # General font size
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# ================= Data loading =================
lat = np.linspace(90, -60, 150)  # Latitude array

base_path1 = '.../pr_variablity/integrate_trend/warm_season/daily_pr_variablity/obs_1980_2023/'
pr_con = np.load(base_path1 + 'daily_pr_variablity_trend_mean_1980_2023.npy')[:150, :] * 1000

base_path = '.../drydown_speed/trend_all/warm_season/duration_10_50/obs_1980_2023/'
drydown_speed = np.load(base_path + 'decline_rate_trend_mean_1980_2023.npy')[:150, :] * 100000

region = np.load('.../koppen_geiger/koppen_geiger_climatezones_1991_2020_100.npy')[:150, :]
bounds = [1, 4, 8, 17, 29]   # Four climate zones

# Colors for the four climate zones
tab20b = cm.get_cmap("tab20b")
tab20c = cm.get_cmap("tab20c")

colors_all = [
    tab20c.colors[8],   # Tropical
    tab20b.colors[11],  # Arid
    tab20c.colors[13],  # Temperate
    tab20b.colors[14]   # Cold
]

# Data to be plotted
data_list = [pr_con, drydown_speed]

sub_labels = ['C', 'F']  # Subplot labels

# ================= Plotting ====================

# Prepare figure
fig, axes = plt.subplots(2, 1, figsize=(3, 9))

# Increase spacing between the two subplots
plt.subplots_adjust(hspace=0.4)

for idx, (data, ax) in enumerate(zip(data_list, axes)):

    violin_data = []
    for i in range(4):
        mask = (region >= bounds[i]) & (region < bounds[i + 1])
        selected = data[mask]
        selected = selected[np.isfinite(selected)]
        violin_data.append(selected)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    if idx == 0:
        ax.set_ylim(-30, 30)  # y-axis range for precipitation concentration
    else:
        ax.set_ylim(-6, 6)    # y-axis range for drydown speed

    # Violin plot
    v = ax.violinplot(
        violin_data,
        showmeans=False,
        showextrema=False,
        showmedians=False
    )

    # Set violin colors
    for i, body in enumerate(v["bodies"]):
        body.set_facecolor(colors_all[i])
        body.set_edgecolor("black")
        body.set_alpha(0.9)

    # Overlay boxplot
    bp = ax.boxplot(
        violin_data,
        widths=0.15,
        patch_artist=True,
        showcaps=False,
        whis=1.5,
        showfliers=False,
        whiskerprops=dict(color="black", linewidth=1),
        medianprops=dict(color="black", linewidth=1),
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1)
    )

    # Keep only vertical whisker lines
    for whisker in bp["whiskers"]:
        whisker.set_linestyle("-")

    # X-axis labels
    ax.set_xticks(np.arange(1, 5))
    ax.set_xticklabels(
        ["Tropical", "Arid", "Temperate", "Cold"],
        rotation=45,
        ha="right",
        fontsize=font_size
    )

    if idx == 0:
        ax.set_ylabel(
            'Trend of Pr$_{Concentration}$\n($10^{-3}$z-score yr$^{-1}$)',
            fontsize=14
        )
    else:
        ax.set_ylabel(
            'Trend of Drydown$_{Speed}$\n($10^{-5}$m$^{3}$m$^{-3}$day$^{-1}$yr$^{-1}$)',
            fontsize=14
        )

    ax.tick_params(axis="y", labelsize=font_size)

    # Subplot label
    ax.text(
        -0.45,
        1.05,
        sub_labels[idx],
        transform=ax.transAxes,
        fontsize=15,
        fontweight='bold',
        va='top',
        ha='left'
    )

save_path = '.../Pr_SM_spatial_trend/'
plt.savefig(
    save_path + "Pr_con_Drydown_speed_koppen_region_1980_2023.jpg",
    dpi=500,
    bbox_inches='tight'
)