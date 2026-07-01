import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FuncFormatter

# -------------------- Font settings ---------------------
font_size = 14  # General font size
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# ================= Data loading =================
lat = np.linspace(90, -60, 150)  # Latitude array

base_path = '.../drydown_speed/'

base_path1 = '.../pr_variablity/integrate_trend/warm_season/daily_pr_variablity/obs_1980_2023/'
trend = np.load(base_path1 + 'daily_pr_variablity_trend_all_1980_2023.npy')[:, :150, :] * 1000

base_path = '.../drydown_speed/trend_all/warm_season/duration_10_50/obs_1980_2023/'
corr = np.load(base_path + 'decline_rate_trend_all_1980_2023.npy')[:, :150, :] * 100000

# ================= Calculate zonal statistics =================
def zonal_stats(data):
    """
    Input: data shape = (Nmodel, lat, lon)
    Return:
        mean_lat shape = (lat,)
        p05_lat  shape = (lat,)
        p95_lat  shape = (lat,)
    """
    # Calculate zonal mean along longitude, shape becomes (Nmodel, lat)
    zonal_mean = np.nanmean(data, axis=2)

    # Calculate mean and percentiles across datasets
    mean_lat = np.nanmean(zonal_mean, axis=0)
    p05_lat = np.nanpercentile(zonal_mean, 5, axis=0)
    p95_lat = np.nanpercentile(zonal_mean, 95, axis=0)

    return mean_lat, p05_lat, p95_lat

trend_mean, trend_p05, trend_p95 = zonal_stats(trend)
corr_mean, corr_p05, corr_p95 = zonal_stats(corr)

# ============ Plotting ======================

# General function: keep only the left and bottom axes and set appropriate tick parameters
def clean_axis(ax):

    # Major tick parameters
    ax.tick_params(
        axis="x",
        which='major',
        bottom=True,
        top=False,
        direction='out',
        length=6,
        width=0.8
    )
    ax.tick_params(
        axis="y",
        which='major',
        left=True,
        right=False,
        direction='out',
        length=6,
        width=0.8
    )

    # Minor tick parameters
    ax.tick_params(
        axis="x",
        which='minor',
        direction='out',
        length=3,
        width=0.8
    )
    ax.tick_params(
        axis="y",
        which='minor',
        direction='out',
        length=3,
        width=0.8
    )

    ax.tick_params(axis="x", which='major', labelsize=14)
    ax.tick_params(axis="y", which='major', labelsize=14)

# Custom formatter: display zero or near-zero values as "0", and remove redundant trailing zeros
def make_formatter(ndigits):
    def fmt(x, pos=None):
        # Display values close to zero as "0"
        if np.isclose(x, 0.0, atol=1e-8):
            return "0"

        # Otherwise, use fixed decimal places and remove trailing zeros and decimal points
        s = f"{x:.{ndigits}f}".rstrip('0').rstrip('.')
        return s

    return FuncFormatter(fmt)

fig, axes = plt.subplots(2, 1, figsize=(2, 9), sharex=False, sharey=True)

# Increase spacing between the two subplots
plt.subplots_adjust(hspace=0.4)

# ---------------- Figure 1: Trend ----------------
ax = axes[0]
line_color_trend = "brown"  # tab:blue

# Add a vertical dashed line at x = 0
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax.plot(trend_mean, lat, linewidth=2, color=line_color_trend, zorder=3)
ax.fill_betweenx(
    lat,
    trend_p05,
    trend_p95,
    alpha=0.25,
    color=line_color_trend,
    zorder=2,
    edgecolor='none'
)

ax.set_xlabel(
    'Trend of Pr$_{Concentration}$\n($10^{-3}$z-score yr$^{-1}$)',
    fontsize=14
)
ax.xaxis.set_label_coords(0.4, -0.12)
ax.set_ylabel("Latitude ($^\circ$N)", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.4)

# X-axis range
ax.set_xlim(-12, 12)

# Major and minor ticks
ax.xaxis.set_major_locator(MultipleLocator(6))    # Major ticks every 6 units
# ax.xaxis.set_minor_locator(MultipleLocator(3))  # Minor ticks every 3 units
ax.yaxis.set_major_locator(MultipleLocator(30))   # Latitude major ticks every 30 degrees
# ax.yaxis.set_minor_locator(MultipleLocator(10)) # Latitude minor ticks every 10 degrees

# Apply custom formatter: keep up to two decimal places for the trend panel
ax.xaxis.set_major_formatter(make_formatter(2))
clean_axis(ax)

ax.text(
    -0.5,
    1,
    'B',
    transform=ax.transAxes,
    fontsize=15,
    fontweight='bold',
    va='top',
    ha='left'
)

# ---------------- Figure 2: Corr ----------------
ax = axes[1]
line_color_corr = "brown"

# Add a vertical dashed line at x = 0
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax.plot(corr_mean, lat, linewidth=2, color=line_color_corr, zorder=3)
ax.fill_betweenx(
    lat,
    corr_p05,
    corr_p95,
    alpha=0.25,
    color=line_color_corr,
    zorder=2,
    edgecolor='none'
)

ax.set_xlabel(
    'Trend of Drydown$_{Speed}$\n($10^{-5}$m$^{3}$m$^{-3}$day$^{-1}$yr$^{-1}$)',
    fontsize=14
)
ax.set_ylabel("Latitude ($^\circ$N)", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.4)

# X-axis range
ax.set_xlim(-2, 2)

# Major and minor ticks
ax.xaxis.set_major_locator(MultipleLocator(1))      # Major ticks every 1 unit
# ax.xaxis.set_minor_locator(MultipleLocator(0.5))  # Minor ticks every 0.5 units
ax.yaxis.set_major_locator(MultipleLocator(30))
# ax.yaxis.set_minor_locator(MultipleLocator(10))

# Apply custom formatter: keep up to two decimal places for the corr panel
ax.xaxis.set_major_formatter(make_formatter(2))
clean_axis(ax)

ax.text(
    -0.5,
    1,
    'E',
    transform=ax.transAxes,
    fontsize=15,
    fontweight='bold',
    va='top',
    ha='left'
)

save_path = '.../Pr_SM_spatial_trend/'
plt.savefig(
    save_path + "Pr_concentration_Drydown_speed_latitude_1980_2023.jpg",
    dpi=500,
    bbox_inches='tight'
)