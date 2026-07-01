import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib import cm
import matplotlib.colors as mcolors

# -------------------- Font settings ---------------------
font_size = 14
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

_nature_colors_div = [
    '#2166ac',  # Dark blue
    '#4393c3',  # Medium blue
    '#92c5de',  # Light blue
    '#d1e5f0',  # Very light blue
    '#f7f7f7',  # Near white, neutral
    '#fddbc7',  # Very light red
    '#f4a582',  # Light red
    '#d6604d',  # Medium red
    '#b2182b',  # Dark red
]
nature_div_cmap = LinearSegmentedColormap.from_list('nature_div', _nature_colors_div, N=256)

# ===== Example data =====
lat = np.linspace(90, -60, 150)
lon = np.linspace(-180, 180, 360)

reference = np.load('.../reference/reference.npy')[:150, :]

trend = np.full((150, 360, 2), np.nan)
sig = np.full((150, 360, 2), np.nan)

base_path1 = '.../integrate_trend/warm_season/daily_pr_variablity/obs_1980_2023/'
trend[:, :, 0] = np.load(base_path1 + 'daily_pr_variablity_trend_mean_1980_2023.npy')[:150, :] * 1000
sig[:, :, 0] = np.load(base_path1 + 'daily_pr_variablity_sig_median_1980_2023.npy')[:150, :] * 1000

base_path = '.../drydown_speed/trend_all/warm_season/duration_10_50/obs_1980_2023/'
trend[:, :, 1] = np.load(base_path + 'decline_rate_trend_mean_1980_2023.npy')[:150, :] * 100000
sig[:, :, 1] = np.load(base_path + 'decline_rate_sig_median_1980_2023.npy')[:150, :] * 100000

sub_labels = ['A', 'D']

labels = [
    r'Trend of Pr$_{Concentration}$  ($10^{-3}$z-score yr$^{-1}$)',
    r'Trend of Drydown$_{Speed}$ ($10^{-5}$m$^3$m$^{-3}$day$^{-1}$yr$^{-1}$)'
]

# ===== Projection and subplots =====
proj = ccrs.Robinson()
fig, axes = plt.subplots(
    nrows=2, ncols=1,
    figsize=(8, 9),
    subplot_kw={'projection': proj}
)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

lon2d, lat2d = np.meshgrid(lon, lat)

vmin = [-10, -2.5]
vmax = [10, 2.5]
cmaps = [nature_div_cmap, nature_div_cmap]

INC_SIG_COLOR = '#b2182b'   # Significant increase
INC_NS_COLOR = '#f4a582'    # Non-significant increase
DEC_SIG_COLOR = '#2166ac'   # Significant decrease
DEC_NS_COLOR = '#92c5de'    # Non-significant decrease

for i, ax in enumerate(axes.flat[0:]):
    ax.set_extent([-179.999, 179.999, -60, 90], crs=ccrs.PlateCarree())

    levels = np.linspace(vmin[i], vmax[i], 11)
    norm = BoundaryNorm(levels, ncolors=256, extend='both')
    pcm = ax.pcolormesh(
        lon2d, lat2d, trend[:, :, i],
        norm=norm,
        cmap=cmaps[i],
        shading='auto',
        transform=ccrs.PlateCarree()
    )

    # Coastlines, thinner and dark gray
    ax.coastlines(linewidth=0.6)

    # Significance hatching
    sig_mask = np.where(~np.isnan(sig[:, :, i]), 1, 0)
    ax.contourf(
        lon2d,
        lat2d,
        sig_mask,
        levels=[-0.5, 0.5, 1.5],
        hatches=[None, '\\\\\\\\'],
        colors='none',
        transform=ccrs.PlateCarree()
    )

    # Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=1,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )
    gl.xlocator = MultipleLocator(60)
    gl.ylocator = MultipleLocator(30)
    gl.top_labels = gl.right_labels = gl.bottom_labels = gl.left_labels = False

    # Colorbar
    if i == 0:
        cbar_ax = fig.add_axes([0.2, 0.525, 0.62, 0.017])
    else:
        cbar_ax = fig.add_axes([0.2, 0.075, 0.62, 0.017])

    cbar = plt.colorbar(pcm, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label(labels[i], fontsize=font_size, labelpad=4)
    cbar.ax.tick_params(labelsize=font_size - 1, width=0.6, length=3)

    # Thinner colorbar outline
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.5)

    def zero_only_formatter(x, pos):
        return "0" if abs(x) < 1e-10 else f"{x:g}"

    cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(zero_only_formatter))

    # ========== Inset bar chart, lower-left corner ==========
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    trend_i = trend[:, :, i]
    sig_i = sig[:, :, i]

    lat_rad = np.radians(lat)
    weight2d = np.tile(np.cos(lat_rad)[:, np.newaxis], (1, 360))

    valid_mask = ~np.isnan(trend_i)
    signif_mask = ~np.isnan(sig_i)

    inc_sig_mask = (trend_i > 0) & signif_mask & valid_mask
    inc_ns_mask = (trend_i > 0) & ~signif_mask & valid_mask
    dec_sig_mask = (trend_i < 0) & signif_mask & valid_mask
    dec_ns_mask = (trend_i < 0) & ~signif_mask & valid_mask

    total_weight = np.sum(weight2d[valid_mask])
    if total_weight > 0:
        inc_sig_ratio = np.sum(weight2d[inc_sig_mask]) / total_weight
        inc_ns_ratio = np.sum(weight2d[inc_ns_mask]) / total_weight
        dec_sig_ratio = np.sum(weight2d[dec_sig_mask]) / total_weight
        dec_ns_ratio = np.sum(weight2d[dec_ns_mask]) / total_weight
    else:
        inc_sig_ratio = inc_ns_ratio = dec_sig_ratio = dec_ns_ratio = 0.0

    ax_pos = ax.get_position()
    offset_x = 0.09
    offset_y = 0.04
    bar_width = 0.08
    bar_height = 0.18

    left = ax_pos.x0 + offset_x
    bottom = ax_pos.y0 + offset_y
    bar_ax = fig.add_axes([left, bottom, bar_width, bar_height])

    # Transparent background
    bar_ax.set_facecolor('none')
    bar_ax.patch.set_alpha(0)
    bar_ax.set_xlim(-0.5, 1.5)

    x_pos = [0, 1]
    width = 0.6

    # Increase, left bar: non-significant at bottom and significant on top
    bar_ax.bar(
        x_pos[0],
        inc_ns_ratio,
        width,
        color=INC_NS_COLOR,
        edgecolor='white',
        linewidth=1
    )
    bar_ax.bar(
        x_pos[0],
        inc_sig_ratio,
        width,
        bottom=inc_ns_ratio,
        color=INC_SIG_COLOR,
        edgecolor='white',
        linewidth=1
    )

    # Decrease, right bar
    bar_ax.bar(
        x_pos[1],
        dec_ns_ratio,
        width,
        color=DEC_NS_COLOR,
        edgecolor='white',
        linewidth=1
    )
    bar_ax.bar(
        x_pos[1],
        dec_sig_ratio,
        width,
        bottom=dec_ns_ratio,
        color=DEC_SIG_COLOR,
        edgecolor='white',
        linewidth=1
    )

    bar_ax.set_ylim(0, 0.75)
    bar_ax.set_yticks([0, 0.25, 0.5])
    bar_ax.set_yticklabels(['0', '25', '50'], fontsize=12)
    bar_ax.set_ylabel('Proportion (%)', fontsize=12, labelpad=2)
    bar_ax.set_xticks(x_pos)
    bar_ax.set_xticklabels(['+', '−'], fontsize=14, fontweight='bold')
    bar_ax.tick_params(axis='both', labelsize=12, width=0.6, length=3)

    # Keep only left and bottom spines
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    bar_ax.spines['left'].set_linewidth(0.6)
    bar_ax.spines['bottom'].set_linewidth(0.6)

    # Percentage labels
    def add_percentage(ax, x, bottom, height, ratio):
        if ratio > 0.03:
            ax.text(
                x,
                bottom + height / 2,
                f'{round(ratio * 100)}',
                ha='center',
                va='center',
                fontsize=12,
                color='k'
            )

    add_percentage(bar_ax, x_pos[0], 0, inc_ns_ratio, inc_ns_ratio)
    add_percentage(bar_ax, x_pos[0], inc_ns_ratio, inc_sig_ratio, inc_sig_ratio)
    add_percentage(bar_ax, x_pos[1], 0, dec_ns_ratio, dec_ns_ratio)
    add_percentage(bar_ax, x_pos[1], dec_ns_ratio, dec_sig_ratio, dec_sig_ratio)

    # Subplot label
    ax.text(
        0,
        1,
        sub_labels[i],
        transform=ax.transAxes,
        fontsize=15,
        fontweight='bold',
        va='top',
        ha='left',
        color='#111111'
    )

# Save figure
save_path = '.../Pr_SM_spatial_trend/'
plt.savefig(
    save_path + "pr_concerntration_Drydown_speed_mk_spatial_1980_2023.jpg",
    dpi=500,
    bbox_inches='tight',
    facecolor='white'
)