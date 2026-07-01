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

_og_colors = [
    '#b35806',  # Dark orange
    '#e08214',  # Medium-dark orange
    '#fdb863',  # Medium orange
    '#fee0b6',  # Light orange
    '#fff5eb',  # Very light orange
    '#f7f7f7',  # Neutral white
    '#d9f0d3',  # Very light green
    '#a6dba0',  # Light green
    '#5aae61',  # Medium green
    '#1b7837',  # Dark green
    '#00441b',  # Very dark green
]
nature_og_cmap = LinearSegmentedColormap.from_list('nature_og', _og_colors, N=256)
nature_og_r_cmap = LinearSegmentedColormap.from_list('nature_og_r', _og_colors[::-1], N=256)

INC_SIG_G = '#1b7837'
INC_NS_G = '#a6dba0'
DEC_SIG_O = '#e08214'
DEC_NS_O = '#fdb863'

INC_SIG_O = '#e08214'
INC_NS_O = '#fdb863'
DEC_SIG_G = '#1b7837'
DEC_NS_G = '#a6dba0'

cmaps = [nature_og_cmap, nature_og_r_cmap, nature_og_r_cmap, nature_og_r_cmap]

# ===== Example data =====
lat = np.linspace(90, -60, 150)
lon = np.linspace(-180, 180, 360)

# ================ Soil moisture changes ===================
base_path = '.../drydown_speed/trend_all/warm_season/duration_10_50/obs_1980_2023/'
trend = np.full((150, 360, 4), np.nan)
sig = np.full((150, 360, 4), np.nan)

trend[:, :, 0] = np.load(
    base_path + 'rise_magnitude_trend_mean_1980_2023.npy'
)[:150, :] * 10000
sig[:, :, 0] = np.load(
    base_path + 'rise_magnitude_sig_median_1980_2023.npy'
)[:150, :] * 10000

base_path1 = '.../drivers/warm_season/'
trend[:, :, 1] = np.load(
    base_path1 + 'drydown_E/mk_trend/duration_10_50/drydown_E_trend_mean_1980_2023.npy'
)[:150, :] * 1000
sig[:, :, 1] = np.load(
    base_path1 + 'drydown_E/mk_trend/duration_10_50/drydown_E_sig_median_1980_2023.npy'
)[:150, :] * 1000

trend[:, :, 2] = np.load(
    base_path1 + 'drydown_Rs_tas/mk_trend/duration_10_50/drydown_rs_trend_mean_1980_2023.npy'
)[:150, :] * 10
sig[:, :, 2] = np.load(
    base_path1 + 'drydown_Rs_tas/mk_trend/duration_10_50/drydown_rs_sig_median_1980_2023.npy'
)[:150, :] * 10

trend[:, :, 3] = np.load(
    base_path1 + 'drydown_Rs_tas/mk_trend/duration_10_50/drydown_tas_trend_mean_1980_2023.npy'
)[:150, :] * 100
sig[:, :, 3] = np.load(
    base_path1 + 'drydown_Rs_tas/mk_trend/duration_10_50/drydown_tas_sig_median_1980_2023.npy'
)[:150, :] * 100

labels = [
    r'Trend of Pulse$_{SM}$ ($10^{-4}$m$^3$m$^{-3}$day$^{-1}$yr$^{-1}$)',
    r'Trend of Drydown$_{SW}$ ($10^{-3}$mm day$^{-1}$yr$^{-1}$)',
    r'Trend of Drydown$_{ET}$ ($10^{-1}$W m$^{-2}$yr$^{-1}$)',
    r'Trend of Drydown$_{Tas}$ ($10^{-2}$°C yr$^{-1}$)'
]

sub_labels = ['A', 'B', 'C', 'D']

# ===== Projection and subplots =====
proj = ccrs.Robinson()
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(14, 9),
    subplot_kw={'projection': proj}
)
plt.subplots_adjust(hspace=0.05, wspace=0.05)

lon2d, lat2d = np.meshgrid(lon, lat)

vmin = [-6, -18, -6, -6]
vmax = [6, 18, 6, 6]

for i, ax in enumerate(axes.flat[0:4]):
    ax.set_extent([-179.999, 179.999, -60, 90], crs=ccrs.PlateCarree())

    levels = np.linspace(vmin[i], vmax[i], 13)
    norm = BoundaryNorm(levels, ncolors=256, extend='both')
    pcm = ax.pcolormesh(
        lon2d,
        lat2d,
        trend[:, :, i],
        norm=norm,
        cmap=cmaps[i],
        shading='auto',
        transform=ccrs.PlateCarree()
    )

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
    cbar = plt.colorbar(
        pcm,
        ax=ax,
        orientation='horizontal',
        pad=0.05,
        aspect=35,
        extend='both',
        shrink=0.9
    )
    cbar.set_label(labels[i], fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size, width=0.6, length=3)

    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.5)

    def zero_only_formatter(x, pos):
        return "0" if abs(x) < 1e-10 else f"{x:g}"

    cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(zero_only_formatter))

    # ========== Inset bar chart ==========
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
    offset_x = 0.05
    offset_y = 0.035
    bar_width = 0.045
    bar_height = 0.15

    left = ax_pos.x0 + offset_x
    bottom = ax_pos.y0 + offset_y
    bar_ax = fig.add_axes([left, bottom, bar_width, bar_height])

    bar_ax.set_facecolor('none')
    bar_ax.patch.set_alpha(0)
    bar_ax.set_xlim(-0.5, 1.5)

    x_pos = [0, 1]
    width = 0.6

    if i == 0:
        inc_ns_color = INC_NS_G
        inc_sig_color = INC_SIG_G
        dec_ns_color = DEC_NS_O
        dec_sig_color = DEC_SIG_O
    else:
        inc_ns_color = INC_NS_O
        inc_sig_color = INC_SIG_O
        dec_ns_color = DEC_NS_G
        dec_sig_color = DEC_SIG_G

    bar_ax.bar(
        x_pos[0],
        inc_ns_ratio,
        width,
        color=inc_ns_color,
        edgecolor='white',
        linewidth=1
    )
    bar_ax.bar(
        x_pos[0],
        inc_sig_ratio,
        width,
        bottom=inc_ns_ratio,
        color=inc_sig_color,
        edgecolor='white',
        linewidth=1
    )
    bar_ax.bar(
        x_pos[1],
        dec_ns_ratio,
        width,
        color=dec_ns_color,
        edgecolor='white',
        linewidth=1
    )
    bar_ax.bar(
        x_pos[1],
        dec_sig_ratio,
        width,
        bottom=dec_ns_ratio,
        color=dec_sig_color,
        edgecolor='white',
        linewidth=1
    )

    if i == 3:
        bar_ax.set_ylim(0, 1)
        bar_ax.set_yticks([0, 0.5, 1])
        bar_ax.set_yticklabels(['0', '50', '100'], fontsize=12)
    else:
        bar_ax.set_ylim(0, 0.8)
        bar_ax.set_yticks([0, 0.4, 0.8])
        bar_ax.set_yticklabels(['0', '40', '80'], fontsize=12)

    bar_ax.set_ylabel('Proportion (%)', fontsize=12)
    bar_ax.set_xticks(x_pos)
    bar_ax.set_xticklabels(['+', '-'], fontsize=100)
    bar_ax.tick_params(axis='both', labelsize=12, width=0.6, length=3)
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)
    bar_ax.spines['left'].set_linewidth(0.6)
    bar_ax.spines['bottom'].set_linewidth(0.6)

    def add_percentage(ax, x, bottom, height, ratio):
        if ratio > 0:
            y_center = bottom + height / 2
            ax.text(
                x,
                y_center,
                f'{round(ratio * 100)}',
                ha='center',
                va='center',
                fontsize=12,
                color='black'
            )

    if i == 3:
        add_percentage(bar_ax, x_pos[0], 0, inc_ns_ratio, inc_ns_ratio)
        add_percentage(bar_ax, x_pos[0], inc_ns_ratio, inc_sig_ratio, inc_sig_ratio)
        add_percentage(bar_ax, x_pos[1], 0, dec_ns_ratio, dec_ns_ratio)
    else:
        add_percentage(bar_ax, x_pos[0], 0, inc_ns_ratio, inc_ns_ratio)
        add_percentage(bar_ax, x_pos[0], inc_ns_ratio, inc_sig_ratio, inc_sig_ratio)
        add_percentage(bar_ax, x_pos[1], 0, dec_ns_ratio, dec_ns_ratio)
        add_percentage(bar_ax, x_pos[1], dec_ns_ratio, dec_sig_ratio, dec_sig_ratio)

    ax.text(
        0,
        1,
        sub_labels[i],
        transform=ax.transAxes,
        fontsize=15,
        fontweight='bold',
        va='top',
        ha='left'
    )

# Save figure
save_path = '.../Figure/drivers/'
plt.savefig(
    save_path + "pr_variability_drivers_mk_spatial_1980_2023.jpg",
    dpi=500,
    bbox_inches='tight',
    facecolor='white'
)