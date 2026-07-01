import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# -------------------- Font settings --------------------
font_size = 14
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# ==================== Configuration parameters ====================
base_path = '.../Daily_rate_change/warm_season/daily_trend_koppen/daily_rate_duration/duration_10_50'
data_sources = ['GLEAM', 'MERRA2', 'ERA5']
zones = ['Zone1', 'Zone2', 'Zone3', 'Zone4']
zone_names = ['Tropical', 'Arid', 'Temperate', 'Cold']
L_min, L_max = 10, 50
d_max = 50
alpha = 0.05

freq_base_dir = '..../Daily_rate_change/warm_season/daily_trend_koppen/daily_rate_duration/heatmaps/duration_fre/'

# ==================== Load and combine trend data ====================
def load_source_data(source, zone):
    trends_file = os.path.join(base_path, source, zone, 'trends.npy')
    if not os.path.exists(trends_file):
        print(f"⚠️ Missing file: {trends_file}")
        return None, None

    trends_arr = np.load(trends_file)
    slope_all = trends_arr[:, :, 0] * 100000
    p_all = trends_arr[:, :, 1]
    slope = slope_all[0:41, 0:50]
    p = p_all[0:41, 0:50]
    return slope, p

combined_slope = {zone: None for zone in zones}
combined_sig = {zone: None for zone in zones}

for zone in zones:
    slopes, ps = [], []
    for src in data_sources:
        sl, pv = load_source_data(src, zone)
        if sl is not None:
            slopes.append(sl)
            ps.append(pv)

    if len(slopes) < 2:
        print(f"⚠️ Fewer than two data sources available for {zone}; skipped")
        continue

    slope_mean = np.mean(slopes, axis=0)
    p_stack = np.stack(ps, axis=0)
    sig_combined = np.sum(p_stack < alpha, axis=0) >= 2
    combined_slope[zone] = slope_mean
    combined_sig[zone] = sig_combined

# ==================== Color ranges for subplots ====================
vmin = [-2.5, -2.5, -2.5, -2.5]
vmax = [2.5, 2.5, 2.5, 2.5]

sub_labels = ['A', 'B', 'C', 'D']

# ==================== Plotting ====================
fig, axes = plt.subplots(2, 2, figsize=(13, 12))
axes = axes.flatten()

nL = L_max - L_min + 1
nd = 50

for idx, zone in enumerate(zones):
    ax = axes[idx]
    slope = combined_slope[zone]
    sig = combined_sig[zone]

    if slope is None:
        ax.text(0.5, 0.5, f'{zone}\nNo data', ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    data = np.full((nL, nd), np.nan)
    for i, L in enumerate(range(L_min, L_max + 1)):
        max_d = L - 1
        for j, d in enumerate(range(1, d_max + 1)):
            if d <= max_d:
                data[i, j] = slope[i, j]

    levels = np.linspace(vmin[idx], vmax[idx], 11)
    norm = BoundaryNorm(levels, ncolors=256, extend='both')

    im = ax.imshow(
        data,
        origin='upper',
        aspect='auto',
        cmap='PuOr_r',
        norm=norm,
        extent=[0.5, 50.5, 0.5, nL + 0.5]
    )

    y_idx, x_idx = np.where(sig[:nL, :] & ~np.isnan(data))
    ax.scatter(
        x_idx + 1.0,
        nL - y_idx,
        s=6,
        c='black',
        marker='.',
        alpha=0.7
    )

    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_xticklabels([1, 10, 20, 30, 40, 50], fontsize=14)
    ax.set_xlabel('Days since drydown start (days)', fontsize=14, labelpad=6)

    y_tick_pos = [nL, nL - 10, nL - 20, nL - 30, nL - 40]
    y_tick_labels = [10, 20, 30, 40, 50]
    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(y_tick_labels, fontsize=14)
    ax.set_ylabel('Drydown duration (days)', fontsize=14, labelpad=6)
    ax.set_title(zone_names[idx], fontsize=15, pad=10)

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation='horizontal',
        pad=0.17,
        fraction=0.04,
        extend='both'
    )
    cbar.set_label(
        r'Daily rate trend ($10^{-4}$ m$^3$ m$^{-3}$ day$^{-1}$ yr$^{-1}$)',
        fontsize=14,
        labelpad=2
    )
    cbar.ax.tick_params(labelsize=14)

    if idx in (0, 1):
        cbar.ax.set_visible(False)

    # ========== Multi-source mean frequency distribution, upper-right corner, full period ==========
    total_freqs_all = []
    L_plot = None

    for src in data_sources:
        npz_path = os.path.join(freq_base_dir, src, f'{zone}_event_counts.npz')

        if not os.path.exists(npz_path):
            print(f"⚠️ Missing frequency file: {npz_path}")
            continue

        freq_data = np.load(npz_path)
        L_all = freq_data['L']
        early_counts = freq_data['early']
        late_counts = freq_data['late']
        n_pixels = freq_data['n_pixels']
        n_early = freq_data['n_years_early']
        n_late = freq_data['n_years_late']

        mask = (L_all >= L_min) & (L_all <= L_max)
        L_plot = L_all[mask]

        # Combine early and late periods into the full period
        total_counts = early_counts[mask] + late_counts[mask]
        total_years = n_early + n_late
        total_freq = total_counts / (total_years * n_pixels)

        total_freqs_all.append(total_freq)

    if total_freqs_all:
        total_avg = np.mean(total_freqs_all, axis=0)

        ax_ins = ax.inset_axes([0.62, 0.63, 0.35, 0.28])
        ax_ins.plot(L_plot, total_avg, color='#b2182b', linewidth=1.2)
        ax_ins.tick_params(labelsize=12)
        ax_ins.spines['top'].set_visible(False)
        ax_ins.spines['right'].set_visible(False)
        ax_ins.set_ylim(0, 0.4)
        ax_ins.set_xticks([10, 20, 30, 40, 50])
        ax_ins.set_yticks([0, 0.2, 0.4])
        ax_ins.set_yticklabels(['0', '0.2', '0.4'])
        ax_ins.set_xlabel('Duration', fontsize=12, labelpad=2)
        ax_ins.set_ylabel('Frequency', fontsize=12, labelpad=2)

    ax.text(
        -0.14,
        1.085,
        sub_labels[idx],
        transform=ax.transAxes,
        fontsize=15,
        fontweight='bold',
        va='top',
        ha='left'
    )

plt.subplots_adjust(hspace=0.02, wspace=0.25, left=0.15)

save_path = '.../daily_drydown_rate/'
plt.savefig(
    save_path + "daily_drydown_rate_duration_1980_2023.jpg",
    dpi=500,
    bbox_inches='tight'
)