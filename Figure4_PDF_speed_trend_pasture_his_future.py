import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ===========================================================
# 0. Font settings
# ===========================================================
font_size = 14
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# ===========================================================
# 1. Configure file paths
# ===========================================================
HIST_PATH = ".../warm_season/period_mean/duration_10_50/Ensemble_drydown_speed_yearmean_1980_2023.npy"
FUT1_PATH = ".../SSP585/2025_2060_mean/ensemble_mean_2025_2060_decline_rate.npy"
FUT2_PATH = ".../warm_season/SSP585/2061_2100_mean/ensemble_mean_2061_2100_decline_rate.npy"

# Historical trend
TREND_HIST_MEAN = ".../trend_all/warm_season/duration_10_50/obs/decline_rate_trend_mean_1980_2023.npy"
TREND_HIST_SIG = ".../trend_all/warm_season/duration_10_50/obs/decline_rate_sig_median_1980_2023.npy"

# Future trend: 2025–2060
TREND_FUT1_MEAN = ".../warm_season/trends_2025_2060/trend_all/SSP585/decline_rate_trend_mean_2025_2060.npy"
TREND_FUT1_SIG = ".../warm_season/trends_2025_2060/trend_all/SSP585/decline_rate_sig_consensus_2025_2060.npy"

# Future trend: 2061–2100
TREND_FUT2_MEAN = ".../warm_season/trends_2061_2100/trend_all/SSP585/decline_rate_trend_mean_2061_2100.npy"
TREND_FUT2_SIG = ".../warm_season/trends_2061_2100/trend_all/SSP585/decline_rate_sig_consensus_2061_2100.npy"

# half-drydown paths
HALF_HIST_PATH = ".../drydown_half/distribution_change/warm_season/duration_10_50/OBS/1980_2023/ensemble_mean/period_mean_half_time_1980_2022.npy"
HALF_FUT1_PATH = ".../drydown_half/distribution_change/warm_season/duration_10_50/CMIP6/2025_2060/ensemble_mean/ensemble_mean_2025_2060_half_time.npy"
HALF_FUT2_PATH = ".../drydown_half/distribution_change/warm_season/duration_10_50/CMIP6/2061_2100/ensemble_mean/ensemble_mean_2061_2100_half_time.npy"

RAINFED_PATH = ".../pasture_cropland/extent/Pasture_100.npy"

X_MIN, X_MAX = 0, 7
N_GRID = 1000

# ===========================================================
# 2. Rainfed pasture mask
# ===========================================================
rainfed = np.load(RAINFED_PATH)
rainfed_mask = (rainfed < 0.1) & ~np.isnan(rainfed)

# ===========================================================
# 3. Construct latitude-based area weighting matrix
# ===========================================================
def build_area_weights(shape):
    nlat, nlon = shape
    lat_res = 180.0 / nlat
    lats = 90.0 - lat_res * (np.arange(nlat) + 0.5)
    weights_1d = np.clip(np.cos(np.deg2rad(lats)), 0, None)
    return np.tile(weights_1d[:, np.newaxis], (1, nlon))


AREA_WEIGHTS = build_area_weights(rainfed.shape)

# ===========================================================
# 4. Utility functions
# ===========================================================
def extract_valid(arr, mask):
    vals = arr[mask]
    wts = AREA_WEIGHTS[mask]
    valid = ~np.isnan(vals)
    return vals[valid], wts[valid]


def compute_kde(values, weights, x_grid):
    if len(values) < 10:
        return np.full_like(x_grid, np.nan)

    w = weights / weights.sum()
    n_eff = 1.0 / np.sum(w ** 2)
    mu = np.sum(w * values)
    std = np.sqrt(np.sum(w * (values - mu) ** 2))
    bw = 1.06 * std * n_eff ** (-0.2)

    if bw <= 0:
        return np.full_like(x_grid, np.nan)

    diff = (x_grid[:, np.newaxis] - values[np.newaxis, :]) / bw
    kernels = np.exp(-0.5 * diff ** 2) / (np.sqrt(2 * np.pi) * bw)
    return kernels @ w


def kde_median(x_grid, kde_vals):
    dx = x_grid[1] - x_grid[0]
    cdf = np.cumsum(kde_vals) * dx
    cdf /= cdf[-1]
    idx = np.searchsorted(cdf, 0.5)
    return x_grid[min(idx, len(x_grid) - 1)]


def trend_proportions(trend_mean, sig, mask,
                      half_hist=None, half_fut=None):
    """
    Calculate area-weighted proportions of trend categories.

    For future periods, when both half_hist and half_fut are provided:
    if the trend is negative but (half_fut - half_hist) < 0,
    meaning that future half-drydown time is shorter than historical
    half-drydown time and drying is actually accelerated, the trend
    sign is flipped before calculating the proportions.

    Parameters
    ----------
    trend_mean : ndarray
        Mean trend over the global grid.
    sig : ndarray
        Significance mask, with NaN indicating non-significant trends.
    mask : ndarray
        Boolean mask for rainfed pasture.
    half_hist : ndarray or None
        Historical half-drydown time over the global grid.
    half_fut : ndarray or None
        Future half-drydown time over the global grid.
    """
    valid = mask & ~np.isnan(trend_mean)

    if valid.sum() == 0:
        return [0, 0, 0, 0]

    # Copy trend values before modification
    t = trend_mean.copy()

    # ---------- Core adjustment ----------
    if half_hist is not None and half_fut is not None:

        flip_mask = (
            valid
            & (trend_mean < 0)
            & ~np.isnan(half_hist)
            & ~np.isnan(half_fut)
            & ((half_fut - half_hist) < 0)
        )

        # Flip the sign: negative to positive
        t[flip_mask] = -t[flip_mask]
    # ------------------------------------

    t_v = t[valid]
    s_v = sig[valid]
    w_v = AREA_WEIGHTS[valid]
    W = w_v.sum()

    sig_pos = w_v[(t_v > 0) & ~np.isnan(s_v)].sum() / W * 100
    nsig_pos = w_v[(t_v > 0) & np.isnan(s_v)].sum() / W * 100
    nsig_neg = w_v[(t_v < 0) & np.isnan(s_v)].sum() / W * 100
    sig_neg = w_v[(t_v < 0) & ~np.isnan(s_v)].sum() / W * 100

    return [sig_pos, nsig_pos, nsig_neg, sig_neg]

# ===========================================================
# 5. Load data and compute KDE
# ===========================================================
x_grid = np.linspace(X_MIN, X_MAX, N_GRID)

hist_vals, hist_wts = extract_valid(np.load(HIST_PATH) * 1000, rainfed_mask)
fut1_vals, fut1_wts = extract_valid(np.load(FUT1_PATH) * 1000, rainfed_mask)
fut2_vals, fut2_wts = extract_valid(np.load(FUT2_PATH) * 1000, rainfed_mask)

hist_kde = compute_kde(hist_vals, hist_wts, x_grid)
fut1_kde = compute_kde(fut1_vals, fut1_wts, x_grid)
fut2_kde = compute_kde(fut2_vals, fut2_wts, x_grid)

med_hist = kde_median(x_grid, hist_kde)
med_fut1 = kde_median(x_grid, fut1_kde)
med_fut2 = kde_median(x_grid, fut2_kde)

# ===========================================================
# 6. Load trend and half-drydown data and calculate proportions
# ===========================================================
hist_trend = np.load(TREND_HIST_MEAN)
hist_sig = np.load(TREND_HIST_SIG)
fut1_trend = np.load(TREND_FUT1_MEAN)
fut1_sig = np.load(TREND_FUT1_SIG)
fut2_trend = np.load(TREND_FUT2_MEAN)
fut2_sig = np.load(TREND_FUT2_SIG)

half_hist = np.load(HALF_HIST_PATH)
half_fut1 = np.load(HALF_FUT1_PATH)
half_fut2 = np.load(HALF_FUT2_PATH)

# Historical trend: no half-drydown adjustment
props_hist = trend_proportions(hist_trend, hist_sig, rainfed_mask)

# Future trends: apply half-drydown adjustment
props_fut1 = trend_proportions(
    fut1_trend,
    fut1_sig,
    rainfed_mask,
    half_hist=half_hist,
    half_fut=half_fut1
)

props_fut2 = trend_proportions(
    fut2_trend,
    fut2_sig,
    rainfed_mask,
    half_hist=half_hist,
    half_fut=half_fut2
)

TREND_LABELS = ["Sig. +", "+", "−", "Sig. −"]
TREND_COLORS = ['#b2182b', '#f4a582', '#92c5de', '#2166ac']

# ===========================================================
# 7. Plotting
# ===========================================================
C_HIST = "#4E79A7"
C_FUT1 = "#F28E2B"
C_FUT2 = "#E15759"

lw = 1.5
med_lw = 1.0
med_ls = (0, (4, 3))

fig, ax_main = plt.subplots(figsize=(6, 5))


def draw_curve(ax, kde_vals, color, label, linestyle="-", median=None):
    ax.plot(x_grid, kde_vals, color=color, lw=lw, linestyle=linestyle, label=label)

    if median is not None:
        ymax = float(np.interp(median, x_grid, kde_vals))
        ax.vlines(median, 0, ymax, color=color, lw=med_lw, linestyle=med_ls)


draw_curve(ax_main, hist_kde, C_HIST, "1980–2023", median=med_hist)
draw_curve(ax_main, fut1_kde, C_FUT1, "2025–2060", median=med_fut1)
draw_curve(ax_main, fut2_kde, C_FUT2, "2061–2100", median=med_fut2)

ax_main.set_xlabel(
    r'Pasture drydown$_{\mathrm{Speed}}$'
    r'($10^{-3}$ m$^{3}$ m$^{-3}$ day$^{-1}$)',
    fontsize=14
)
ax_main.set_ylabel("Probability density", fontsize=14, labelpad=10)
ax_main.set_xlim(X_MIN, X_MAX)
ax_main.set_ylim(0, 0.8)
ax_main.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax_main.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax_main.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_main.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
ax_main.tick_params(axis='both', labelsize=14)


def zero_only_formatter(x, pos):
    return "0" if abs(x) < 1e-10 else f"{x:.1f}"


ax_main.yaxis.set_major_formatter(ticker.FuncFormatter(zero_only_formatter))
ax_main.legend(
    loc="upper left",
    ncol=1,
    handlelength=1.8,
    borderpad=0.5,
    fontsize=12,
    frameon=False
)

# ---------- Inset stacked bar chart with three periods ----------
ax_bar = ax_main.inset_axes([0.68, 0.28, 0.3, 0.5])
x_pos = np.array([0, 1, 2])
bar_w = 0.45
bottoms = np.zeros(3)
bar_data = np.array([props_hist, props_fut1, props_fut2])

for i, (label, color) in enumerate(zip(TREND_LABELS, TREND_COLORS)):
    vals = bar_data[:, i]

    ax_bar.bar(
        x_pos,
        vals,
        bottom=bottoms,
        width=bar_w,
        color=color,
        label=label,
        edgecolor='white',
        linewidth=0.4
    )

    for xi, (v, b) in enumerate(zip(vals, bottoms)):
        if v < 3:
            bottoms[xi] += v
            continue

        ax_bar.text(
            x_pos[xi],
            b + v / 2,
            f"{v:.0f}",
            ha='center',
            va='center',
            fontsize=12,
            color='k'
        )

    bottoms += vals

ax_bar.set_xlim(-0.5, 2.5)
ax_bar.set_ylim(0, 100)
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(
    ["1980–\n2023", "2025–\n2060", "2061–\n2100"],
    fontsize=10,
    ha='center'
)
ax_bar.set_ylabel("Trend proportion (%)", fontsize=12)
ax_bar.tick_params(axis='y', labelsize=10)
ax_bar.tick_params(axis='x', length=0)
ax_bar.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.set_facecolor('white')
ax_bar.legend(
    loc="lower left",
    bbox_to_anchor=(-0.15, 1.05),
    ncol=2,
    fontsize=12,
    handlelength=1.0,
    handletextpad=0.3,
    columnspacing=0.6,
    borderpad=0.3,
    frameon=False
)

ax_main.text(
    -0.15,
    1.02,
    'B',
    transform=ax_main.transAxes,
    fontsize=15,
    fontweight='bold',
    va='top',
    ha='left'
)

save_path = '.../cropland_pasture/'
plt.savefig(
    save_path + "Pasture_Drydown_speed_pdf.jpg",
    dpi=500,
    bbox_inches='tight'
)