import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex

# -------------------- Font settings ---------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# ===========================================================
# 1. Load data
# ===========================================================
area_ratio = np.load(".../crop_pasture/cropland/crop_sig_acceleration_area_ratio.npy")
production_data = np.load(".../crop_pasture/cropland/crop_production_sig_acceleration_ratio.npy")

CROP_NAMES = ["Maize", "Wheat", "Rice", "Soybean"]
period_labels = ["1980–2023", "2025–2060", "2061–2100"]

# ===========================================================
# 2. Helper function: generate darker edge colors
# ===========================================================
def darken_color(color, factor=0.6):
    """Darken a color by multiplying each RGB channel by factor."""
    rgb = to_rgb(color)
    darker = [c * factor for c in rgb]
    return to_hex(darker)

# ===========================================================
# 3. Plot parameter settings
# ===========================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

colors = ["#9b81b7", "#f5928a", "#dc5356"]  # Fill colors
y_positions = np.arange(len(CROP_NAMES))

# ===========================================================
# 4. Main plotting loop
# ===========================================================
for i, crop in enumerate(CROP_NAMES):

    # Area proportions for the three periods
    hist_area = area_ratio[i, 0]
    mid_area = area_ratio[i, 1]
    fut_area = area_ratio[i, 2]

    # Corresponding production data for annotation
    prod_hist = production_data[i, 0]
    prod_mid = production_data[i, 1]
    prod_fut = production_data[i, 2]

    # a. Dumbbell line connecting the three points
    ax.hlines(
        y=i,
        xmin=hist_area,
        xmax=fut_area,
        colors='#cdb4db',
        linewidth=2.5,
        zorder=1
    )

    # b. Plot points for the three periods with darker edges
    for j, (area, prod, fill_color) in enumerate(zip(
        [hist_area, mid_area, fut_area],
        [prod_hist, prod_mid, prod_fut],
        colors
    )):
        # Calculate edge color
        edge_color = darken_color(fill_color, factor=0.7)

        # Plot circles
        ax.scatter(
            area,
            i,
            color=fill_color,
            s=180,
            edgecolors=edge_color,
            linewidths=2,
            zorder=2
        )

        # Add production value above the circle
        offset_y = -0.15
        ax.text(
            area,
            i + offset_y,
            f"{prod:.0f}%",
            ha='center',
            va='bottom',
            fontsize=12,
            color='black',
            fontweight='normal'
        )

# ===========================================================
# 5. Visual refinement
# ===========================================================
ax.set_ylim(-0.5, 3.5)
ax.set_yticks(y_positions)
ax.set_yticklabels(CROP_NAMES, fontsize=14)
ax.invert_yaxis()

ax.set_xlabel(
    "Area with significantly increased drydown$_{\mathrm{Speed}}$ (%)",
    fontsize=14,
    labelpad=10
)
ax.set_xlim(5, 35)
ax.tick_params(axis='x', labelsize=14)

ax.grid(axis='x', linestyle='--', alpha=0.5, color='grey')

# Legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label=period_labels[0],
        markerfacecolor=colors[0],
        markersize=10
    ),
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label=period_labels[1],
        markerfacecolor=colors[1],
        markersize=10
    ),
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label=period_labels[2],
        markerfacecolor=colors[2],
        markersize=10
    )
]

ax.legend(handles=legend_elements, loc='lower left', frameon=False, fontsize=12)

ax.text(
    -0.15,
    1,
    'C',
    transform=ax.transAxes,
    fontsize=15,
    fontweight='bold',
    va='top',
    ha='left'
)

# ===========================================================
# 6. Save and show
# ===========================================================
plt.savefig(
    ".../Figure/cropland_pasture/crop_type_effect.jpg",
    bbox_inches='tight',
    dpi=500
)