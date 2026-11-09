import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

# Set global font to DejaVu Serif (similar to Times New Roman)
plt.rcParams["font.family"] = "DejaVu Serif"
sns.set(style="whitegrid", font_scale=1.2, font="DejaVu Serif")

# ========== User Configuration ==========
json_path = "Data/lib5_organized_072125.json"
output_base = "Output/visual/g"
label_x = "EFA"  # Change to your desired label
label_y = "EFA_cce"  # Change to your desired label
dg_len_threshold = 15  # Only keep entries with dg_len >= this value
label_x_unit = "eV/atom"  # Custom unit for label_x
label_y_unit = "eV/atom"  # Custom unit for label_y

# Helper to format axis label (lowercase except EFA, d2h uses LaTeX)
def format_label(label, unit=None):
    if label.lower() == "d2h":
        base = r"$\langle \Delta H_{\mathrm{hull}} \rangle$"
        if unit:
            return f"{base} ({unit})"
        return base
    elif label.upper().startswith("EFA"):
        base = "EFA" + label[3:].lower()
    else:
        base = label.lower()
    if unit:
        return f"{base} ({unit})"
    return base

def format_title(label):
    # For titles, all lower except EFA; d2h用普通字符串
    if label.upper().startswith("EFA"):
        return "EFA" + label[3:].lower()
    elif label.lower() == "d2h":
        return "d2h"
    else:
        return label.lower()

def format_legend(label):
    # For legend, EFA特殊处理，d2h用普通字符串，元素首字母大写
    if label.upper().startswith("EFA"):
        return "EFA" + label[3:].lower()
    elif label.lower() == "d2h":
        return "d2h"
    else:
        return label.capitalize()

# ========== Load Data ==========
with open(json_path, 'r') as f:
    data = json.load(f)

# ========== Filter Data by dg_len ==========
data = [entry for entry in data if entry.get('dg_len', 0) >= dg_len_threshold]

# ========== Extract Data ==========
x = []
y = []
for entry in data:
    if label_x in entry and label_y in entry:
        x.append(entry[label_x])
        y.append(entry[label_y])

# ========== Parity Plot ==========
plt.figure(figsize=(8, 8))

# Rank-difference coloring (signed)
from scipy.stats import rankdata
rank_x = rankdata(x, method='average') / len(x)
rank_y = rankdata(y, method='average') / len(y)
rank_diff = rank_x - rank_y  # signed difference
scatter = plt.scatter(x, y, c=rank_diff, cmap="coolwarm", s=40, alpha=0.7)
plt.colorbar(scatter)  # No label

# Regression line (no scatter)
sns.regplot(x=x, y=y, scatter=False, line_kws={'color': 'red'})

# Calculate correlation coefficient
corr = np.corrcoef(x, y)[0, 1]

# Annotate correlation
plt.annotate(f"Pearson r = {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=14, ha='left', va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))

plt.xlabel(format_label(label_x, label_x_unit))
plt.ylabel(format_label(label_y, label_y_unit))
plt.title(f"parity plot: {format_title(label_x)} vs {format_title(label_y)} (color: rank diff)")
plt.tight_layout()
plt.savefig(f"{output_base}_{label_x}_vs_{label_y}_parity_rankdiff_signed.png", dpi=300)
plt.show()

# ========== Distribution Plots ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(x, kde=True, ax=axes[0], color='skyblue', stat="density")
axes[0].set_title(f"distribution of {format_title(label_x)}")
axes[0].set_xlabel(format_label(label_x, label_x_unit))
axes[0].set_ylabel("density")
sns.histplot(y, kde=True, ax=axes[1], color='salmon', stat="density")
axes[1].set_title(f"distribution of {format_title(label_y)}")
axes[1].set_xlabel(format_label(label_y, label_y_unit))
axes[1].set_ylabel("density")
plt.tight_layout()
plt.savefig(f"{output_base}_{label_x}_vs_{label_y}_dists.png", dpi=300)
plt.show()

# ========== Per-Element Difference Distribution ==========
from collections import defaultdict

element_diff = defaultdict(list)
for entry in data:
    if label_x in entry and label_y in entry and 'Elements' in entry:
        diff = entry[label_x] - entry[label_y]
        for elem in entry['Elements']:
            element_diff[elem].append(diff)

# Plot each element's diff distribution (only for elements with enough data)
min_count = 10  # Only plot elements with at least 10 samples
selected_elements = [e for e, v in element_diff.items() if len(v) >= min_count]

# Subplot arrangement: single row, multiple columns, all share x-axis
n_elems = len(selected_elements)
if n_elems > 0:
    fig, axes = plt.subplots(n_elems, 1, figsize=(10, 2*n_elems), sharex=True, squeeze=False)
    axes = axes[:, 0]  # flatten to 1D
    # Find global xlim for all elements
    all_diffs = [d for elem in selected_elements for d in element_diff[elem]]
    x_min, x_max = min(all_diffs), max(all_diffs)
    # Use seaborn color palette for distinct colors
    palette = itertools.cycle(sns.color_palette("tab10", n_elems))
    for idx, elem in enumerate(selected_elements):
        ax = axes[idx]
        color = next(palette)
        sns.kdeplot(element_diff[elem], label=format_legend(elem), fill=True, alpha=0.3, ax=ax, color=color)
        ax.set_xlabel(f"{format_label(label_x)} - {format_label(label_y)}")
        ax.set_ylabel("density")
        # No title for subplots
        ax.set_xlim(x_min, x_max)
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_base}_{label_x}_vs_{label_y}_element_diff.png", dpi=300)
    plt.show()

# ========== Save Data for Visualization ==========
import pandas as pd

# 1. chem_id, label_x, label_y, diff, rank_x, rank_y, rank_diff
chem_ids = []
x_vals = []
y_vals = []
diffs = []
rank_xs = []
rank_ys = []
rank_diffs = []
for i, entry in enumerate(data):
    if label_x in entry and label_y in entry and 'chem_id' in entry:
        chem_ids.append(entry['chem_id'])
        x_vals.append(entry[label_x])
        y_vals.append(entry[label_y])
        diffs.append(entry[label_x] - entry[label_y])
        rank_xs.append(rank_x[i])
        rank_ys.append(rank_y[i])
        rank_diffs.append(rank_x[i] - rank_y[i])
df_labels = pd.DataFrame({
    'chem_id': chem_ids,
    label_x: x_vals,
    label_y: y_vals,
    'diff': diffs,
    f'rank_{label_x}': rank_xs,
    f'rank_{label_y}': rank_ys,
    'rank_diff': rank_diffs
})
df_labels.to_csv(f"{output_base}_{label_x}_vs_{label_y}_chem_diff.csv", index=False)

# 2. element, chem_id, diff (long format)
elem_rows = []
for entry in data:
    if label_x in entry and label_y in entry and 'chem_id' in entry and 'Elements' in entry:
        diff = entry[label_x] - entry[label_y]
        for elem in entry['Elements']:
            elem_rows.append({'element': elem, 'chem_id': entry['chem_id'], 'diff': diff})
df_elem = pd.DataFrame(elem_rows)
df_elem.to_csv(f"{output_base}_{label_x}_vs_{label_y}_element_diff.csv", index=False)