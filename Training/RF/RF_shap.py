# -*- coding: utf-8 -*-
"""
SHAP Feature Importance Analysis for Random Forest

Created by: Guangshuai Han
Year: 2025
Affiliation: Johns Hopkins University
Purpose:
    - Compute SHAP values for trained Random Forest models.
    - Store individual SHAP values per feature before aggregation.
    - Visualize and rank SHAP-based feature importance.
"""

import os
import pickle
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys

# ---------------- PATH CONFIGURATION ----------------
# Specify model output directory (contains model and training_info.json)
output_dir = "Output/Training/2026-03-19/17-33-55"  # Update as needed
saving_dir = output_dir + "/shap"
#make saving_dir
os.makedirs(saving_dir, exist_ok=True)
model_file = os.path.join(output_dir, "best_model.pkl")
info_file = os.path.join(output_dir, "training_info.json")
shap_importance_plot = os.path.join(saving_dir , "shap_feature_importance.png")
shap_detailed_plot = os.path.join(saving_dir , "shap_detailed_feature_importance.png")
shap_raw_csv_file = os.path.join(saving_dir , "shap_raw_values.csv")
shap_aggregated_csv_file = os.path.join(saving_dir , "shap_feature_importance.csv")
# ----------------------------------------------------

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from mat_encod.data_util import ele_df, json_loader

# ===== Load Training Info =====
with open(info_file, "r") as f:
    info_dict = json.load(f)

# Dynamically build stats_labels and marker styles
stats_labels = []
marker_styles = {}
marker_map = {
    "Avg": ("use_avg", "o"),
    "Var": ("use_var", "s"),
    "Max": ("use_max", "^"),
    "Min": ("use_min", "d"),
    "wAvg": ("use_weighted_avg", "P"),
    "wVar": ("use_weighted_var", "X"),
    "Ratio": ("use_ratio", "*"),
    "Delta": ("use_delta", "v"),
}

for label, (json_key, marker) in marker_map.items():
    if info_dict.get(json_key, False):
        stats_labels.append(label)
        marker_styles[label] = marker

stats_count = len(stats_labels)
print(f"Detected statistical features: {stats_labels}")
print(f"Corresponding marker styles: {marker_styles}")

# Load model
with open(model_file, "rb") as f:
    model = pickle.load(f)

# Load features
ele_DF = ele_df(prop_list=info_dict['prop_list'],
                total_e_list=info_dict['unique_elements'],
                normalize=info_dict['normalize'])

loader = json_loader(info_dict['json_path'], df_data=ele_DF)
train_x, train_y, _ = loader.load_data_nocomp(
    target_key=info_dict['target_key'],
    split=False,
    compositions=info_dict['compositions'],
    use_avg=info_dict['use_avg'],
    use_var=info_dict['use_var'],
    use_min=info_dict['use_min'],
    use_max=info_dict['use_max'],
    use_weighted_avg=info_dict['use_weighted_avg'],
    use_weighted_var=info_dict['use_weighted_var'],
    use_ratio=info_dict['use_ratio'],
    use_delta=info_dict['use_delta'],
)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(np.array(train_x))

# Save raw SHAP values
raw_shap_dict = {f"{prop}_{stat}": shap_values[:, i * stats_count + j] 
                 for i, prop in enumerate(info_dict['prop_list']) 
                 for j, stat in enumerate(stats_labels)}
pd.DataFrame(raw_shap_dict).to_csv(shap_raw_csv_file, index=False)
print(f"Raw SHAP values saved to: {shap_raw_csv_file}")

# Aggregate SHAP values by property and statistic
aggregated_dict = {prop: [np.mean(shap_values[:, i * stats_count + j]) 
                          for j in range(stats_count)]
                   for i, prop in enumerate(info_dict['prop_list'])}
agg_df = pd.DataFrame(aggregated_dict, index=stats_labels).T.reset_index()
agg_df.rename(columns={"index": "Property"}, inplace=True)
agg_df.to_csv(shap_aggregated_csv_file, index=False)
print(f"Aggregated SHAP feature importance saved to: {shap_aggregated_csv_file}")

# Plot summary bar chart
agg_df["Mean SHAP"] = agg_df[stats_labels].mean(axis=1)
agg_df_sorted = agg_df.sort_values(by="Mean SHAP", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(y=agg_df_sorted["Property"], x=agg_df_sorted["Mean SHAP"], palette="RdBu", edgecolor="black")
plt.xlabel("Mean SHAP Value")
plt.ylabel("Element Property")
plt.title("SHAP Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(shap_importance_plot, dpi=300)
plt.close()
print(f"SHAP bar plot saved to {shap_importance_plot}")

# ===== Detailed Marker Plot =====
property_names = np.repeat(agg_df["Property"].values, stats_count)
shap_vals = np.concatenate([agg_df[stat].values for stat in stats_labels])
stat_labels = [s for s in stats_labels for _ in range(len(agg_df))]
x_positions = np.tile(np.arange(len(agg_df)), stats_count)

# Marker styles
marker_styles = {
    "Avg": "o", "Var": "s", "Max": "^", "Min": "d",
    "wAvg": "P", "wVar": "X", "Ratio": "*", "Delta": "v"
}

norm = plt.Normalize(vmin=-max(abs(shap_vals)), vmax=max(abs(shap_vals)))
fig, ax = plt.subplots(figsize=(14, 8))

for stat in stats_labels:
    mask = np.array(stat_labels) == stat
    ax.scatter(
        x_positions[mask], shap_vals[mask],
        c=shap_vals[mask], cmap='RdBu_r', norm=norm,
        s=100, edgecolors="black", alpha=0.8,
        marker=marker_styles.get(stat, "o"), label=stat
    )

fig.colorbar(ax.collections[0], ax=ax, label="SHAP Value Intensity")
ax.set_xlabel("Element Properties")
ax.set_ylabel("SHAP Value")
ax.set_title("Detailed SHAP Feature Importance")
ax.set_xticks(np.arange(len(agg_df)))
ax.set_xticklabels(agg_df["Property"].values, rotation=45, ha="right")
ax.axhline(0, color="black", linestyle="--")
ax.legend(title="SHAP Statistic")
ax.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(shap_detailed_plot, dpi=300)
plt.close()
print(f"Detailed SHAP feature plot saved to {shap_detailed_plot}")
