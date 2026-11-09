import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import itertools

# ================= Configuration ================= #
old_training_path = "Output/Training/2025-07-23/EFA/test_results.csv"
new_training_path = "Output/Training/2026-03-19/EFA/test_results.csv"
inference_path = "Output/Validation/2026-03-19/17-50-29/predictions.csv"

# Output directory setup
timestamp = datetime.now().strftime("%Y-%m-%d")
output_dir = os.path.join("Output/visual", timestamp)
os.makedirs(output_dir, exist_ok=True)

# Styling
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 26,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 30,
    'font.family': 'sans-serif'
})
sns.set_style("whitegrid", {'grid.linestyle': '--'})
colors = ["#2A4E6E", "#E63946", "#457B9D", "#A8DADC", "#F1FAEE"]

# ================= Helper Functions ================= #
def get_elements_from_row(row, columns):
    return [str(row[col]).strip() for col in columns if pd.notnull(row[col]) and str(row[col]).strip() != ""]

def extract_element_counts(df):
    elem_cols = [col for col in df.columns if 'Element' in col]
    all_elements = []
    for _, row in df.iterrows():
        all_elements.extend(get_elements_from_row(row, elem_cols))
    return pd.Series(all_elements).value_counts()

def generate_cooccurrence_matrix(df, elem_cols, min_freq=0):
    all_pairs = []
    unique_elements = set()
    
    for _, row in df.iterrows():
        elements = sorted(get_elements_from_row(row, elem_cols))
        unique_elements.update(elements)
        if len(elements) >= 2:
            all_pairs.extend(list(itertools.combinations(elements, 2)))
    
    unique_elements = sorted(list(unique_elements))
    matrix = pd.DataFrame(0, index=unique_elements, columns=unique_elements)
    
    for p1, p2 in all_pairs:
        matrix.loc[p1, p2] += 1
        matrix.loc[p2, p1] += 1
    
    # Filter elements that are rarely used if needed
    if min_freq > 0:
        freq = matrix.sum()
        keep = freq[freq >= min_freq].index
        matrix = matrix.loc[keep, keep]
        
    return matrix

# ================= Load Data ================= #
df_old = pd.read_csv(old_training_path)
df_new = pd.read_csv(new_training_path)
df_inf = pd.read_csv(inference_path)

train_elem_cols = [col for col in df_new.columns if 'Element' in col]
inf_elem_cols = [col for col in df_inf.columns if 'Element' in col]

# ================= Task 1: Element Increment Analysis ================= #
print("Analyzing Task 1: Element Increment...")
counts_old = extract_element_counts(df_old)
counts_new = extract_element_counts(df_new)

comparison_df = pd.DataFrame({
    'Old': counts_old,
    'New': counts_new
}).fillna(0)
comparison_df['Increment'] = comparison_df['New'] - comparison_df['Old']
comparison_df = comparison_df.sort_values(by='New', ascending=False)

# Plot Enhanced Task 1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 16))
top_inc = comparison_df.sort_values(by='Increment', ascending=False).head(20)
sns.barplot(x=top_inc.index, y=top_inc['Increment'], ax=ax1, palette="magma", hue=top_inc.index, legend=False)
ax1.set_title('Top 20 Elements by Absolute Increment (July 2025 -> March 2026)')
ax1.set_ylabel('Incremental Count')

# Average EFA per element
elem_efa = []
for _, row in df_new.iterrows():
    elems = get_elements_from_row(row, train_elem_cols)
    efa = row['True']
    for e in elems:
        elem_efa.append({'Element': e, 'EFA': efa})
avg_efa_df = pd.DataFrame(elem_efa).groupby('Element')['EFA'].mean()
comparison_df['Avg_EFA'] = avg_efa_df

top_efa = comparison_df.sort_values(by='Avg_EFA', ascending=False).head(20)
sns.barplot(x=top_efa.index, y=top_efa['Avg_EFA'], ax=ax2, palette="coolwarm", hue=top_efa.index, legend=False)
ax2.set_title('Top 20 Elements by Average Calculated EFA (New Dataset)')
ax2.set_ylabel('Average EFA (eV/atom)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'task1_enhanced_analysis.png'), dpi=300)
plt.close()

# ================= Task 2: Heatmaps (Co-occurrence) ================= #
print("Analyzing Task 2: Heatmaps...")

# 1. Training Co-occurrence Heatmap
plt.figure(figsize=(20, 18))
train_matrix = generate_cooccurrence_matrix(df_new, train_elem_cols)
# Filter for better visualization if too many elements
if len(train_matrix) > 40:
    top_elements = extract_element_counts(df_new).head(30).index
    train_matrix = train_matrix.loc[top_elements, top_elements]

sns.heatmap(train_matrix, cmap="YlGnBu", annot=False, cbar_kws={'label': 'Number of Co-occurrences'})
plt.title('Training Data: Element Co-occurrence Heatmap (Top 30 Elements)', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'task2_training_heatmap.png'), dpi=300)
plt.close()

# 2. High EFA Inference Co-occurrence Heatmap
threshold = df_inf['ML_Predicted'].quantile(0.90)
df_high = df_inf[df_inf['ML_Predicted'] >= threshold].copy()

plt.figure(figsize=(20, 18))
high_matrix = generate_cooccurrence_matrix(df_high, inf_elem_cols)
if len(high_matrix) > 30:
    top_high_elements = extract_element_counts(df_high).head(25).index
    high_matrix = high_matrix.loc[top_high_elements, top_high_elements]

sns.heatmap(high_matrix, cmap="Reds", annot=False, cbar_kws={'label': 'Frequency in High EFA Candidates'})
plt.title(f'High EFA (Top 10%) Inference: Element Co-occurrence (Top 25 Elements)', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'task2_high_efa_heatmap.png'), dpi=300)
plt.close()

# ================= Task 3: Sr, Al, Sn Separate Analysis ================= #
print("Analyzing Task 3: Specific Elements (Sr, Al, Sn) Separately...")
target_elements = ['Sr', 'Al', 'Sn']
fig, axes = plt.subplots(len(target_elements), 2, figsize=(20, 6 * len(target_elements)))

for idx, target in enumerate(target_elements):
    df_new[f'has_{target}'] = df_new.apply(lambda r: target in get_elements_from_row(r, train_elem_cols), axis=1)
    df_inf[f'has_{target}'] = df_inf.apply(lambda r: target in get_elements_from_row(r, inf_elem_cols), axis=1)
    
    sns.kdeplot(df_new['True'], fill=True, ax=axes[idx, 0], label='Overall Calc.', color='gray', alpha=0.3)
    if df_new[f'has_{target}'].any():
        sns.kdeplot(df_new[df_new[f'has_{target}']]['True'], fill=True, ax=axes[idx, 0], label=f'With {target}', color='#E63946', alpha=0.6)
    axes[idx, 0].set_title(f'{target} in Calculated (Training)')
    axes[idx, 0].legend()

    sns.kdeplot(df_inf['ML_Predicted'], fill=True, ax=axes[idx, 1], label='Overall Extrap.', color='gray', alpha=0.3)
    if df_inf[f'has_{target}'].any():
        sns.kdeplot(df_inf[df_inf[f'has_{target}']]['ML_Predicted'], fill=True, ax=axes[idx, 1], label=f'With {target}', color='#457B9D', alpha=0.6)
    axes[idx, 1].set_title(f'{target} in Extrapolated (Inference)')
    axes[idx, 1].legend()

plt.suptitle('Individual Element Distribution Analysis', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'task3_separate_elements_dist.png'), dpi=300)
plt.close()

# ================= Task 4: High EFA Extrapolation & Basis Analysis ================= #
print("Analyzing Task 4: High EFA and Basis Analysis...")
high_elem_counts = extract_element_counts(df_high)
train_counts = extract_element_counts(df_new)
basis_df = pd.DataFrame({'Inference_Frequency': high_elem_counts, 'Training_Basis': train_counts}).fillna(0)

plt.figure(figsize=(15, 10))
top_high = basis_df.sort_values(by='Inference_Frequency', ascending=False).head(20)
sns.barplot(x=top_high.index, y=top_high['Inference_Frequency'], palette="viridis", hue=top_high.index, legend=False)
plt.title(f'Element Frequency in Top 10% EFA Predictions (EFA > {threshold:.2f})', pad=20)
plt.ylabel('Frequency in High EFA Samples')

for i, elem in enumerate(top_high.index):
    basis = top_high.loc[elem, 'Training_Basis']
    if basis == 0:
        plt.text(i, top_high.loc[elem, 'Inference_Frequency'], 'NO BASIS', color='red', fontweight='bold', ha='center', va='bottom', fontsize=12, rotation=45)
    elif basis < 10:
        plt.text(i, top_high.loc[elem, 'Inference_Frequency'], f'LOW({int(basis)})', color='orange', fontweight='bold', ha='center', va='bottom', fontsize=10, rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'task4_high_efa_basis.png'), dpi=300)
plt.close()

print(f"Visualization completed. Files saved in {output_dir}")
