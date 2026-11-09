import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import sys

# ===== Setup Project Path =====
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from mat_encod.data_util import ele_df, json_loader

# ========== USER INPUT ==========
model_dir = "Output/Training/2026-03-19/17-33-55"   # Folder containing training_info.json, model, etc.
json_val_path = "Data/comp_pool/generated_compositions.json"  # Path to the new JSON file for validation
output_base = "Output/Validation"
# ================================

# ========== Load training_info.json ==========
info_path = os.path.join(model_dir, "training_info.json")
with open(info_path, "r") as f:
    info_dict = json.load(f)

# ========== Output Paths ==========
timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
output_dir = os.path.join(output_base, timestamp)
os.makedirs(output_dir, exist_ok=True)

csv_output_path = os.path.join(output_dir, "predictions.csv")
dist_plot_path = os.path.join(output_dir, "distribution_plot.png")
best_model_file = os.path.join(output_dir, "best_model.pkl")

# ========== Load Element Properties ==========
ele_DF = ele_df(prop_list=info_dict['prop_list'],
                total_e_list=info_dict['unique_elements'],
                normalize=info_dict['normalize'])

# ========== Load Training Data ==========
# Load high-fidelity data only
train_loader = json_loader(info_dict['json_path'], df_data=ele_DF)
train_x, train_y, train_elements = train_loader.load_data_nocomp(
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

# ========== Load Best Parameters ==========
best_param_path = os.path.join(model_dir, "best_params.txt")
best_params = {}
with open(best_param_path, "r") as f:
    lines = f.readlines()
    # Skip header and separator lines
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "=====================":
            start_idx = i + 2
            break
    
    # Read parameters
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        if line == "Model Performance:" or line == "=================":
            break
        key, value = line.split(": ")
        # Only process model parameters (ignore performance metrics)
        if key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
            if value == "None":
                best_params[key] = None
            elif value.isdigit():
                best_params[key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                best_params[key] = float(value)
            else:
                best_params[key] = value
print(f"Loaded best parameters from {best_param_path}")

# ========== Train Model ==========
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(train_x, train_y)

# Get predictions for training data
train_predictions = model.predict(train_x)

# Save training predictions to CSV
train_csv_path = os.path.join(output_dir, "train_predictions.csv")
train_df = pd.DataFrame({
    "True_Value": train_y,
    "ML_Predicted": train_predictions,
    "Composition": ["-".join(e) for e in train_elements]
})

# Add individual elements as columns
train_ele_length = len(train_elements[0])
for i in range(train_ele_length):
    train_df[f"Element_{i+1}"] = [e[i] if len(e) > i else None for e in train_elements]

# Save training results
train_df.to_csv(train_csv_path, index=False)
print(f"Training predictions saved to {train_csv_path}")

# Plot training predictions
train_plot_path = os.path.join(output_dir, "train_predictions_plot.png")
plt.figure(figsize=(10, 10))
plt.scatter(train_y, train_predictions, alpha=0.5, color='blue')
plt.plot([min(train_y), max(train_y)], [min(train_y), max(train_y)], 'r--', label='y=x')

# Add regression line
z = np.polyfit(train_y, train_predictions, 1)
p = np.poly1d(z)
plt.plot(train_y, p(train_y), 'g-', label='Fit')

plt.xlabel(info_dict['xlabel'])
plt.ylabel(info_dict['ylabel'])
plt.title('Training Data: True vs Predicted Values')
plt.legend()
plt.grid(True, alpha=0.3)

# Calculate and display metrics
train_mae = mean_absolute_error(train_y, train_predictions)
train_r2 = r2_score(train_y, train_predictions)
plt.text(0.05, 0.95, f'MAE: {train_mae:.3f}\nR²: {train_r2:.3f}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(train_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Training predictions plot saved to {train_plot_path}")

# Save model
with open(best_model_file, "wb") as f:
    pickle.dump(model, f)
print(f"Model trained and saved to {best_model_file}")

# ========== Load Validation Data ==========
with open(json_val_path, 'r') as f:
    ele_j_data = json.load(f)

# 只用 Elements 字段
if 'Elements' in ele_j_data[0]:
    element_field = 'Elements'
else:
    raise ValueError("No Elements field found in JSON.")

ele_length = len(ele_j_data[0][element_field])
val_elements = [entry[element_field] for entry in ele_j_data]
val_chemid = [entry.get('chem_id', None) for entry in ele_j_data]
val_DFT = [entry.get('DFT', None) for entry in ele_j_data]
val_EXP = [entry.get('EXP', None) for entry in ele_j_data]
val_single_phase = [entry.get('single_phase', None) for entry in ele_j_data]

# 重新生成 ele_DF
unique_elements = list(set([item for entry in ele_j_data for item in entry[element_field]]))
ele_DF = ele_df(prop_list=info_dict['prop_list'],
                total_e_list=unique_elements,
                normalize=info_dict['normalize'])
val_loader = json_loader(json_val_path, df_data=ele_DF)

val_x, _ = val_loader.load_data_nocomp(
    with_target=False,
    compositions=None,
    use_avg=info_dict['use_avg'],
    use_var=info_dict['use_var'],
    use_min=info_dict['use_min'],
    use_max=info_dict['use_max'],
    use_weighted_avg=info_dict['use_weighted_avg'],
    use_weighted_var=info_dict['use_weighted_var'],
    use_ratio=info_dict['use_ratio'],
    use_delta=info_dict['use_delta'],
)

# ========== Make Predictions ==========
predictions = model.predict(val_x)

# ========== Save Results to CSV ==========
df = pd.DataFrame(predictions, columns=["ML_Predicted"])
for i in range(ele_length):
    df[f"Element_{i+1}"] = [e[i] if len(e) > i else None for e in val_elements]

# 新增三列：chem_id, DFT, EXP
df["chem_id"] = val_chemid
df["DFT"] = val_DFT
df["EXP"] = val_EXP
df["single_phase"] = val_single_phase

df.to_csv(csv_output_path, index=False)
print(f"Predictions saved to {csv_output_path}")

