import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import sys

# ===== Setup Project Path =====
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from mat_encod.data_util import ele_df, json_loader, p_ele_encod

# ========== USER INPUT ==========
exp_json_paths = [
    "Data/exp_truth/lib5_exp_truth.json",
    "Data/exp_truth/lib6_exp_truth.json",
]
model_dir = "Output/Training/2026-05-13/13-55-28" # Updated to latest RF model
json_val_path = "Data/comp_pool/generated_compositions.json"
output_base = model_dir + "/Modelupdate_exp_feedback"
REWARD_YES = 0.5  # single_phase=Yes reward
REWARD_NO = 2.0   # single_phase=No reward

# ========== Load training_info.json ==========
info_path = os.path.join(model_dir, "training_info.json")
with open(info_path, "r") as f:
    info_dict = json.load(f)

# ========== Collect All Elements ==========
all_elements = set(info_dict['unique_elements'])
exp_entries = []
for exp_path in exp_json_paths:
    if not os.path.exists(exp_path):
        continue
    with open(exp_path, 'r') as f:
        data = json.load(f)
        exp_entries.extend(data)
        for entry in data:
            all_elements.update(entry['Elements'])

# ========== Load Element Properties ==========
ele_DF = ele_df(prop_list=info_dict['prop_list'], total_e_list=list(all_elements), normalize=info_dict['normalize'])

# ========== Prepare exp_truth Features ==========
class InMemoryJsonLoader(json_loader):
    def __init__(self, entries, df_data):
        self.T_data = entries
        self.p_encod = p_ele_encod(df_data)

exp_loader = InMemoryJsonLoader(exp_entries, ele_DF)
exp_x, exp_elements = exp_loader.load_data_nocomp(
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

# ========== Load Original Model ==========
with open(os.path.join(model_dir, "best_model.pkl"), "rb") as f:
    original_model = pickle.load(f)

# ========== Predict and Apply Reward ==========
exp_pred = original_model.predict(exp_x)
exp_single_phase = [entry['single_phase'] for entry in exp_entries]
exp_chem_id = [entry.get('chem_id', None) for entry in exp_entries]
exp_elements_str = ["-".join(entry['Elements']) for entry in exp_entries]
exp_composition = [entry.get('Composition', None) for entry in exp_entries]
exp_rewarded_y = np.array([
    pred * REWARD_YES if str(phase).lower() == "yes" else pred * REWARD_NO
    for pred, phase in zip(exp_pred, exp_single_phase)
])

# ========== Save exp_truth predictions and rewards to CSV ==========
exp_pred_df = pd.DataFrame({
    "chem_id": exp_chem_id,
    "Elements": exp_elements_str,
    "single_phase": exp_single_phase,
    "model_pred": exp_pred,
    "model_pred_rewarded": exp_rewarded_y,
    "Composition": exp_composition
})

# ========== Load Best Parameters ==========
best_param_path = os.path.join(model_dir, "best_params.txt")
best_params = {}
if os.path.exists(best_param_path):
    with open(best_param_path, "r") as f:
        lines = f.readlines()
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == "=====================":
                start_idx = i + 2
                break
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line == "Model Performance:" or line == "=================":
                break
            if ": " in line:
                key, value = line.split(": ")
                if key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                    if value == "None":
                        best_params[key] = None
                    elif value.replace('.', '', 1).isdigit():
                        best_params[key] = float(value) if '.' in value else int(value)
                    else:
                        best_params[key] = value

# ========== Retrain Model with Feedback ==========
# Note: Random Forest doesn't support incremental fitting like XGBoost.
# We retrain a new model on the rewarded experimental data.
updated_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
updated_model.fit(exp_x, exp_rewarded_y)

# ========== Save Model ==========
timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
output_dir = os.path.join(output_base, timestamp)
os.makedirs(output_dir, exist_ok=True)
updated_model_file = os.path.join(output_dir, "updated_model.pkl")
with open(updated_model_file, "wb") as f:
    pickle.dump(updated_model, f)

# Save exp_pred_df to CSV
exp_pred_csv_path = os.path.join(output_dir, "exp_truth_pred_and_rewarded.csv")
exp_pred_df.to_csv(exp_pred_csv_path, index=False)
print(f"exp_truth predictions and rewards saved to {exp_pred_csv_path}")

# ========== Visualization: scatter plot ==========
plt.figure(figsize=(8,6))
for label, color in [("Single Phase", "#1f77b4"), ("Not Single Phase", "#ff7f0e")]:
    idx = [i for i, phase in enumerate(exp_single_phase) if str(phase).lower() == ("yes" if label == "Single Phase" else "no")]
    if idx:
        plt.scatter(np.array(exp_pred)[idx], np.array(exp_rewarded_y)[idx], c=color, label=label, s=60, alpha=0.8, edgecolors='k')
plt.xlabel("Original Model Prediction")
plt.ylabel("Rewarded/penalized Value")
plt.title("exp_truth: Model Prediction vs Rewarded Value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
scatter_path = os.path.join(output_dir, "exp_truth_rewarded_scatter.png")
plt.savefig(scatter_path, dpi=200)
plt.close()
print(f"exp_truth reward scatter plot saved to {scatter_path}")

# ========== External Validation ==========
with open(json_val_path, 'r') as f:
    val_j_data = json.load(f)
val_elements = [entry['Elements'] for entry in val_j_data]
val_chemid = [entry.get('chem_id', None) for entry in val_j_data]
val_DFT = [entry.get('DFT', None) for entry in val_j_data]
val_EXP = [entry.get('EXP', None) for entry in val_j_data]
val_ele_length = len(val_elements[0])
unique_elements_val = list(set([item for entry in val_j_data for item in entry['Elements']]))
ele_DF_val = ele_df(prop_list=info_dict['prop_list'], total_e_list=unique_elements_val, normalize=info_dict['normalize'])
val_loader = json_loader(json_val_path, df_data=ele_DF_val)
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
val_predictions_updated = updated_model.predict(val_x)
val_csv_path = os.path.join(output_dir, "val_predictions_updated.csv")
val_df = pd.DataFrame({
    "Updated_Model_Predicted": val_predictions_updated
})
for i in range(val_ele_length):
    val_df[f"Element_{i+1}"] = [e[i] if len(e) > i else None for e in val_elements]
val_df["chem_id"] = val_chemid
val_df["DFT"] = val_DFT
val_df["EXP"] = val_EXP
val_single_phase_val = [entry.get('single_phase', None) for entry in val_j_data]
val_df["single_phase"] = val_single_phase_val
val_df.to_csv(val_csv_path, index=False)
print(f"Validation predictions saved to {val_csv_path}")
