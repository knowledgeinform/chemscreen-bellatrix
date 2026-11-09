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

from mat_encod.data_util import ele_df, json_loader

# ========== USER INPUT ==========
model_dir = "Output/Training/2025-07-22/18-18-06"  # Folder containing training_info.json, model, etc.
new_data_path = "Data/lib6_organized_072125.json"  # Path to the new data for model update
json_val_path = "Data/comp_pool/generated_compositions.json"  # Path to the validation JSON file
output_base = model_dir+"/Modelupdate5-6"

# ========== WEIGHTED LEARNING CONFIGURATION ==========
NEW_DATA_WEIGHT = 3  # How many times to repeat new data (weight factor)
# ======================================================

# ========== Load training_info.json ==========
info_path = os.path.join(model_dir, "training_info.json")
with open(info_path, "r") as f:
    info_dict = json.load(f)

# ========== Output Paths ==========
timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
output_dir = os.path.join(output_base, timestamp)
os.makedirs(output_dir, exist_ok=True)

updated_model_file = os.path.join(output_dir, "weighted_model.pkl")

# ========== Collect All Elements from All Data Sources ==========
# First, collect elements from training data
train_elements_set = set(info_dict['unique_elements'])

# Collect elements from new data
print(f"Loading new data from {new_data_path}")
with open(new_data_path, 'r') as f:
    new_j_data = json.load(f)
new_elements_set = set()
for entry in new_j_data:
    if 'Elements' in entry:
        new_elements_set.update(entry['Elements'])

# Collect elements from validation data
with open(json_val_path, 'r') as f:
    val_j_data = json.load(f)
val_elements_set = set()
for entry in val_j_data:
    if 'Elements' in entry:
        val_elements_set.update(entry['Elements'])

# Combine all elements
all_elements = list(train_elements_set | new_elements_set | val_elements_set)
print(f"Total unique elements across all datasets: {len(all_elements)}")

# ========== Load Element Properties ==========
ele_DF = ele_df(prop_list=info_dict['prop_list'],
                total_e_list=all_elements,
                normalize=info_dict['normalize'])

# ========== Load Original Training Data ==========
print("Loading original training data...")
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

# ========== Data Analysis ==========
print(f"\n=== Data Distribution Analysis ===")
train_y_array = np.array(train_y)
print(f"Training data: {len(train_y)} samples")
print(f"Training target range: {train_y_array.min():.3f} to {train_y_array.max():.3f}")
print(f"Training target mean: {train_y_array.mean():.3f}, std: {train_y_array.std():.3f}")

# ========== Load New Data ==========
print("Loading new data...")
new_loader = json_loader(new_data_path, df_data=ele_DF)
new_x, new_y, new_elements = new_loader.load_data_nocomp(
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

new_y_array = np.array(new_y)
print(f"New data: {len(new_y)} samples")
print(f"New target range: {new_y_array.min():.3f} to {new_y_array.max():.3f}")
print(f"New target mean: {new_y_array.mean():.3f}, std: {new_y_array.std():.3f}")

# ========== Train Original Model ==========
print("\n=== Training Original Model ===")
original_model = RandomForestRegressor(**best_params, random_state=42)
original_model.fit(train_x, train_y)
print("Original model trained successfully")

# ========== Evaluate Original Model on New Data ==========
print(f"\n=== Evaluating Original Model on New Data ===")
new_original_predictions = original_model.predict(new_x)
new_original_mae = mean_absolute_error(new_y, new_original_predictions)
new_original_r2 = r2_score(new_y, new_original_predictions)
new_original_rmse = np.sqrt(mean_squared_error(new_y, new_original_predictions))

print(f"Original Model on New Data - MAE: {new_original_mae:.3f}, R²: {new_original_r2:.3f}, RMSE: {new_original_rmse:.3f}")

# ========== Train Weighted Model ==========
print(f"\n=== Training Weighted Model (new data weighted {NEW_DATA_WEIGHT}x) ===")
weighted_x = np.vstack([train_x] + [new_x] * NEW_DATA_WEIGHT)
weighted_y = np.concatenate([train_y] + [new_y] * NEW_DATA_WEIGHT)

weighted_model = RandomForestRegressor(**best_params, random_state=42)
weighted_model.fit(weighted_x, weighted_y)
print("Weighted model trained successfully")

# ========== Evaluate Weighted Model on New Data ==========
print(f"\n=== Evaluating Weighted Model on New Data ===")
new_weighted_predictions = weighted_model.predict(new_x)
new_weighted_mae = mean_absolute_error(new_y, new_weighted_predictions)
new_weighted_r2 = r2_score(new_y, new_weighted_predictions)
new_weighted_rmse = np.sqrt(mean_squared_error(new_y, new_weighted_predictions))

print(f"Weighted Model on New Data - MAE: {new_weighted_mae:.3f}, R²: {new_weighted_r2:.3f}, RMSE: {new_weighted_rmse:.3f}")

# ========== Save Results ==========
# Save new data predictions to CSV
new_data_csv_path = os.path.join(output_dir, "weighted_predictions.csv")
new_data_df = pd.DataFrame({
    "True_Value": new_y,
    "Original_Model_Predicted": new_original_predictions,
    "Weighted_Model_Predicted": new_weighted_predictions
})

# Add element columns
new_ele_length = len(new_elements[0]) if new_elements else 0
for i in range(new_ele_length):
    new_data_df[f"Element_{i+1}"] = [e[i] if len(e) > i else None for e in new_elements]

# Add composition if available
if 'composition' in new_j_data[0]:
    new_data_df["Composition"] = [entry.get('composition', None) for entry in new_j_data]
elif 'Composition' in new_j_data[0]:
    new_data_df["Composition"] = [entry.get('Composition', None) for entry in new_j_data]

# When saving new data predictions to CSV
# Add after extracting composition if available
if 'single_phase' in new_j_data[0]:
    new_data_df["single_phase"] = [entry.get('single_phase', None) for entry in new_j_data]

new_data_df.to_csv(new_data_csv_path, index=False)
print(f"Weighted predictions saved to {new_data_csv_path}")

# Save performance comparison
performance_path = os.path.join(output_dir, "model_comparison.txt")
with open(performance_path, 'w') as f:
    f.write("=== Model Comparison ===\n\n")
    f.write(f"Configuration:\n")
    f.write(f"  NEW_DATA_WEIGHT: {NEW_DATA_WEIGHT}\n\n")
    
    f.write("Original Model:\n")
    f.write(f"  MAE: {new_original_mae:.6f}\n")
    f.write(f"  R²: {new_original_r2:.6f}\n")
    f.write(f"  RMSE: {new_original_rmse:.6f}\n\n")
    
    f.write("Weighted Model:\n")
    f.write(f"  MAE: {new_weighted_mae:.6f}\n")
    f.write(f"  R²: {new_weighted_r2:.6f}\n")
    f.write(f"  RMSE: {new_weighted_rmse:.6f}\n\n")
    
    f.write("Improvement:\n")
    f.write(f"  MAE improvement: {new_original_mae - new_weighted_mae:.6f} ({((new_original_mae - new_weighted_mae) / new_original_mae * 100):.2f}%)\n")
    f.write(f"  R² improvement: {new_weighted_r2 - new_original_r2:.6f} ({((new_weighted_r2 - new_original_r2) / abs(new_original_r2) * 100):.2f}%)\n")
    f.write(f"  RMSE improvement: {new_original_rmse - new_weighted_rmse:.6f} ({((new_original_rmse - new_weighted_rmse) / new_original_rmse * 100):.2f}%)\n")

print(f"Model comparison saved to {performance_path}")

# Create visualization
plot_path = os.path.join(output_dir, "model_comparison_plot.png")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Original vs True
axes[0, 0].scatter(new_y, new_original_predictions, alpha=0.7, color='red')
axes[0, 0].plot([min(new_y), max(new_y)], [min(new_y), max(new_y)], 'k--', label='y=x')
axes[0, 0].set_xlabel('True Values')
axes[0, 0].set_ylabel('Original Model Predictions')
axes[0, 0].set_title(f'Original Model\nMAE: {new_original_mae:.3f}, R²: {new_original_r2:.3f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Weighted vs True
axes[0, 1].scatter(new_y, new_weighted_predictions, alpha=0.7, color='green')
axes[0, 1].plot([min(new_y), max(new_y)], [min(new_y), max(new_y)], 'k--', label='y=x')
axes[0, 1].set_xlabel('True Values')
axes[0, 1].set_ylabel('Weighted Model Predictions')
axes[0, 1].set_title(f'Weighted Model\nMAE: {new_weighted_mae:.3f}, R²: {new_weighted_r2:.3f}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Original vs Weighted predictions
axes[1, 0].scatter(new_original_predictions, new_weighted_predictions, alpha=0.7, color='purple')
axes[1, 0].plot([min(new_original_predictions), max(new_original_predictions)], 
                [min(new_original_predictions), max(new_original_predictions)], 'k--', label='y=x')
axes[1, 0].set_xlabel('Original Model Predictions')
axes[1, 0].set_ylabel('Weighted Model Predictions')
axes[1, 0].set_title('Original vs Weighted Model Predictions')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Error comparison
original_errors = np.abs(new_y - new_original_predictions)
weighted_errors = np.abs(new_y - new_weighted_predictions)
axes[1, 1].hist(original_errors, bins=20, alpha=0.7, color='red', label=f'Original (MAE: {new_original_mae:.3f})')
axes[1, 1].hist(weighted_errors, bins=20, alpha=0.7, color='green', label=f'Weighted (MAE: {new_weighted_mae:.3f})')
axes[1, 1].set_xlabel('Absolute Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Error Distribution Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Model comparison plot saved to {plot_path}")

# Save weighted model
with open(updated_model_file, "wb") as f:
    pickle.dump(weighted_model, f)
print(f"Weighted model saved to {updated_model_file}")

# ========== Load Model and Make Predictions Function ==========
def load_model_and_predict(model_path, new_data_path, output_path=None):
    """
    Load the trained model and make predictions on new data
    
    Parameters:
    - model_path: Path to the saved model (.pkl file)
    - new_data_path: Path to new data JSON file
    - output_path: Optional path to save predictions CSV
    """
    print(f"\n=== Loading Model and Making Predictions ===")
    
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    
    # Load new data
    with open(new_data_path, 'r') as f:
        new_data = json.load(f)
    
    # Prepare data using the same preprocessing
    new_loader = json_loader(new_data_path, df_data=ele_DF)
    new_x, new_y, new_elements = new_loader.load_data_nocomp(
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
    
    # Make predictions
    predictions = model.predict(new_x)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "True_Value": new_y,
        "Predicted_Value": predictions
    })
    
    # Add element columns
    new_ele_length = len(new_elements[0]) if new_elements else 0
    for i in range(new_ele_length):
        results_df[f"Element_{i+1}"] = [e[i] if len(e) > i else None for e in new_elements]
    
    # Add composition if available
    if 'composition' in new_data[0]:
        results_df["Composition"] = [entry.get('composition', None) for entry in new_data]
    elif 'Composition' in new_data[0]:
        results_df["Composition"] = [entry.get('Composition', None) for entry in new_data]
    
    # Calculate metrics
    mae = mean_absolute_error(new_y, predictions)
    r2 = r2_score(new_y, predictions)
    rmse = np.sqrt(mean_squared_error(new_y, predictions))
    
    print(f"Predictions made on {len(new_y)} samples")
    print(f"MAE: {mae:.3f}, R²: {r2:.3f}, RMSE: {rmse:.3f}")
    
    # Save results if output path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    return results_df, mae, r2, rmse

# ========== Demonstrate Model Loading and Prediction ==========
print(f"\n=== Demonstrating Model Loading and Prediction ===")
# This shows how to load the model and make predictions on new data
# demo_output_path = os.path.join(output_dir, "demo_predictions.csv")
# demo_results, demo_mae, demo_r2, demo_rmse = load_model_and_predict(
#     updated_model_file, 
#     new_data_path, 
#     demo_output_path
# )

print(f"\nAll results saved to: {output_dir}")
print(f"\n=== Summary ===")
print(f"Weighted model performance: MAE: {new_weighted_mae:.3f}, R²: {new_weighted_r2:.3f}")
print(f"Improvement: MAE reduced by {((new_original_mae - new_weighted_mae) / new_original_mae * 100):.1f}%")
print(f"Model saved to: {updated_model_file}")
print(f"To use the model later, call: load_model_and_predict('{updated_model_file}', 'your_new_data.json')") 

# ========== Predict on Validation Data and Save CSV ==========
print(f"\n=== Predicting on Validation Data and Saving CSV ===")
with open(json_val_path, 'r') as f:
    val_j_data = json.load(f)

if 'Elements' in val_j_data[0]:
    element_field = 'Elements'
else:
    raise ValueError("No Elements field found in JSON.")

val_elements = [entry[element_field] for entry in val_j_data]
val_chemid = [entry.get('chem_id', None) for entry in val_j_data]
val_DFT = [entry.get('DFT', None) for entry in val_j_data]
val_EXP = [entry.get('EXP', None) for entry in val_j_data]
val_ele_length = len(val_elements[0])

# 重新生成 ele_DF，确保所有元素都在
unique_elements = list(set([item for entry in val_j_data for item in entry[element_field]]))
ele_DF_val = ele_df(prop_list=info_dict['prop_list'],
                    total_e_list=unique_elements,
                    normalize=info_dict['normalize'])
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

val_predictions_weighted = weighted_model.predict(val_x)
val_predictions_original = original_model.predict(val_x)

val_csv_path = os.path.join(output_dir, "val_weighted_predictions.csv")
val_df = pd.DataFrame({
    "Original_Model_Predicted": val_predictions_original,
    "Weighted_Model_Predicted": val_predictions_weighted
})
for i in range(val_ele_length):
    val_df[f"Element_{i+1}"] = [e[i] if len(e) > i else None for e in val_elements]
val_df["chem_id"] = val_chemid
val_df["DFT"] = val_DFT
val_df["EXP"] = val_EXP
# When saving validation predictions to CSV
val_single_phase = [entry.get('single_phase', None) for entry in val_j_data]
val_df["single_phase"] = val_single_phase
val_df.to_csv(val_csv_path, index=False)
print(f"Validation predictions saved to {val_csv_path}") 