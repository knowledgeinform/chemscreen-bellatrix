import os
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import sys

# ===== Setup Project Path =====
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from mat_encod.data_util import ele_df, json_loader

def run_pipeline(model_dir, input_json_path, output_base="Output/Application", top_n=20):
    """
    1. Load compositions from JSON
    2. Deduplicate
    3. Predict using trained model
    4. Recommend top N candidates
    """
    # ========== 1. Load and Deduplicate ==========
    print(f"Loading compositions from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Deduplicate based on chem_id (if available) or sorted elements
    unique_data = {}
    for entry in data:
        if 'chem_id' in entry:
            cid = entry['chem_id']
        else:
            cid = "-".join(sorted(entry['Elements']))
        
        if cid not in unique_data:
            unique_data[cid] = entry
    
    deduped_data = list(unique_data.values())
    print(f"Original: {len(data)}, Unique: {len(deduped_data)}")

    # ========== 2. Load Model and Info ==========
    info_path = os.path.join(model_dir, "training_info.json")
    with open(info_path, "r") as f:
        info_dict = json.load(f)
    
    model_path = os.path.join(model_dir, "best_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # ========== 3. Prepare Features and Predict ==========
    print("Preparing features and making predictions...")
    element_field = 'Elements'
    unique_elements = list(set([item for entry in deduped_data for item in entry[element_field]]))
    
    ele_DF = ele_df(prop_list=info_dict['prop_list'],
                    total_e_list=unique_elements,
                    normalize=info_dict['normalize'])
    
    # Create a temporary file for deduped data to use json_loader
    temp_json = "temp_deduped.json"
    with open(temp_json, 'w') as f:
        json.dump(deduped_data, f)
    
    loader = json_loader(temp_json, df_data=ele_DF)
    x, _ = loader.load_data_nocomp(
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
    os.remove(temp_json)
    
    predictions = model.predict(x)
    
    # ========== 4. Recommendation ==========
    print(f"Generating top {top_n} recommendations...")
    df = pd.DataFrame({
        "ML_Predicted": predictions,
        "chem_id": [entry.get('chem_id', "-".join(entry['Elements'])) for entry in deduped_data],
        "Elements": ["-".join(entry['Elements']) for entry in deduped_data],
        "DFT": [entry.get('DFT', 'No') for entry in deduped_data],
        "EXP": [entry.get('EXP', 'No') for entry in deduped_data]
    })
    
    # Filter out those already in DFT or EXP if desired
    # For now, just sort by prediction
    recommendations = df.sort_values(by="ML_Predicted", ascending=False).head(top_n)
    
    # ========== 5. Save Results ==========
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = os.path.join(output_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    full_preds_path = os.path.join(output_dir, "all_predictions.csv")
    recom_path = os.path.join(output_dir, "top_recommendations.csv")
    
    df.to_csv(full_preds_path, index=False)
    recommendations.to_csv(recom_path, index=False)
    
    print(f"All predictions saved to {full_preds_path}")
    print(f"Top recommendations saved to {recom_path}")
    print("\nTop 5 Recommendations:")
    print(recommendations[['chem_id', 'ML_Predicted']].head(5))

if __name__ == "__main__":
    # Example usage
    # model_dir = "Output/Training/2026-03-19/17-33-55"
    # input_json = "Data/comp_pool/generated_compositions.json"
    
    if len(sys.argv) < 3:
        print("Usage: python run_prediction_and_recommendation.py <model_dir> <input_json_path>")
    else:
        run_pipeline(sys.argv[1], sys.argv[2])
