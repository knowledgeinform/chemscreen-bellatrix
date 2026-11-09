# -*- coding: utf-8 -*-
"""
Created by: Guangshuai Han
Year: 2025
Affiliation: Johns Hopkins University
Purpose: This code is designed to perform cross-validation for a random forest model
         and save the best model parameters and results.
"""

# ---------------- USER CONFIGURATIONS ----------------
info_dict = {
    'metric_name': "EFA",
    'target_key': "EFA",
    'xlabel': r"DFT Calculated DEED (eV/atom)$^{-1}$",
    'ylabel': r"ML Predicted DEED (eV/atom)$^{-1}$",
    'json_path': "Data/lib5_organized_072125.json",
    'base_output_dir': "Output/Training",
    'param_grid': {
        'n_estimators': [50],
        'max_depth': [10],
    },
    'cv_num': 2,
    'random_state': 21,
    'scoring_metrix': 'neg_median_absolute_error',
    'prop_list': [
        "atomic_mass",
        "atomic_number",
        "c6_gb",
        "conductivity_thermal",
        "density",
        "density_PT",
        "electron_affinity_PT",
        "electronegativity_Allen",
        "electronegativity_Ghosh",
        "electronegativity_Pauling",
        "enthalpy_atomization_WE",
        "enthalpy_fusion",
        "enthalpy_vaporization",
        "hardness_chemical_Ghosh",
        "polarizability",
        "radii_Ghosh08",
        "radii_Pyykko",
        "radii_Slatter",
        "radius_covalent",
        "radius_covalent_PT",
        "specific_heat_PT",
        "temperature_melting",
        "valence_PT",
        "valence_d",
        "valence_f",
        "valence_iupac",
        "valence_p",
        "valence_s",
        "valence_std",
        "variance_parameter_mass",
        "volume_molar"
    ],
    'n_jobs': -1,
    'verbose': 1,
    'normalize': 'quantile',
    'use_avg': True,
    'use_var': False,
    'use_min': True,
    'use_max': True,
    'use_weighted_avg': False,
    'use_weighted_var': False,
    'use_ratio': False,
    'use_delta': True,
    'compositions': None,
    'dg_len_threshold': 0, 
    'label_range': [None, 65], 
}

# -----------------------------------------------------

import sys
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import tempfile

# Add the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from mat_encod.data_util import ele_df, p_ele_encod, json_loader


def plot_results(cv_results, title, file_path, axis_limits=None):
    """Plot all CV results together with different colors"""
    plt.figure(figsize=(8, 8))
    plt.gca().set_facecolor('#E6ECF4')
    
    # Calculate axis limits if not provided
    if axis_limits is None:
        all_true = np.concatenate(cv_results['true_values'])
        all_pred = np.concatenate(cv_results['predictions'])
        min_val = min(min(all_true), min(all_pred))
        max_val = max(max(all_true), max(all_pred))
        padding = (max_val - min_val) * 0.05
        min_val -= padding
        max_val += padding
    else:
        min_val, max_val = axis_limits

    # Plot diagonal line
    x_vals = np.linspace(min_val, max_val, 100)
    plt.plot(x_vals, x_vals, '--', color='black', linewidth=1.5, label=r"$y = x$")

    # Plot each CV fold with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(cv_results['true_values'])))
    for i, (true_vals, pred_vals) in enumerate(zip(cv_results['true_values'], cv_results['predictions'])):
        plt.scatter(true_vals, pred_vals, color=colors[i], alpha=0.6, s=20, 
                   edgecolor='none', label=f'CV Fold {i+1}')
        
        # Add linear fit for this fold
        slope, intercept = np.polyfit(true_vals, pred_vals, 1)
        plt.plot(x_vals, slope * x_vals + intercept, color=colors[i], 
                linewidth=1, alpha=0.5)

    # Calculate overall metrics
    overall_true = np.concatenate(cv_results['true_values'])
    overall_pred = np.concatenate(cv_results['predictions'])
    overall_mae = mean_absolute_error(overall_true, overall_pred)
    overall_r2 = r2_score(overall_true, overall_pred)

    # Add metrics text
    metrics_text = f"Overall R² = {overall_r2:.3f}\nOverall MAE = {overall_mae:.3f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
            fontsize=16, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.xlabel(info_dict['xlabel'])
    plt.ylabel(info_dict['ylabel'])
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=12)
    plt.title(title)
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_model_and_params(model, params, output_dir, cv_score=None, final_mae=None, final_r2=None):
    # Save the model
    with open(os.path.join(output_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    # Save the parameters in JSON format
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    # Save the parameters in TXT format
    with open(os.path.join(output_dir, 'best_params.txt'), 'w') as f:
        f.write("Best Model Parameters:\n")
        f.write("=====================\n\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        # Add model performance metrics if provided
        if any([cv_score is not None, final_mae is not None, final_r2 is not None]):
            f.write("\nModel Performance:\n")
            f.write("=================\n")
            if cv_score is not None:
                f.write(f"Best CV Score: {cv_score:.4f}\n")
            if final_mae is not None:
                f.write(f"Final MAE: {final_mae:.4f}\n")
            if final_r2 is not None:
                f.write(f"Final R²: {final_r2:.4f}\n")

if __name__ == "__main__":
    # Load data
    with open(info_dict['json_path'], 'r') as f:
        ele_j_data = json.load(f)
    print(f"Total entries loaded: {len(ele_j_data)}")

    # 过滤dg_len
    dg_len_threshold = info_dict.get('dg_len_threshold', 0.0)
    filtered_data = [d for d in ele_j_data if d.get('dg_len', 0) >= dg_len_threshold]
    print(f"Entries after dg_len filter (threshold={dg_len_threshold}): {len(filtered_data)}")

    # 过滤label范围outlier（支持None灵活配置）
    label_range = info_dict.get('label_range', None)
    target_key = info_dict['target_key']
    if label_range is not None:
        label_min, label_max = label_range
        if label_min is not None and label_max is not None:
            filtered_data = [d for d in filtered_data if label_min <= d.get(target_key, 0) <= label_max]
        elif label_min is not None:
            filtered_data = [d for d in filtered_data if label_min <= d.get(target_key, 0)]
        elif label_max is not None:
            filtered_data = [d for d in filtered_data if d.get(target_key, 0) <= label_max]
    print(f"Entries after label_range filter: {len(filtered_data)}")

    if not filtered_data:
        raise ValueError("No data left after filtering! Please check your filters (dg_len_threshold, label_range) or the input JSON file.")

    # 提取chem_id
    chem_id_list = [d['chem_id'] if 'chem_id' in d else d.get('auid', 'unknown') for d in filtered_data]
    elements_list = [d['Elements'] for d in filtered_data]
    # 写入临时json文件
    with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.json') as tmpf:
        json.dump(filtered_data, tmpf)
        tmp_json_path = tmpf.name
    unique_elements = list(set([item for sublist in elements_list for item in sublist]))
    info_dict['unique_elements'] = unique_elements 
    ele_DF = ele_df(prop_list=info_dict['prop_list'], 
                    total_e_list=unique_elements,
                    normalize=info_dict['normalize'])
    data_loader = json_loader(tmp_json_path, df_data=ele_DF)
    train_x, train_y, train_elements = data_loader.load_data_nocomp(
        target_key=info_dict['target_key'], split=False,
        compositions=info_dict['compositions'], 
        use_avg=info_dict['use_avg'], 
        use_var=info_dict['use_var'], 
        use_min=info_dict['use_min'],
        use_max=info_dict['use_max'], 
        use_weighted_avg=info_dict['use_weighted_avg'], 
        use_weighted_var=info_dict['use_weighted_var'],
        use_ratio=info_dict['use_ratio'], 
        use_delta=info_dict['use_delta']
    )
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_elements_array = np.array(train_elements)
    chem_id_array = np.array(chem_id_list)

    # Create output directory
    output_dir = os.path.join(info_dict['base_output_dir'], datetime.now().strftime("%Y-%m-%d/%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    info_dict['final_dir'] = output_dir

    # KFold split
    kf = KFold(n_splits=info_dict['cv_num'], shuffle=True, random_state=info_dict['random_state'])
    train_x_array = np.array(train_x)
    train_y_array = np.array(train_y)
    train_elements_array = np.array(train_elements)

    fold_params = []  # 记录每fold的best_params
    fold_mae = []
    fold_r2 = []
    fold_indices = []

    all_true = []
    all_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_x_array)):
        x_train, x_val = train_x_array[train_idx], train_x_array[val_idx]
        y_train, y_val = train_y_array[train_idx], train_y_array[val_idx]

        rf = RandomForestRegressor(random_state=info_dict['random_state'])
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=info_dict['param_grid'],
            cv=info_dict['cv_num'],
            scoring=info_dict['scoring_metrix'],
            n_jobs=info_dict['n_jobs'],
            verbose=0
        )
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        fold_params.append(best_params)
        fold_mae.append(mae)
        fold_r2.append(r2)
        fold_indices.append(fold_idx)
        all_true.append(y_val)
        all_pred.append(y_pred)

    # 统计参数出现频率
    from collections import Counter, defaultdict
    param_strs = [str(p) for p in fold_params]
    freq_counter = Counter(param_strs)
    max_freq = max(freq_counter.values())
    candidates = [i for i, s in enumerate(param_strs) if freq_counter[s] == max_freq]
    if len(candidates) == 1:
        best_param_idx = candidates[0]
    else:
        # 多个参数频率一样，选MAE最小的
        best_param_idx = min(candidates, key=lambda i: fold_mae[i])
    best_params_final = fold_params[best_param_idx]

    # 用最终best_params在全体数据上训练
    final_model = RandomForestRegressor(**best_params_final, random_state=info_dict['random_state'])
    final_model.fit(train_x_array, train_y_array)
    final_predictions = final_model.predict(train_x_array)
    final_mae = mean_absolute_error(train_y_array, final_predictions)
    final_r2 = r2_score(train_y_array, final_predictions)

    # 保存每fold的MAE和R2
    summary_df = pd.DataFrame({
        'Fold': list(range(1, len(fold_mae)+1)),
        'MAE': fold_mae,
        'R2': fold_r2
    })
    summary_df.loc['Mean'] = ['Mean', np.mean(fold_mae), np.mean(fold_r2)]
    summary_df.to_csv(os.path.join(output_dir, "cv_summary.csv"), index=False)

    # 画总的scatter图，文件名test_result.png
    cv_results = {
        'true_values': all_true,
        'predictions': all_pred
    }
    all_true_values = np.concatenate(all_true)
    all_predicted_values = np.concatenate(all_pred)
    min_val = min(min(all_true_values), min(all_predicted_values))
    max_val = max(max(all_true_values), max(all_predicted_values))
    padding = (max_val - min_val) * 0.05
    axis_limits = (min_val - padding, max_val + padding)
    plot_results(cv_results, 
                f"Cross-Validation Results (Mean MAE={np.mean(fold_mae):.3f}, Mean R²={np.mean(fold_r2):.3f})", 
                os.path.join(output_dir, "test_result.png"),
                axis_limits=axis_limits)

    # 保存最终模型和参数
    save_model_and_params(final_model, best_params_final, output_dir,
                         cv_score=None,
                         final_mae=final_mae,
                         final_r2=final_r2)
    # 保存训练信息
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(info_dict, f, indent=4)

    # 保存test_results.csv（每个fold的真实值、预测值、fold编号、元素信息、chem_id）
    test_results_list = []
    max_elem_len = max(len(e) for fold in all_true for e in train_elements_array)  # 统计最大元素数
    for fold_idx, (y_true, y_pred) in enumerate(zip(all_true, all_pred)):
        val_idx = list(kf.split(train_x_array))[fold_idx][1]
        fold_elements = train_elements_array[val_idx]
        fold_chem_ids = chem_id_array[val_idx]
        elem_cols = {}
        for i in range(max_elem_len):
            elem_cols[f'Element{i+1}'] = [e[i] if len(e) > i else '' for e in fold_elements]
        df = pd.DataFrame({
            'Fold': fold_idx + 1,
            'True': y_true,
            'Pred': y_pred,
            'chem_id': fold_chem_ids,
            **elem_cols
        })
        test_results_list.append(df)
    test_results_df = pd.concat(test_results_list, ignore_index=True)
    test_results_df.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)

    # 保存train_result.csv（全体数据的真实值、预测值、元素信息、chem_id）
    max_elem_len_train = max(len(e) for e in train_elements_array)
    elem_cols_train = {}
    for i in range(max_elem_len_train):
        elem_cols_train[f'Element{i+1}'] = [e[i] if len(e) > i else '' for e in train_elements_array]
    train_result_df = pd.DataFrame({
        'True': train_y,
        'Pred': final_predictions,
        'chem_id': chem_id_array,
        **elem_cols_train
    })
    train_result_df.to_csv(os.path.join(output_dir, "train_result.csv"), index=False)