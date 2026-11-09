# -*- coding: utf-8 -*-
"""
Generate unique 5-cation compositions with charge balancing and high-entropy composition.

Created by: Guangshuai Han
Year: 2025
Affiliation: Johns Hopkins University
Purpose:
    - Extract unique cations from dataset.
    - Ensure specific elements are included/excluded.
    - Generate 5-cation + fixed anion compositions with balanced charge.
    - Save results in JSON format with strict fields: species, Elements, composition, chem_id, DFT, EXP.
"""

import json
import itertools
import os
from datetime import datetime
import sys

# ================= User Configuration ================= #
json_input_path = "Data/lib6_organized_072125.json"  # Path to DFT dataset
exp_truth_path_list = ["Data/exp_truth/lib5_exp_truth.json", "Data/exp_truth/lib6_exp_truth.json"]
exclude_elements = ["O", "Tc", "I"]  # Elements to exclude
include_elements = ["Sr", "Zr", "Nb", "Mo", "Sn", "Zn", "Fe", "Co", "Ni", "Cu", "Sb","Cr","Al","Ga", 'Ru', 'Rh', 'Pd', 'Ag', 'Ba', 'Pt', 'Pb']  # Elements that must be included
anion_list = ["O"]  # Fixed anion elements
output_base = "Output/DataProcessing"  # Output directory
num_cations = 5  # Select 5 cations per composition
target_charge = 10  # Target charge for charge balance
# ====================================================== #

# Add the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import data loading functions from mat_encod
from mat_encod.data_util import get_atomnum

def generate_chem_id(species, composition):
    comp_floats = [float(c) for c in composition]
    sorted_pairs = sorted(zip(species, comp_floats), key=lambda x: x[0])
    sorted_species = [x[0] for x in sorted_pairs]
    sorted_composition = [x[1] for x in sorted_pairs]
    comp_str = ''.join([str(int(round(c * 1000))) for c in sorted_composition])
    return ''.join(sorted_species) + '_' + comp_str

# Expanded valence database for new elements
valence_database = {
    "Ti": [3, 4], "Mn": [2, 3, 4, 7], "Sc": [3], "Rb": [1], "Ga": [3], "Ag": [1],
    "Zn": [2], "Fe": [2, 3], "Co": [2, 3], "Ni": [2, 3], "Cu": [1, 2], "Sn": [2, 4],
    "Ca": [2], "V": [2, 3, 4, 5], "Mg": [2], "Tc": [7], "Nb": [5], "Cd": [2], "Sr": [2],
    "I": [-1], "O": [-2],
}

# Create output directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(output_base, timestamp)
os.makedirs(output_dir, exist_ok=True)
json_output_path = os.path.join(output_dir, "generated_compositions.json")

# Step 1: Extract unique cations from JSON
def extract_cation_pool(json_input_path, exclude_elements, include_elements):
    with open(json_input_path, "r") as f:
        ele_j_data = json.load(f)
    unique_elements = []
    for entry in ele_j_data:
        unique_elements.extend(entry['Elements'])
    all_cations = list(set(unique_elements))
    # Exclude unwanted elements
    final_cation_pool = [ele for ele in all_cations if ele not in exclude_elements]
    # Add include_elements if not already present
    for ele in include_elements:
        if ele not in final_cation_pool:
            final_cation_pool.append(ele)
    return final_cation_pool

# Step 2: 读取 DFT/EXP chem_id 列表
def get_chem_id_set(json_path_list):
    chem_id_set = set()
    for path in json_path_list:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            try:
                data = json.load(f)
            except Exception:
                continue
            for entry in data:
                chem_id = entry.get('chem_id')
                if chem_id:
                    chem_id_set.add(chem_id)
    return chem_id_set

# Step 2b: Build chem_id to single_phase mapping from EXP data

def get_single_phase_map(json_path_list):
    single_phase_map = {}
    for path in json_path_list:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            try:
                data = json.load(f)
            except Exception:
                continue
            for entry in data:
                chem_id = entry.get('chem_id')
                sp = entry.get('single_phase')
                if chem_id:
                    single_phase_map[chem_id] = sp
    return single_phase_map

# Step 3: 生成所有高熵组合（等分 composition）并价态筛选
def generate_charge_balanced_high_entropy_compositions(cation_pool, valence_db, target_charge, num_cations):
    valid_compositions = []
    for combo in itertools.combinations(cation_pool, num_cations):
        sorted_cations = tuple(sorted(combo, key=get_atomnum))
        valence_combinations = list(itertools.product(*(valence_db.get(ele, [0]) for ele in sorted_cations)))
        for valences in valence_combinations:
            if sum(valences) == target_charge:
                # 组合 species = cations + ['O']，然后排序
                full_species = list(sorted(list(sorted_cations) + ['O'], key=get_atomnum))
                # 组合 composition，O 为 0.5，其余均分 0.5
                comp_dict = {ele: 0.5 / num_cations for ele in sorted_cations}
                comp_dict['O'] = 0.5
                composition = [comp_dict[ele] for ele in full_species]
                chem_id = generate_chem_id(full_species, composition)
                valid_compositions.append({
                    "species": full_species,
                    "Elements": list(sorted_cations),  # Elements字段只包含阳离子，不包含O
                    "composition": composition,
                    "chem_id": chem_id
                })
    return valid_compositions

# Step 4: 标记 DFT/EXP 字段，并添加 single_phase 字段

def mark_dft_exp_singlephase(compositions, dft_set, exp_set, single_phase_map):
    for comp in compositions:
        comp["DFT"] = "yes" if comp["chem_id"] in dft_set else "no"
        comp["EXP"] = "yes" if comp["chem_id"] in exp_set else "no"
        if comp["chem_id"] in single_phase_map:
            sp = single_phase_map[comp["chem_id"]]
            if sp == "Yes":
                comp["single_phase"] = "Yes"
            elif sp == "No":
                comp["single_phase"] = "No"
            else:
                comp["single_phase"] = "Unknown"
        else:
            comp["single_phase"] = "Unknown"
    return compositions

if __name__ == "__main__":
    # 1. 提取阳离子池
    cation_pool = extract_cation_pool(json_input_path, exclude_elements, include_elements)
    print(f"Final Cation Pool: {cation_pool}")

    # 2. 读取 DFT/EXP chem_id
    dft_set = get_chem_id_set([json_input_path, "Data/lib5_organized_072125.json"])
    exp_set = get_chem_id_set(exp_truth_path_list)
    single_phase_map = get_single_phase_map(exp_truth_path_list)

    # 3. 生成所有高熵组合
    compositions = generate_charge_balanced_high_entropy_compositions(cation_pool, valence_database, target_charge, num_cations)
    print(f"Generated {len(compositions)} charge-balanced high-entropy compositions.")

    # Debug: 打印 DFT/EXP chem_id 数量和样例
    print(f"DFT chem_id count: {len(dft_set)}")
    print(f"EXP chem_id count: {len(exp_set)}")
    print("Sample DFT chem_id:", list(dft_set)[:5])
    print("Sample EXP chem_id:", list(exp_set)[:5])

    # Debug: 打印生成的组合 chem_id 样例
    print("Sample generated chem_id:", [c['chem_id'] for c in compositions[:5]])

    # Debug: 检查交集
    generated_ids = set(c['chem_id'] for c in compositions)
    dft_overlap = generated_ids & dft_set
    exp_overlap = generated_ids & exp_set
    print(f"Overlap with DFT: {len(dft_overlap)}")
    print(f"Overlap with EXP: {len(exp_overlap)}")
    if dft_overlap:
        print("Sample overlap DFT chem_id:", list(dft_overlap)[:5])
    if exp_overlap:
        print("Sample overlap EXP chem_id:", list(exp_overlap)[:5])

    # 4. Mark DFT/EXP fields and single_phase
    compositions = mark_dft_exp_singlephase(compositions, dft_set, exp_set, single_phase_map)

    # 5. Deduplicate by chem_id
    unique_compositions = {}
    for comp in compositions:
        if comp["chem_id"] not in unique_compositions:
            unique_compositions[comp["chem_id"]] = comp
    compositions = list(unique_compositions.values())
    print(f"After deduplication, {len(compositions)} unique compositions remain.")

    # 6. Save JSON
    with open(json_output_path, "w") as f:
        json.dump(compositions, f, indent=4)
    print(f"Saved {len(compositions)} compositions to {json_output_path}")
