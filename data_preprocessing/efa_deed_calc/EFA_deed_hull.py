import os
import json
import re
from datetime import datetime

# ================= User Configuration ================= #
json_data_path = "Data/RAW_data/lib5_all.json"  # Path to the raw data
output_base = "Output/DataProcessing"          # Base path for the output directory
num_of_cation = 4
pocc_dg_path = "Data/dg_info/pocc_dg_lib5.json"
dg_length_threshold = 17  # Minimum dg length required to proceed
# ====================================================== #

# Generate a timestamped directory for output
timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
output_dir = os.path.join(output_base, timestamp)
log_dir = os.path.join(output_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, "process_log.txt")
json_output_path = os.path.join(output_dir, "processed_data.json")

# Initialize lists for processed data and log entries
processed_data = []
log_entries = []

#read the pocc_dg_path
with open(pocc_dg_path, 'r') as f:
    pocc_dg = json.load(f)

def extract_hull_energy_from_log(log_file_path):
    """
    Extract the hull energy value before (meV/atom) in the last matching line of the log file.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        float: Extracted hull energy value (in eV/atom), or None if not found.
    """
    hull_energy = None
    try:
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()
            # Search backwards for the last line containing (meV/atom)
            for line in reversed(lines):
                if "(meV/atom)" in line:
                    # Use regex to extract the value before (meV/atom)
                    match = re.search(r"=\s*([\-0-9.]+) \(meV/atom\)", line)
                    if match:
                        hull_energy = float(match.group(1)) / 1000.0  # Convert meV/atom to eV/atom
                        break
    except Exception as e:
        log_entries.append(f"Error parsing log file {log_file_path}: {e}")
    return hull_energy

def DEED_anly(dg_list, eform_list, hf_hull):
    """
    Calculate dh_hull and DEED based on the given lists and hull energy.

    Args:
        dg_list (list of float): List of POCC weights (dg values).
        eform_list (list of float): List of formation enthalpies (enth values).
        hf_hull (float): Hull energy (from CHULL).

    Returns:
        tuple: (dh_hull, DEED)
    """
    dg_tot = sum(dg_list)
    # Calculate h_mix
    h_mix = sum(dg * enth for dg, enth in zip(dg_list, eform_list)) / dg_tot
    # Calculate sigma
    sigma = sum(dg * (enth - h_mix) ** 2.0 for dg, enth in zip(dg_list, eform_list))
    sigma = (sigma / (dg_tot - 1)) ** 0.5
    isigma = sigma ** -1.0
    # Calculate dh_hull
    dh_hull = 0
    for dg, enth in zip(dg_list, eform_list):
        dh_hull += dg * (enth - hf_hull)
    dh_hull = dh_hull / dg_tot
    return dh_hull, isigma

# Iterate through all subdirectories in the RAW data directory
count = 0
with open(json_data_path , 'r') as f:
    data = json.load(f)
    
for i in range(10):
#for i in range(len(data)):
    temp_dict = {}
    temp_key = list(data[i].keys())
    temp_data = data[i][temp_key[0]]
    temp_e_list = temp_data['main']['species']
    # Order the temp_e_list by the order of alphabet
    temp_e_list = sorted(temp_e_list)
    
    temp_dict['species'] = temp_e_list.copy()
    # Build the composition list: O is 0.5, others are cation fraction
    composition_list = []
    for elem in temp_e_list:
        if elem == 'O':
            composition_list.append(0.5)
        else:
            composition_list.append(0.5 / num_of_cation)
    #remove the last piece of the composition_list
    full_composition_list = composition_list
    composition_list = composition_list[:-1]
    temp_e_list_str = ''.join(temp_e_list)
    # Convert composition_list to string for --hull_enthalpy argument
    hull_enthalpy_str = ','.join([f"{x:.4f}" for x in composition_list])
    # Build the aflow_aflux command
    chull_log_path = os.path.join(log_dir, f"aflow_{temp_e_list_str}_hull.log")  # Save CHULL log in dedicated folder
    chull_command = f"aflow_aflux --chull --alloy={temp_e_list_str} --force --hull_enthalpy={hull_enthalpy_str} --keep=log > {chull_log_path}"
    log_entries.append(f"Executing CHULL command: {chull_command}")
    
    
    try:
        temp_dg = []
        temp_eform = []
        temp_eform_cce = []
        for i in range(len(temp_data['sub_pocc'])):
            temp_proto = temp_data['sub_pocc'][i]['other']['prototype']
            #split the temp_proto by '.' and get the last part
            temp_proto = temp_proto.split('.')[-1]
            temp_dg.append(pocc_dg[temp_proto])
            temp_eform.append(temp_data['sub_pocc'][i]['other']['enthalpy_formation_atom'])
            temp_cce_tileseform = temp_data['sub_pocc'][i]['other']['enthalpy_formation_cce_0K_atom']
            #append the temp_cce_tileseform if it can be a float, else append 'enthalpy_formation_cce_0K_atom'
            if isinstance(temp_cce_tileseform, float):
                temp_eform_cce.append(temp_data['sub_pocc'][i]['other']['enthalpy_formation_cce_0K_atom'])
            else:
                temp_eform_cce.append(temp_data['sub_pocc'][i]['other']['enthalpy_formation_atom'])
    except Exception as e:
        log_entries.append(f"Error building dg/eform lists for elements {temp_e_list}, prototype {temp_proto}: {e}")
        continue

    if len(temp_dg) < dg_length_threshold:
        log_entries.append(f"Skipped entry for elements {temp_e_list} due to insufficient dg length: {len(temp_dg)}")
        continue
    os.system(chull_command)
    hull_energy = extract_hull_energy_from_log(chull_log_path)

    try:
        temp_d2h, temp_EFA = DEED_anly(temp_dg, temp_eform, hull_energy)
        temp_d2h, temp_EFA_cce = DEED_anly(temp_dg, temp_eform_cce, hull_energy)
        temp_deed = (temp_EFA / temp_d2h) ** 0.5
        temp_deed_cce = (temp_EFA_cce / temp_d2h) ** 0.5
        print(temp_deed)
    except Exception as e:
        log_entries.append(f"Error in DEED_anly for elements {temp_e_list}: {e}")
        continue
    
    temp_dict['Composition'] = full_composition_list
    print(temp_e_list)
    try:
        temp_e_list.remove('S')
    except:
        pass
    try:
        temp_e_list.remove('O')
    except:
        pass
    
    temp_dict['Elements'] = temp_e_list
    # Check for complex results and handle them
    def check_and_log_complex(val, name, element_list):
        if isinstance(val, complex):
            log_entries.append(f"Complex value encountered for {name} in elements {element_list}: {val}")
            return None
        return val

    temp_dict['EFA'] = check_and_log_complex(temp_EFA, 'EFA', temp_e_list)
    temp_dict['DEED'] = check_and_log_complex(temp_deed, 'DEED', temp_e_list)
    temp_dict['d2h'] = check_and_log_complex(temp_d2h, 'd2h', temp_e_list)
    temp_dict['deed_cce'] = check_and_log_complex(temp_deed_cce, 'deed_cce', temp_e_list)
    temp_dict['EFA_cce'] = check_and_log_complex(temp_EFA_cce, 'EFA_cce', temp_e_list)
    temp_dict['dg_list'] = temp_dg
    temp_dict['eform_list'] = temp_eform
    temp_dict['eform_cce_list'] = temp_eform_cce
    temp_dict['dg_len'] = len(temp_dg)
    processed_data.append(temp_dict)
    count += 1
    
# Save the processed data to a JSON file
with open(json_output_path, "w") as f:
    json.dump(processed_data, f, indent=4)

# Save log entries to a log file
with open(log_file_path, "w") as f:
    for entry in log_entries:
        f.write(entry + "\n")

print(f"Data processing completed. JSON saved to {json_output_path}")
print(f"Log file saved to {log_file_path}")

os.system("rm -f aflow*")  