import os
import json
import re
from datetime import datetime

# ================= User Configuration ================= #
json_data_path = "Data/RAW_data/processed_data.json"  # Path to the raw data
output_base = "Output/DataProcessing"          # Base path for the output directory
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

def extract_dist2hull_from_log(log_file_path):
    """
    Extracts dist2hull value from the log file.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        float: Extracted dist2hull value in eV/atom, or None if not found.
    """
    dist2hull = None
    try:
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                # Search for the line containing 'dist2hull' and '(meV/atom)'
                if "dist2hull" in line and "(meV/atom)" in line:
                    # Use regex to extract the numeric value before '(meV/atom)'
                    match = re.search(r"dist2hull = ([\-0-9.]+) \(meV/atom\)", line)
                    if match:
                        dist2hull = float(match.group(1)) / 1000.0  # Convert from meV/atom to eV/atom
                        break
    except Exception as e:
        log_entries.append(f"Error parsing log file {log_file_path}: {e}")
    return dist2hull

# Iterate through all subdirectories in the RAW data directory
count = 0
with open(json_data_path , 'r') as f:
    data = json.load(f)

#for i in range(len(data)):
for i in range(5):
    temp_dict = {}
    temp_key = list(data[i].keys())
    temp_data = data[i][temp_key[0]]
    temp_e_list = temp_data['main']['species']
    temp_EFA = temp_data['main']['EFA']
    temp_auid = temp_data['main']['auid']
    chull_log_path = os.path.join(log_dir, f"aflow_{temp_auid}_hull.log")  # Save CHULL log in dedicated folder
    
    chull_command = f"/usr/local/bin/aflow_aflux --chull --alloy={''.join(temp_e_list)} --force --d2h={temp_auid } --keep=log > {chull_log_path}"
    print(chull_command)
    log_entries.append(f"Executing CHULL command: {chull_command}")
    os.system(chull_command)

    # Extract the dist2hull value
    dist2hull = extract_dist2hull_from_log(chull_log_path)

    # Calculate DEED
    deed_value = None
    
    #remove the 'S' and 'O' element
    try:
        temp_e_list.remove('S')
    except:
        pass
    try:
        temp_e_list.remove('O')
    except:
        pass
    
    if temp_EFA and dist2hull:
        try:
            deed_value = (temp_EFA / abs(dist2hull)) ** 0.5
        except Exception as e:
            log_entries.append(f"Failed to calculate DEED for {temp_e_list}: {e}")

        # Build the JSON record
        temp_dict['Elements'] = temp_e_list
        
        temp_dict['Elements'] = temp_e_list
        temp_dict['Composition'] = ['0.2', '0.2', '0.2', '0.2', '0.2']
        temp_dict['EFA'] = temp_data['main']['EFA']
        print(temp_data['main']['EFA'])
        temp_dict['auid'] = temp_data['main']['auid']
        temp_dict['DEED'] = deed_value
        temp_dict['d2h'] = dist2hull
        processed_data.append(temp_dict)
    
        log_entries.append(f"Processed successfully.")
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