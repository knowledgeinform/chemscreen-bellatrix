import json
from typing import List, Any

# ====== User configuration: set your file paths here ======
input_json = "Data/lib5_organized_072125.json"      # Path to your input JSON file
output_json = None             # Set to None to overwrite input file, or specify a new file name
# =========================================================

def generate_chem_id(species: List[str], composition: List[Any]) -> str:
    """
    Generate a chemical stoichiometry id, e.g., AlCoFeNiO_100200300200200
    species: list of str, element symbols
    composition: list of float/int, corresponding to species
    """
    comp_floats = [float(c) for c in composition]
    # Sort by element symbol, keep composition order consistent
    sorted_pairs = sorted(zip(species, comp_floats), key=lambda x: x[0])
    sorted_species = [x[0] for x in sorted_pairs]
    sorted_composition = [x[1] for x in sorted_pairs]
    comp_str = ''.join([str(int(round(c * 1000))) for c in sorted_composition])
    return ''.join(sorted_species) + '_' + comp_str

def find_species_and_composition(entry: dict):
    """
    General function to find species and composition fields in an entry.
    Prefer 'species', otherwise use 'Elements'. Composition field must be 'Composition'.
    Returns: (species_list, composition_list)
    """
    if 'species' in entry and 'Composition' in entry:
        return entry['species'], entry['Composition']
    elif 'Elements' in entry and 'Composition' in entry:
        return entry['Elements'], entry['Composition']
    else:
        raise ValueError('Entry does not contain recognizable species/Elements and Composition fields.')

def add_chem_id_to_json(input_json_path, output_json_path=None):
    """
    Add chem_id to each entry in the JSON file. If output_json_path is None, overwrite the input file.
    """
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        try:
            species, composition = find_species_and_composition(entry)
            chem_id = generate_chem_id(species, composition)
            entry['chem_id'] = chem_id
        except Exception as e:
            entry['chem_id'] = None  # Mark error for later inspection
            entry['chem_id_error'] = str(e)

    if output_json_path is None:
        output_json_path = input_json_path
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    add_chem_id_to_json(input_json, output_json)
    print(f"Done! chem_id added to each entry in {output_json if output_json else input_json}") 