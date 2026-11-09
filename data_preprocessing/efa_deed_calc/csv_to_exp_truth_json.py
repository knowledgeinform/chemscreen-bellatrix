import csv
import json
import os
from typing import List, Any

# ====== User configuration: set your file paths here ======
input_csv = "Data/exp_truth/exp_raw.csv"  # Path to your input CSV file
output_dir = "Data/exp_truth/"  # Directory to save output JSON files
# =========================================================

def parse_list_field(field: str) -> List[str]:
    """
    Parse a comma-separated string into a list, stripping whitespace.
    """
    return [x.strip() for x in field.split(',') if x.strip()]

def parse_composition_field(field: str) -> List[float]:
    """
    Parse a comma-separated string into a list of floats, stripping whitespace.
    """
    return [float(x.strip()) for x in field.split(',') if x.strip()]

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

def process_csv(input_csv_path, output_dir):
    """
    Read the CSV, process each row, and write to JSON files grouped by cation count.
    """
    # Dictionary to group entries by cation count
    cation_groups = {}
    total_rows = 0
    used_rows = 0
    skipped_rows = 0

    with open(input_csv_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_rows += 1
            # Strip whitespace and quotes from all field values
            clean_row = {k.strip().replace('\ufeff', ''): (v.strip().strip('"') if v is not None else v) for k, v in row.items()}
            if total_rows <= 3:
                print(f"Row {total_rows} raw: {row}")
                print(f"Row {total_rows} cleaned: {clean_row}")
            species_raw = clean_row.get('species') or clean_row.get('Species')
            composition_raw = clean_row.get('composition') or clean_row.get('Composition')
            if not species_raw or not composition_raw:
                print(f'Skipping row {total_rows}: missing species or composition')
                skipped_rows += 1
                continue  # Skip rows without required fields
            species = parse_list_field(species_raw)
            composition = parse_composition_field(composition_raw)
            # Elements: species without 'O'
            elements = [el for el in species if el != 'O']
            cation_count = len(elements)
            # Generate chem_id
            chem_id = generate_chem_id(species, composition)
            # Prepare entry
            entry = {
                'species': species,
                'Elements': elements,
                'Composition': composition,
                'chem_id': chem_id
            }
            # Add other experimental fields
            for k, v in clean_row.items():
                if k not in ['species', 'Species', 'composition', 'Composition']:
                    entry[k] = v
            # Group by cation count
            if cation_count not in cation_groups:
                cation_groups[cation_count] = []
            cation_groups[cation_count].append(entry)
            used_rows += 1

    print(f"Total rows: {total_rows}, Used: {used_rows}, Skipped: {skipped_rows}")
    print(f"Cation groups found: {list(cation_groups.keys())}")

    # Write each group to a separate JSON file
    for cation_count, entries in cation_groups.items():
        out_name = f"lib{cation_count+1}_exp_truth.json"  # +1 for O
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, 'w') as f:
            json.dump(entries, f, indent=4)
        print(f"Saved {len(entries)} entries to {out_path}")

if __name__ == "__main__":
    process_csv(input_csv, output_dir)
    print("Done! Experimental truth JSON files generated.") 