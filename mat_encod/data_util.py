# -*- coding: utf-8 -*-
"""
Created by: Guangshuai (Jerry) Han @S4E Lab
Year: 2025
Affiliation: Johns Hopkins University
Purpose: This code is designed to process and normalize element data for the purpose of
         chemical composition feature extraction.
         It includes functionalities for:
         - Loading and normalizing element data.
         - Encoding chemical elements based on their properties.
         - Generating statistical features for machine learning tasks.
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Load element data
json_path = 'Data/ele_data/Aflow_e_data_GH.json'
with open(json_path, 'r') as file:
    ele_data = json.load(file)

e_list = list(ele_data.keys())
#remove the "AAA_notes"
e_list.remove("AAA_notes")
def get_atomnum(name):
    """
    Retrieve the atomic number for a given element name using the original method.
    """
    try:
        return int(ele_data[name]['atomic_number'])
    except:
        return None

# function that create a dataframe from element data
# extract the specific properties from the json element data
# fill the null values with the mean of the column and nomalize the data

def ele_df(prop_list=['atomic_number', 'atomic_mass', 'density', 
                'electronegativity_Allen', 'radii_Ghosh08', 'radii_Pyykko',
                'radii_Slatter', 'radius_covalent','polarizability',
                'thermal_conductivity_25C'], total_e_list=e_list, normalize='Z-score'):

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import quantile_transform

    # Initialize element dataframe and missing value tracker
    ele_df = pd.DataFrame(columns=prop_list, index=total_e_list)
    missing_report = pd.DataFrame(0, index=total_e_list, columns=prop_list)

    # Load data from element dictionary
    for ele in total_e_list:
        for prop in prop_list:
            try:
                ele_df.loc[ele, prop] = ele_data[ele][prop]
            except:
                ele_df.loc[ele, prop] = None
                missing_report.loc[ele, prop] = 1

    # Convert to numeric and detect missing values
    ele_df = ele_df.apply(pd.to_numeric, errors='coerce')
    missing_before = ele_df.isna()

    # Fill missing values with column mean and summarize fill counts
    ele_df = ele_df.apply(lambda x: x.fillna(x.mean()), axis=0)
    filled_report = missing_before & ~ele_df.isna()
    fill_counts = filled_report.sum(axis=1)
    fill_summary = fill_counts[fill_counts > 0].sort_values(ascending=False)

    # Print missing fill report
    print("\n[Missing Value Fill Report]")
    print("Number of elements with filled values:", len(fill_summary))
    #print(fill_summary.to_string())

    # Apply normalization based on method selected
    if normalize == 'Z-score':
        ele_df = (ele_df - ele_df.mean()) / ele_df.std()

    elif normalize == 'min-max':
        ele_df = (ele_df - ele_df.min()) / (ele_df.max() - ele_df.min())

    elif normalize == 'min-max-0.1-0.9':
        ele_df = 0.1 + 0.8 * (ele_df - ele_df.min()) / (ele_df.max() - ele_df.min())

    elif normalize == 'log-zscore':
        ele_df = np.log(ele_df + 1e-6)
        ele_df = (ele_df - ele_df.mean()) / ele_df.std()

    elif normalize == 'quantile':
        ele_df = pd.DataFrame(quantile_transform(ele_df, n_quantiles=100, output_distribution='uniform', copy=True),
                              index=ele_df.index, columns=ele_df.columns)

    elif normalize == 'none':
        pass  # No normalization applied

    else:
        raise ValueError("Unsupported normalization method. Choose from 'Z-score', 'min-max', 'min-max-0.1-0.9', 'log-zscore', 'quantile', or 'none'.")

    # Transpose the dataframe: elements as columns, properties as rows
    ele_df = ele_df.transpose()

    return ele_df


    
class p_ele_encod:
    def __init__(self, ele_df_data):
        """
        Initialize the encoder with a normalized element dataframe.
        """
        self.ele_df_data = ele_df_data

    def get_ele_stats_inp(self, ele_list, compositions=None, 
                          use_avg=True, 
                          use_var=True, 
                          use_min=True, 
                          use_max=True,
                          use_weighted_avg=True, #weighted average
                          use_weighted_var=True, #weighted variance
                          use_ratio=True, #max/min ratio
                          use_delta=True #mismatch
                          ):
        """
        Generate statistical input features based on the element properties.

        Parameters:
        - ele_list: list of element symbols
        - compositions: list of composition ratios (same length as ele_list), optional
        - use_avg, use_var, use_min, use_max: whether to include those statistics
        - use_weighted_avg, use_weighted_var: include weighted statistics
        - use_ratio: include max/min ratio
        - use_delta: include sqrt of weighted variance (delta)

        Returns:
        - A single concatenated feature vector (numpy array)
        """
        import numpy as np

        # Default uniform composition
        if compositions is None:
            compositions = [1 / len(ele_list)] * len(ele_list)

        # Sort and get unique elements
        elist = sorted(ele_list, key=get_atomnum)
        el_list = list(set(elist))

        metal = []
        for ele in el_list:
            try:
                meta = self.ele_df_data[ele].to_list()
                metal.append(np.array(meta))
            except Exception as e:
                print(f"Error processing element {ele}: {e}")

        metal = np.array(metal)

        # Initialize feature container
        features = []

        if use_avg:
            features.append(np.mean(metal, axis=0))

        if use_var:
            features.append(np.var(metal, axis=0))

        if use_min:
            features.append(np.min(metal, axis=0))

        if use_max:
            features.append(np.max(metal, axis=0))

        if use_weighted_avg:
            features.append(np.average(metal, axis=0, weights=compositions))

        if use_weighted_var:
            weighted_avg = np.average(metal, axis=0, weights=compositions)
            weighted_var = np.average((metal - weighted_avg) ** 2, axis=0, weights=compositions)
            features.append(weighted_var)

        if use_ratio:
            max_val = np.max(metal, axis=0)
            min_val = np.min(metal, axis=0)
            ratio = np.divide(min_val, max_val, out=np.zeros_like(min_val), where=max_val!=0)
            features.append(ratio)

        if use_delta:
            weighted_avg = np.average(metal, axis=0, weights=compositions)
            delta = np.sqrt(np.sum([c * (v - weighted_avg) ** 2 for c, v in zip(compositions, metal)], axis=0))
            features.append(delta)

        # Concatenate all selected features
        ele_inp = np.concatenate(features, axis=0)
        
        return ele_inp


    def get_ele_stats_comp(self, ele_list, comp_list):
        """
        Generate statistical composition features based on the element properties and it's 
        atomic ratio in the composition.
        """
        metal = []
        return metal
    

#Data loader for training

class json_loader:
    """
    Load and process JSON file for ChemScreen training.
    """
    def __init__(self, json_path, df_data):
        with open(json_path, 'r') as file:
            self.T_data = json.load(file)

        self.p_encod = p_ele_encod(df_data)
        
    
    
    def load_data_nocomp(self, 
                      target_key='EFA', 
                      with_target=True, 
                      split=False, 
                      test_size=0.2,compositions=None, 
                      use_avg=True, use_var=True, use_min=True, 
                      use_max=True, use_weighted_avg=False, 
                      use_weighted_var=False, use_ratio=True, 
                      use_delta=True
                      
                    ):
        """
        General-purpose loader for training or inference.
        
        Parameters:
            - with_target: If True, load target property from JSON
            - split: If True, split into train/test (only if with_target is True)
        """
        inputs = []
        targets = []
        element_lists = []

        for entry in self.T_data:

            temp_e_list = entry['Elements']
            
            composition = (
                entry.get('composition') or
                entry.get('Composition') or
                entry.get('Compositions')
            )
            if composition is None:
                raise KeyError("No composition field found in entry!")
            
            # major limitation: need to load the function from the multiple times ... May need to fix this once database is larger
            tem_inputs = self.p_encod.get_ele_stats_inp(temp_e_list, compositions=compositions, 
                                                        use_avg=use_avg, use_var=use_var, use_min=use_min, use_max=use_max, 
                                                        use_weighted_avg=use_weighted_avg, use_weighted_var=use_weighted_var, 
                                                        use_ratio=use_ratio, use_delta=use_delta)
            


            inputs.append(tem_inputs)
            element_lists.append(temp_e_list)

            if with_target:
                targets.append(entry[target_key])

        # Return different structures based on mode
        if with_target:
            if split:
                train_x, test_x, train_y, test_y, train_elements, test_elements = train_test_split(
                    inputs, targets, element_lists, test_size=test_size, random_state=42
                )
                return train_x, test_x, train_y, test_y, train_elements, test_elements
            else:
                return inputs, targets, element_lists
        else:
            return inputs, element_lists
