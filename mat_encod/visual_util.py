import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np

# Define periodic table positions
periodic_table = {
    "H": (1, 1), "He": (18, 1),
    "Li": (1, 2), "Be": (2, 2), "B": (13, 2), "C": (14, 2), "N": (15, 2), "O": (16, 2), "F": (17, 2), "Ne": (18, 2),
    "Na": (1, 3), "Mg": (2, 3), "Al": (13, 3), "Si": (14, 3), "P": (15, 3), "S": (16, 3), "Cl": (17, 3), "Ar": (18, 3),
    "K": (1, 4), "Ca": (2, 4), "Sc": (3, 4), "Ti": (4, 4), "V": (5, 4), "Cr": (6, 4), "Mn": (7, 4), "Fe": (8, 4),
    "Co": (9, 4), "Ni": (10, 4), "Cu": (11, 4), "Zn": (12, 4), "Ga": (13, 4), "Ge": (14, 4), "As": (15, 4),
    "Se": (16, 4), "Br": (17, 4), "Kr": (18, 4),
    "Rb": (1, 5), "Sr": (2, 5), "Y": (3, 5), "Zr": (4, 5), "Nb": (5, 5), "Mo": (6, 5), "Tc": (7, 5), "Ru": (8, 5),
    "Rh": (9, 5), "Pd": (10, 5), "Ag": (11, 5), "Cd": (12, 5), "In": (13, 5), "Sn": (14, 5), "Sb": (15, 5),
    "Te": (16, 5), "I": (17, 5), "Xe": (18, 5),
    "Cs": (1, 6), "Ba": (2, 6), "La": (3, 9), "Ce": (4, 9), "Pr": (5, 9), "Nd": (6, 9), "Pm": (7, 9), "Sm": (8, 9),
    "Eu": (9, 9), "Gd": (10, 9), "Tb": (11, 9), "Dy": (12, 9), "Ho": (13, 9), "Er": (14, 9), "Tm": (15, 9),
    "Yb": (16, 9), "Lu": (17, 9), "Hf": (4, 6), "Ta": (5, 6), "W": (6, 6), "Re": (7, 6), "Os": (8, 6), "Ir": (9, 6),
    "Pt": (10, 6), "Au": (11, 6), "Hg": (12, 6), "Tl": (13, 6), "Pb": (14, 6), "Bi": (15, 6), "Po": (16, 6),
    "At": (17, 6), "Rn": (18, 6),
    "Fr": (1, 7), "Ra": (2, 7), "Ac": (3, 10), "Th": (4, 10), "Pa": (5, 10), "U": (6, 10), "Np": (7, 10),
    "Pu": (8, 10), "Am": (9, 10), "Cm": (10, 10), "Bk": (11, 10), "Cf": (12, 10), "Es": (13, 10), "Fm": (14, 10),
    "Md": (15, 10), "No": (16, 10), "Lr": (17, 10)
}

def plot_periodic_heatmap(elements, values, cmap="seismic", save_path=None):
    """
    Plots a periodic table heatmap based on element frequency.

    Args:
        elements (list): List of element symbols.
        values (list): Corresponding frequency values for each element.
        cmap (str): Colormap for heatmap visualization.
        save_path (str, optional): File path to save the figure. If None, only show.
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Normalize colormap
    norm = Normalize(vmin=0, vmax=max(values) if values else 1)
    colormap = cm.get_cmap(cmap)

    # Create dictionary for values
    element_values = {ele: val for ele, val in zip(elements, values)}

    # Plot elements
    for element, (group, period) in periodic_table.items():
        value = element_values.get(element, 0)  # Default to 0 if element not in data
        color = colormap(norm(value)) if value > 0 else "lightgrey"

        ax.add_patch(plt.Rectangle((group, period), 1, 1, color=color, edgecolor='black', linewidth=2.5))
        ax.text(group + 0.5, period + 0.5, element, ha='center', va='center',
                fontsize=12, fontweight='bold', color='black')

    # Configure axis limits
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 11.5)
    ax.invert_yaxis()
    ax.axis('off')

    # Add colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label("Element Frequency")

    # Save or show figure
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
    plt.close()
