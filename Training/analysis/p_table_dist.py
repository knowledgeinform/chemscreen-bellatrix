import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Set global font to DejaVu Serif (similar to Times New Roman)
plt.rcParams["font.family"] = "DejaVu Serif"

# ========== User Configuration ==========
DFT_json_path_list = [
    "Data/lib5_organized_072125.json",
    "Data/lib6_organized_072125.json",
]

EXP_json_path_list = [
    "Data/exp_truth/lib5_exp_truth.json",
    "Data/exp_truth/lib6_exp_truth.json",
]

explor_json_path = ["Data/comp_pool/generated_compositions.json"]

# ========== Periodic Table Visualization ========== 
from matplotlib.patches import Wedge, Rectangle
from matplotlib.lines import Line2D

# ========== Custom Colors for Legend ==========
# You can customize these colors as needed
dft_color = '#002FA7'      
exp_color = '#429A67'       
explor_color = '#F2BB44'   


# Define elements to exclude (lanthanides, actinides, noble gases, O, F, non-metals)
exclude_elements = {
    # Noble gases
    'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn',
    # O and F
    'O', 'F',
    # Non-metals (chalcogens and halogens)
    'S', 'Se', 'Te', 'Po', 'Cl', 'Br', 'I', 'At',
    # Lanthanides
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    # Actinides
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    # Superheavy elements (period 7)
    'Fr', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
}

# Simplified periodic table layout (excluding unwanted elements)
ptable_layout = {
    # 1st period
    'H': (0, 0),
    # 2nd period
    'Li': (1, 0), 'Be': (1, 1), 'B': (1, 12), 'C': (1, 13), 'N': (1, 14),
    # 3rd period
    'Na': (2, 0), 'Mg': (2, 1), 'Al': (2, 12), 'Si': (2, 13), 'P': (2, 14),
    # 4th period
    'K': (3, 0), 'Ca': (3, 1), 'Sc': (3, 2), 'Ti': (3, 3), 'V': (3, 4), 'Cr': (3, 5), 'Mn': (3, 6), 'Fe': (3, 7), 
    'Co': (3, 8), 'Ni': (3, 9), 'Cu': (3, 10), 'Zn': (3, 11), 'Ga': (3, 12), 'Ge': (3, 13), 'As': (3, 14),
    # 5th period
    'Rb': (4, 0), 'Sr': (4, 1), 'Y': (4, 2), 'Zr': (4, 3), 'Nb': (4, 4), 'Mo': (4, 5), 'Tc': (4, 6), 'Ru': (4, 7), 
    'Rh': (4, 8), 'Pd': (4, 9), 'Ag': (4, 10), 'Cd': (4, 11), 'In': (4, 12), 'Sn': (4, 13), 'Sb': (4, 14),
    # 6th period (main group only, no lanthanides)
    'Cs': (5, 0), 'Ba': (5, 1), 'Hf': (5, 3), 'Ta': (5, 4), 'W': (5, 5), 'Re': (5, 6), 'Os': (5, 7), 'Ir': (5, 8),
    'Pt': (5, 9), 'Au': (5, 10), 'Hg': (5, 11), 'Tl': (5, 12), 'Pb': (5, 13), 'Bi': (5, 14)
}

# Helper: get all unique elements in a json file list
def get_elements_from_json(json_path_list):
    elements = set()
    for path in json_path_list:
        with open(path, 'r') as f:
            data = json.load(f)
        for entry in data:
            if isinstance(entry, dict) and 'Elements' in entry:
                elements.update(entry['Elements'])
            elif isinstance(entry, list):
                elements.update(entry)
    return elements

def get_elements_from_large_json(json_path):
    elements = set()
    with open(json_path, 'r') as f:
        for line in f:
            if '"Elements"' in line:
                try:
                    elems = eval(line.split(':',1)[1].strip().rstrip(',').replace('null','None'))
                    if isinstance(elems, list):
                        elements.update(elems)
                except Exception:
                    continue
    return elements

DFT_elements = get_elements_from_json(DFT_json_path_list)
EXP_elements = get_elements_from_json(EXP_json_path_list)
explor_elements = get_elements_from_json(explor_json_path)

# Filter out excluded elements
DFT_elements = DFT_elements - exclude_elements
EXP_elements = EXP_elements - exclude_elements
explor_elements = explor_elements - exclude_elements

# Debug print (可删除)
print("DFT_elements:", DFT_elements)
print("EXP_elements:", EXP_elements)
print("explor_elements:", explor_elements)

all_elements = set(ptable_layout.keys())
used_elements = DFT_elements | EXP_elements | explor_elements

presence = {}
for elem in all_elements:
    presence[elem] = [elem in DFT_elements, elem in EXP_elements, elem in explor_elements]

# Calculate optimal figure size based on actual elements
max_row = max(pos[0] for pos in ptable_layout.values()) + 1
max_col = max(pos[1] for pos in ptable_layout.values()) + 1
fig_width = max_col * 0.8 + 2  # Add space for legend
fig_height = max_row * 0.8 + 1

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_aspect('equal')
ax.set_xlim(-0.5, max_col + 0.5)
ax.set_ylim(-0.5, max_row + 0.5)
ax.invert_yaxis()  # 反转y轴，使周期表正向显示
ax.axis('off')

# Box size and spacing parameters
box_size = 0.7
spacing = 0.1
border_width = 2.5

for elem, (row, col) in ptable_layout.items():
    # Calculate position with spacing
    x = col + spacing
    y = row + spacing
    
    # Draw rectangle with thicker border and gaps
    rect = Rectangle((x, y), box_size, box_size, 
                    linewidth=border_width, 
                    edgecolor='black', 
                    facecolor='white', 
                    zorder=1)
    ax.add_patch(rect)
    
    # Calculate center for wedges and text
    cx, cy = x + box_size/2, y + box_size/2
    r = box_size/2 - 0.05  # Slightly smaller radius for wedges
    
    # DFT: 左 (custom color)
    if presence[elem][0]:
        wedge = Wedge((cx, cy), r, 120, 240, 
                     facecolor=dft_color, 
                     edgecolor='none', 
                     alpha=0.8, 
                     zorder=2)
        ax.add_patch(wedge)
    
    # EXP: 右 (custom color)
    if presence[elem][1]:
        wedge = Wedge((cx, cy), r, 0, 120, 
                     facecolor=exp_color, 
                     edgecolor='none', 
                     alpha=0.8, 
                     zorder=2)
        ax.add_patch(wedge)
    
    # Explor: 下 (custom color)
    if presence[elem][2]:
        wedge = Wedge((cx, cy), r, 240, 360, 
                     facecolor=explor_color, 
                     edgecolor='none', 
                     alpha=0.8, 
                     zorder=2)
        ax.add_patch(wedge)
    
    # Text color: black for used elements, gray for unused
    text_color = 'black' if elem in used_elements else 'gray'
    ax.text(cx, cy, elem, ha='center', va='center', 
            fontsize=12, fontweight='bold', color=text_color, zorder=3)

# Create custom legend with custom colors and better labels
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='DFT database', 
           markerfacecolor=dft_color, markersize=12, markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='o', color='w', label='experimental database', 
           markerfacecolor=exp_color, markersize=12, markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='o', color='w', label='chemical space exploration', 
           markerfacecolor=explor_color, markersize=12, markeredgecolor='black', markeredgewidth=1)
]

# Position legend just above transition metals, each on a new line
ax.legend(handles=legend_elements, loc='upper center', 
         bbox_to_anchor=(0.5, 0.85), fontsize=14, frameon=True,
         framealpha=0.9, fancybox=True, shadow=True, ncol=1)
plt.tight_layout()

os.makedirs('Output/visual', exist_ok=True)  # 确保文件夹存在
plt.savefig('Output/visual/element_periodic_table_venn.png', dpi=300, bbox_inches='tight')
plt.show()
