import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
colors = {
    'main': '#2E86AB',
    'sharing': '#A23B72', 
    'portals': '#F18F01',
    'devices': '#C73E1D',
    'trust': '#7209B7',
    'social': '#2D5016',
    'other': '#8B4513'
}

# Main index box
main_box = FancyBboxPatch((3, 9), 4, 1.5, boxstyle="round,pad=0.1", 
                         facecolor=colors['main'], edgecolor='black', linewidth=2)
ax.add_patch(main_box)
ax.text(5, 9.75, 'Privacy Caution Index', ha='center', va='center', 
        fontsize=16, fontweight='bold', color='white')

# Sub-dimension boxes
subdims = [
    ('Sharing Willingness', 1, 7, colors['sharing'], [
        'WillingShareData_HCP2',
        'SharedHealthDeviceInfo2', 
        'SocMed_SharedPers',
        'SocMed_SharedGen'
    ]),
    ('Portal Usage', 3, 7, colors['portals'], [
        'AccessOnlineRecord3',
        'OnlinePortal_PCP',
        'OnlinePortal_OthHCP',
        'OnlinePortal_Insurer',
        'OnlinePortal_Lab',
        'OnlinePortal_Pharmacy',
        'OnlinePortal_Hospital'
    ]),
    ('Device Usage', 5, 7, colors['devices'], [
        'UseDevice_Computer',
        'UseDevice_SmPhone',
        'UseDevice_Tablet',
        'UseDevice_SmWatch'
    ]),
    ('Trust Levels', 7, 7, colors['trust'], [
        'TrustHCSystem',
        'CancerTrustDoctor',
        'CancerTrustScientists',
        'CancerTrustFamily'
    ]),
    ('Social Media', 1, 4, colors['social'], [
        'SocMed_Visited',
        'MisleadingHealthInfo'
    ]),
    ('Other Privacy', 3, 4, colors['other'], [
        'ConfidentMedForms',
        'WillingUseTelehealth'
    ])
]

# Draw sub-dimension boxes
for name, x, y, color, vars_list in subdims:
    # Sub-dimension box
    sub_box = FancyBboxPatch((x, y), 1.8, 2, boxstyle="round,pad=0.05",
                            facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(sub_box)
    
    # Sub-dimension title
    ax.text(x+0.9, y+1.7, name, ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    
    # Variable count
    ax.text(x+0.9, y+1.4, f'{len(vars_list)} variables', ha='center', va='center',
            fontsize=8, color='white')
    
    # Variable list (truncated)
    var_text = '\n'.join([v[:15] + '...' if len(v) > 15 else v for v in vars_list[:3]])
    if len(vars_list) > 3:
        var_text += f'\n+{len(vars_list)-3} more'
    ax.text(x+0.9, y+0.8, var_text, ha='center', va='center',
            fontsize=7, color='white')

# Arrows from sub-dimensions to main index
arrow_positions = [
    (1.9, 8, 4.2, 9.5),  # Sharing
    (3.9, 8, 4.8, 9.5),  # Portals  
    (5.9, 8, 5.4, 9.5),  # Devices
    (7.9, 8, 5.8, 9.5),  # Trust
    (1.9, 6, 4.2, 9.2),  # Social
    (3.9, 6, 4.8, 9.2)   # Other
]

for x1, y1, x2, y2 in arrow_positions:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

# Scoring explanation
scoring_box = FancyBboxPatch((0.5, 0.5), 9, 2.5, boxstyle="round,pad=0.1",
                            facecolor='lightgray', edgecolor='black', linewidth=1)
ax.add_patch(scoring_box)

ax.text(5, 2.7, 'Scoring System (0-1 Scale)', ha='center', va='center',
        fontsize=14, fontweight='bold')

scoring_text = """
• 0 = Low Caution (More sharing, trusting, digital engagement)
• 1 = High Caution (Less sharing, distrusting, limited digital engagement)  
• 0.5 = Moderate Caution (Default for missing values)

Examples:
• WillingShareData_HCP2: Yes=0, No=1
• TrustHCSystem: A lot=0, Some=0.3, A little=0.7, Not at all=1
• UseDevice_Computer: Yes=0, No=1
"""

ax.text(5, 1.5, scoring_text, ha='center', va='center', fontsize=10)

# Results box
results_box = FancyBboxPatch((0.5, 3.2), 9, 0.8, boxstyle="round,pad=0.1",
                            facecolor='lightblue', edgecolor='black', linewidth=1)
ax.add_patch(results_box)

results_text = "Results: Diabetic (0.476) vs Non-Diabetic (0.467) | Difference: +0.010"
ax.text(5, 3.6, results_text, ha='center', va='center', fontsize=12, fontweight='bold')

# Title
ax.text(5, 11.5, 'Privacy Caution Index Construction', ha='center', va='center',
        fontsize=20, fontweight='bold')

# Legend
legend_text = "Legend: Each sub-dimension is averaged, then all sub-dimensions are averaged to create the final index"
ax.text(5, 0.2, legend_text, ha='center', va='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('privacy_index_construction_diagram.png', dpi=300, bbox_inches='tight')
print("Privacy Index Construction Diagram saved as: privacy_index_construction_diagram.png")
