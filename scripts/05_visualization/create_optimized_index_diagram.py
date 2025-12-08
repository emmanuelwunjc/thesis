import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure with better proportions
fig, ax = plt.subplots(figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 16)
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

# Main title
ax.text(9, 15, 'Privacy Caution Index Construction', ha='center', va='center',
        fontsize=22, fontweight='bold')

# Main index box (larger and centered)
main_box = FancyBboxPatch((6, 12), 6, 2, boxstyle="round,pad=0.2", 
                         facecolor=colors['main'], edgecolor='black', linewidth=3)
ax.add_patch(main_box)
ax.text(9, 13, 'Privacy Caution Index', ha='center', va='center', 
        fontsize=18, fontweight='bold', color='white')
ax.text(9, 12.5, '(0-1 Scale: Higher = More Cautious)', ha='center', va='center',
        fontsize=12, color='white')

# Sub-dimension boxes with better spacing
subdims = [
    ('Sharing\nWillingness', 1, 9, colors['sharing'], [
        'WillingShareData_HCP2',
        'SharedHealthDeviceInfo2', 
        'SocMed_SharedPers',
        'SocMed_SharedGen'
    ]),
    ('Portal\nUsage', 4.5, 9, colors['portals'], [
        'AccessOnlineRecord3',
        'OnlinePortal_PCP',
        'OnlinePortal_OthHCP',
        'OnlinePortal_Insurer',
        'OnlinePortal_Lab',
        'OnlinePortal_Pharmacy',
        'OnlinePortal_Hospital'
    ]),
    ('Device\nUsage', 8, 9, colors['devices'], [
        'UseDevice_Computer',
        'UseDevice_SmPhone',
        'UseDevice_Tablet',
        'UseDevice_SmWatch'
    ]),
    ('Trust\nLevels', 11.5, 9, colors['trust'], [
        'TrustHCSystem',
        'CancerTrustDoctor',
        'CancerTrustScientists',
        'CancerTrustFamily'
    ]),
    ('Social\nMedia', 15, 9, colors['social'], [
        'SocMed_Visited',
        'MisleadingHealthInfo'
    ]),
    ('Other\nPrivacy', 1, 5.5, colors['other'], [
        'ConfidentMedForms',
        'WillingUseTelehealth'
    ])
]

# Draw sub-dimension boxes with better sizing
for name, x, y, color, vars_list in subdims:
    # Sub-dimension box (larger)
    sub_box = FancyBboxPatch((x, y), 2.8, 3, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(sub_box)
    
    # Sub-dimension title
    ax.text(x+1.4, y+2.3, name, ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    
    # Variable count
    ax.text(x+1.4, y+2, f'{len(vars_list)} variables', ha='center', va='center',
            fontsize=10, color='white')
    
    # Variable list (better formatted)
    var_text = '\n'.join([v[:18] + '...' if len(v) > 18 else v for v in vars_list])
    ax.text(x+1.4, y+1, var_text, ha='center', va='center',
            fontsize=8, color='white', linespacing=1.2)

# Arrows from sub-dimensions to main index (better positioned)
arrow_positions = [
    (2.4, 12, 7.5, 12.8),   # Sharing
    (5.9, 12, 8.2, 12.8),   # Portals  
    (9.4, 12, 8.9, 12.8),   # Devices
    (12.9, 12, 9.6, 12.8),  # Trust
    (16.4, 12, 10.3, 12.8), # Social
    (2.4, 8.5, 7.5, 12.2)   # Other
]

for x1, y1, x2, y2 in arrow_positions:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=3, color='gray'))

# Scoring explanation (better positioned and formatted)
scoring_box = FancyBboxPatch((1, 1), 16, 3.5, boxstyle="round,pad=0.2",
                            facecolor='lightgray', edgecolor='black', linewidth=2)
ax.add_patch(scoring_box)

ax.text(9, 4, 'Scoring System (0-1 Scale)', ha='center', va='center',
        fontsize=16, fontweight='bold')

scoring_text = """• 0 = Low Caution (More sharing, trusting, digital engagement)
• 1 = High Caution (Less sharing, distrusting, limited digital engagement)  
• 0.5 = Moderate Caution (Default for missing values)

Examples: WillingShareData_HCP2: Yes=0, No=1 | TrustHCSystem: A lot=0, Not at all=1 | UseDevice_Computer: Yes=0, No=1"""

ax.text(9, 2.5, scoring_text, ha='center', va='center', fontsize=11, linespacing=1.3)

# Results box (better positioned)
results_box = FancyBboxPatch((1, 4.8), 16, 1, boxstyle="round,pad=0.1",
                            facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(results_box)

results_text = "Results: Diabetic Group (0.476) vs Non-Diabetic Group (0.467) | Difference: +0.010 (Diabetics slightly more cautious)"
ax.text(9, 5.3, results_text, ha='center', va='center', fontsize=13, fontweight='bold')

plt.tight_layout()

# Save figure with correct path
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
output_path = repo_root / 'figures' / 'privacy_index_construction_diagram_optimized.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Optimized Privacy Index Construction Diagram saved: {output_path}")
plt.close()
