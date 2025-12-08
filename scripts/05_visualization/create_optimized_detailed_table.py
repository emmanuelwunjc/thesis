import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create detailed table with better layout
fig, ax = plt.subplots(figsize=(16, 12))
ax.axis('off')

# Data for the table with better formatting
data = [
    ['Sub-dimension', 'Variables\nCount', 'Key Variables', 'Scoring Logic', 'Example Mapping'],
    ['Sharing Willingness', '4', 'WillingShareData_HCP2\nSharedHealthDeviceInfo2', 'Share = Less Cautious\n(More willing to share)', 'Yes = 0\nNo = 1'],
    ['Portal Usage', '7', 'AccessOnlineRecord3\nOnlinePortal_PCP\nOnlinePortal_Pharmacy', 'Use Portals = Less Cautious\n(More digital engagement)', 'Selected = 0\nNot selected = 1'],
    ['Device Usage', '4', 'UseDevice_Computer\nUseDevice_SmPhone\nUseDevice_SmWatch', 'Use Devices = Less Cautious\n(More tech adoption)', 'Yes = 0\nNo = 1'],
    ['Trust Levels', '4', 'TrustHCSystem\nCancerTrustDoctor\nCancerTrustScientists', 'Trust = Less Cautious\n(More trusting)', 'A lot = 0\nNot at all = 1'],
    ['Social Media', '2', 'SocMed_Visited\nMisleadingHealthInfo', 'Use Social Media = Less Cautious\n(More social engagement)', 'Yes = 0\nNo = 1'],
    ['Other Privacy', '2', 'ConfidentMedForms\nWillingUseTelehealth', 'Confident/Willing = Less Cautious\n(More confident)', 'Yes = 0\nNo = 1']
]

# Create table with better spacing
table = ax.table(cellText=data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.15, 0.25, 0.25, 0.15])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Color header row
for i in range(5):
    table[(0, i)].set_facecolor('#4A90E2')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color alternating rows
for i in range(1, len(data)):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

# Add title
ax.text(0.5, 0.95, 'Privacy Caution Index: Detailed Sub-dimension Structure', 
        ha='center', va='center', fontsize=18, fontweight='bold', 
        transform=ax.transAxes)

# Add formula
formula_text = 'Final Index = (Sharing + Portals + Devices + Trust + Social + Other) ÷ 6'
ax.text(0.5, 0.88, formula_text, ha='center', va='center', fontsize=14, 
        style='italic', transform=ax.transAxes)

# Add interpretation with better formatting
interpretation = """
Interpretation Guidelines:
• 0.0 - 0.3: Low Caution (High digital engagement, frequent sharing, high trust)
• 0.3 - 0.7: Moderate Caution (Balanced digital behavior, selective sharing)  
• 0.7 - 1.0: High Caution (Limited digital engagement, privacy-focused, cautious sharing)

Current Results: Diabetic patients show slightly higher caution (+0.010) compared to non-diabetic patients
"""
ax.text(0.5, 0.15, interpretation, ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7),
        transform=ax.transAxes)

# Add summary box
summary_text = "Total Variables: 23 | Sub-dimensions: 6 | Scale: 0-1 | Missing Value Default: 0.5"
ax.text(0.5, 0.05, summary_text, ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
        transform=ax.transAxes)

plt.tight_layout()

# Save figure with correct path
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
output_path = repo_root / 'figures' / 'privacy_index_detailed_table_optimized.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Optimized Detailed Index Table saved: {output_path}")
plt.close()
