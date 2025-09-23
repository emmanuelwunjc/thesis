import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create detailed table
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# Data for the table
data = [
    ['Sub-dimension', 'Variables', 'Scoring Logic', 'Example Mapping'],
    ['Sharing Willingness', '4 variables', 'Share = Less Cautious', 'Yes=0, No=1'],
    ['Portal Usage', '7 variables', 'Use Portals = Less Cautious', 'Selected=0, Not selected=1'],
    ['Device Usage', '4 variables', 'Use Devices = Less Cautious', 'Yes=0, No=1'],
    ['Trust Levels', '4 variables', 'Trust = Less Cautious', 'A lot=0, Not at all=1'],
    ['Social Media', '2 variables', 'Use Social Media = Less Cautious', 'Yes=0, No=1'],
    ['Other Privacy', '2 variables', 'Confident/Willing = Less Cautious', 'Yes=0, No=1']
]

# Create table
table = ax.table(cellText=data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header row
for i in range(4):
    table[(0, i)].set_facecolor('#4A90E2')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color alternating rows
for i in range(1, len(data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

# Add title
ax.text(0.5, 0.95, 'Privacy Caution Index: Sub-dimension Structure', 
        ha='center', va='center', fontsize=16, fontweight='bold', 
        transform=ax.transAxes)

# Add formula
formula_text = 'Final Index = (Sharing + Portals + Devices + Trust + Social + Other) / 6'
ax.text(0.5, 0.88, formula_text, ha='center', va='center', fontsize=12, 
        style='italic', transform=ax.transAxes)

# Add interpretation
interpretation = """
Interpretation: Higher values indicate greater privacy caution
• 0.0-0.3: Low caution (high digital engagement, sharing, trust)
• 0.3-0.7: Moderate caution (balanced digital behavior)  
• 0.7-1.0: High caution (limited digital engagement, privacy-focused)
"""
ax.text(0.5, 0.15, interpretation, ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7),
        transform=ax.transAxes)

plt.tight_layout()
plt.savefig('privacy_index_detailed_table.png', dpi=300, bbox_inches='tight')
print("Detailed Index Table saved as: privacy_index_detailed_table.png")
