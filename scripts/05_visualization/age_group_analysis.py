#!/usr/bin/env python3
"""
Age Group Analysis: Compare age distributions between diabetic and non-diabetic groups.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up matplotlib for English labels only
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# Get repository root
repo_root = Path(__file__).parent.parent.parent

# Read data
json_path = repo_root / 'analysis' / 'results' / 'diabetes_demographics_crosstabs.json'
with open(json_path, 'r', encoding='utf-8') as f:
    demo_data = json.load(f)

age_data = demo_data['crosstabs']['age']['count']

# Define age groups
def get_age_group(age):
    if age < 30:
        return '18-29'
    elif age < 40:
        return '30-39'
    elif age < 50:
        return '40-49'
    elif age < 60:
        return '50-59'
    elif age < 70:
        return '60-69'
    elif age < 80:
        return '70-79'
    else:
        return '80+'

# Aggregate data by age group
age_groups = {}
for age_str, counts in age_data.items():
    try:
        age = int(age_str)
        if 18 <= age <= 100:
            group = get_age_group(age)
            if group not in age_groups:
                age_groups[group] = {'diabetic': 0, 'non_diabetic': 0}
            age_groups[group]['diabetic'] += counts['Diabetic']
            age_groups[group]['non_diabetic'] += counts['Non-Diabetic']
    except ValueError:
        continue

# Prepare data
groups = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
diabetic_counts = [age_groups[g]['diabetic'] for g in groups]
non_diabetic_counts = [age_groups[g]['non_diabetic'] for g in groups]
total_counts = [diabetic_counts[i] + non_diabetic_counts[i] for i in range(len(groups))]

# Calculate percentages
diabetic_percentages = [count/sum(diabetic_counts)*100 for count in diabetic_counts]
total_percentages = [count/sum(total_counts)*100 for count in total_counts]

# Create charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Bar chart comparison
x_pos = np.arange(len(groups))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, diabetic_percentages, width, 
                label='Diabetic Group', color='#E74C3C', alpha=0.7, edgecolor='darkred')
bars2 = ax1.bar(x_pos + width/2, total_percentages, width, 
                label='Full Database', color='#3498DB', alpha=0.7, edgecolor='darkblue')

ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax1.set_title('Age Group Distribution Comparison (Bar Chart)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(groups, rotation=45, ha='right')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Right plot: Line chart comparison
ax2.plot(groups, diabetic_percentages, 'o-', color='#E74C3C', linewidth=2, 
         markersize=6, label='Diabetic Group', markerfacecolor='#E74C3C', markeredgecolor='darkred')
ax2.plot(groups, total_percentages, 's-', color='#3498DB', linewidth=2, 
         markersize=6, label='Full Database', markerfacecolor='#3498DB', markeredgecolor='darkblue')

ax2.set_xlabel('Age Group', fontsize=12, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Age Group Distribution Comparison (Line Chart)', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(groups)))
ax2.set_xticklabels(groups, rotation=45, ha='right')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = repo_root / 'figures' / 'age_group_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Age group comparison plot saved: {output_path}")

# Print detailed statistics
print(f"\nAge Group Distribution Statistics:")
print(f"{'Age Group':<10} {'Diabetic':<12} {'Full Database':<15} {'Diabetic %':<12}")
print("-" * 55)
for i, group in enumerate(groups):
    diabetic_pct = diabetic_percentages[i]
    total_pct = total_percentages[i]
    diabetic_ratio = diabetic_counts[i] / total_counts[i] * 100 if total_counts[i] > 0 else 0
    print(f"{group:<10} {diabetic_pct:>8.2f}%    {total_pct:>10.2f}%    {diabetic_ratio:>8.2f}%")

plt.close()
