#!/usr/bin/env python3
"""
Age Distribution Plot: Compare age distributions between diabetic and full database.
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

# Extract age data
age_data = demo_data['crosstabs']['age']['count']

# Filter normal age range (18-100 years)
normal_ages = {}
for age_str, counts in age_data.items():
    try:
        age = int(age_str)
        if 18 <= age <= 100:  # Only keep ages 18-100
            normal_ages[age] = counts
    except ValueError:
        continue

# Sort by age
ages = sorted(normal_ages.keys())
diabetic_counts = [normal_ages[age]['Diabetic'] for age in ages]
non_diabetic_counts = [normal_ages[age]['Non-Diabetic'] for age in ages]
total_counts = [diabetic_counts[i] + non_diabetic_counts[i] for i in range(len(ages))]

# Calculate percentages
diabetic_percentages = [count/sum(diabetic_counts)*100 for count in diabetic_counts]
total_percentages = [count/sum(total_counts)*100 for count in total_counts]

# Create chart
fig, ax = plt.subplots(figsize=(14, 8))

# Draw bar chart
width = 0.35
x_pos = np.arange(len(ages))

bars1 = ax.bar(x_pos - width/2, diabetic_percentages, width, 
               label='Diabetic Group', color='#E74C3C', alpha=0.7, edgecolor='darkred')
bars2 = ax.bar(x_pos + width/2, total_percentages, width, 
               label='Full Database', color='#3498DB', alpha=0.7, edgecolor='darkblue')

# Set labels and title
ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Diabetic Group vs Full Database: Age Distribution Comparison', 
             fontsize=16, fontweight='bold', pad=20)

# Set x-axis labels (show every 5 years)
step = 5
ax.set_xticks(x_pos[::step])
ax.set_xticklabels(ages[::step], rotation=45, ha='right')

# Add legend
ax.legend(fontsize=11, loc='upper right')

# Add grid
ax.grid(True, alpha=0.3, axis='y')

# Adjust layout
plt.tight_layout()

# Save chart
output_path = repo_root / 'figures' / 'age_distribution_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Age distribution comparison plot saved: {output_path}")

# Display statistics
print(f"\nStatistics:")
print(f"Age range: {min(ages)}-{max(ages)} years")
print(f"Diabetic group total: {sum(diabetic_counts):,}")
print(f"Full database total: {sum(total_counts):,}")
print(f"Diabetic group mean age: {np.average(ages, weights=diabetic_counts):.1f} years")
print(f"Full database mean age: {np.average(ages, weights=total_counts):.1f} years")

# Find peak ages
diabetic_peak_age = ages[np.argmax(diabetic_percentages)]
total_peak_age = ages[np.argmax(total_percentages)]
print(f"Diabetic group peak age: {diabetic_peak_age} years ({max(diabetic_percentages):.2f}%)")
print(f"Full database peak age: {total_peak_age} years ({max(total_percentages):.2f}%)")

plt.close()
