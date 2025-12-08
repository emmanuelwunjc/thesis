#!/usr/bin/env python3
"""
Visualize Diabetes Analysis Data

Creates visualizations for diabetes prevalence and age distribution.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
summary_path = repo_root / 'analysis' / 'results' / 'diabetes_summary.json'
demo_path = repo_root / 'analysis' / 'results' / 'diabetes_demographics_crosstabs.json'

try:
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    with open(demo_path, 'r', encoding='utf-8') as f:
        demo = json.load(f)
except FileNotFoundError as e:
    print(f"❌ Data file not found: {e}")
    print(f"Looking for: {summary_path} or {demo_path}")
    sys.exit(1)

# 1. Diabetes prevalence pie chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
diabetic = summary['summary']['num_diabetic']
non_diabetic = summary['summary']['num_non_diabetic']
ax1.pie([diabetic, non_diabetic], labels=['Diabetic', 'Non-Diabetic'], 
        autopct='%1.1f%%', colors=['#E74C3C', '#3498DB'], startangle=90)
ax1.set_title('Diabetes Prevalence Distribution', fontsize=14, fontweight='bold', pad=15)

# 2. Age distribution bar chart (select normal age range)
age_data = demo['crosstabs']['age']['count']
normal_ages = {k: v for k, v in age_data.items() if k.isdigit() and 18 <= int(k) <= 80}
ages = sorted([int(k) for k in normal_ages.keys()])
diabetic_counts = [normal_ages[str(age)]['Diabetic'] for age in ages]
non_diabetic_counts = [normal_ages[str(age)]['Non-Diabetic'] for age in ages]

x = range(len(ages))
width = 0.35
ax2.bar([i - width/2 for i in x], diabetic_counts, width, 
        label='Diabetic', alpha=0.8, color='#E74C3C')
ax2.bar([i + width/2 for i in x], non_diabetic_counts, width, 
        label='Non-Diabetic', alpha=0.8, color='#3498DB')
ax2.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title('Age Distribution Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x[::5])  # Show label every 5 years
ax2.set_xticklabels(ages[::5], rotation=45, ha='right')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save figure
output_path = repo_root / 'figures' / 'diabetes_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Diabetes analysis plot saved: {output_path}")

plt.close()
