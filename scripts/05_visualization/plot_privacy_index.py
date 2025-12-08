#!/usr/bin/env python3
"""
Plot Privacy Caution Index Analysis

Creates visualizations comparing privacy caution index between diabetic and non-diabetic patients.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up matplotlib for English labels only
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# Get repository root
repo_root = Path(__file__).parent.parent.parent

# Load results
results_path = repo_root / 'analysis' / 'results' / 'privacy_caution_index.json'
with open(results_path, 'r', encoding='utf-8') as f:
    result = json.load(f)

# Load individual data
data_path = repo_root / 'analysis' / 'data' / 'privacy_caution_index_individual.csv'
df = pd.read_csv(data_path)

# 1. Overall distribution comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Histogram comparison
dia_data = df[df['diabetic']==1]['privacy_caution_index'].dropna()
non_data = df[df['diabetic']==0]['privacy_caution_index'].dropna()

ax1.hist(dia_data, bins=30, alpha=0.7, color='#E74C3C', label='Diabetic', density=True)
ax1.hist(non_data, bins=30, alpha=0.7, color='#3498DB', label='Non-Diabetic', density=True)
ax1.set_xlabel('Privacy Caution Index', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Privacy Caution Index Distribution Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Box plot
box_data = [dia_data, non_data]
bp = ax2.boxplot(box_data, labels=['Diabetic', 'Non-Diabetic'], patch_artist=True)
bp['boxes'][0].set_facecolor('#E74C3C')
bp['boxes'][1].set_facecolor('#3498DB')
ax2.set_ylabel('Privacy Caution Index', fontsize=12)
ax2.set_title('Privacy Caution Index Box Plot', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Sub-dimension differences bar chart
subdims = result['privacy_caution_index']['subdimensions']
names = list(subdims.keys())
# Map subdimension names to readable labels
name_labels = {
    'sharing': 'Sharing',
    'portals': 'Portals',
    'devices': 'Devices',
    'trust': 'Trust',
    'social': 'Social',
    'other': 'Other'
}
labels = [name_labels.get(name, name.title()) for name in names]
diffs = [subdims[name]['difference'] for name in names]
colors = ['#E74C3C' if d > 0 else '#3498DB' for d in diffs]

ax3.barh(labels, diffs, color=colors, alpha=0.7)
ax3.set_xlabel('Group Difference (Diabetic - Non-Diabetic)', fontsize=12)
ax3.set_title('Sub-dimension Differences', fontsize=14, fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax3.grid(True, alpha=0.3, axis='x')

# Sub-dimension means comparison
dia_means = [subdims[name]['diabetic_mean'] for name in names]
non_means = [subdims[name]['non_diabetic_mean'] for name in names]

x = np.arange(len(labels))
width = 0.35
ax4.bar(x - width/2, dia_means, width, label='Diabetic', color='#E74C3C', alpha=0.7)
ax4.bar(x + width/2, non_means, width, label='Non-Diabetic', color='#3498DB', alpha=0.7)
ax4.set_xlabel('Sub-dimension', fontsize=12)
ax4.set_ylabel('Mean Index Value', fontsize=12)
ax4.set_title('Sub-dimension Means Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(labels, rotation=45, ha='right')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save figure
output_path = repo_root / 'figures' / 'privacy_caution_index_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Privacy caution index analysis plot saved: {output_path}")

# 2. Regression preparation data summary
print("\n=== Regression Analysis Preparation ===")
print("Dependent Variable: WillingShareData_HCP2 (Willingness to share data with HCP)")
print("Independent Variables:")
print("- diabetic: Diabetes status dummy variable")
print("- privacy_caution_index: Privacy caution index (0-1 scale)")
print("- demographics: Age, education, region, urban/rural, etc.")

# Data quality check
print(f"\n=== Data Quality ===")
print(f"Total sample: {len(df):,}")
print(f"Diabetic sample: {df['diabetic'].sum():,}")
print(f"Missing privacy index: {df['privacy_caution_index'].isna().sum():,}")
print(f"Privacy index range: {df['privacy_caution_index'].min():.3f} - {df['privacy_caution_index'].max():.3f}")

plt.close()
