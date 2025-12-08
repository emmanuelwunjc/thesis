#!/usr/bin/env python3
"""
Plot Privacy Differences Between Diabetic and Non-Diabetic Patients

Creates visualizations showing top privacy-related variable differences.
"""

import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up matplotlib for English labels only
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# Get repository root
repo_root = Path(__file__).parent.parent.parent

# Load comparison data
json_path = repo_root / 'analysis' / 'results' / 'privacy_dummies_compare.json'
j = json.loads(json_path.read_text(encoding='utf-8'))
comp = j['comparison']

rows = []
for var, dummies in comp.items():
    for lvl, stats in dummies.items():
        diff = stats.get('diff')
        md = stats.get('mean_diabetic')
        mn = stats.get('mean_non_diabetic')
        if diff is None or not (diff==diff):  # Check for NaN
            continue
        rows.append((abs(diff), diff, var, lvl, md, mn))
rows.sort(reverse=True)

def save_top10_diff_chart():
    """Create top 10 differences chart."""
    top = rows[:10]
    labels = [f"{v} | {lvl}" for _,_,v,lvl,_,_ in top]
    diffs = [d for _,d,_,_,_,_ in top]
    colors = ['#E74C3C' if d>0 else '#3498DB' for d in diffs]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(top))
    ax.barh(y, diffs, color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Difference (Diabetic - Non-Diabetic)', fontsize=12)
    ax.set_title('Privacy/Data-Related Variables: Top-10 Group Differences (Weighted Proportions)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, d in enumerate(diffs):
        ax.text(d + (0.005 if d>=0 else -0.005), i, f"{d:+.3f}", 
                va='center', ha='left' if d>=0 else 'right', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    output_path = repo_root / 'figures' / 'privacy_top10_diffs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Top 10 differences chart saved: {output_path}")


def grouped_bar_for_var(var_name, max_levels=4, filename='tmp.png'):
    """Create grouped bar chart for a specific variable."""
    dd = comp.get(var_name, {})
    # Pick top levels by diabetic+non mean sum
    items = []
    for lvl, stats in dd.items():
        md = stats.get('mean_diabetic')
        mn = stats.get('mean_non_diabetic')
        if any(v is None or not (v==v) for v in [md,mn]):  # Check for NaN
            continue
        items.append((md+mn, lvl, md, mn))
    items.sort(reverse=True)
    items = items[:max_levels]
    
    if not items:
        return
    
    lvls = [it[1] for it in items]
    md = [it[2] for it in items]
    mn = [it[3] for it in items]
    
    # Truncate long labels
    lvls = [lvl[:30] + '...' if len(lvl) > 30 else lvl for lvl in lvls]
    
    x = np.arange(len(lvls))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, [v*100 for v in md], w, label='Diabetic', 
           color='#E74C3C', alpha=0.8, edgecolor='darkred')
    ax.bar(x + w/2, [v*100 for v in mn], w, label='Non-Diabetic', 
           color='#3498DB', alpha=0.8, edgecolor='darkblue')
    ax.set_xticks(x)
    ax.set_xticklabels(lvls, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'{var_name}: Distribution Comparison (Weighted)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    
    output_path = repo_root / 'figures' / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Chart saved: {output_path}")

# Generate top 10 differences chart
save_top10_diff_chart()

# Selected detailed plots
selected = [
    ('SharedHealthDeviceInfo2', 'privacy_shared_device.png'),
    ('UseDevice_Computer', 'privacy_use_computer.png'),
    ('UseDevice_SmWatch', 'privacy_use_watch.png'),
    ('TrustHCSystem', 'privacy_trust_hcsystem.png'),
    ('CancerTrustScientists', 'privacy_trust_scientists.png'),
    ('OnlinePortal_Pharmacy', 'privacy_portal_pharmacy.png'),
]

for var, fn in selected:
    grouped_bar_for_var(var, filename=fn)

print('\n✅ All privacy difference plots generated successfully!')
