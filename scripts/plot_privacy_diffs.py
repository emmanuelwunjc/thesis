import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

j = json.loads(Path('privacy_dummies_compare.json').read_text())
comp = j['comparison']

rows = []
for var, dummies in comp.items():
    for lvl, stats in dummies.items():
        diff = stats.get('diff')
        md = stats.get('mean_diabetic')
        mn = stats.get('mean_non_diabetic')
        if diff is None or not (diff==diff):
            continue
        rows.append((abs(diff), diff, var, lvl, md, mn))
rows.sort(reverse=True)

def save_top10_diff_chart():
    top = rows[:10]
    labels = [f"{v} | {lvl}" for _,_,v,lvl,_,_ in top]
    diffs = [d for _,d,_,_,_,_ in top]
    colors = ['red' if d>0 else 'blue' for d in diffs]
    fig, ax = plt.subplots(figsize=(10,6))
    y = np.arange(len(top))
    ax.barh(y, diffs, color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('差异 (糖尿病 - 非糖)')
    ax.set_title('隐私/数据相关变量：Top-10 组间差（加权比例）')
    for i, d in enumerate(diffs):
        ax.text(d + (0.005 if d>=0 else -0.005), i, f"{d:+.3f}", va='center', ha='left' if d>=0 else 'right')
    plt.tight_layout()
    plt.savefig('privacy_top10_diffs.png', dpi=300, bbox_inches='tight')


def grouped_bar_for_var(var_name, max_levels=4, filename='tmp.png'):
    dd = comp.get(var_name, {})
    # pick top levels by diabetic+non mean sum
    items = []
    for lvl, stats in dd.items():
        md = stats.get('mean_diabetic')
        mn = stats.get('mean_non_diabetic')
        if any(v is None or not (v==v) for v in [md,mn]):
            continue
        items.append((md+mn, lvl, md, mn))
    items.sort(reverse=True)
    items = items[:max_levels]
    if not items:
        return
    lvls = [it[1] for it in items]
    md = [it[2] for it in items]
    mn = [it[3] for it in items]
    x = np.arange(len(lvls))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - w/2, [v*100 for v in md], w, label='糖尿病', color='red', alpha=0.8, edgecolor='darkred')
    ax.bar(x + w/2, [v*100 for v in mn], w, label='非糖尿病', color='blue', alpha=0.8, edgecolor='darkblue')
    ax.set_xticks(x)
    ax.set_xticklabels(lvls, rotation=20, ha='right')
    ax.set_ylabel('百分比 (%)')
    ax.set_title(f'{var_name} 分布对比（加权）')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')

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

print('Generated: privacy_top10_diffs.png and detailed privacy_* plots')
