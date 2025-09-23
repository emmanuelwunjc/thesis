import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
with open('diabetes_demographics_crosstabs.json', 'r') as f:
    demo_data = json.load(f)

age_data = demo_data['crosstabs']['age']['count']

# 定义年龄组
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

# 按年龄组汇总数据
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

# 准备数据
groups = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
diabetic_counts = [age_groups[g]['diabetic'] for g in groups]
non_diabetic_counts = [age_groups[g]['non_diabetic'] for g in groups]
total_counts = [diabetic_counts[i] + non_diabetic_counts[i] for i in range(len(groups))]

# 计算百分比
diabetic_percentages = [count/sum(diabetic_counts)*100 for count in diabetic_counts]
total_percentages = [count/sum(total_counts)*100 for count in total_counts]

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：柱状图对比
x_pos = np.arange(len(groups))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, diabetic_percentages, width, 
                label='糖尿病组', color='red', alpha=0.7, edgecolor='darkred')
bars2 = ax1.bar(x_pos + width/2, total_percentages, width, 
                label='完整数据库', color='blue', alpha=0.7, edgecolor='darkblue')

ax1.set_xlabel('年龄组', fontsize=12, fontweight='bold')
ax1.set_ylabel('百分比 (%)', fontsize=12, fontweight='bold')
ax1.set_title('年龄组分布对比 (柱状图)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(groups, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 右图：折线图对比
ax2.plot(groups, diabetic_percentages, 'o-', color='red', linewidth=2, 
         markersize=6, label='糖尿病组', markerfacecolor='red', markeredgecolor='darkred')
ax2.plot(groups, total_percentages, 's-', color='blue', linewidth=2, 
         markersize=6, label='完整数据库', markerfacecolor='blue', markeredgecolor='darkblue')

ax2.set_xlabel('年龄组', fontsize=12, fontweight='bold')
ax2.set_ylabel('百分比 (%)', fontsize=12, fontweight='bold')
ax2.set_title('年龄组分布对比 (折线图)', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(groups)))
ax2.set_xticklabels(groups, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('age_group_comparison.png', dpi=300, bbox_inches='tight')
print("年龄组对比图已保存为: age_group_comparison.png")

# 打印详细统计
print(f"\n年龄组分布统计:")
print(f"{'年龄组':<8} {'糖尿病组':<12} {'完整数据库':<12} {'糖尿病占比':<10}")
print("-" * 50)
for i, group in enumerate(groups):
    diabetic_pct = diabetic_percentages[i]
    total_pct = total_percentages[i]
    diabetic_ratio = diabetic_counts[i] / total_counts[i] * 100 if total_counts[i] > 0 else 0
    print(f"{group:<8} {diabetic_pct:>8.2f}%    {total_pct:>8.2f}%    {diabetic_ratio:>8.2f}%")

plt.show()
