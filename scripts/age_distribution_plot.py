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

# 提取年龄数据
age_data = demo_data['crosstabs']['age']['count']

# 过滤正常年龄范围 (18-100岁)
normal_ages = {}
for age_str, counts in age_data.items():
    try:
        age = int(age_str)
        if 18 <= age <= 100:  # 只保留18-100岁的数据
            normal_ages[age] = counts
    except ValueError:
        continue

# 按年龄排序
ages = sorted(normal_ages.keys())
diabetic_counts = [normal_ages[age]['Diabetic'] for age in ages]
non_diabetic_counts = [normal_ages[age]['Non-Diabetic'] for age in ages]
total_counts = [diabetic_counts[i] + non_diabetic_counts[i] for i in range(len(ages))]

# 计算百分比
diabetic_percentages = [count/sum(diabetic_counts)*100 for count in diabetic_counts]
total_percentages = [count/sum(total_counts)*100 for count in total_counts]

# 创建图表
fig, ax = plt.subplots(figsize=(14, 8))

# 绘制柱状图
width = 0.35
x_pos = np.arange(len(ages))

bars1 = ax.bar(x_pos - width/2, diabetic_percentages, width, 
               label='糖尿病组', color='red', alpha=0.7, edgecolor='darkred')
bars2 = ax.bar(x_pos + width/2, total_percentages, width, 
               label='完整数据库', color='blue', alpha=0.7, edgecolor='darkblue')

# 设置标签和标题
ax.set_xlabel('年龄 (岁)', fontsize=12, fontweight='bold')
ax.set_ylabel('百分比 (%)', fontsize=12, fontweight='bold')
ax.set_title('糖尿病组 vs 完整数据库年龄分布对比', fontsize=16, fontweight='bold', pad=20)

# 设置x轴标签 (每5年显示一个)
step = 5
ax.set_xticks(x_pos[::step])
ax.set_xticklabels(ages[::step], rotation=45)

# 添加图例
ax.legend(fontsize=11, loc='upper right')

# 添加网格
ax.grid(True, alpha=0.3, axis='y')

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('age_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("年龄分布对比图已保存为: age_distribution_comparison.png")

# 显示统计信息
print(f"\n统计信息:")
print(f"年龄范围: {min(ages)}-{max(ages)}岁")
print(f"糖尿病组总人数: {sum(diabetic_counts):,}")
print(f"完整数据库总人数: {sum(total_counts):,}")
print(f"糖尿病组平均年龄: {np.average(ages, weights=diabetic_counts):.1f}岁")
print(f"完整数据库平均年龄: {np.average(ages, weights=total_counts):.1f}岁")

# 找出峰值年龄
diabetic_peak_age = ages[np.argmax(diabetic_percentages)]
total_peak_age = ages[np.argmax(total_percentages)]
print(f"糖尿病组峰值年龄: {diabetic_peak_age}岁 ({max(diabetic_percentages):.2f}%)")
print(f"完整数据库峰值年龄: {total_peak_age}岁 ({max(total_percentages):.2f}%)")

plt.show()
