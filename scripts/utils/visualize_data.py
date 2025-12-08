import json
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
with open('diabetes_summary.json', 'r') as f:
    summary = json.load(f)

with open('diabetes_demographics_crosstabs.json', 'r') as f:
    demo = json.load(f)

# 1. 糖尿病患病率饼图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 饼图
diabetic = summary['summary']['num_diabetic']
non_diabetic = summary['summary']['num_non_diabetic']
ax1.pie([diabetic, non_diabetic], labels=['糖尿病', '非糖尿病'], autopct='%1.1f%%')
ax1.set_title('糖尿病患病率分布')

# 2. 年龄分布柱状图 (选择正常年龄范围)
age_data = demo['crosstabs']['age']['count']
normal_ages = {k: v for k, v in age_data.items() if k.isdigit() and 18 <= int(k) <= 80}
ages = sorted([int(k) for k in normal_ages.keys()])
diabetic_counts = [normal_ages[str(age)]['Diabetic'] for age in ages]
non_diabetic_counts = [normal_ages[str(age)]['Non-Diabetic'] for age in ages]

x = range(len(ages))
width = 0.35
ax2.bar([i - width/2 for i in x], diabetic_counts, width, label='糖尿病', alpha=0.8)
ax2.bar([i + width/2 for i in x], non_diabetic_counts, width, label='非糖尿病', alpha=0.8)
ax2.set_xlabel('年龄')
ax2.set_ylabel('人数')
ax2.set_title('年龄分布对比')
ax2.set_xticks(x[::5])  # 每5年显示一个标签
ax2.set_xticklabels(ages[::5])
ax2.legend()

plt.tight_layout()
plt.savefig('diabetes_analysis.png', dpi=300, bbox_inches='tight')
print("图表已保存为 diabetes_analysis.png")
