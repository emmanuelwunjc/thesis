import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load results
with open('privacy_caution_index.json', 'r') as f:
    result = json.load(f)

# Load individual data
df = pd.read_csv('privacy_caution_index_individual.csv')

# 1. 总体分布对比
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 直方图对比
dia_data = df[df['diabetic']==1]['privacy_caution_index'].dropna()
non_data = df[df['diabetic']==0]['privacy_caution_index'].dropna()

ax1.hist(dia_data, bins=30, alpha=0.7, color='red', label='糖尿病', density=True)
ax1.hist(non_data, bins=30, alpha=0.7, color='blue', label='非糖尿病', density=True)
ax1.set_xlabel('隐私谨慎指数')
ax1.set_ylabel('密度')
ax1.set_title('隐私谨慎指数分布对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 箱线图
box_data = [dia_data, non_data]
ax2.boxplot(box_data, labels=['糖尿病', '非糖尿病'])
ax2.set_ylabel('隐私谨慎指数')
ax2.set_title('隐私谨慎指数箱线图')
ax2.grid(True, alpha=0.3)

# 子维度差异条形图
subdims = result['privacy_caution_index']['subdimensions']
names = list(subdims.keys())
diffs = [subdims[name]['difference'] for name in names]
colors = ['red' if d > 0 else 'blue' for d in diffs]

ax3.barh(names, diffs, color=colors, alpha=0.7)
ax3.set_xlabel('组间差异 (糖尿病 - 非糖尿病)')
ax3.set_title('子维度差异')
ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax3.grid(True, alpha=0.3)

# 子维度均值对比
dia_means = [subdims[name]['diabetic_mean'] for name in names]
non_means = [subdims[name]['non_diabetic_mean'] for name in names]

x = np.arange(len(names))
width = 0.35
ax4.bar(x - width/2, dia_means, width, label='糖尿病', color='red', alpha=0.7)
ax4.bar(x + width/2, non_means, width, label='非糖尿病', color='blue', alpha=0.7)
ax4.set_xlabel('子维度')
ax4.set_ylabel('均值')
ax4.set_title('子维度均值对比')
ax4.set_xticks(x)
ax4.set_xticklabels(names, rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('privacy_caution_index_analysis.png', dpi=300, bbox_inches='tight')
print("隐私谨慎指数分析图已保存: privacy_caution_index_analysis.png")

# 2. 回归准备数据
print("\n=== 回归分析准备 ===")
print("因变量: WillingShareData_HCP2 (是否愿意与HCP分享数据)")
print("自变量:")
print("- diabetic: 糖尿病虚拟变量")
print("- privacy_caution_index: 隐私谨慎指数 (0-1)")
print("- demographics: 年龄、教育、地区、城乡等")

# 检查数据质量
print(f"\n=== 数据质量 ===")
print(f"总样本: {len(df)}")
print(f"糖尿病样本: {df['diabetic'].sum()}")
print(f"隐私指数缺失: {df['privacy_caution_index'].isna().sum()}")
print(f"隐私指数范围: {df['privacy_caution_index'].min():.3f} - {df['privacy_caution_index'].max():.3f}")

plt.show()
