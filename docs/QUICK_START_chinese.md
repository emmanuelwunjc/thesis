# HINTS 7 糖尿病隐私分析 - 快速恢复指南

## 🚀 快速开始

### 1. 运行完整分析
```bash
python3 scripts/wrangle.py
```

### 2. 构建隐私谨慎指数
```bash
python3 scripts/build_privacy_index.py
```

### 3. 生成可视化
```bash
python3 scripts/plot_privacy_index.py
```

## 📁 关键文件位置

### 数据文件
- `data/hints7_public copy.rda` - 原始HINTS 7数据

### 分析脚本
- `scripts/wrangle.py` - 主分析脚本
- `scripts/build_privacy_index.py` - 隐私指数构建
- `scripts/plot_privacy_index.py` - 可视化脚本

### 结果文件
- `analysis/privacy_caution_index_individual.csv` - 回归用数据
- `analysis/privacy_caution_index.json` - 指数汇总统计
- `analysis/diabetes_summary.json` - 基础统计
- `analysis/diabetes_privacy_analysis.json` - 隐私分析

### 可视化
- `figures/privacy_caution_index_analysis.png` - 指数分析图
- `figures/privacy_index_construction_diagram_optimized.png` - 指数构建图

## 🔍 核心发现速览

- **糖尿病患病率**: 21.08% (1,534/7,278)
- **隐私指数差异**: +0.010 (糖尿病组更谨慎)
- **最大差异**: 设备使用 (-0.084, 糖尿病组使用更少)
- **数据分享**: 糖尿病组更愿意分享 (-0.045)

## 📊 回归分析数据

使用 `analysis/privacy_caution_index_individual.csv` 进行回归分析：

```python
import pandas as pd
df = pd.read_csv('analysis/privacy_caution_index_individual.csv')
# 包含: HHID, diabetic, privacy_caution_index, subindex_*, weight
```

## 🎯 下一步

1. 运行加权回归分析
2. 探索年龄交互效应
3. 生成政策建议报告

---
*最后更新: 2024-09-23*
