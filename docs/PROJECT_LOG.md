# HINTS 7 糖尿病与隐私关注分析项目日志

## 项目概述
- **数据源**: HINTS 7 Public Dataset (`data/hints7_public copy.rda`)
- **样本规模**: 7,278 行 × 470 列
- **主要目标**: 分析糖尿病患者的隐私关注、数据分享意愿和数字健康行为差异
- **分析日期**: 2024-09-17

## 核心发现

### 1. 糖尿病患病率
- **总患病率**: 21.08% (1,534/7,278)
- **识别依据**: `MedConditions_Diabetes` 字段
- **分类**: 全部为 "No meds reported"（未检测到药物字段）

### 2. 关键隐私/数据行为差异

#### 数据分享意愿
- **SharedHealthDeviceInfo2**: 糖尿病组更愿意分享设备数据 (+0.175)
- **WillingShareData_HCP2**: 糖尿病组更愿意与HCP分享数据

#### 数字设备使用
- **UseDevice_Computer**: 糖尿病组使用率较低 (-0.129, 0.694 vs 0.823)
- **UseDevice_SmWatch**: 糖尿病组使用率较低 (-0.105, 0.258 vs 0.363)
- **UseDevice_SmPhone**: 糖尿病组使用率略低 (-0.067, 0.858 vs 0.925)

#### 信任与社媒
- **TrustHCSystem**: 糖尿病组对医疗系统信任度更高 (+0.055)
- **CancerTrustScientists**: 糖尿病组对科学家信任度较低 (-0.083)
- **MisleadingHealthInfo**: 糖尿病组更可能不使用社媒 (+0.080)

#### 在线记录/门户
- **OnlinePortal_Pharmacy**: 糖尿病组更常使用药房门户 (+0.060)
- **HowAccessOnlineRecord2**: 糖尿病组更少通过网站访问记录 (-0.059)

## 构建的指数

### 隐私谨慎指数 (Privacy Caution Index)
- **定义**: 0-1 连续变量，越高表示越谨慎（不分享、不信任、少数字参与）
- **子维度**: 6个
  1. **分享意愿** (sharing): 糖尿病组更愿意分享 (-0.045)
  2. **门户使用** (portals): 糖尿病组更常使用 (-0.011)
  3. **设备使用** (devices): 糖尿病组更谨慎 (+0.084)
  4. **信任度** (trust): 糖尿病组略更谨慎 (+0.010)
  5. **社媒使用** (social): 糖尿病组更谨慎 (+0.020)
  6. **其他隐私** (other): 无显著差异 (+0.000)

- **总体差异**: 糖尿病组 0.476 vs 非糖尿病组 0.467 (差异 +0.010)

## 识别的关键变量

### 人口学变量
- `Age`: 年龄
- `Education`: 教育水平
- `CENSREG`: 地区
- `RUC2003`: 城乡分类

### 隐私相关变量 (46个)
- **信任类**: CancerTrustDoctor, TrustHCSystem, CancerTrustScientists等
- **分享类**: WillingShareData_HCP2, SharedHealthDeviceInfo2, LabShare2_*等
- **设备类**: UseDevice_Computer, UseDevice_SmPhone, UseDevice_Tablet等
- **门户类**: OnlinePortal_*, AccessOnlineRecord3, RecordsOnline2_*等

## 文件结构

### 核心分析脚本
- `scripts/wrangle.py`: 主分析脚本，支持多种分析模式
- `scripts/build_privacy_index.py`: 隐私谨慎指数构建脚本
- `scripts/plot_privacy_index.py`: 指数可视化脚本

### 数据输出文件
- `analysis/diabetes_summary.json`: 基础统计汇总
- `analysis/diabetes_demographics_crosstabs.json`: 人口学交叉表
- `analysis/diabetes_privacy_analysis.json`: 隐私变量分析
- `analysis/privacy_dummies_compare.json`: 加权隐私变量对比
- `analysis/privacy_caution_index.json`: 隐私谨慎指数汇总
- `analysis/privacy_caution_index_individual.csv`: 个体指数数据

### 年龄带分析
- `analysis/age_band_analyses.json`: 58-78岁和IQR年龄带分析

### 可视化文件
- `figures/privacy_top10_diffs.png`: Top-10隐私差异图
- `figures/privacy_shared_device.png`: 设备数据分享对比
- `figures/privacy_use_computer.png`: 电脑使用对比
- `figures/privacy_use_watch.png`: 智能手表使用对比
- `figures/privacy_trust_hcsystem.png`: 医疗系统信任对比
- `figures/privacy_trust_scientists.png`: 科学家信任对比
- `figures/privacy_portal_pharmacy.png`: 药房门户使用对比
- `figures/privacy_caution_index_analysis.png`: 隐私谨慎指数综合分析
- `figures/privacy_index_construction_diagram_optimized.png`: 指数构建图
- `figures/privacy_index_detailed_table_optimized.png`: 指数详细表

## 分析方法

### 数据加载
- 使用Rscript作为pyreadr的备选方案
- 自动检测年龄、教育、地区等人口学变量
- 支持样本权重和jackknife复制权重

### 隐私变量识别
- 基于关键词匹配：privacy, share, records, portal, trust, confidence等
- 自动生成indexed dummies
- 支持加权均值和jackknife标准误

### 指数构建
- 子维度分组避免共线性
- 方向统一：0=不谨慎，1=最谨慎
- 加权计算支持复杂抽样设计

## 回归分析框架

### 主回归方程
```
WillingShareData_HCP2 = β₀ + β₁×diabetic + β₂×privacy_caution_index + β₃×demographics + ε
```

### 预期系数
- β₁ > 0: 糖尿病更愿意分享数据
- β₂ < 0: 隐私谨慎指数越高，越不愿意分享
- 可考虑交互项: diabetic × privacy_caution_index

### 控制变量
- 年龄 (Age)
- 教育 (Education) 
- 地区 (CENSREG)
- 城乡 (RUC2003)

## 可用命令

### 基础分析
```bash
python3 scripts/wrangle.py
```

### 年龄带分析
```bash
python3 scripts/wrangle.py --age-band 58 78 --age-iqr
```

### 隐私变量加权对比
```bash
python3 scripts/wrangle.py --privacy-dummies
```

### 重新生成隐私指数
```bash
python3 scripts/build_privacy_index.py
```

### 生成可视化
```bash
python3 scripts/plot_privacy_index.py
```

## 下一步建议

1. **回归分析**: 使用构建的隐私谨慎指数进行加权回归
2. **年龄分层**: 进一步分析不同年龄组的差异模式
3. **交互效应**: 探索糖尿病与隐私指数的交互作用
4. **稳健性检验**: 使用不同权重方案验证结果
5. **政策含义**: 基于发现提出数字健康政策建议

## 技术细节

### 权重处理
- 主权重: 自动检测为 `Weight`
- 复制权重: 未检测到jackknife权重列
- 加权统计: 支持复杂抽样设计

### 缺失值处理
- 同子维度内≥半数非缺失才计入指数
- 默认中等谨慎度(0.5)处理未知类别

### 数据质量
- 隐私指数范围: 0.233 - 0.780
- 无缺失值
- 分布接近正态

## 回归分析结果 (2024-09-23)

### 主回归模型
- **样本规模**: 2,421个有效观测值
- **模型R²**: 0.1736
- **方法**: 加权最小二乘法

### 关键系数
- **糖尿病效应**: +0.0278 (p=0.161, 不显著)
- **隐私谨慎指数**: -2.3159 (p<0.001, 高度显著) ⭐
- **年龄效应**: +0.0024 (p<0.001, 高度显著)
- **教育效应**: -0.0149 (p=0.129, 不显著)

### 交互效应模型
- **糖尿病×隐私交互**: +0.4896 (p=0.038, 显著)
- **模型R²**: 0.1753

### 年龄分组分析
- **18-35岁**: 糖尿病效应-0.0475, 隐私效应-2.8936
- **36-50岁**: 糖尿病效应+0.0558, 隐私效应-2.6580
- **51-65岁**: 糖尿病效应+0.0164, 隐私效应-2.3663
- **65+岁**: 糖尿病效应+0.0112, 隐私效应-1.6398

### 主要发现
1. **隐私关注是影响数据分享的最强预测因子**
2. **糖尿病患者的隐私-分享权衡与非糖尿病患者不同**
3. **年轻患者对隐私更敏感**
4. **年龄对数据分享意愿有正向影响**

### 政策含义
1. **隐私保护政策**: 应优先考虑透明度和患者控制
2. **糖尿病管理**: 需要针对性的隐私沟通策略
3. **年龄差异化**: 不同年龄组需要不同的隐私教育方法
4. **系统设计**: 实施隐私保护设计原则

---
*最后更新: 2024-09-23*
*分析工具: Python + pandas + matplotlib + scipy*
*数据来源: HINTS 7 Public Dataset*
*项目结构: 已重新组织为专业目录结构*
*回归分析: 已完成加权回归和交互效应分析*
