# HINTS 7 Diabetes Privacy Analysis

A comprehensive analysis of diabetes patients' privacy concerns and data sharing behaviors using the HINTS 7 Public Dataset.

## 📊 Project Overview

This project analyzes privacy-related behaviors and attitudes among diabetic vs non-diabetic patients, focusing on:
- Data sharing willingness
- Digital device usage patterns  
- Trust in healthcare systems
- Online portal engagement
- Social media behavior

## 🔍 Key Findings

- **Diabetes Prevalence**: 21.08% (1,534/7,278)
- **Privacy Index Difference**: +0.010 (diabetics slightly more cautious)
- **Largest Difference**: Device usage (+0.084, diabetics use fewer devices)
- **Data Sharing**: Diabetics more willing to share health data (-0.045)

## 📁 Project Structure

```
├── data/                    # Raw data files
│   └── hints7_public copy.rda
├── scripts/                 # Analysis scripts
│   ├── wrangle.py          # Main analysis pipeline
│   ├── build_privacy_index.py  # Privacy index construction
│   └── *.py               # Supporting analysis scripts
├── analysis/               # Analysis outputs
│   ├── *.json             # Statistical results
│   └── *.csv              # Individual-level data
├── figures/               # Visualizations
│   └── *.png             # Charts and diagrams
└── docs/                  # Documentation
    ├── PROJECT_LOG.md     # Complete project log
    ├── QUICK_START.md     # Quick start guide
    └── HINTS 7*.pdf      # Original documentation
```

## 🚀 Quick Start

1. **Run complete analysis**:
   ```bash
   python3 scripts/wrangle.py
   ```

2. **Build privacy index**:
   ```bash
   python3 scripts/build_privacy_index.py
   ```

3. **View results**:
   - Check `analysis/privacy_caution_index_individual.csv` for regression data
   - Review `figures/` for visualizations
   - Read `docs/PROJECT_LOG.md` for detailed findings

## 📈 Privacy Caution Index

A composite index (0-1 scale) measuring privacy caution across 6 dimensions:
- **Sharing Willingness** (4 variables)
- **Portal Usage** (7 variables) 
- **Device Usage** (4 variables)
- **Trust Levels** (4 variables)
- **Social Media** (2 variables)
- **Other Privacy** (2 variables)

Higher values indicate greater privacy caution.

## 🔬 Regression Framework

```
WillingShareData_HCP2 = β₀ + β₁×diabetic + β₂×privacy_caution_index + β₃×demographics + ε
```

Expected coefficients:
- β₁ > 0: Diabetics more willing to share
- β₂ < 0: Higher privacy caution reduces sharing willingness

## 📋 Requirements

- Python 3.7+
- pandas
- matplotlib
- numpy
- R (for data loading)

## 📄 Documentation

- `docs/PROJECT_LOG.md` - Complete project documentation
- `docs/QUICK_START.md` - Quick recovery guide
- `docs/HINTS 7*.pdf` - Original HINTS documentation

## 🎯 Next Steps

1. Run weighted regression analysis
2. Explore age interaction effects
3. Generate policy recommendations
4. Extend to other chronic conditions

## 📊 Available Analyses

### Basic Analysis
```bash
python3 scripts/wrangle.py
```

### Age Band Analysis
```bash
python3 scripts/wrangle.py --age-band 58 78 --age-iqr
```

### Weighted Privacy Comparisons
```bash
python3 scripts/wrangle.py --privacy-dummies
```

### Privacy Index Construction
```bash
python3 scripts/build_privacy_index.py
```

### Generate Visualizations
```bash
python3 scripts/plot_privacy_index.py
```

---
*Last updated: 2024-09-23*
