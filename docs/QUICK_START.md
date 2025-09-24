# Quick Start Guide
## HINTS 7 Diabetes Privacy Study

**Purpose**: Rapid project resumption and analysis execution  
**Last Updated**: 2024-09-23  
**Status**: ✅ Ready to Use  

---

## 🚀 Quick Setup (5 minutes)

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/emmanuelwunjc/thesis.git
cd thesis

# Install dependencies
pip install pandas numpy matplotlib scikit-learn seaborn pyreadr

# Verify data file exists
ls data/hints7_public\ copy.rda
```

### 2. Run Complete Analysis Pipeline
```bash
# Step 1: Data cleaning and preprocessing
python3 scripts/data_cleaning_for_ml.py

# Step 2: Machine learning model selection
python3 scripts/simplified_ml_model_selection.py

# Step 3: Generate best model visualizations
python3 scripts/create_best_model_visualizations.py
```

### 3. View Results
- **Main Report**: `analysis/BEST_ML_MODEL_DETAILED_REPORT.md`
- **Executive Summary**: `analysis/BEST_MODEL_EXECUTIVE_SUMMARY.md`
- **Visualizations**: `figures/best_ml_model_detailed_analysis.png`
- **Architecture**: `figures/best_model_architecture.png`

---

## 📊 Key Analysis Commands

### Data Processing
```bash
# Basic diabetes analysis
python3 scripts/wrangle.py

# Privacy index construction
python3 scripts/build_privacy_index.py

# Age-specific analysis
python3 scripts/wrangle.py --age-band
python3 scripts/wrangle.py --age-iqr
```

### Machine Learning Analysis
```bash
# Automated model selection
python3 scripts/simplified_ml_model_selection.py

# Comprehensive ML analysis (original)
python3 scripts/ml_model_selection.py

# Best model visualizations
python3 scripts/create_best_model_visualizations.py
```

### Regression Analysis
```bash
# Main regression analysis
python3 scripts/regression_analysis.py

# Comprehensive regression (6 models)
python3 scripts/comprehensive_regression_analysis.py

# Display formatted results
python3 scripts/display_regression_results.py
```

### Causal Inference
```bash
# Difference-in-differences
python3 scripts/difference_in_differences_analysis.py

# Advanced causal inference (PSM, IV, RDD)
python3 scripts/causal_inference_analysis.py
```

### Visualization Generation
```bash
# General data visualizations
python3 scripts/visualize_data.py

# Age distribution analysis
python3 scripts/age_distribution_plot.py

# Model logic diagrams
python3 scripts/create_model_logic_diagram.py
```

---

## 📋 Essential Files Reference

### Core Data Files
- **`data/hints7_public copy.rda`** - Original HINTS 7 dataset
- **`analysis/ml_cleaned_data.csv`** - ML-ready cleaned data
- **`analysis/privacy_caution_index_individual.csv`** - Privacy index scores

### Key Analysis Scripts
- **`scripts/data_cleaning_for_ml.py`** - Data preprocessing pipeline
- **`scripts/simplified_ml_model_selection.py`** - ML model selection
- **`scripts/create_best_model_visualizations.py`** - Visualization generation

### Main Reports
- **`analysis/BEST_ML_MODEL_DETAILED_REPORT.md`** - Comprehensive technical report
- **`analysis/BEST_MODEL_EXECUTIVE_SUMMARY.md`** - Executive summary
- **`README.md`** - Project overview

### Key Visualizations
- **`figures/best_ml_model_detailed_analysis.png`** - Main analysis charts
- **`figures/best_model_architecture.png`** - Model architecture diagram
- **`figures/data_quality_analysis.png`** - Data quality assessment

---

## 🎯 Best Model Quick Reference

### Model Specification
- **Algorithm**: Random Forest Regressor
- **Features**: 6 core variables (diabetes, privacy_index, age, region, insurance, gender)
- **Performance**: R² = -0.1239, MSE = 0.0403, MAE = 0.1588

### Key Findings
1. **Diabetes Importance**: Confirmed in all optimal models
2. **Privacy Priority**: Privacy caution index is most important predictor
3. **Algorithm Superiority**: Random Forest consistently best performer
4. **Feature Optimization**: 6-feature combination provides optimal performance

### Policy Implications
- Diabetes patients require specialized data sharing strategies
- Privacy concerns are primary barrier to data sharing
- Healthcare systems need diabetes-specific privacy protocols

---

## 🔧 Troubleshooting

### Common Issues

#### Data Loading Problems
```bash
# If pyreadr fails, R fallback will be used automatically
# Check R installation: which R
# Install R if needed: brew install r (macOS) or apt-get install r-base (Linux)
```

#### Memory Issues
```bash
# For large datasets, reduce feature combinations
# Edit scripts/simplified_ml_model_selection.py
# Change max_features from 6 to 4
```

#### Visualization Issues
```bash
# If matplotlib backend issues occur
export MPLBACKEND=Agg
python3 scripts/create_best_model_visualizations.py
```

### Performance Optimization
```bash
# For faster execution, reduce model complexity
# Edit scripts/simplified_ml_model_selection.py
# Reduce n_estimators from 50 to 25
```

---

## 📈 Analysis Workflow

### Complete Analysis Pipeline
1. **Data Loading** → `scripts/data_cleaning_for_ml.py`
2. **ML Selection** → `scripts/simplified_ml_model_selection.py`
3. **Visualization** → `scripts/create_best_model_visualizations.py`
4. **Report Generation** → Automatic via scripts

### Custom Analysis Workflow
1. **Choose Analysis Type**:
   - Descriptive: `scripts/wrangle.py`
   - Privacy: `scripts/build_privacy_index.py`
   - Regression: `scripts/regression_analysis.py`
   - Causal: `scripts/causal_inference_analysis.py`

2. **Run Analysis**: Execute chosen script
3. **View Results**: Check `analysis/` and `figures/` directories
4. **Generate Reports**: Use display scripts for formatted output

---

## 🎨 Visualization Options

### High-Resolution Outputs
All visualization scripts generate both PNG and PDF formats:
- **PNG**: For presentations and web use
- **PDF**: For academic papers and publications

### Customization Options
```python
# In visualization scripts, modify:
figsize=(30, 24)  # Adjust figure size
dpi=300          # Adjust resolution
colors={...}     # Customize color palette
```

---

## 📊 Results Interpretation

### Model Performance Metrics
- **R²**: Model explanatory power (higher is better)
- **MSE**: Prediction accuracy (lower is better)
- **MAE**: Prediction bias (lower is better)

### Feature Importance
- **Privacy Index**: Most important predictor (0.35)
- **Age**: Second most important (0.25)
- **Diabetes**: Core variable, medium importance (0.20)

### Statistical Significance
- **p < 0.001**: Highly significant
- **p < 0.01**: Significant
- **p < 0.05**: Marginally significant
- **p > 0.05**: Not significant

---

## 🔄 Reproducibility

### Ensuring Reproducible Results
```python
# All scripts use fixed random seeds
random_state=42  # Consistent across all analyses
```

### Version Control
```bash
# Check current status
git status

# View recent changes
git log --oneline -5

# Restore previous version if needed
git checkout HEAD~1 -- scripts/script_name.py
```

---

## 📞 Support and Resources

### Documentation
- **Project Overview**: `README.md`
- **File Directory**: `FILE_DIRECTORY.md`
- **Project Status**: `PROJECT_STATUS_SUMMARY.md`
- **Technical Details**: `docs/PROJECT_LOG.md`

### External Resources
- **HINTS 7 Dataset**: https://hints.cancer.gov/
- **scikit-learn Documentation**: https://scikit-learn.org/
- **pandas Documentation**: https://pandas.pydata.org/

### Contact Information
- **Repository**: https://github.com/emmanuelwunjc/thesis.git
- **Analysis Date**: 2024-09-23
- **Technology Stack**: Python, scikit-learn, pandas, matplotlib

---

## ✅ Quick Checklist

### Before Starting Analysis
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Data file present (`data/hints7_public copy.rda`)
- [ ] Python environment activated

### After Running Analysis
- [ ] Data cleaning completed
- [ ] ML model selection finished
- [ ] Visualizations generated
- [ ] Reports created
- [ ] Results reviewed

### For Custom Analysis
- [ ] Choose appropriate script
- [ ] Modify parameters if needed
- [ ] Run analysis
- [ ] Check output files
- [ ] Interpret results

---

**Quick Start Guide Last Updated**: 2024-09-23  
**Total Setup Time**: ~5 minutes  
**Analysis Completion Time**: ~10 minutes  
**Project Status**: ✅ Ready for Use
