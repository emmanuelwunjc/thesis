# Chronic Disease Privacy Research: HINTS 7 Analysis

**Research Question**: How do chronic diseases requiring daily tracking affect privacy concerns and data sharing behavior?

**Key Finding**: Chronic disease patients (diabetes, hypertension, heart conditions, depression, lung disease) are more willing to share health data and less privacy-concerned compared to non-chronic disease individuals.

---

## 🚀 Quick Start

### For First-Time Visitors
1. **Start Here**: Read this README for overview
2. **Navigation Guide**: [`NAVIGATION_GUIDE.md`](NAVIGATION_GUIDE.md) - Complete directory structure and navigation
3. **Key Findings**: [`analysis/summaries/english/CHRONIC_DISEASE_ANALYSIS_SUMMARY.md`](analysis/summaries/english/CHRONIC_DISEASE_ANALYSIS_SUMMARY.md)

### For Supervisors
1. **Supervisor Guide**: [`docs/guides/SUPERVISOR_GUIDE.md`](docs/guides/SUPERVISOR_GUIDE.md)
2. **Executive Summary**: [`analysis/summaries/english/BEST_MODEL_EXECUTIVE_SUMMARY.md`](analysis/summaries/english/BEST_MODEL_EXECUTIVE_SUMMARY.md)

### For Researchers
1. **Thesis Outline**: [`THESIS_OUTLINE.md`](THESIS_OUTLINE.md)
2. **Methodology**: [`analysis/summaries/english/MODEL_LOGIC_SUMMARY.md`](analysis/summaries/english/MODEL_LOGIC_SUMMARY.md)
3. **Results**: [`analysis/summaries/english/REGRESSION_RESULTS_SUMMARY.md`](analysis/summaries/english/REGRESSION_RESULTS_SUMMARY.md)

---

## 📁 Repository Structure

```
thesis/
├── README.md                          # This file - main entry point
├── NAVIGATION_GUIDE.md                 # Complete navigation guide
├── THESIS_OUTLINE.md                   # Full thesis document
│
├── data/                               # All data files
│   ├── raw/                           # Original HINTS 7 dataset
│   ├── processed/                     # Cleaned/processed data
│   └── intermediate/                   # Intermediate files
│
├── scripts/                            # Analysis scripts (organized by function)
│   ├── 01_data_preparation/           # Data loading, cleaning, index building
│   ├── 02_regression/                  # Regression analysis
│   ├── 03_machine_learning/           # ML model selection
│   ├── 04_causal_inference/           # Causal inference methods
│   ├── 05_visualization/              # Plotting and visualization
│   └── utils/                         # Utility functions
│
├── analysis/                            # Analysis outputs
│   ├── results/                        # JSON results files
│   ├── summaries/                     # Markdown summaries
│   │   ├── english/                   # English summaries
│   │   └── chinese/                   # Chinese summaries
│   └── data/                           # Processed CSV files
│
├── figures/                            # All visualizations
│   └── [organized by analysis type]
│
└── docs/                               # Documentation
    ├── guides/                         # User guides
    ├── methodology/                    # Methodology docs
    └── references/                     # Reference materials
```

**See [`NAVIGATION_GUIDE.md`](NAVIGATION_GUIDE.md) for detailed directory structure and file locations.**

---

## 🎯 Key Findings

### Main Result
**Chronic disease patients show different privacy-sharing trade-offs:**
- ✅ More willing to share data (all conditions, p < 0.01)
- ✅ Less privacy-concerned (consistent pattern)
- ✅ Similar effect sizes across conditions (OR: 1.43-2.38)

### Conditions Analyzed
1. **Diabetes**: 472 cases, OR=1.53, p=0.0010
2. **Hypertension**: 900 cases, OR=1.50, p<0.0001 ⭐ (strongest interaction)
3. **Heart Condition**: 197 cases, OR=2.38, p<0.0001 (largest effect)
4. **Depression**: 693 cases, OR=1.43, p=0.0011
5. **Lung Disease**: 312 cases, OR=1.53, p=0.0061

### Generalizability
✅ **Confirmed**: Findings apply to chronic disease management broadly, not just diabetes-specific.

---

## 📊 Analysis Pipeline

### 1. Data Preparation
```bash
# Load and clean data
python scripts/01_data_preparation/wrangle.py
python scripts/01_data_preparation/build_privacy_index.py
python scripts/01_data_preparation/data_cleaning_for_ml.py
```

### 2. Regression Analysis
```bash
# Run regression models
python scripts/02_regression/regression_analysis.py
python scripts/02_regression/comprehensive_regression_analysis.py
```

### 3. Machine Learning
```bash
# ML model selection
python scripts/03_machine_learning/simplified_ml_model_selection.py
```

### 4. Causal Inference
```bash
# Causal methods
python scripts/04_causal_inference/causal_inference_analysis.py
```

### 5. Multi-Condition Analysis
```bash
# Analyze all chronic diseases
python scripts/03_machine_learning/multi_chronic_disease_analysis.py
```

---

## 📚 Key Documents

### Understanding the Research
- **Overview**: [`README.md`](README.md) (this file)
- **Navigation**: [`NAVIGATION_GUIDE.md`](NAVIGATION_GUIDE.md)
- **Thesis**: [`THESIS_OUTLINE.md`](THESIS_OUTLINE.md)
- **Findings**: [`analysis/summaries/english/CHRONIC_DISEASE_ANALYSIS_SUMMARY.md`](analysis/summaries/english/CHRONIC_DISEASE_ANALYSIS_SUMMARY.md)

### Understanding the Methods
- **Model Logic**: [`analysis/summaries/english/MODEL_LOGIC_SUMMARY.md`](analysis/summaries/english/MODEL_LOGIC_SUMMARY.md)
- **Regression**: [`analysis/summaries/english/REGRESSION_RESULTS_SUMMARY.md`](analysis/summaries/english/REGRESSION_RESULTS_SUMMARY.md)
- **ML**: [`analysis/summaries/english/BEST_ML_MODEL_DETAILED_REPORT.md`](analysis/summaries/english/BEST_ML_MODEL_DETAILED_REPORT.md)

### Understanding the Results
- **Executive Summary**: [`analysis/summaries/english/BEST_MODEL_EXECUTIVE_SUMMARY.md`](analysis/summaries/english/BEST_MODEL_EXECUTIVE_SUMMARY.md)
- **Causal Inference**: [`analysis/summaries/english/CAUSAL_INFERENCE_SUMMARY.md`](analysis/summaries/english/CAUSAL_INFERENCE_SUMMARY.md)
- **Multi-Condition**: [`analysis/summaries/english/MULTI_CHRONIC_DISEASE_FINDINGS.md`](analysis/summaries/english/MULTI_CHRONIC_DISEASE_FINDINGS.md)

### Understanding the Significance
- **Diabetes Role**: [`analysis/summaries/english/DIABETES_ROLE_AND_SIGNIFICANCE.md`](analysis/summaries/english/DIABETES_ROLE_AND_SIGNIFICANCE.md)
- **Analysis Decisions**: [`analysis/summaries/english/DATA_ANALYSIS_DECISIONS.md`](analysis/summaries/english/DATA_ANALYSIS_DECISIONS.md)

---

## 🔬 Methodology

### Data Source
- **HINTS 7 Public Dataset**: 7,278 individuals, 48 variables
- **Chronic Disease Patients**: 2,574 across 5 conditions
- **Target Variable**: Data sharing willingness (binary)

### Methods
1. **Regression Analysis**: 6 models (main, interaction, stratified, mediation, multiple outcomes, severity)
2. **Machine Learning**: Automated model selection (1,020 configurations, 4 algorithms)
3. **Causal Inference**: PSM, IV, RDD, DiD methods
4. **Multi-Condition Analysis**: Replicated analysis across 5 chronic diseases

### Key Innovation
- **First exhaustive search** in chronic disease privacy research
- **Automated model selection** ensuring core variables included
- **Multi-condition validation** confirming generalizability

---

## 💡 Policy Implications

### For Healthcare Systems
- **Chronic disease-specific privacy protocols** needed
- **Tailored data sharing controls** for daily-tracking conditions
- **Privacy education programs** for chronic disease patients

### For Policy Makers
- **Broaden scope**: Chronic disease-specific (not just diabetes) privacy policies
- **Priority conditions**: Hypertension, heart conditions, depression, lung disease
- **Regulatory updates**: HIPAA modifications for chronic disease management

### For Researchers
- **Theoretical contribution**: Chronic disease management privacy patterns
- **Methodological innovation**: Automated model selection approach
- **Future directions**: Longitudinal studies, intervention research

---

## 📈 Research Contributions

### Theoretical
- First comprehensive study of chronic disease management privacy patterns
- Validates privacy protection motivation theory in chronic disease context
- Demonstrates heterogeneity in privacy behavior by health status

### Methodological
- Automated model selection reduces subjective bias
- Exhaustive search ensures optimal model identification
- Multi-condition analysis validates generalizability

### Empirical
- Confirms chronic disease effect on privacy behavior
- Quantifies privacy importance (strongest predictor)
- Identifies policy-relevant patterns across conditions

---

## 🛠️ Technology Stack

- **Python 3.x**: Main analysis language
- **Libraries**: pandas, numpy, scikit-learn, statsmodels, matplotlib
- **Data**: HINTS 7 Public Dataset (R format)
- **Visualization**: matplotlib, seaborn

---

## 📞 Repository Information

- **Data Source**: [HINTS 7 Public Dataset](https://hints.cancer.gov/)
- **Analysis Date**: 2024
- **Sample Size**: 7,278 (HINTS 7), 2,574 (chronic disease patients)
- **Conditions Analyzed**: 5 chronic diseases requiring daily tracking

---

## ✅ Project Status

- [x] Data cleaning completed
- [x] Privacy index constructed
- [x] Regression analysis completed (6 models)
- [x] ML model selection completed
- [x] Causal inference analysis completed
- [x] Multi-condition analysis completed
- [x] Documentation organized
- [x] Repository reorganized

---

## 🔍 Finding Information

**Looking for something specific?** Check [`NAVIGATION_GUIDE.md`](NAVIGATION_GUIDE.md) for:
- Complete directory structure
- File locations by function
- Typical workflows
- Common searches

---

*Last Updated: 2024*  
*Repository Status: Organized and documented*
