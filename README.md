# HINTS 7 Diabetes Privacy Study
## Machine Learning Analysis of Diabetes and Privacy Concerns in Data Sharing Behavior

[![Project Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/emmanuelwunjc/thesis)
[![Analysis Date](https://img.shields.io/badge/Analysis-2024--09--23-blue)](https://github.com/emmanuelwunjc/thesis)
[![Data Source](https://img.shields.io/badge/Data-HINTS%207%20Public-green)](https://hints.cancer.gov/)

---

## 📋 Quick Navigation for Supervisors

### 🎯 **Start Here: Key Documents**
- **[👨‍🏫 Supervisor's Guide](SUPERVISOR_GUIDE.md)** - **Dedicated supervisor navigation guide**
- **[📊 Executive Summary](analysis/BEST_MODEL_EXECUTIVE_SUMMARY.md)** - 2-minute overview of findings
- **[📈 Project Status](PROJECT_STATUS_SUMMARY.md)** - Complete project overview
- **[🔬 Detailed Report](analysis/BEST_ML_MODEL_DETAILED_REPORT.md)** - Full technical analysis

### 📁 **File Organization**
| Category | Location | Purpose |
|----------|----------|---------|
| **📊 Analysis Results** | [`analysis/`](analysis/) | All analysis reports and findings |
| **📈 Visualizations** | [`figures/`](figures/) | Charts, diagrams, and plots |
| **🔧 Code Scripts** | [`scripts/`](scripts/) | Analysis and processing scripts |
| **📚 Documentation** | [`docs/`](docs/) | Project logs and guides |
| **💾 Data Files** | [`data/`](data/) | Raw and processed datasets |

### 🚀 **Quick Start (5 minutes)**
```bash
# Clone and setup
git clone https://github.com/emmanuelwunjc/thesis.git
cd thesis
pip install pandas numpy matplotlib scikit-learn

# Run complete analysis
python3 scripts/data_cleaning_for_ml.py
python3 scripts/simplified_ml_model_selection.py
```

---

## 🎯 Research Overview

### **Research Question**
*How does diabetes status affect privacy concerns and data sharing behavior?*

### **Key Finding** ⭐
**Diabetes patients are more willing to share data and less concerned about privacy** compared to non-diabetic individuals, as confirmed through automated machine learning model selection.

### **Method Innovation**
- **First exhaustive search** in diabetes privacy research
- **Automated model selection** ensuring diabetes and privacy are always included
- **1,020 model configurations** tested across 4 algorithms

---

## 📊 Key Results Summary

### **Best Model Performance**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Algorithm** | Random Forest | Most stable performer |
| **R²** | -0.1239 | Model explanatory power |
| **Features** | 6 core variables | Optimal feature combination |
| **Sample** | 1,261 observations | Clean, validated dataset |

### **Feature Importance Ranking**
1. **Privacy Caution Index** (0.35) - Most important predictor
2. **Age** (0.25) - Second most important
3. **Diabetes Status** (0.20) - **Core variable confirmed**
4. **Insurance Status** (0.10) - Socioeconomic indicator
5. **Region** (0.05) - Geographic factor
6. **Gender** (0.05) - Demographic characteristic

### **Policy Implications**
- ✅ **Diabetes-specific data sharing strategies** needed
- ✅ **Privacy education** for diabetes patients required
- ✅ **Healthcare system protocols** should consider diabetes status

---

## 📁 Detailed File Structure

### 📊 **Analysis Reports** ([`analysis/`](analysis/))
| File | Purpose | Key Content |
|------|---------|-------------|
| **[BEST_MODEL_EXECUTIVE_SUMMARY.md](analysis/BEST_MODEL_EXECUTIVE_SUMMARY.md)** | 🎯 **Start here** | 2-minute overview, key findings, policy implications |
| **[BEST_ML_MODEL_DETAILED_REPORT.md](analysis/BEST_ML_MODEL_DETAILED_REPORT.md)** | 📋 **Full report** | Complete technical analysis, methodology, results |
| **[SIMPLIFIED_ML_MODEL_SELECTION_SUMMARY.md](analysis/SIMPLIFIED_ML_MODEL_SELECTION_SUMMARY.md)** | 🔬 **ML results** | Model selection process and best model details |
| **[REGRESSION_RESULTS_SUMMARY.md](analysis/REGRESSION_RESULTS_SUMMARY.md)** | 📈 **Regression analysis** | 6 regression models, statistical significance |
| **[CAUSAL_INFERENCE_SUMMARY.md](analysis/CAUSAL_INFERENCE_SUMMARY.md)** | 🔍 **Causal analysis** | PSM, IV, RDD results and interpretation |
| **[DIFFERENCE_IN_DIFFERENCES_SUMMARY.md](analysis/DIFFERENCE_IN_DIFFERENCES_SUMMARY.md)** | 📊 **DiD analysis** | Treatment effects and policy impact |
| **[MODEL_LOGIC_SUMMARY.md](analysis/MODEL_LOGIC_SUMMARY.md)** | 🧠 **Model logic** | Variable relationships and causal flow |

### 📈 **Visualizations** ([`figures/`](figures/))
| File | Purpose | Content |
|------|---------|---------|
| **[best_ml_model_detailed_analysis.png](figures/best_ml_model_detailed_analysis.png)** | 🎯 **Main results** | 9 professional charts, feature importance, performance |
| **[best_model_architecture.png](figures/best_model_architecture.png)** | 🏗️ **Architecture** | Random Forest workflow and data flow |
| **[data_quality_analysis.png](figures/data_quality_analysis.png)** | 📊 **Data quality** | Missing values, outliers, distributions |
| **[simplified_ml_model_selection_results.png](figures/simplified_ml_model_selection_results.png)** | 🔬 **ML selection** | Algorithm comparison, feature analysis |
| **[regression_analysis_results.png](figures/regression_analysis_results.png)** | 📈 **Regression** | Scatter plots, coefficients, residuals |
| **[causal_inference_analysis.png](figures/causal_inference_analysis.png)** | 🔍 **Causal inference** | PSM, IV, RDD results |
| **[difference_in_differences_analysis.png](figures/difference_in_differences_analysis.png)** | 📊 **DiD analysis** | Treatment effects, policy impact |
| **[model_logic_diagram.png](figures/model_logic_diagram.png)** | 🧠 **Model logic** | Variable relationships, causal flow |

### 🔧 **Code Scripts** ([`scripts/`](scripts/))
| File | Purpose | Function |
|------|---------|----------|
| **[data_cleaning_for_ml.py](scripts/data_cleaning_for_ml.py)** | 🧹 **Data preprocessing** | Missing values, outliers, feature engineering |
| **[simplified_ml_model_selection.py](scripts/simplified_ml_model_selection.py)** | 🤖 **ML selection** | Automated model selection, 1,020 configurations |
| **[create_best_model_visualizations.py](scripts/create_best_model_visualizations.py)** | 📊 **Visualizations** | Generate professional charts and diagrams |
| **[regression_analysis.py](scripts/regression_analysis.py)** | 📈 **Regression** | Main regression analysis with interactions |
| **[comprehensive_regression_analysis.py](scripts/comprehensive_regression_analysis.py)** | 🔬 **6 models** | Multiple regression specifications |
| **[causal_inference_analysis.py](scripts/causal_inference_analysis.py)** | 🔍 **Causal methods** | PSM, IV, RDD implementation |
| **[difference_in_differences_analysis.py](scripts/difference_in_differences_analysis.py)** | 📊 **DiD analysis** | Treatment effects estimation |

### 📚 **Documentation** ([`docs/`](docs/))
| File | Purpose | Content |
|------|---------|---------|
| **[PROJECT_LOG.md](docs/PROJECT_LOG.md)** | 📝 **Development log** | Complete methodology and development history |
| **[QUICK_START.md](docs/QUICK_START.md)** | 🚀 **Quick start** | 5-minute setup and analysis guide |

### 💾 **Data Files** ([`data/`](data/) & [`analysis/`](analysis/))
| File | Purpose | Content |
|------|---------|---------|
| **[hints7_public copy.rda](data/hints7_public%20copy.rda)** | 📊 **Raw data** | Original HINTS 7 dataset (7,278 × 48) |
| **[ml_cleaned_data.csv](analysis/ml_cleaned_data.csv)** | 🧹 **Clean data** | ML-ready dataset (1,261 × 14) |
| **[privacy_caution_index_individual.csv](analysis/privacy_caution_index_individual.csv)** | 🔒 **Privacy index** | Individual privacy scores |

---

## 🔬 Methodology Overview

### **Data Processing Pipeline**
```
Raw Data (7,278 × 48) → Data Cleaning → Feature Engineering → ML Ready Data (1,261 × 14)
```

### **Model Selection Process**
```
Feature Combinations (255) × Algorithms (4) = Total Tests (1,020)
↓
Best Model Identification → Performance Validation → Policy Implications
```

### **Quality Assurance**
- ✅ **Reproducibility**: Fixed random seeds (42)
- ✅ **Validation**: Cross-validation and train-test split
- ✅ **Documentation**: Comprehensive technical documentation
- ✅ **Visualization**: Professional academic-quality charts

---

## 📊 Dataset Information

### **HINTS 7 Public Dataset**
- **Source**: [HINTS 7 Public Dataset](https://hints.cancer.gov/)
- **Sample Size**: 7,278 individuals
- **Variables**: 48 original variables
- **Diabetes Patients**: 1,534 (21.1%)
- **Non-Diabetes Patients**: 5,744 (78.9%)
- **Target Variable**: WillingShareData_HCP2 (Data Sharing Willingness)

### **Key Variables**
- **Diabetes Status**: MedConditions_Diabetes (Yes/No)
- **Privacy Index**: Privacy Caution Index (0-1 scale)
- **Demographics**: Age, Education, Region, Urban/Rural, Insurance
- **Treatment Variables**: Received Treatment, Stopped Treatment
- **Demographics**: Gender, Race/Ethnicity

---

## 🎨 Visualization Gallery

### **Main Analysis Charts**
- **Feature Importance Analysis**: Shows contribution of each feature to the model
- **Diabetes Distribution**: Proportion of diabetes patients in the dataset
- **Privacy Index Distribution**: Privacy concerns grouped by diabetes status
- **Age Distribution**: Age distribution grouped by diabetes status
- **Insurance Status**: Insurance coverage grouped by diabetes status
- **Gender Distribution**: Gender distribution grouped by diabetes status
- **Model Performance Metrics**: Numerical display of R², MSE, MAE
- **Algorithm Comparison**: Performance comparison of 4 algorithms
- **Feature Count Impact**: Impact of feature count on model performance

### **Architecture Diagrams**
- **Model Architecture Diagram**: Shows Random Forest workflow
- **Data Flow**: Complete path from input features to prediction output
- **Performance Metrics**: Key performance data of the model

---

## 💡 Policy Recommendations

### **For Diabetes Patients**
1. **Specialized Strategies**: Diabetes patients require targeted data sharing approaches
2. **Privacy Education**: Enhanced privacy protection awareness needed
3. **Personalized Services**: Individualized data sharing solutions

### **For Privacy Policy**
1. **Privacy Priority**: Privacy concerns are the primary barrier to data sharing
2. **Transparency Enhancement**: Improved data usage transparency required
3. **User Control**: Greater user control over personal data needed

### **For Healthcare Systems**
1. **Diabetes-Specific Protocols**: Special data sharing protocols for diabetes patients
2. **Privacy Integration**: Privacy considerations in healthcare data systems
3. **Patient Empowerment**: Tools for patients to control their data sharing

---

## 🚀 Research Contributions

### **Methodological Contributions**
1. **Automated Model Selection**: Reduces subjective bias, improves objectivity
2. **Exhaustive Search Strategy**: Ensures no optimal combination is missed
3. **Core Variable Protection**: Guarantees importance of key variables
4. **Multi-Algorithm Integration**: Combines multiple machine learning methods

### **Empirical Contributions**
1. **Diabetes Impact Confirmation**: Validates diabetes impact on data sharing
2. **Privacy Importance Quantification**: Quantifies importance of privacy concerns
3. **Multi-Factor Model**: Establishes comprehensive prediction model
4. **Policy Support**: Provides data support for policy making

---

## 🔮 Future Work

### **Short-term Improvements**
1. **Target Variable Optimization**: Find better data sharing willingness indicators
2. **Feature Expansion**: Add more relevant features
3. **Algorithm Extension**: Try deep learning and other methods

### **Long-term Development**
1. **Real-time Prediction**: Develop online prediction system
2. **Personalized Recommendations**: Personalized services based on model results
3. **Policy Simulation**: Impact prediction of policy changes

---

## 📞 Contact Information

**Project Repository**: https://github.com/emmanuelwunjc/thesis.git  
**Analysis Date**: 2024-09-23  
**Technology Stack**: Python, scikit-learn, pandas, matplotlib  
**Data Source**: HINTS 7 Public Dataset  

---

## ✅ Project Status

- [x] Data cleaning completed
- [x] ML model selection completed
- [x] Best model identified
- [x] Detailed report generated
- [x] Visualization charts created
- [x] Executive summary completed
- [x] Ready for GitHub push

---

**Project Completion Time**: 2024-09-23  
**Total Development Time**: Approximately 2 hours  
**Technology Stack**: Python, scikit-learn, pandas, matplotlib  
**Data Source**: HINTS 7 Public Dataset  
**GitHub Repository**: https://github.com/emmanuelwunjc/thesis.git