# Repository Navigation Guide

## 🗺️ Quick Start Paths

### For First-Time Visitors
1. **Start Here**: [`README.md`](README.md) - Overview and quick start
2. **Thesis Document**: [`THESIS_OUTLINE.md`](THESIS_OUTLINE.md) - Complete thesis outline
3. **Key Findings**: [`analysis/summaries/english/CHRONIC_DISEASE_ANALYSIS_SUMMARY.md`](analysis/summaries/english/CHRONIC_DISEASE_ANALYSIS_SUMMARY.md)

### For Supervisors
1. **Supervisor Guide**: [`docs/guides/SUPERVISOR_GUIDE.md`](docs/guides/SUPERVISOR_GUIDE.md)
2. **Executive Summary**: [`analysis/summaries/english/BEST_MODEL_EXECUTIVE_SUMMARY.md`](analysis/summaries/english/BEST_MODEL_EXECUTIVE_SUMMARY.md)
3. **Project Status**: [`docs/guides/PROJECT_STATUS_SUMMARY.md`](docs/guides/PROJECT_STATUS_SUMMARY.md)

### For Researchers
1. **Methodology**: [`analysis/summaries/english/MODEL_LOGIC_SUMMARY.md`](analysis/summaries/english/MODEL_LOGIC_SUMMARY.md)
2. **Results**: [`analysis/summaries/english/REGRESSION_RESULTS_SUMMARY.md`](analysis/summaries/english/REGRESSION_RESULTS_SUMMARY.md)
3. **Causal Inference**: [`analysis/summaries/english/CAUSAL_INFERENCE_SUMMARY.md`](analysis/summaries/english/CAUSAL_INFERENCE_SUMMARY.md)

---

## 📁 Directory Structure

### `/data/` - Data Files
- **`raw/`** - Original HINTS 7 dataset
- **`processed/`** - Cleaned and processed data
- **`intermediate/`** - Intermediate processing files

### `/scripts/` - Analysis Scripts (Organized by Function)

#### `01_data_preparation/` - Data Loading & Cleaning
- `wrangle.py` - Main data loading utility
- `build_privacy_index.py` - Privacy index construction
- `data_cleaning_for_ml.py` - ML data preparation
- `prepare_regression_data.py` - Regression data preparation

#### `02_regression/` - Regression Analysis
- `regression_analysis.py` - Main regression analysis
- `comprehensive_regression_analysis.py` - 6 regression models
- `privacy_as_dependent_analysis.py` - Privacy as outcome
- `privacy_index_correlation_analysis.py` - Correlation analysis

#### `03_machine_learning/` - ML Model Selection
- `ml_model_selection.py` - Full ML analysis
- `simplified_ml_model_selection.py` - Simplified ML selection
- `exhaustive_variable_search.py` - Variable search
- `multi_chronic_disease_analysis.py` - Multi-condition analysis

#### `04_causal_inference/` - Causal Methods
- `causal_inference_analysis.py` - PSM, IV, RDD
- `difference_in_differences_analysis.py` - DiD analysis
- `panel_difference_in_differences_analysis.py` - Panel DiD
- `true_difference_in_differences_analysis.py` - True panel DiD

#### `05_visualization/` - Plotting & Visualization
- `plot_privacy_index.py` - Privacy index plots
- `plot_privacy_diffs.py` - Privacy difference plots
- `create_*.py` - Various visualization creation scripts

#### `utils/` - Utility Functions
- `explore_data.py` - Data exploration
- `visualize_data.py` - General visualization

### `/analysis/` - Analysis Outputs

#### `results/` - JSON Results Files
- All `.json` files with analysis results
- `.tex` files with LaTeX tables

#### `summaries/english/` - English Summaries
- `BEST_MODEL_EXECUTIVE_SUMMARY.md` - ⭐ Start here
- `REGRESSION_RESULTS_SUMMARY.md` - Regression results
- `CAUSAL_INFERENCE_SUMMARY.md` - Causal inference
- `MULTI_CHRONIC_DISEASE_FINDINGS.md` - Multi-condition analysis
- `DIABETES_ROLE_AND_SIGNIFICANCE.md` - Diabetes significance
- `DATA_ANALYSIS_DECISIONS.md` - Analysis decisions

#### `summaries/chinese/` - Chinese Summaries
- All Chinese versions of summaries

#### `data/` - Processed Data Files
- All `.csv` files with processed data

### `/figures/` - Visualizations
- Organized by analysis type (regression, ml, causal, exploratory)

### `/docs/` - Documentation
- `guides/` - User guides and documentation
- `methodology/` - Methodology documentation
- `references/` - Reference materials

---

## 🔄 Typical Workflow

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
python scripts/04_causal_inference/difference_in_differences_analysis.py
```

### 5. Visualization
```bash
# Create plots
python scripts/05_visualization/plot_privacy_index.py
```

---

## 📊 Key Documents by Topic

### Understanding the Research
- **Overview**: [`README.md`](README.md)
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

## 🔍 Finding Specific Information

### Looking for...
- **Data files**: Check `/data/` (raw, processed, intermediate)
- **Analysis scripts**: Check `/scripts/` (organized by function)
- **Results**: Check `/analysis/results/` (JSON files)
- **Summaries**: Check `/analysis/summaries/english/` (Markdown)
- **Plots**: Check `/figures/` (organized by type)
- **Documentation**: Check `/docs/guides/`

### Common Searches
- **Privacy index**: `scripts/01_data_preparation/build_privacy_index.py`
- **Regression results**: `analysis/summaries/english/REGRESSION_RESULTS_SUMMARY.md`
- **ML models**: `scripts/03_machine_learning/` or `analysis/summaries/english/BEST_ML_MODEL_DETAILED_REPORT.md`
- **Chronic diseases**: `scripts/03_machine_learning/multi_chronic_disease_analysis.py`
- **Causal inference**: `scripts/04_causal_inference/` or `analysis/summaries/english/CAUSAL_INFERENCE_SUMMARY.md`

---

## 📝 File Naming Conventions

### Scripts
- `*_analysis.py` - Main analysis scripts
- `*_preparation.py` - Data preparation
- `plot_*.py` - Visualization scripts
- `create_*.py` - Diagram creation

### Results
- `*_results.json` - Analysis results
- `*_SUMMARY.md` - Summary documents
- `*_REPORT.md` - Detailed reports
- `*_FINDINGS.md` - Research findings

### Data
- `*_cleaned.csv` - Cleaned data
- `*_individual.csv` - Individual-level data
- `*_dataset.csv` - Processed datasets

---

*Last Updated: 2024*  
*Repository Structure: Organized by function and type*



## 📁 Current File Structure

### `/scripts/` - Analysis Scripts

#### `01_data_preparation/` (4 scripts)
- `build_privacy_index.py`
- `data_cleaning_for_ml.py`
- `prepare_regression_data.py`
- `wrangle.py`

#### `02_regression/` (5 scripts)
- `comprehensive_privacy_analysis.py`
- `comprehensive_regression_analysis.py`
- `privacy_as_dependent_analysis.py`
- `privacy_index_correlation_analysis.py`
- `regression_analysis.py`

#### `03_machine_learning/` (5 scripts)
- `exhaustive_variable_search.py`
- `ml_model_selection.py`
- `multi_chronic_disease_analysis.py`
- `quick_high_r2_search.py`
- `simplified_ml_model_selection.py`

#### `04_causal_inference/` (4 scripts)
- `causal_inference_analysis.py`
- `difference_in_differences_analysis.py`
- `panel_difference_in_differences_analysis.py`
- `true_difference_in_differences_analysis.py`

#### `05_visualization/` (16 scripts)
- `age_distribution_plot.py`
- `age_group_analysis.py`
- `create_best_model_visualizations.py`
- `create_conceptual_framework_diagram.py`
- `create_descriptive_statistics.py`
- `create_descriptive_statistics_figure.py`
- `create_detailed_index_table.py`
- `create_index_diagram.py`
- `create_model_logic_diagram.py`
- `create_optimized_detailed_table.py`
- ... and 6 more

#### `utils/` (6 scripts)
- `check_and_fix_all_figures.py`
- `display_regression_results.py`
- `explore_data.py`
- `generate_latex_tables.py`
- `update_documentation.py`
- `visualize_data.py`

### `/analysis/` - Analysis Outputs

#### `results/` (18 JSON files)
- `age_band_analyses.json`
- `causal_inference_results.json`
- `comprehensive_regression_results.json`
- `data_quality_report.json`
- `diabetes_demographics_crosstabs.json`
- ... and 13 more

#### `summaries/english/` (11 summaries)
- `BEST_ML_MODEL_DETAILED_REPORT.md`
- `BEST_MODEL_EXECUTIVE_SUMMARY.md`
- `CAUSAL_INFERENCE_SUMMARY.md`
- `CHRONIC_DISEASE_ANALYSIS_SUMMARY.md`
- `DATA_ANALYSIS_DECISIONS.md`
- ... and 6 more


*Last Updated: 2026-01-14 16:12:40*
