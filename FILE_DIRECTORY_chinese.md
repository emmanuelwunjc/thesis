# HINTS 7 Diabetes Privacy Analysis - File Directory

## 📁 Project Structure Overview

This document provides a comprehensive guide to all files in the project, explaining their purpose, origin, and usage.

---

## 📊 Data Files (`data/`)

### `hints7_public copy.rda`
- **Source**: Original HINTS 7 Public Dataset
- **Purpose**: Raw survey data from Health Information National Trends Survey
- **Content**: 7,278 observations × 470 variables
- **Usage**: Primary data source for all analyses
- **Note**: Requires R or pyreadr to load

---

## 🔬 Analysis Scripts (`scripts/`)

### Core Analysis Scripts

#### `wrangle.py`
- **Purpose**: Main analysis pipeline
- **Function**: 
  - Loads HINTS 7 data
  - Detects diabetic patients
  - Performs demographic analysis
  - Analyzes privacy-related variables
  - Generates cross-tabulations
- **Outputs**: 
  - `diabetes_summary.json`
  - `diabetes_demographics_crosstabs.json`
  - `diabetes_privacy_analysis.json`
  - `privacy_dummies_compare.json`
- **Usage**: `python3 scripts/wrangle.py`

#### `build_privacy_index.py`
- **Purpose**: Constructs multi-dimensional privacy caution index
- **Function**:
  - Groups privacy variables into 6 sub-dimensions
  - Creates standardized 0-1 scale index
  - Calculates weighted statistics
- **Outputs**:
  - `privacy_caution_index.json`
  - `privacy_caution_index_individual.csv`
- **Usage**: `python3 scripts/build_privacy_index.py`

#### `regression_analysis.py`
- **Purpose**: Original regression analysis
- **Function**:
  - Main regression model
  - Interaction effects analysis
  - Subgroup analysis by age
  - Creates regression visualizations
- **Outputs**:
  - `regression_results.json`
  - `regression_analysis_results.png/pdf`
- **Usage**: `python3 scripts/regression_analysis.py`

#### `comprehensive_regression_analysis.py` ⭐ **NEW**
- **Purpose**: 6 different regression model specifications
- **Function**:
  - Model 1: Diabetes as moderator of privacy concerns
  - Model 2: Stratified analysis by diabetes status
  - Model 3: Diabetes-centered interaction model
  - Model 4: Mediation analysis
  - Model 5: Multiple outcomes analysis
  - Model 6: Diabetes severity analysis
- **Outputs**:
  - `comprehensive_regression_results.json`
  - `diabetes_effects_comparison.png`
- **Usage**: `python3 scripts/comprehensive_regression_analysis.py`

### Data Preparation Scripts

#### `prepare_regression_data.py`
- **Purpose**: Prepares comprehensive regression dataset
- **Function**:
  - Merges privacy index with original HINTS variables
  - Creates demographic variables
  - Handles missing values
  - Recodes variables for regression
- **Outputs**:
  - `regression_dataset.csv`
  - `hints_regression_vars.csv`
- **Usage**: `python3 scripts/prepare_regression_data.py`

### Visualization Scripts

#### `plot_privacy_index.py`
- **Purpose**: Creates privacy index visualizations
- **Function**: Generates charts for privacy index analysis
- **Usage**: `python3 scripts/plot_privacy_index.py`

#### `plot_privacy_diffs.py`
- **Purpose**: Creates privacy difference visualizations
- **Function**: Generates charts comparing privacy behaviors
- **Usage**: `python3 scripts/plot_privacy_diffs.py`

#### `age_distribution_plot.py`
- **Purpose**: Creates age distribution comparisons
- **Function**: Generates age distribution charts
- **Usage**: `python3 scripts/age_distribution_plot.py`

#### `age_group_analysis.py`
- **Purpose**: Performs age group analysis
- **Function**: Analyzes privacy behaviors by age groups
- **Usage**: `python3 scripts/age_group_analysis.py`

#### `explore_data.py`
- **Purpose**: Data exploration and summary statistics
- **Function**: Generates basic data summaries
- **Usage**: `python3 scripts/explore_data.py`

#### `visualize_data.py`
- **Purpose**: General data visualization
- **Function**: Creates various data visualizations
- **Usage**: `python3 scripts/visualize_data.py`

### Index Construction Scripts

#### `create_index_diagram.py`
- **Purpose**: Creates privacy index construction diagram
- **Function**: Generates visual representation of index structure
- **Outputs**: `privacy_index_construction_diagram.png`

#### `create_optimized_index_diagram.py`
- **Purpose**: Creates optimized privacy index diagram
- **Function**: Generates improved visual representation
- **Outputs**: `privacy_index_construction_diagram_optimized.png`

#### `create_detailed_index_table.py`
- **Purpose**: Creates detailed privacy index table
- **Function**: Generates comprehensive index breakdown
- **Outputs**: `privacy_index_detailed_table.png`

#### `create_optimized_detailed_table.py`
- **Purpose**: Creates optimized detailed index table
- **Function**: Generates improved index breakdown
- **Outputs**: `privacy_index_detailed_table_optimized.png`

### Results Display Scripts

#### `display_regression_results.py`
- **Purpose**: Displays formatted regression results
- **Function**: 
  - Formats results for console display
  - Creates academic-style tables
  - Provides policy implications
- **Usage**: `python3 scripts/display_regression_results.py`

#### `generate_latex_tables.py`
- **Purpose**: Generates LaTeX tables for academic papers
- **Function**: 
  - Creates publication-ready tables
  - Formats significance levels
  - Adds proper citations
- **Outputs**: `regression_tables_latex.tex`
- **Usage**: `python3 scripts/generate_latex_tables.py`

---

## 📈 Analysis Results (`analysis/`)

### Core Results

#### `diabetes_summary.json`
- **Source**: `wrangle.py`
- **Content**: Basic diabetes statistics and prevalence
- **Usage**: Quick reference for diabetes rates

#### `diabetes_demographics_crosstabs.json`
- **Source**: `wrangle.py`
- **Content**: Cross-tabulations of diabetes vs demographics
- **Usage**: Demographic analysis results

#### `diabetes_privacy_analysis.json`
- **Source**: `wrangle.py`
- **Content**: Detailed privacy behavior analysis
- **Usage**: Privacy variable comparisons

#### `privacy_dummies_compare.json`
- **Source**: `wrangle.py`
- **Content**: Weighted comparisons of privacy dummies
- **Usage**: Privacy behavior differences

### Privacy Index Results

#### `privacy_caution_index.json`
- **Source**: `build_privacy_index.py`
- **Content**: Privacy index summary statistics
- **Usage**: Index construction results

#### `privacy_caution_index_individual.csv`
- **Source**: `build_privacy_index.py`
- **Content**: Individual-level privacy index scores
- **Usage**: Regression analysis input data

### Regression Results

#### `regression_results.json`
- **Source**: `regression_analysis.py`
- **Content**: Original regression analysis results
- **Usage**: Main regression findings

#### `comprehensive_regression_results.json` ⭐ **NEW**
- **Source**: `comprehensive_regression_analysis.py`
- **Content**: Results from all 6 regression models
- **Usage**: Comprehensive analysis findings

#### `regression_dataset.csv`
- **Source**: `prepare_regression_data.py`
- **Content**: Prepared dataset for regression analysis
- **Usage**: Regression analysis input data

#### `hints_regression_vars.csv`
- **Source**: `prepare_regression_data.py`
- **Content**: Extracted HINTS variables for regression
- **Usage**: Intermediate data file

### Academic Outputs

#### `regression_tables_latex.tex`
- **Source**: `generate_latex_tables.py`
- **Content**: LaTeX tables for academic papers
- **Usage**: Publication-ready tables

#### `REGRESSION_RESULTS_SUMMARY.md`
- **Source**: Manual creation
- **Content**: Comprehensive results summary
- **Usage**: Results documentation

---

## 📊 Visualizations (`figures/`)

### Regression Analysis Charts

#### `regression_analysis_results.png/pdf`
- **Source**: `regression_analysis.py`
- **Content**: 4-panel regression analysis visualization
- **Usage**: Main regression results presentation

#### `diabetes_effects_comparison.png` ⭐ **NEW**
- **Source**: `comprehensive_regression_analysis.py`
- **Content**: Comparison of diabetes effects across models
- **Usage**: Model comparison visualization

### Privacy Index Charts

#### `privacy_caution_index_analysis.png`
- **Source**: `plot_privacy_index.py`
- **Content**: Privacy index analysis charts
- **Usage**: Index analysis presentation

#### `privacy_index_construction_diagram_optimized.png`
- **Source**: `create_optimized_index_diagram.py`
- **Content**: Privacy index structure diagram
- **Usage**: Index construction explanation

#### `privacy_index_detailed_table_optimized.png`
- **Source**: `create_optimized_detailed_table.py`
- **Content**: Detailed privacy index breakdown
- **Usage**: Index component explanation

### Privacy Behavior Charts

#### `privacy_top10_diffs.png`
- **Source**: `plot_privacy_diffs.py`
- **Content**: Top 10 privacy behavior differences
- **Usage**: Key differences presentation

#### `privacy_shared_device.png`
- **Source**: `plot_privacy_diffs.py`
- **Content**: Device sharing behavior comparison
- **Usage**: Device sharing analysis

#### `privacy_use_computer.png`
- **Source**: `plot_privacy_diffs.py`
- **Content**: Computer usage comparison
- **Usage**: Computer usage analysis

#### `privacy_use_watch.png`
- **Source**: `plot_privacy_diffs.py`
- **Content**: Smart watch usage comparison
- **Usage**: Smart watch usage analysis

#### `privacy_trust_hcsystem.png`
- **Source**: `plot_privacy_diffs.py`
- **Content**: Healthcare system trust comparison
- **Usage**: Trust analysis

#### `privacy_trust_scientists.png`
- **Source**: `plot_privacy_diffs.py`
- **Content**: Scientist trust comparison
- **Usage**: Trust analysis

#### `privacy_portal_pharmacy.png`
- **Source**: `plot_privacy_diffs.py`
- **Content**: Pharmacy portal usage comparison
- **Usage**: Portal usage analysis

### Age Analysis Charts

#### `age_distribution_comparison.png`
- **Source**: `age_distribution_plot.py`
- **Content**: Age distribution comparison
- **Usage**: Age analysis presentation

#### `age_group_comparison.png`
- **Source**: `age_group_analysis.py`
- **Content**: Age group comparison
- **Usage**: Age group analysis

#### `diabetes_analysis.png`
- **Source**: `explore_data.py`
- **Content**: General diabetes analysis
- **Usage**: Basic analysis presentation

---

## 📚 Documentation (`docs/`)

### Project Documentation

#### `PROJECT_LOG.md`
- **Purpose**: Complete project documentation
- **Content**: 
  - Methodology
  - Findings
  - Analysis results
  - Project history
- **Usage**: Comprehensive project reference

#### `QUICK_START.md`
- **Purpose**: Quick project recovery guide
- **Content**: 
  - Key commands
  - File locations
  - Quick analysis steps
- **Usage**: Fast project setup

### Original Documentation

#### `HINTS 7 Annotated English.pdf`
- **Source**: Original HINTS documentation
- **Content**: English survey annotations
- **Usage**: Survey question reference

#### `HINTS 7 Annotated Spanish.pdf`
- **Source**: Original HINTS documentation
- **Content**: Spanish survey annotations
- **Usage**: Survey question reference

#### `HINTS 7 History Document.pdf`
- **Source**: Original HINTS documentation
- **Content**: Survey history and methodology
- **Usage**: Methodology reference

#### `HINTS 7 Methodology Report.pdf`
- **Source**: Original HINTS documentation
- **Content**: Detailed methodology
- **Usage**: Methodology reference

#### `HINTS 7 Public Codebook.pdf`
- **Source**: Original HINTS documentation
- **Content**: Variable codebook
- **Usage**: Variable reference

#### `HINTS 7 Survey Overview Data Analysis Recommendations.pdf`
- **Source**: Original HINTS documentation
- **Content**: Analysis recommendations
- **Usage**: Analysis guidance

---

## 📋 Project Management Files

### `README.md`
- **Purpose**: Project overview and quick start
- **Content**: 
  - Project description
  - Key findings
  - Usage instructions
  - Project status
- **Usage**: First point of contact

### `PROJECT_STATUS_SUMMARY.md`
- **Purpose**: Comprehensive project status
- **Content**: 
  - Achievements
  - Findings
  - Deliverables
  - Next steps
- **Usage**: Project progress tracking

### `FILE_DIRECTORY.md` (This File)
- **Purpose**: Complete file directory and usage guide
- **Content**: 
  - File descriptions
  - Usage instructions
  - Source information
- **Usage**: File navigation and understanding

### `.gitignore`
- **Purpose**: Git ignore configuration
- **Content**: 
  - Python cache files
  - Temporary files
  - OS-specific files
- **Usage**: Version control configuration

---

## 🚀 Quick Start Guide

### For New Users
1. Start with `README.md` for project overview
2. Read `PROJECT_LOG.md` for detailed methodology
3. Check `PROJECT_STATUS_SUMMARY.md` for current status
4. Use this file (`FILE_DIRECTORY.md`) to understand file structure

### For Analysis
1. Run `python3 scripts/wrangle.py` for basic analysis
2. Run `python3 scripts/build_privacy_index.py` for privacy index
3. Run `python3 scripts/comprehensive_regression_analysis.py` for full regression analysis
4. Check `analysis/` folder for results

### For Visualization
1. Check `figures/` folder for all charts
2. Use `scripts/display_regression_results.py` for formatted results
3. Use `scripts/generate_latex_tables.py` for academic tables

### For Documentation
1. Check `docs/` folder for project documentation
2. Use `analysis/REGRESSION_RESULTS_SUMMARY.md` for results summary
3. Use `analysis/regression_tables_latex.tex` for academic tables

---

## 📞 File Dependencies

### Data Flow
```
hints7_public copy.rda → wrangle.py → diabetes_*.json
                    → prepare_regression_data.py → regression_dataset.csv
                    → build_privacy_index.py → privacy_caution_index_*.json/csv
```

### Analysis Flow
```
regression_dataset.csv → regression_analysis.py → regression_results.json
                     → comprehensive_regression_analysis.py → comprehensive_regression_results.json
```

### Visualization Flow
```
*.json results → display_regression_results.py → console output
              → generate_latex_tables.py → regression_tables_latex.tex
              → plot_*.py → figures/*.png
```

---

*Last Updated: 2024-09-23*  
*Total Files: 50+ files across 5 main directories*  
*Project Status: Complete with comprehensive regression analysis*
