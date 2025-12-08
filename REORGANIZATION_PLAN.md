# Repository Reorganization Plan

## Current Problems
- Too many files in root directory
- Scripts not organized by function
- Analysis outputs mixed (JSON, CSV, MD)
- No clear navigation path
- Duplicate files (English/Chinese versions)

## New Structure

```
thesis/
├── README.md                          # Main entry point with navigation
├── THESIS_OUTLINE.md                  # Thesis document
│
├── data/                              # All data files
│   ├── raw/                          # Original data
│   ├── processed/                    # Cleaned/processed data
│   └── intermediate/                 # Intermediate processing files
│
├── scripts/                           # All analysis scripts
│   ├── 01_data_preparation/          # Data loading, cleaning, index building
│   ├── 02_regression/                # Regression analysis
│   ├── 03_machine_learning/          # ML model selection
│   ├── 04_causal_inference/          # Causal inference methods
│   ├── 05_visualization/              # Plotting and visualization
│   └── utils/                        # Utility functions
│
├── analysis/                          # Analysis outputs
│   ├── results/                      # JSON results files
│   ├── summaries/                    # Markdown summaries
│   │   ├── english/                 # English summaries
│   │   └── chinese/                 # Chinese summaries
│   └── data/                         # Processed CSV files
│
├── figures/                           # All visualizations
│   ├── regression/                   # Regression plots
│   ├── ml/                           # ML visualizations
│   ├── causal/                       # Causal inference plots
│   └── exploratory/                  # Exploratory plots
│
├── docs/                              # Documentation
│   ├── guides/                       # User guides
│   ├── methodology/                  # Methodology docs
│   └── references/                   # Reference materials
│
└── temp/                              # Temporary files (gitignored)
```

## File Mapping

### Scripts Organization
- Data preparation: wrangle.py, build_privacy_index.py, data_cleaning_for_ml.py, prepare_regression_data.py
- Regression: regression_analysis.py, comprehensive_regression_analysis.py, privacy_as_dependent_analysis.py
- ML: ml_model_selection.py, simplified_ml_model_selection.py, exhaustive_variable_search.py
- Causal: causal_inference_analysis.py, difference_in_differences_analysis.py, panel_difference_in_differences_analysis.py, true_difference_in_differences_analysis.py
- Visualization: All plot_*.py, create_*.py scripts
- Utils: explore_data.py, visualize_data.py

### Analysis Organization
- Results: All *.json files
- Summaries/English: All *_SUMMARY.md, *_REPORT.md (English)
- Summaries/Chinese: All *_chinese.md files
- Data: All *.csv files

### Root Cleanup
- Keep: README.md, THESIS_OUTLINE.md
- Move to docs/: PROJECT_STATUS_SUMMARY.md, SUPERVISOR_GUIDE.md, FILE_DIRECTORY.md
- Delete: temp files, backup files

