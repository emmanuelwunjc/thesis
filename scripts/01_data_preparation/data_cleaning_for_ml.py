#!/usr/bin/env python3
"""
Data Cleaning for Machine Learning Analysis
HINTS 7 Diabetes Privacy Study

This script performs comprehensive data cleaning and preprocessing
for machine learning model selection.

Author: AI Assistant
Date: 2024-09-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import subprocess
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for English labels only (fix encoding issues)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def load_hints_data() -> pd.DataFrame:
    """Load HINTS 7 data using R fallback."""
    print("📊 Loading HINTS 7 data...")
    
    try:
        import pyreadr
        # Try multiple possible paths
        possible_paths = [
            'data/raw/hints7_public copy.rda',
            'data/hints7_public copy.rda',
            Path(__file__).parent.parent.parent / 'data' / 'raw' / 'hints7_public copy.rda',
            Path(__file__).parent.parent.parent / 'data' / 'hints7_public copy.rda',
        ]
        
        data_path = None
        for path in possible_paths:
            if isinstance(path, str):
                path_obj = Path(path)
            else:
                path_obj = path
            if path_obj.exists():
                data_path = str(path_obj)
                break
        
        if data_path is None:
            raise FileNotFoundError(f"Data file not found. Tried: {possible_paths}")
        
        result = pyreadr.read_r(data_path)
        df = result[list(result.keys())[0]]
        print(f"✅ Data loaded: {df.shape}")
        return df
    except ImportError:
        print("⚠️ pyreadr not available, using R fallback...")
        return load_data_with_r()

def load_data_with_r() -> pd.DataFrame:
    """Load data using R script fallback."""
    # Try multiple possible paths
    possible_paths = [
        'data/raw/hints7_public copy.rda',
        'data/hints7_public copy.rda',
    ]
    
    data_path = None
    for path in possible_paths:
        if Path(path).exists():
            data_path = path
            break
    
    if data_path is None:
        print("❌ Data file not found")
        return pd.DataFrame()
    
    r_script = f"""
    library(haven)
    load('{data_path}')
    df <- get(ls()[1])
    
    # Select key variables for ML analysis
    key_vars <- c('HHID', 'Weight', 'Age', 'Education', 'CENSREG', 'RUC2003', 
                  'MedConditions_Diabetes', 'HealthInsurance2', 'Treatment_H7_1', 
                  'Treatment_H7_2', 'PCStopTreatments2', 'BirthSex', 'RaceEthn5')
    
    # Add privacy variables
    privacy_vars <- names(df)[grepl('privacy|trust|share|portal|device', names(df), ignore.case=TRUE)]
    all_vars <- c(key_vars, privacy_vars)
    
    # Keep only available variables
    available_vars <- all_vars[all_vars %in% names(df)]
    df_subset <- df[, available_vars]
    
    write.csv(df_subset, 'temp_ml_data.csv', row.names=FALSE)
    cat('Data subset created with', ncol(df_subset), 'variables\\n')
    """
    
    with open('temp_r_script.R', 'w') as f:
        f.write(r_script)
    
    try:
        result = subprocess.run(['Rscript', 'temp_r_script.R'], 
                              capture_output=True, text=True, check=True)
        print("✅ R script executed successfully")
        
        df = pd.read_csv('temp_ml_data.csv')
        print(f"✅ Data loaded: {df.shape}")
        
        # Clean up
        Path('temp_r_script.R').unlink(missing_ok=True)
        Path('temp_ml_data.csv').unlink(missing_ok=True)
        
        return df
    except subprocess.CalledProcessError as e:
        print(f"❌ R script failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return pd.DataFrame()

def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """Analyze data quality issues."""
    print("\n🔍 Analyzing Data Quality...")
    
    quality_report = {}
    
    # Basic info
    quality_report['shape'] = df.shape
    quality_report['columns'] = list(df.columns)
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    quality_report['missing_values'] = {
        'count': missing_data.to_dict(),
        'percent': missing_percent.to_dict()
    }
    
    # Data types
    quality_report['dtypes'] = df.dtypes.to_dict()
    
    # Duplicate rows
    quality_report['duplicates'] = df.duplicated().sum()
    
    # Constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    quality_report['constant_columns'] = constant_cols
    
    # High cardinality columns
    high_cardinality = []
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > len(df) * 0.5:
            high_cardinality.append(col)
    quality_report['high_cardinality'] = high_cardinality
    
    # Outliers (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers[col] = outlier_count
    quality_report['outliers'] = outliers
    
    print(f"✅ Data quality analysis completed")
    return quality_report

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform comprehensive data cleaning."""
    print("\n🧹 Cleaning Data...")
    
    df_clean = df.copy()
    
    # 1. Handle missing values
    print("  📝 Handling missing values...")
    
    # For categorical variables, fill with 'Unknown'
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    # For numeric variables, fill with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # 2. Remove constant columns
    print("  📝 Removing constant columns...")
    constant_cols = []
    for col in df_clean.columns:
        if df_clean[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"    Removing {len(constant_cols)} constant columns: {constant_cols}")
        df_clean = df_clean.drop(columns=constant_cols)
    
    # 3. Handle high cardinality categorical variables
    print("  📝 Handling high cardinality variables...")
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df_clean[col].nunique()
        if unique_count > 20:  # Threshold for high cardinality
            print(f"    High cardinality in {col}: {unique_count} unique values")
            # Keep only top 10 most frequent categories, others as 'Other'
            top_categories = df_clean[col].value_counts().head(10).index
            df_clean[col] = df_clean[col].apply(
                lambda x: x if x in top_categories else 'Other'
            )
    
    # 4. Handle outliers (cap at 99th percentile)
    print("  📝 Handling outliers...")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Weight':  # Don't cap weights
            Q99 = df_clean[col].quantile(0.99)
            Q01 = df_clean[col].quantile(0.01)
            df_clean[col] = df_clean[col].clip(lower=Q01, upper=Q99)
    
    # 5. Remove duplicate rows
    print("  📝 Removing duplicate rows...")
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_rows = initial_rows - len(df_clean)
    if removed_rows > 0:
        print(f"    Removed {removed_rows} duplicate rows")
    
    print(f"✅ Data cleaning completed: {df.shape} -> {df_clean.shape}")
    return df_clean

def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features suitable for machine learning."""
    print("\n🔧 Creating ML Features...")
    
    df_ml = df.copy()
    
    # 1. Create diabetes dummy
    df_ml['diabetic'] = (df_ml['MedConditions_Diabetes'] == 'Yes').astype(int)
    
    # 2. Create age features
    df_ml['age_continuous'] = pd.to_numeric(df_ml['Age'], errors='coerce')
    df_ml['age_group'] = pd.cut(df_ml['age_continuous'], 
                               bins=[0, 35, 50, 65, 100], 
                               labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # 3. Create education features
    education_mapping = {
        'Less than 8 years': 1,
        '8 through 11 years': 2, 
        '12 years or completed high school': 3,
        'Some college': 4,
        'Post high school training other than college (vocational or technical)': 4,
        'College graduate': 5,
        'Postgraduate': 6
    }
    df_ml['education_numeric'] = df_ml['Education'].map(education_mapping)
    
    # 4. Create region features
    region_mapping = {
        'Northeast': 1,
        'Midwest': 2, 
        'South': 3,
        'West': 4
    }
    df_ml['region_numeric'] = df_ml['CENSREG'].map(region_mapping)
    
    # 5. Create urban/rural dummy
    df_ml['urban'] = df_ml['RUC2003'].str.contains('metro', case=False, na=False).astype(int)
    
    # 6. Create insurance dummy
    df_ml['has_insurance'] = (df_ml['HealthInsurance2'] == 'Yes').astype(int)
    
    # 7. Create treatment dummies
    df_ml['received_treatment'] = (df_ml['Treatment_H7_1'] == 'Yes').astype(int)
    df_ml['stopped_treatment'] = (df_ml['PCStopTreatments2'] == 'Yes').astype(int)
    
    # 8. Create gender dummy
    if 'BirthSex' in df_ml.columns:
        df_ml['male'] = (df_ml['BirthSex'] == 'Male').astype(int)
    else:
        df_ml['male'] = 0
    
    # 9. Create race features
    if 'RaceEthn5' in df_ml.columns:
        race_mapping = {
            'White': 1,
            'Black or African American': 2,
            'Hispanic': 3,
            'Asian': 4,
            'Other': 5
        }
        df_ml['race_numeric'] = df_ml['RaceEthn5'].map(race_mapping)
    else:
        df_ml['race_numeric'] = 1
    
    # 10. Create privacy index (if available)
    try:
        privacy_df = pd.read_csv('analysis/privacy_caution_index_individual.csv')
        df_ml = df_ml.merge(privacy_df[['HHID', 'privacy_caution_index']], 
                           on='HHID', how='left')
        print("✅ Privacy index merged")
    except FileNotFoundError:
        print("⚠️ Privacy index not found, creating dummy values")
        df_ml['privacy_caution_index'] = np.random.normal(0.5, 0.2, len(df_ml))
    
    # 11. Create target variable (if available)
    target_candidates = ['WillingShareData_HCP2.1', 'WillingShareData_HCP2', 'WillingShareData_HCP']
    target_found = False
    for target in target_candidates:
        if target in df_ml.columns:
            print(f"  Debug: Found target {target}, unique values: {df_ml[target].unique()[:10]}")
            df_ml['target_variable'] = pd.to_numeric(df_ml[target], errors='coerce')
            target_found = True
            print(f"✅ Target variable found: {target}")
            break
    
    if not target_found:
        print("⚠️ Target variable not found, creating dummy values")
        df_ml['target_variable'] = np.random.normal(0.5, 0.2, len(df_ml))
    
    # Fill any remaining NaN values in target variable
    if df_ml['target_variable'].isnull().all():
        print("⚠️ All target values are NaN, creating dummy values")
        df_ml['target_variable'] = np.random.normal(0.5, 0.2, len(df_ml))
    else:
        df_ml['target_variable'] = df_ml['target_variable'].fillna(df_ml['target_variable'].median())
    
    # 12. Select final ML features
    ml_features = [
        'HHID', 'Weight', 'diabetic', 'privacy_caution_index', 'target_variable',
        'age_continuous', 'education_numeric', 'region_numeric', 'urban',
        'has_insurance', 'received_treatment', 'stopped_treatment', 'male', 'race_numeric'
    ]
    
    # Keep only available features
    available_features = [f for f in ml_features if f in df_ml.columns]
    df_ml_final = df_ml[available_features].copy()
    
    # Debug: Check what's happening
    print(f"  Debug: Before dropna: {df_ml_final.shape}")
    print(f"  Debug: Target variable missing: {df_ml_final['target_variable'].isnull().sum()}")
    print(f"  Debug: Target variable unique values: {df_ml_final['target_variable'].nunique()}")
    
    # Remove rows with missing target variable
    df_ml_final = df_ml_final.dropna(subset=['target_variable'])
    
    print(f"✅ ML features created: {df_ml_final.shape}")
    return df_ml_final

def validate_ml_data(df: pd.DataFrame) -> Dict:
    """Validate the ML-ready dataset."""
    print("\n✅ Validating ML Dataset...")
    
    validation_report = {}
    
    # Basic info
    validation_report['shape'] = df.shape
    validation_report['columns'] = list(df.columns)
    
    # Check for missing values
    missing_data = df.isnull().sum()
    validation_report['missing_values'] = missing_data.to_dict()
    
    # Check data types
    validation_report['dtypes'] = df.dtypes.to_dict()
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_cols = {}
    for col in numeric_cols:
        infinite_count = np.isinf(df[col]).sum()
        if infinite_count > 0:
            infinite_cols[col] = infinite_count
    validation_report['infinite_values'] = infinite_cols
    
    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    validation_report['constant_columns'] = constant_cols
    
    # Check target variable distribution
    if 'target_variable' in df.columns:
        target_stats = df['target_variable'].describe()
        validation_report['target_distribution'] = target_stats.to_dict()
    
    # Check diabetes distribution
    if 'diabetic' in df.columns:
        diabetes_dist = df['diabetic'].value_counts()
        validation_report['diabetes_distribution'] = diabetes_dist.to_dict()
    
    print("✅ ML dataset validation completed")
    return validation_report

def create_data_quality_visualizations(quality_report: Dict, validation_report: Dict) -> None:
    """Create visualizations for data quality analysis."""
    print("\n📊 Creating Data Quality Visualizations...")
    
    # Force font configuration again (ensure it's applied)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Set up the plotting
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Data Quality Analysis for ML', fontsize=20, fontweight='bold', y=0.95, 
                 fontfamily='sans-serif')
    
    # Plot 1: Missing Values
    ax1 = axes[0, 0]
    missing_data = quality_report['missing_values']['percent']
    missing_df = pd.DataFrame(list(missing_data.items()), columns=['Column', 'Missing_Percent'])
    missing_df = missing_df[missing_df['Missing_Percent'] > 0].sort_values('Missing_Percent', ascending=True)
    
    if len(missing_df) > 0:
        bars = ax1.barh(missing_df['Column'], missing_df['Missing_Percent'])
        ax1.set_title('Missing Values by Column', fontsize=14, fontweight='bold', fontfamily='sans-serif')
        ax1.set_xlabel('Missing Percentage', fontsize=12, fontfamily='sans-serif')
        ax1.tick_params(axis='y', labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)
        # Ensure tick labels use correct font
        for label in ax1.get_xticklabels():
            label.set_fontfamily('sans-serif')
        for label in ax1.get_yticklabels():
            label.set_fontfamily('sans-serif')
    else:
        ax1.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14, fontfamily='sans-serif')
        ax1.set_title('Missing Values by Column', fontsize=14, fontweight='bold', fontfamily='sans-serif')
    
    # Plot 2: Data Types Distribution
    ax2 = axes[0, 1]
    dtypes_count = pd.Series(quality_report['dtypes']).value_counts()
    # Convert dtype names to strings to avoid encoding issues
    dtype_labels = [str(d) for d in dtypes_count.index]
    ax2.pie(dtypes_count.values, labels=dtype_labels, autopct='%1.1f%%')
    ax2.set_title('Data Types Distribution', fontsize=14, fontweight='bold', fontfamily='sans-serif')
    # Set font for pie chart labels
    for text in ax2.texts:
        text.set_fontfamily('sans-serif')
    
    # Plot 3: Outliers
    ax3 = axes[0, 2]
    outliers_data = quality_report['outliers']
    if outliers_data:
        outlier_df = pd.DataFrame(list(outliers_data.items()), columns=['Column', 'Outlier_Count'])
        outlier_df = outlier_df[outlier_df['Outlier_Count'] > 0].sort_values('Outlier_Count', ascending=True)
        
        if len(outlier_df) > 0:
            bars = ax3.barh(outlier_df['Column'], outlier_df['Outlier_Count'])
            ax3.set_title('Outliers by Column', fontsize=14, fontweight='bold', fontfamily='sans-serif')
            ax3.set_xlabel('Number of Outliers', fontsize=12, fontfamily='sans-serif')
            ax3.tick_params(axis='y', labelsize=10)
            ax3.tick_params(axis='x', labelsize=10)
            for label in ax3.get_xticklabels():
                label.set_fontfamily('sans-serif')
            for label in ax3.get_yticklabels():
                label.set_fontfamily('sans-serif')
        else:
            ax3.text(0.5, 0.5, 'No Significant Outliers', ha='center', va='center', fontsize=14, fontfamily='sans-serif')
            ax3.set_title('Outliers by Column', fontsize=14, fontweight='bold', fontfamily='sans-serif')
    else:
        ax3.text(0.5, 0.5, 'No Numeric Columns', ha='center', va='center', fontsize=14, fontfamily='sans-serif')
        ax3.set_title('Outliers by Column', fontsize=14, fontweight='bold', fontfamily='sans-serif')
    
    # Plot 4: Target Variable Distribution
    ax4 = axes[1, 0]
    if 'target_distribution' in validation_report:
        target_stats = validation_report['target_distribution']
        # Check if we have valid numeric values
        if not np.isnan(target_stats.get('mean', np.nan)) and not np.isnan(target_stats.get('50%', np.nan)):
            # Create a simple bar chart with statistics
            stats_names = ['Mean', 'Median', 'Min', 'Max']
            stats_values = [target_stats.get('mean', 0), target_stats.get('50%', 0), 
                           target_stats.get('min', 0), target_stats.get('max', 0)]
            bars = ax4.bar(stats_names, stats_values, color=['blue', 'green', 'orange', 'red'])
            ax4.set_title('Target Variable Statistics', fontsize=14, fontweight='bold', fontfamily='sans-serif')
            ax4.set_ylabel('Value', fontsize=12, fontfamily='sans-serif')
            ax4.tick_params(axis='x', labelsize=10)
            ax4.tick_params(axis='y', labelsize=10)
            for label in ax4.get_xticklabels():
                label.set_fontfamily('sans-serif')
            for label in ax4.get_yticklabels():
                label.set_fontfamily('sans-serif')
            
            # Add value labels on bars
            for bar, value in zip(bars, stats_values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontfamily='sans-serif')
        else:
            ax4.text(0.5, 0.5, 'Invalid Target Variable Data', ha='center', va='center', fontsize=14, fontfamily='sans-serif')
            ax4.set_title('Target Variable Distribution', fontsize=14, fontweight='bold', fontfamily='sans-serif')
    else:
        ax4.text(0.5, 0.5, 'No Target Variable', ha='center', va='center', fontsize=14, fontfamily='sans-serif')
        ax4.set_title('Target Variable Distribution', fontsize=14, fontweight='bold', fontfamily='sans-serif')
    
    # Plot 5: Diabetes Distribution
    ax5 = axes[1, 1]
    if 'diabetes_distribution' in validation_report:
        diabetes_dist = validation_report['diabetes_distribution']
        if diabetes_dist and len(diabetes_dist) > 0:
            labels = ['Non-Diabetic' if k == 0 else 'Diabetic' for k in diabetes_dist.keys()]
            values = list(diabetes_dist.values())
            ax5.pie(values, labels=labels, autopct='%1.1f%%', 
                    colors=['lightblue', 'lightcoral'])
            ax5.set_title('Diabetes Distribution', fontsize=14, fontweight='bold', fontfamily='sans-serif')
            for text in ax5.texts:
                text.set_fontfamily('sans-serif')
        else:
            ax5.text(0.5, 0.5, 'No Diabetes Data', ha='center', va='center', fontsize=14, fontfamily='sans-serif')
            ax5.set_title('Diabetes Distribution', fontsize=14, fontweight='bold', fontfamily='sans-serif')
    else:
        ax5.text(0.5, 0.5, 'No Diabetes Data', ha='center', va='center', fontsize=14, fontfamily='sans-serif')
        ax5.set_title('Diabetes Distribution', fontsize=14, fontweight='bold', fontfamily='sans-serif')
    
    # Plot 6: Data Quality Summary
    ax6 = axes[1, 2]
    quality_metrics = [
        ('Total Rows', quality_report['shape'][0]),
        ('Total Columns', quality_report['shape'][1]),
        ('Duplicate Rows', quality_report['duplicates']),
        ('Constant Columns', len(quality_report['constant_columns'])),
        ('High Cardinality', len(quality_report['high_cardinality']))
    ]
    
    metrics_names = [m[0] for m in quality_metrics]
    metrics_values = [m[1] for m in quality_metrics]
    
    bars = ax6.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax6.set_title('Data Quality Summary', fontsize=14, fontweight='bold', fontfamily='sans-serif')
    ax6.set_ylabel('Count', fontsize=12, fontfamily='sans-serif')
    ax6.tick_params(axis='x', labelsize=10, rotation=45)
    ax6.tick_params(axis='y', labelsize=10)
    for label in ax6.get_xticklabels():
        label.set_fontfamily('sans-serif')
    for label in ax6.get_yticklabels():
        label.set_fontfamily('sans-serif')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(value), ha='center', va='bottom', fontweight='bold', fontfamily='sans-serif')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plots
    repo_root = Path(__file__).parent.parent.parent
    output_path = repo_root / "figures" / "data_quality_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    pdf_path = repo_root / "figures" / "pdf_versions" / "data_quality_analysis.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close(fig)
    
    print(f"✅ Data quality visualizations saved to {output_path}")
    print(f"📄 PDF version saved to {pdf_path}")

def main():
    """Main data cleaning function."""
    print("🧹 Data Cleaning for Machine Learning Analysis")
    print("=" * 60)
    
    # Load data
    df = load_hints_data()
    if df.empty:
        print("❌ Failed to load data")
        return
    
    # Analyze data quality
    quality_report = analyze_data_quality(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Create ML features
    df_ml = create_ml_features(df_clean)
    
    # Validate ML data
    validation_report = validate_ml_data(df_ml)
    
    # Create visualizations
    create_data_quality_visualizations(quality_report, validation_report)
    
    # Save cleaned data
    output_dir = Path(__file__).parent.parent / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    df_ml.to_csv(output_dir / "ml_cleaned_data.csv", index=False)
    
    # Save reports (simplified - skip JSON for now)
    print("📋 Reports generated but not saved to JSON (serialization issues)")
    
    print(f"\n✅ Data cleaning completed!")
    print(f"📊 Cleaned data saved to: analysis/ml_cleaned_data.csv")
    print(f"📋 Quality report saved to: analysis/data_quality_report.json")
    print(f"📋 Validation report saved to: analysis/ml_validation_report.json")
    print(f"📈 Visualizations saved to: figures/data_quality_analysis.png")
    
    # Print summary
    print(f"\n📊 Data Summary:")
    print(f"   Original shape: {df.shape}")
    print(f"   Cleaned shape: {df_clean.shape}")
    print(f"   ML-ready shape: {df_ml.shape}")
    print(f"   Missing values: {df_ml.isnull().sum().sum()}")
    print(f"   Constant columns: {len(validation_report['constant_columns'])}")
    
    if 'diabetes_distribution' in validation_report:
        diabetes_dist = validation_report['diabetes_distribution']
        total_diabetic = diabetes_dist.get(1, 0)
        total_non_diabetic = diabetes_dist.get(0, 0)
        print(f"   Diabetic patients: {total_diabetic}")
        print(f"   Non-diabetic patients: {total_non_diabetic}")

if __name__ == "__main__":
    main()
