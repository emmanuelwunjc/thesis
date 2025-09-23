#!/usr/bin/env python3
"""
Prepare comprehensive regression dataset by merging privacy index data 
with original HINTS 7 variables needed for regression analysis.
"""

import pandas as pd
import numpy as np
import subprocess
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_hints_data():
    """Load the original HINTS 7 data using R."""
    print("📊 Loading HINTS 7 data...")
    
    # Create R script to load and export data
    r_script = """
    # Load the data
    load('data/hints7_public copy.rda')
    df <- get(ls()[1])
    
    # Select variables needed for regression
    regression_vars <- c(
        'HHID',
        'WillingShareData_HCP2',
        'Age',
        'Education', 
        'CENSREG',
        'RUC2003',
        'Weight'
    )
    
    # Check which variables exist
    available_vars <- regression_vars[regression_vars %in% names(df)]
    cat('Available variables:', paste(available_vars, collapse=', '), '\\n')
    
    # Create subset
    df_subset <- df[, available_vars, drop=FALSE]
    
    # Recode WillingShareData_HCP2 to numeric (Yes=1, No=0, others=NA)
    if('WillingShareData_HCP2' %in% names(df_subset)) {
        df_subset$WillingShareData_HCP2_numeric <- ifelse(
            df_subset$WillingShareData_HCP2 == 'Yes', 1,
            ifelse(df_subset$WillingShareData_HCP2 == 'No', 0, NA)
        )
    }
    
    # Write to CSV
    write.csv(df_subset, 'analysis/hints_regression_vars.csv', row.names=FALSE)
    
    cat('Data exported with', nrow(df_subset), 'rows and', ncol(df_subset), 'columns\\n')
    """
    
    # Write and run R script
    with open('temp_load_data.R', 'w') as f:
        f.write(r_script)
    
    try:
        result = subprocess.run(['Rscript', 'temp_load_data.R'], 
                              capture_output=True, text=True, check=True)
        print("✅ HINTS data loaded successfully")
        print(result.stdout)
        
        # Load the exported data
        df = pd.read_csv('analysis/hints_regression_vars.csv')
        print(f"📊 Loaded {len(df)} observations with {len(df.columns)} variables")
        
        return df
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error loading HINTS data: {e}")
        print(f"STDERR: {e.stderr}")
        return None
    finally:
        # Clean up temp file
        if Path('temp_load_data.R').exists():
            Path('temp_load_data.R').unlink()

def merge_with_privacy_index(hints_df, privacy_df):
    """Merge HINTS data with privacy index data."""
    print("🔗 Merging datasets...")
    
    # Merge on HHID
    merged_df = hints_df.merge(privacy_df, on='HHID', how='inner')
    print(f"✅ Merged dataset: {len(merged_df)} observations")
    
    # Check for missing values in key variables
    key_vars = ['WillingShareData_HCP2_numeric', 'diabetic', 'privacy_caution_index']
    for var in key_vars:
        if var in merged_df.columns:
            missing_count = merged_df[var].isna().sum()
            print(f"  {var}: {missing_count} missing values")
    
    return merged_df

def prepare_final_dataset(df):
    """Prepare final dataset for regression."""
    print("🔧 Preparing final regression dataset...")
    
    # Rename variables for consistency
    df = df.copy()
    
    # Rename columns to match regression script expectations
    column_mapping = {
        'WillingShareData_HCP2_numeric': 'WillingShareData_HCP2',
        'Age': 'age',
        'Education': 'education', 
        'CENSREG': 'region',
        'RUC2003': 'urban_rural'
    }
    
    df = df.rename(columns=column_mapping)
    
    # WillingShareData_HCP2_numeric is already numeric from the merge
    # No additional conversion needed
    
    # Create additional variables
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 35, 50, 65, 100], 
                            labels=['18-35', '36-50', '51-65', '65+'])
    
    # Convert education to numeric (1-5 scale)
    education_mapping = {
        'Less than 8 years': 1,
        '8 through 11 years': 2, 
        '12 years or completed high school': 3,
        'Some college': 4,
        'Post high school training other than college (vocational or technical)': 4,
        'College graduate': 5,
        'Postgraduate': 5
    }
    
    df['education_numeric'] = df['education'].map(education_mapping)
    
    # Create education groups
    df['education_group'] = pd.cut(df['education_numeric'], 
                                   bins=[0, 2, 3, 4, 5], 
                                   labels=['<HS', 'HS/Some College', 'College', 'Graduate'])
    
    # Create region dummies
    df['region_northeast'] = (df['region'] == 1).astype(int)
    df['region_midwest'] = (df['region'] == 2).astype(int)
    df['region_south'] = (df['region'] == 3).astype(int)
    df['region_west'] = (df['region'] == 4).astype(int)
    
    # Create urban/rural dummy
    df['urban'] = (df['urban_rural'] == 1).astype(int)
    
    print("✅ Final dataset prepared")
    return df

def main():
    """Main function to prepare regression dataset."""
    print("🚀 Preparing Comprehensive Regression Dataset")
    print("=" * 50)
    
    # Load privacy index data
    privacy_path = Path(__file__).parent.parent / "analysis" / "privacy_caution_index_individual.csv"
    if not privacy_path.exists():
        print(f"❌ Privacy index data not found at {privacy_path}")
        return
    
    privacy_df = pd.read_csv(privacy_path)
    print(f"📊 Loaded privacy index data: {len(privacy_df)} observations")
    
    # Load HINTS data
    hints_df = load_hints_data()
    if hints_df is None:
        print("❌ Failed to load HINTS data")
        return
    
    # Merge datasets
    merged_df = merge_with_privacy_index(hints_df, privacy_df)
    
    # Prepare final dataset
    final_df = prepare_final_dataset(merged_df)
    
    # Save final dataset
    output_path = Path(__file__).parent.parent / "analysis" / "regression_dataset.csv"
    final_df.to_csv(output_path, index=False)
    print(f"✅ Final regression dataset saved to: {output_path}")
    
    # Print summary
    print(f"\n📊 Final Dataset Summary:")
    print(f"  Observations: {len(final_df)}")
    print(f"  Variables: {len(final_df.columns)}")
    print(f"  Diabetic patients: {final_df['diabetic'].sum()}")
    print(f"  Non-diabetic patients: {(final_df['diabetic'] == 0).sum()}")
    
    # Check key variables
    print(f"\n🔍 Key Variables Summary:")
    print(f"  WillingShareData_HCP2: {final_df['WillingShareData_HCP2'].isna().sum()} missing values")
    print(f"  diabetic: {final_df['diabetic'].sum()} diabetic patients")
    print(f"  privacy_caution_index: {final_df['privacy_caution_index'].mean():.3f} (mean)")
    print(f"  age: {final_df['age'].mean():.1f} (mean)")
    print(f"  education_numeric: {final_df['education_numeric'].mean():.1f} (mean)")

if __name__ == "__main__":
    main()
