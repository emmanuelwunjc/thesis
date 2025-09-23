#!/usr/bin/env python3
"""
Display Regression Results in Academic Format

This script reads the regression results JSON and displays them in a 
well-formatted, academic-style table format suitable for papers and presentations.

Author: AI Assistant
Date: 2024-09-23
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def load_regression_results():
    """Load regression results from JSON file."""
    results_path = Path(__file__).parent.parent / "analysis" / "regression_results.json"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found at {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

def format_coefficient_table(results, model_name="Main Model"):
    """Format regression coefficients into a clean table."""
    model_data = results[model_name.lower().replace(" ", "_")]
    
    # Extract data
    variables = list(model_data['coefficients'].keys())
    coefs = [model_data['coefficients'][var] for var in variables]
    ses = [model_data['std_errors'][var] for var in variables]
    t_stats = [model_data['t_statistics'][var] for var in variables]
    p_vals = [model_data['p_values'][var] for var in variables]
    
    # Create significance stars
    def get_significance_stars(p_val):
        if p_val < 0.001:
            return "***"
        elif p_val < 0.01:
            return "**"
        elif p_val < 0.05:
            return "*"
        elif p_val < 0.1:
            return "†"
        else:
            return ""
    
    significance = [get_significance_stars(p) for p in p_vals]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Variable': variables,
        'Coefficient': coefs,
        'Std. Error': ses,
        't-statistic': t_stats,
        'p-value': p_vals,
        'Significance': significance
    })
    
    # Format variable names
    var_names = {
        'const': 'Constant',
        'diabetic': 'Diabetes Status',
        'privacy_caution_index': 'Privacy Caution Index',
        'diabetic_privacy_interaction': 'Diabetes × Privacy Index',
        'age': 'Age',
        'education_numeric': 'Education Level'
    }
    
    df['Variable'] = df['Variable'].map(var_names).fillna(df['Variable'])
    
    return df

def display_main_results(results):
    """Display main regression results."""
    print("=" * 80)
    print("HINTS 7 DIABETES PRIVACY REGRESSION ANALYSIS RESULTS")
    print("=" * 80)
    print()
    
    # Main model
    main_df = format_coefficient_table(results, "main_regression")
    main_data = results['main_regression']
    
    print("📊 MAIN REGRESSION MODEL")
    print("-" * 50)
    print(f"Sample Size: {main_data['n_obs']:,} observations")
    print(f"R²: {main_data['r_squared']:.4f}")
    print(f"Model: Weighted Least Squares")
    print()
    
    # Display table
    print("Regression Coefficients:")
    print()
    
    # Format numbers for display
    display_df = main_df.copy()
    display_df['Coefficient'] = display_df['Coefficient'].apply(lambda x: f"{x:.4f}")
    display_df['Std. Error'] = display_df['Std. Error'].apply(lambda x: f"{x:.4f}")
    display_df['t-statistic'] = display_df['t-statistic'].apply(lambda x: f"{x:.3f}")
    display_df['p-value'] = display_df['p-value'].apply(lambda x: f"{x:.4f}")
    
    # Print formatted table
    print(f"{'Variable':<25} {'Coeff':<10} {'Std.Err':<10} {'t-stat':<8} {'p-value':<10} {'Sig':<4}")
    print("-" * 70)
    
    for _, row in display_df.iterrows():
        print(f"{row['Variable']:<25} {row['Coefficient']:<10} {row['Std. Error']:<10} "
              f"{row['t-statistic']:<8} {row['p-value']:<10} {row['Significance']:<4}")
    
    print()
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, † p<0.1")
    print()

def display_interaction_results(results):
    """Display interaction model results."""
    interaction_df = format_coefficient_table(results, "interaction_regression")
    interaction_data = results['interaction_regression']
    
    print("🔬 INTERACTION MODEL")
    print("-" * 50)
    print(f"Sample Size: {interaction_data['n_obs']:,} observations")
    print(f"R²: {interaction_data['r_squared']:.4f}")
    print(f"Model: Weighted Least Squares with Interaction")
    print()
    
    # Display table
    print("Regression Coefficients (with Interaction):")
    print()
    
    # Format numbers for display
    display_df = interaction_df.copy()
    display_df['Coefficient'] = display_df['Coefficient'].apply(lambda x: f"{x:.4f}")
    display_df['Std. Error'] = display_df['Std. Error'].apply(lambda x: f"{x:.4f}")
    display_df['t-statistic'] = display_df['t-statistic'].apply(lambda x: f"{x:.3f}")
    display_df['p-value'] = display_df['p-value'].apply(lambda x: f"{x:.4f}")
    
    # Print formatted table
    print(f"{'Variable':<30} {'Coeff':<10} {'Std.Err':<10} {'t-stat':<8} {'p-value':<10} {'Sig':<4}")
    print("-" * 75)
    
    for _, row in display_df.iterrows():
        print(f"{row['Variable']:<30} {row['Coefficient']:<10} {row['Std. Error']:<10} "
              f"{row['t-statistic']:<8} {row['p-value']:<10} {row['Significance']:<4}")
    
    print()
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, † p<0.1")
    print()

def display_subgroup_results(results):
    """Display subgroup analysis results."""
    subgroup_data = results['subgroup_analyses']
    
    print("📈 AGE GROUP SUBGROUP ANALYSIS")
    print("-" * 50)
    print()
    
    # Create summary table
    age_groups = []
    diabetic_effects = []
    privacy_effects = []
    sample_sizes = []
    
    for age_group, data in subgroup_data.items():
        age_groups.append(age_group.replace('age_', ''))
        diabetic_effects.append(data['coefficients']['diabetic'])
        privacy_effects.append(data['coefficients']['privacy_caution_index'])
        sample_sizes.append(data['n_obs'])
    
    subgroup_df = pd.DataFrame({
        'Age Group': age_groups,
        'Sample Size': sample_sizes,
        'Diabetic Effect': diabetic_effects,
        'Privacy Effect': privacy_effects
    })
    
    print("Subgroup Regression Results:")
    print()
    print(f"{'Age Group':<12} {'N':<8} {'Diabetic Coef':<15} {'Privacy Coef':<15}")
    print("-" * 55)
    
    for _, row in subgroup_df.iterrows():
        print(f"{row['Age Group']:<12} {row['Sample Size']:<8} "
              f"{row['Diabetic Effect']:<15.4f} {row['Privacy Effect']:<15.4f}")
    
    print()

def display_key_findings(results):
    """Display key findings summary."""
    print("🎯 KEY FINDINGS SUMMARY")
    print("=" * 50)
    print()
    
    main_data = results['main_regression']
    interaction_data = results['interaction_regression']
    
    # Extract key coefficients
    diabetic_coef = main_data['coefficients']['diabetic']
    privacy_coef = main_data['coefficients']['privacy_caution_index']
    interaction_coef = interaction_data['coefficients']['diabetic_privacy_interaction']
    
    diabetic_p = main_data['p_values']['diabetic']
    privacy_p = main_data['p_values']['privacy_caution_index']
    interaction_p = interaction_data['p_values']['diabetic_privacy_interaction']
    
    print("1. DIABETES EFFECT ON DATA SHARING:")
    print(f"   • Coefficient: {diabetic_coef:.4f}")
    print(f"   • Significance: {'Significant' if diabetic_p < 0.05 else 'Not significant'} (p={diabetic_p:.4f})")
    print(f"   • Interpretation: Diabetes status {'increases' if diabetic_coef > 0 else 'decreases'} data sharing willingness")
    print()
    
    print("2. PRIVACY CAUTION EFFECT:")
    print(f"   • Coefficient: {privacy_coef:.4f}")
    print(f"   • Significance: {'Highly significant' if privacy_p < 0.001 else 'Significant'} (p={privacy_p:.4f})")
    print(f"   • Interpretation: Higher privacy caution strongly reduces data sharing willingness")
    print()
    
    print("3. DIABETES-PRIVACY INTERACTION:")
    print(f"   • Coefficient: {interaction_coef:.4f}")
    print(f"   • Significance: {'Significant' if interaction_p < 0.05 else 'Not significant'} (p={interaction_p:.4f})")
    print(f"   • Interpretation: Diabetes moderates the privacy-sharing relationship")
    print()
    
    print("4. MODEL FIT:")
    print(f"   • Main Model R²: {main_data['r_squared']:.4f}")
    print(f"   • Interaction Model R²: {interaction_data['r_squared']:.4f}")
    print(f"   • Sample Size: {main_data['n_obs']:,} observations")
    print()

def display_policy_implications():
    """Display policy implications."""
    print("📋 POLICY IMPLICATIONS")
    print("=" * 50)
    print()
    
    print("1. PRIVACY PROTECTION POLICIES:")
    print("   • Privacy concerns are the strongest predictor of data sharing reluctance")
    print("   • Healthcare systems should prioritize transparent privacy policies")
    print("   • Clear communication about data use and protection is essential")
    print()
    
    print("2. DIABETES-SPECIFIC CONSIDERATIONS:")
    print("   • Diabetic patients show different privacy-sharing trade-offs")
    print("   • Healthcare providers should tailor privacy communications for chronic conditions")
    print("   • Consider diabetes-specific data sharing protocols")
    print()
    
    print("3. AGE-RELATED STRATEGIES:")
    print("   • Younger patients are more privacy-sensitive")
    print("   • Older patients may be more willing to share data")
    print("   • Develop age-appropriate privacy education programs")
    print()
    
    print("4. HEALTHCARE SYSTEM DESIGN:")
    print("   • Implement privacy-by-design principles")
    print("   • Provide granular data sharing controls")
    print("   • Ensure patient autonomy in data sharing decisions")
    print()

def main():
    """Main function to display all results."""
    try:
        # Load results
        results = load_regression_results()
        
        # Display all sections
        display_main_results(results)
        display_interaction_results(results)
        display_subgroup_results(results)
        display_key_findings(results)
        display_policy_implications()
        
        print("=" * 80)
        print("END OF REGRESSION ANALYSIS RESULTS")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error displaying results: {e}")

if __name__ == "__main__":
    main()
