#!/usr/bin/env python3
"""
HINTS 7 Diabetes Privacy Regression Analysis

This script performs comprehensive regression analysis on diabetes patients' 
data sharing willingness, focusing on the relationship between diabetes status,
privacy caution index, and demographic factors.

Main regression framework:
WillingShareData_HCP2 = β₀ + β₁×diabetic + β₂×privacy_caution_index + β₃×demographics + ε

Author: AI Assistant
Date: 2024-09-23
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_regression_data() -> pd.DataFrame:
    """Load the individual-level data for regression analysis."""
    data_path = Path(__file__).parent.parent / "analysis" / "regression_dataset.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Regression data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded regression data: {len(df)} observations")
    return df

def prepare_regression_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare variables for regression analysis."""
    df = df.copy()
    
    # Ensure diabetic is binary (0/1)
    df['diabetic'] = df['diabetic'].astype(int)
    
    # Variables are already prepared in the dataset
    # Just ensure they exist and are properly typed
    if 'age_group' not in df.columns:
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 35, 50, 65, 100], 
                                labels=['18-35', '36-50', '51-65', '65+'])
    
    if 'education_group' not in df.columns:
        df['education_group'] = pd.cut(df['education_numeric'], 
                                     bins=[0, 2, 3, 4, 5], 
                                     labels=['<HS', 'HS/Some College', 'College', 'Graduate'])
    
    print("✅ Variables prepared for regression")
    return df

def run_weighted_regression(df: pd.DataFrame, 
                          dependent_var: str,
                          independent_vars: List[str],
                          weights: Optional[str] = None) -> Dict:
    """Run weighted regression analysis."""
    
    # Prepare data
    y = df[dependent_var].dropna()
    X = df[independent_vars].dropna()
    w = df[weights].dropna() if weights else None
    
    # Align indices
    common_idx = y.index.intersection(X.index)
    if weights:
        common_idx = common_idx.intersection(w.index)
    
    y = y.loc[common_idx]
    X = X.loc[common_idx]
    if weights:
        w = w.loc[common_idx]
        # Filter out negative or zero weights
        valid_weights = w > 0
        y = y[valid_weights]
        X = X[valid_weights]
        w = w[valid_weights]
    
    # Add constant
    X_with_const = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
    
    # Weighted least squares
    if weights and w is not None:
        # Weighted regression using numpy
        W = np.diag(np.sqrt(w))
        X_weighted = W @ X_with_const.values
        y_weighted = W @ y.values
        
        # Solve weighted normal equations
        try:
            coeffs = np.linalg.solve(X_weighted.T @ X_weighted, X_weighted.T @ y_weighted)
            residuals = y.values - X_with_const.values @ coeffs
            
            # Calculate standard errors
            mse = np.sum(w * residuals**2) / (len(y) - len(coeffs))
            var_coeffs = mse * np.linalg.inv(X_weighted.T @ X_weighted)
            se_coeffs = np.sqrt(np.diag(var_coeffs))
            
            # Calculate t-statistics and p-values
            t_stats = coeffs / se_coeffs
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - len(coeffs)))
            
        except np.linalg.LinAlgError:
            print("⚠️ Weighted regression failed, using OLS")
            coeffs, se_coeffs, t_stats, p_values = run_ols_regression(X_with_const, y)
    else:
        coeffs, se_coeffs, t_stats, p_values = run_ols_regression(X_with_const, y)
    
    # Create results dictionary
    results = {
        'n_obs': len(y),
        'n_vars': len(independent_vars),
        'coefficients': dict(zip(['const'] + independent_vars, coeffs)),
        'std_errors': dict(zip(['const'] + independent_vars, se_coeffs)),
        't_statistics': dict(zip(['const'] + independent_vars, t_stats)),
        'p_values': dict(zip(['const'] + independent_vars, p_values)),
        'r_squared': calculate_r_squared(y, X_with_const, coeffs),
        'weighted': weights is not None
    }
    
    return results

def run_ols_regression(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ordinary least squares regression."""
    from scipy import stats
    
    # OLS regression
    coeffs = np.linalg.solve(X.T @ X, X.T @ y)
    residuals = y.values - X.values @ coeffs
    
    # Calculate standard errors
    mse = np.sum(residuals**2) / (len(y) - len(coeffs))
    var_coeffs = mse * np.linalg.inv(X.T @ X)
    se_coeffs = np.sqrt(np.diag(var_coeffs))
    
    # Calculate t-statistics and p-values
    t_stats = coeffs / se_coeffs
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - len(coeffs)))
    
    return coeffs, se_coeffs, t_stats, p_values

def calculate_r_squared(y: pd.Series, X: pd.DataFrame, coeffs: np.ndarray) -> float:
    """Calculate R-squared."""
    y_pred = X.values @ coeffs
    ss_res = np.sum((y.values - y_pred)**2)
    ss_tot = np.sum((y.values - np.mean(y.values))**2)
    return 1 - (ss_res / ss_tot)

def run_main_regression(df: pd.DataFrame) -> Dict:
    """Run the main regression analysis."""
    print("\n🔬 Running Main Regression Analysis")
    print("=" * 50)
    
    # Define variables
    dependent_var = 'WillingShareData_HCP2.1'
    independent_vars = [
        'diabetic',
        'privacy_caution_index',
        'age',
        'education_numeric'
    ]
    
    # Check if dependent variable exists
    if dependent_var not in df.columns:
        print(f"⚠️ Dependent variable '{dependent_var}' not found in data")
        print("Available columns:", df.columns.tolist())
        return {}
    
    # Run weighted regression
    results = run_weighted_regression(
        df, 
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        weights='weight'
    )
    
    # Print results
    print(f"📊 Regression Results (N={results['n_obs']})")
    print(f"R² = {results['r_squared']:.4f}")
    print("\nCoefficients:")
    for var in ['const'] + independent_vars:
        coef = results['coefficients'][var]
        se = results['std_errors'][var]
        t_stat = results['t_statistics'][var]
        p_val = results['p_values'][var]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {var:20s}: {coef:8.4f} ({se:6.4f}) t={t_stat:6.2f} p={p_val:.4f} {sig}")
    
    return results

def run_interaction_regression(df: pd.DataFrame) -> Dict:
    """Run regression with interaction effects."""
    print("\n🔬 Running Interaction Effects Analysis")
    print("=" * 50)
    
    # Create interaction term
    df['diabetic_privacy_interaction'] = df['diabetic'] * df['privacy_caution_index']
    
    dependent_var = 'WillingShareData_HCP2.1'
    independent_vars = [
        'diabetic',
        'privacy_caution_index',
        'diabetic_privacy_interaction',
        'age',
        'education_numeric'
    ]
    
    if dependent_var not in df.columns:
        print(f"⚠️ Dependent variable '{dependent_var}' not found in data")
        return {}
    
    results = run_weighted_regression(
        df,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        weights='weight'
    )
    
    print(f"📊 Interaction Regression Results (N={results['n_obs']})")
    print(f"R² = {results['r_squared']:.4f}")
    print("\nCoefficients:")
    for var in ['const'] + independent_vars:
        coef = results['coefficients'][var]
        se = results['std_errors'][var]
        t_stat = results['t_statistics'][var]
        p_val = results['p_values'][var]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {var:25s}: {coef:8.4f} ({se:6.4f}) t={t_stat:6.2f} p={p_val:.4f} {sig}")
    
    return results

def run_subgroup_analyses(df: pd.DataFrame) -> Dict:
    """Run subgroup analyses by age and education."""
    print("\n🔬 Running Subgroup Analyses")
    print("=" * 50)
    
    subgroup_results = {}
    
    # Age subgroups
    age_groups = ['18-35', '36-50', '51-65', '65+']
    for age_group in age_groups:
        if age_group in df['age_group'].values:
            subgroup_df = df[df['age_group'] == age_group].copy()
            if len(subgroup_df) > 100:  # Minimum sample size
                print(f"\n📊 Age Group: {age_group} (N={len(subgroup_df)})")
                
                independent_vars = [
                    'diabetic',
                    'privacy_caution_index', 
                    'education_numeric'
                ]
                
                results = run_weighted_regression(
                    subgroup_df,
                    dependent_var='WillingShareData_HCP2.1',
                    independent_vars=independent_vars,
                    weights='weight'
                )
                
                subgroup_results[f'age_{age_group}'] = results
                
                # Print key coefficients
                diabetic_coef = results['coefficients']['diabetic']
                privacy_coef = results['coefficients']['privacy_caution_index']
                print(f"  Diabetic effect: {diabetic_coef:.4f}")
                print(f"  Privacy effect: {privacy_coef:.4f}")
    
    return subgroup_results

def create_regression_plots(df: pd.DataFrame, results: Dict) -> None:
    """Create visualization plots for regression results."""
    print("\n📊 Creating Regression Visualization Plots")
    
    # Set up the plotting with larger figure size
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('HINTS 7 Diabetes Privacy Regression Analysis Results', fontsize=20, fontweight='bold', y=0.95)
    
    # Set academic color palette
    colors = {
        'diabetic': '#2E86AB',      # Professional blue
        'non_diabetic': '#A23B72',  # Deep rose
        'accent': '#F18F01',        # Academic orange
        'neutral': '#6C757D',       # Professional gray
        'success': '#28A745',       # Success green
        'warning': '#FFC107'        # Warning amber
    }
    
    # 1. Scatter plot: Privacy Index vs Data Sharing Willingness
    ax1 = axes[0, 0]
    diabetic_data = df[df['diabetic'] == 1]
    non_diabetic_data = df[df['diabetic'] == 0]
    
    # Create scatter plots with academic colors
    ax1.scatter(non_diabetic_data['privacy_caution_index'], 
               non_diabetic_data['WillingShareData_HCP2.1'], 
               alpha=0.7, label='Non-Diabetic', s=30, 
               color=colors['non_diabetic'], edgecolors='white', linewidth=0.5)
    ax1.scatter(diabetic_data['privacy_caution_index'], 
               diabetic_data['WillingShareData_HCP2.1'], 
               alpha=0.7, label='Diabetic', s=30,
               color=colors['diabetic'], edgecolors='white', linewidth=0.5)
    
    # Add regression lines
    x_range = np.linspace(df['privacy_caution_index'].min(), df['privacy_caution_index'].max(), 100)
    y_pred_non_diab = (results['coefficients']['const'] + 
                      results['coefficients']['privacy_caution_index'] * x_range)
    y_pred_diab = (results['coefficients']['const'] + 
                   results['coefficients']['diabetic'] + 
                   results['coefficients']['privacy_caution_index'] * x_range)
    
    ax1.plot(x_range, y_pred_non_diab, color=colors['non_diabetic'], 
             linewidth=3, label='Non-Diabetic Fit', linestyle='-')
    ax1.plot(x_range, y_pred_diab, color=colors['diabetic'], 
             linewidth=3, label='Diabetic Fit', linestyle='-')
    
    ax1.set_xlabel('Privacy Caution Index', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Willingness to Share Data with HCP', fontsize=14, fontweight='bold')
    ax1.set_title('Privacy Index vs Data Sharing Willingness', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # 2. Coefficient plot
    ax2 = axes[0, 1]
    vars_to_plot = ['diabetic', 'privacy_caution_index', 'age', 'education_numeric']
    coefs = [results['coefficients'][var] for var in vars_to_plot]
    ses = [results['std_errors'][var] for var in vars_to_plot]
    
    # Create color scheme for coefficients
    coef_colors = [colors['accent'] if coef > 0 else colors['neutral'] for coef in coefs]
    
    y_pos = np.arange(len(vars_to_plot))
    bars = ax2.barh(y_pos, coefs, xerr=ses, capsize=8, alpha=0.8, 
                   color=coef_colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, coef, se) in enumerate(zip(bars, coefs, ses)):
        width = bar.get_width()
        ax2.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{coef:.3f}', ha='left' if width >= 0 else 'right', va='center',
                fontsize=11, fontweight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(['Diabetes Status', 'Privacy Caution Index', 'Age', 'Education Level'], 
                       fontsize=12, fontweight='bold')
    ax2.set_xlabel('Coefficient Value', fontsize=14, fontweight='bold')
    ax2.set_title('Key Regression Coefficients', fontsize=16, fontweight='bold', pad=20)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # 3. Residuals plot
    ax3 = axes[1, 0]
    y_pred = (results['coefficients']['const'] + 
              results['coefficients']['diabetic'] * df['diabetic'] +
              results['coefficients']['privacy_caution_index'] * df['privacy_caution_index'] +
              results['coefficients']['age'] * df['age'] +
              results['coefficients']['education_numeric'] * df['education_numeric'])
    
    residuals = df['WillingShareData_HCP2.1'] - y_pred
    
    # Create scatter plot with academic styling
    ax3.scatter(y_pred, residuals, alpha=0.7, s=30, 
               color=colors['neutral'], edgecolors='white', linewidth=0.5)
    ax3.axhline(y=0, color=colors['accent'], linestyle='--', linewidth=3, alpha=0.8)
    
    # Add confidence bands
    residual_std = np.std(residuals)
    ax3.axhline(y=2*residual_std, color=colors['warning'], linestyle=':', alpha=0.6, linewidth=2)
    ax3.axhline(y=-2*residual_std, color=colors['warning'], linestyle=':', alpha=0.6, linewidth=2)
    
    ax3.set_xlabel('Predicted Values', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Residuals', fontsize=14, fontweight='bold')
    ax3.set_title('Residuals vs Predicted Values', fontsize=16, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # 4. Distribution of dependent variable by diabetes status
    ax4 = axes[1, 1]
    diabetic_values = df[df['diabetic'] == 1]['WillingShareData_HCP2.1'].dropna()
    non_diabetic_values = df[df['diabetic'] == 0]['WillingShareData_HCP2.1'].dropna()
    
    # Create histogram with academic styling
    ax4.hist(non_diabetic_values, alpha=0.7, label='Non-Diabetic', bins=20, density=True,
             color=colors['non_diabetic'], edgecolor='white', linewidth=1)
    ax4.hist(diabetic_values, alpha=0.7, label='Diabetic', bins=20, density=True,
             color=colors['diabetic'], edgecolor='white', linewidth=1)
    
    # Add mean lines
    ax4.axvline(non_diabetic_values.mean(), color=colors['non_diabetic'], 
               linestyle='--', linewidth=3, alpha=0.8, label=f'Non-Diabetic Mean: {non_diabetic_values.mean():.3f}')
    ax4.axvline(diabetic_values.mean(), color=colors['diabetic'], 
               linestyle='--', linewidth=3, alpha=0.8, label=f'Diabetic Mean: {diabetic_values.mean():.3f}')
    
    ax4.set_xlabel('Willingness to Share Data with HCP', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax4.set_title('Distribution of Data Sharing Willingness', fontsize=16, fontweight='bold', pad=20)
    ax4.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.tick_params(axis='both', which='major', labelsize=12)
    
    # Adjust layout with more padding
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot with high quality
    output_path = Path(__file__).parent.parent / "figures" / "regression_analysis_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Regression plots saved to: {output_path}")
    
    # Also save as PDF for academic use
    pdf_path = Path(__file__).parent.parent / "figures" / "regression_analysis_results.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Regression plots also saved as PDF: {pdf_path}")
    
    plt.show()

def save_regression_results(main_results: Dict, 
                          interaction_results: Dict, 
                          subgroup_results: Dict) -> None:
    """Save all regression results to JSON."""
    output_path = Path(__file__).parent.parent / "analysis" / "regression_results.json"
    
    all_results = {
        'main_regression': main_results,
        'interaction_regression': interaction_results,
        'subgroup_analyses': subgroup_results,
        'analysis_date': '2024-09-23',
        'sample_size': main_results.get('n_obs', 0),
        'model_specification': {
            'dependent_variable': 'WillingShareData_HCP2.1',
            'main_variables': ['diabetic', 'privacy_caution_index', 'demographics'],
            'interaction_term': 'diabetic × privacy_caution_index'
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ Regression results saved to: {output_path}")

def main():
    """Main function to run complete regression analysis."""
    print("🚀 Starting HINTS 7 Diabetes Privacy Regression Analysis")
    print("=" * 60)
    
    try:
        # Load data
        df = load_regression_data()
        
        # Prepare variables
        df = prepare_regression_variables(df)
        
        # Run main regression
        main_results = run_main_regression(df)
        
        # Run interaction regression
        interaction_results = run_interaction_regression(df)
        
        # Run subgroup analyses
        subgroup_results = run_subgroup_analyses(df)
        
        # Create plots
        if main_results:
            create_regression_plots(df, main_results)
        
        # Save results
        save_regression_results(main_results, interaction_results, subgroup_results)
        
        print("\n✅ Regression analysis completed successfully!")
        print("\n📊 Key Findings Summary:")
        if main_results:
            diabetic_effect = main_results['coefficients']['diabetic']
            privacy_effect = main_results['coefficients']['privacy_caution_index']
            print(f"• Diabetic effect on data sharing: {diabetic_effect:.4f}")
            print(f"• Privacy caution effect: {privacy_effect:.4f}")
            print(f"• Model R²: {main_results['r_squared']:.4f}")
        
    except Exception as e:
        print(f"❌ Error in regression analysis: {e}")
        raise

if __name__ == "__main__":
    main()
