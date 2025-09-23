#!/usr/bin/env python3
"""
Comprehensive Regression Analysis - Multiple Model Specifications

This script implements 6 different regression approaches to highlight 
the importance of diabetes in privacy and data sharing behaviors.

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

# Set academic style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the regression dataset."""
    data_path = Path(__file__).parent.parent / "analysis" / "regression_dataset.csv"
    df = pd.read_csv(data_path)
    print(f"✅ Loaded data: {len(df)} observations")
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
        try:
            W = np.diag(np.sqrt(w))
            X_weighted = W @ X_with_const.values
            y_weighted = W @ y.values
            
            coeffs = np.linalg.solve(X_weighted.T @ X_weighted, X_weighted.T @ y_weighted)
            residuals = y.values - X_with_const.values @ coeffs
            
            mse = np.sum(w * residuals**2) / (len(y) - len(coeffs))
            var_coeffs = mse * np.linalg.inv(X_weighted.T @ X_weighted)
            se_coeffs = np.sqrt(np.diag(var_coeffs))
            
            t_stats = coeffs / se_coeffs
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - len(coeffs)))
            
        except np.linalg.LinAlgError:
            print("⚠️ Weighted regression failed, using OLS")
            coeffs, se_coeffs, t_stats, p_values = run_ols_regression(X_with_const, y)
    else:
        coeffs, se_coeffs, t_stats, p_values = run_ols_regression(X_with_const, y)
    
    # Calculate R-squared
    y_pred = X_with_const.values @ coeffs
    ss_res = np.sum((y.values - y_pred)**2)
    ss_tot = np.sum((y.values - np.mean(y.values))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Create results dictionary
    results = {
        'n_obs': len(y),
        'n_vars': len(independent_vars),
        'coefficients': dict(zip(['const'] + independent_vars, coeffs)),
        'std_errors': dict(zip(['const'] + independent_vars, se_coeffs)),
        't_statistics': dict(zip(['const'] + independent_vars, t_stats)),
        'p_values': dict(zip(['const'] + independent_vars, p_values)),
        'r_squared': r_squared,
        'weighted': weights is not None
    }
    
    return results

def run_ols_regression(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ordinary least squares regression."""
    coeffs = np.linalg.solve(X.T @ X, X.T @ y)
    residuals = y.values - X.values @ coeffs
    
    mse = np.sum(residuals**2) / (len(y) - len(coeffs))
    var_coeffs = mse * np.linalg.inv(X.T @ X)
    se_coeffs = np.sqrt(np.diag(var_coeffs))
    
    t_stats = coeffs / se_coeffs
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - len(coeffs)))
    
    return coeffs, se_coeffs, t_stats, p_values

def model_1_moderator_analysis(df: pd.DataFrame) -> Dict:
    """Model 1: Diabetes as moderator of privacy concerns."""
    print("\n🔬 Model 1: Diabetes as Moderator of Privacy Concerns")
    print("=" * 60)
    print("Model: Privacy_Caution_Index = β₀ + β₁×diabetic + β₂×age + β₃×education + β₄×diabetic×age + ε")
    
    # Create interaction term
    df['diabetic_age_interaction'] = df['diabetic'] * df['age']
    
    independent_vars = ['diabetic', 'age', 'education_numeric', 'diabetic_age_interaction']
    
    results = run_weighted_regression(
        df,
        dependent_var='privacy_caution_index',
        independent_vars=independent_vars,
        weights='weight'
    )
    
    print(f"📊 Results (N={results['n_obs']})")
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

def model_2_stratified_analysis(df: pd.DataFrame) -> Dict:
    """Model 2: Stratified analysis by diabetes status."""
    print("\n🔬 Model 2: Stratified Analysis by Diabetes Status")
    print("=" * 60)
    print("Separate regressions for diabetic and non-diabetic groups")
    
    results = {}
    
    # Diabetic group
    diabetic_df = df[df['diabetic'] == 1].copy()
    if len(diabetic_df) > 100:
        print(f"\n📊 Diabetic Group (N={len(diabetic_df)})")
        independent_vars = ['privacy_caution_index', 'age', 'education_numeric']
        
        diabetic_results = run_weighted_regression(
            diabetic_df,
            dependent_var='WillingShareData_HCP2.1',
            independent_vars=independent_vars,
            weights='weight'
        )
        
        results['diabetic_group'] = diabetic_results
        
        print(f"R² = {diabetic_results['r_squared']:.4f}")
        print("Coefficients:")
        for var in ['const'] + independent_vars:
            coef = diabetic_results['coefficients'][var]
            se = diabetic_results['std_errors'][var]
            t_stat = diabetic_results['t_statistics'][var]
            p_val = diabetic_results['p_values'][var]
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"  {var:25s}: {coef:8.4f} ({se:6.4f}) t={t_stat:6.2f} p={p_val:.4f} {sig}")
    
    # Non-diabetic group
    non_diabetic_df = df[df['diabetic'] == 0].copy()
    if len(non_diabetic_df) > 100:
        print(f"\n📊 Non-Diabetic Group (N={len(non_diabetic_df)})")
        independent_vars = ['privacy_caution_index', 'age', 'education_numeric']
        
        non_diabetic_results = run_weighted_regression(
            non_diabetic_df,
            dependent_var='WillingShareData_HCP2.1',
            independent_vars=independent_vars,
            weights='weight'
        )
        
        results['non_diabetic_group'] = non_diabetic_results
        
        print(f"R² = {non_diabetic_results['r_squared']:.4f}")
        print("Coefficients:")
        for var in ['const'] + independent_vars:
            coef = non_diabetic_results['coefficients'][var]
            se = non_diabetic_results['std_errors'][var]
            t_stat = non_diabetic_results['t_statistics'][var]
            p_val = non_diabetic_results['p_values'][var]
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"  {var:25s}: {coef:8.4f} ({se:6.4f}) t={t_stat:6.2f} p={p_val:.4f} {sig}")
    
    return results

def model_3_diabetes_centered_interactions(df: pd.DataFrame) -> Dict:
    """Model 3: Diabetes-centered interaction model."""
    print("\n🔬 Model 3: Diabetes-Centered Interaction Model")
    print("=" * 60)
    print("Model: WillingShareData_HCP2 = β₀ + β₁×diabetic + β₂×privacy_index + β₃×diabetic×privacy_index + β₄×diabetic×age + β₅×diabetic×education + ε")
    
    # Create interaction terms
    df['diabetic_privacy_interaction'] = df['diabetic'] * df['privacy_caution_index']
    df['diabetic_age_interaction'] = df['diabetic'] * df['age']
    df['diabetic_education_interaction'] = df['diabetic'] * df['education_numeric']
    
    independent_vars = [
        'diabetic',
        'privacy_caution_index', 
        'diabetic_privacy_interaction',
        'diabetic_age_interaction',
        'diabetic_education_interaction'
    ]
    
    results = run_weighted_regression(
        df,
        dependent_var='WillingShareData_HCP2.1',
        independent_vars=independent_vars,
        weights='weight'
    )
    
    print(f"📊 Results (N={results['n_obs']})")
    print(f"R² = {results['r_squared']:.4f}")
    print("\nCoefficients:")
    for var in ['const'] + independent_vars:
        coef = results['coefficients'][var]
        se = results['std_errors'][var]
        t_stat = results['t_statistics'][var]
        p_val = results['p_values'][var]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {var:30s}: {coef:8.4f} ({se:6.4f}) t={t_stat:6.2f} p={p_val:.4f} {sig}")
    
    return results

def model_4_mediation_analysis(df: pd.DataFrame) -> Dict:
    """Model 4: Mediation analysis."""
    print("\n🔬 Model 4: Mediation Analysis")
    print("=" * 60)
    print("Step 1: Diabetes → Privacy Index")
    print("Step 2: Diabetes + Privacy Index → Data Sharing")
    
    results = {}
    
    # Step 1: Diabetes → Privacy Index
    print("\n📊 Step 1: Diabetes → Privacy Index")
    independent_vars = ['diabetic', 'age', 'education_numeric']
    
    step1_results = run_weighted_regression(
        df,
        dependent_var='privacy_caution_index',
        independent_vars=independent_vars,
        weights='weight'
    )
    
    results['step1'] = step1_results
    
    print(f"R² = {step1_results['r_squared']:.4f}")
    print("Coefficients:")
    for var in ['const'] + independent_vars:
        coef = step1_results['coefficients'][var]
        se = step1_results['std_errors'][var]
        t_stat = step1_results['t_statistics'][var]
        p_val = step1_results['p_values'][var]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {var:25s}: {coef:8.4f} ({se:6.4f}) t={t_stat:6.2f} p={p_val:.4f} {sig}")
    
    # Step 2: Diabetes + Privacy Index → Data Sharing
    print("\n📊 Step 2: Diabetes + Privacy Index → Data Sharing")
    independent_vars = ['diabetic', 'privacy_caution_index', 'age', 'education_numeric']
    
    step2_results = run_weighted_regression(
        df,
        dependent_var='WillingShareData_HCP2.1',
        independent_vars=independent_vars,
        weights='weight'
    )
    
    results['step2'] = step2_results
    
    print(f"R² = {step2_results['r_squared']:.4f}")
    print("Coefficients:")
    for var in ['const'] + independent_vars:
        coef = step2_results['coefficients'][var]
        se = step2_results['std_errors'][var]
        t_stat = step2_results['t_statistics'][var]
        p_val = step2_results['p_values'][var]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {var:25s}: {coef:8.4f} ({se:6.4f}) t={t_stat:6.2f} p={p_val:.4f} {sig}")
    
    # Calculate mediation effect
    diabetes_to_privacy = step1_results['coefficients']['diabetic']
    privacy_to_sharing = step2_results['coefficients']['privacy_caution_index']
    mediation_effect = diabetes_to_privacy * privacy_to_sharing
    
    print(f"\n🔍 Mediation Effect:")
    print(f"Diabetes → Privacy: {diabetes_to_privacy:.4f}")
    print(f"Privacy → Sharing: {privacy_to_sharing:.4f}")
    print(f"Indirect Effect: {mediation_effect:.4f}")
    
    results['mediation_effect'] = mediation_effect
    
    return results

def model_5_multiple_outcomes(df: pd.DataFrame) -> Dict:
    """Model 5: Multiple outcomes analysis."""
    print("\n🔬 Model 5: Multiple Outcomes Analysis")
    print("=" * 60)
    print("Diabetes effects on multiple privacy-related outcomes")
    
    results = {}
    
    # Create additional outcome variables
    # Device usage index (simplified)
    device_vars = ['UseDevice_Computer', 'UseDevice_SmPhone', 'UseDevice_Tablet', 'UseDevice_SmWatch']
    available_device_vars = [var for var in device_vars if var in df.columns]
    
    if available_device_vars:
        # Create device usage index
        device_data = df[available_device_vars].copy()
        # Convert to numeric (assuming 0/1 or similar)
        for var in available_device_vars:
            device_data[var] = pd.to_numeric(device_data[var], errors='coerce')
        df['device_usage_index'] = device_data.mean(axis=1)
    
    # Define outcomes
    outcomes = {
        'privacy_caution_index': 'Privacy Caution Index',
        'WillingShareData_HCP2.1': 'Data Sharing Willingness'
    }
    
    if 'device_usage_index' in df.columns:
        outcomes['device_usage_index'] = 'Device Usage Index'
    
    # Run regressions for each outcome
    for outcome_var, outcome_name in outcomes.items():
        print(f"\n📊 Outcome: {outcome_name}")
        independent_vars = ['diabetic', 'age', 'education_numeric']
        
        outcome_results = run_weighted_regression(
            df,
            dependent_var=outcome_var,
            independent_vars=independent_vars,
            weights='weight'
        )
        
        results[outcome_var] = outcome_results
        
        print(f"R² = {outcome_results['r_squared']:.4f}")
        print("Coefficients:")
        for var in ['const'] + independent_vars:
            coef = outcome_results['coefficients'][var]
            se = outcome_results['std_errors'][var]
            t_stat = outcome_results['t_statistics'][var]
            p_val = outcome_results['p_values'][var]
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"  {var:25s}: {coef:8.4f} ({se:6.4f}) t={t_stat:6.2f} p={p_val:.4f} {sig}")
    
    return results

def model_6_diabetes_severity(df: pd.DataFrame) -> Dict:
    """Model 6: Diabetes severity analysis."""
    print("\n🔬 Model 6: Diabetes Severity Analysis")
    print("=" * 60)
    print("Model: WillingShareData_HCP2 = β₀ + β₁×diabetes_severity + β₂×privacy_index + β₃×age + β₄×education + ε")
    
    # Create diabetes severity variable (simplified approach)
    # In real analysis, this would be based on clinical indicators
    df['diabetes_severity'] = df['diabetic'] * (1 + df['age'] / 100)  # Simple severity proxy
    
    independent_vars = ['diabetes_severity', 'privacy_caution_index', 'age', 'education_numeric']
    
    results = run_weighted_regression(
        df,
        dependent_var='WillingShareData_HCP2.1',
        independent_vars=independent_vars,
        weights='weight'
    )
    
    print(f"📊 Results (N={results['n_obs']})")
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

def create_comparison_plot(all_results: Dict) -> None:
    """Create comparison plot of diabetes effects across models."""
    print("\n📊 Creating Model Comparison Visualization")
    
    # Extract diabetes coefficients from different models
    diabetes_effects = {}
    
    # Model 1: Moderator analysis
    if 'model_1' in all_results:
        diabetes_effects['Model 1\n(Moderator)'] = all_results['model_1']['coefficients']['diabetic']
    
    # Model 2: Stratified analysis (average of both groups)
    if 'model_2' in all_results:
        if 'diabetic_group' in all_results['model_2'] and 'non_diabetic_group' in all_results['model_2']:
            # Calculate difference in privacy effects
            diab_privacy = all_results['model_2']['diabetic_group']['coefficients']['privacy_caution_index']
            non_diab_privacy = all_results['model_2']['non_diabetic_group']['coefficients']['privacy_caution_index']
            diabetes_effects['Model 2\n(Stratified)'] = diab_privacy - non_diab_privacy
    
    # Model 3: Diabetes-centered interactions
    if 'model_3' in all_results:
        diabetes_effects['Model 3\n(Centered)'] = all_results['model_3']['coefficients']['diabetic']
    
    # Model 4: Mediation analysis
    if 'model_4' in all_results:
        diabetes_effects['Model 4\n(Mediation)'] = all_results['model_4']['step2']['coefficients']['diabetic']
    
    # Model 5: Multiple outcomes (privacy index)
    if 'model_5' in all_results and 'privacy_caution_index' in all_results['model_5']:
        diabetes_effects['Model 5\n(Multiple)'] = all_results['model_5']['privacy_caution_index']['coefficients']['diabetic']
    
    # Model 6: Diabetes severity
    if 'model_6' in all_results:
        diabetes_effects['Model 6\n(Severity)'] = all_results['model_6']['coefficients']['diabetes_severity']
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    models = list(diabetes_effects.keys())
    effects = list(diabetes_effects.values())
    
    colors = ['#2E86AB' if effect > 0 else '#A23B72' for effect in effects]
    
    bars = ax.bar(models, effects, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, effect in zip(bars, effects):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{effect:.4f}', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Diabetes Effect Size', fontsize=14, fontweight='bold')
    ax.set_title('Diabetes Effects Across Different Model Specifications', fontsize=16, fontweight='bold', pad=20)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2E86AB', alpha=0.8, label='Positive Effect'),
                      Patch(facecolor='#A23B72', alpha=0.8, label='Negative Effect')]
    ax.legend(handles=legend_elements, fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent.parent / "figures" / "diabetes_effects_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Comparison plot saved to: {output_path}")
    
    plt.show()

def save_all_results(all_results: Dict) -> None:
    """Save all results to JSON."""
    output_path = Path(__file__).parent.parent / "analysis" / "comprehensive_regression_results.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ All results saved to: {output_path}")

def main():
    """Run all regression models."""
    print("🚀 Comprehensive Regression Analysis - Multiple Model Specifications")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    all_results = {}
    
    # Run all models
    try:
        all_results['model_1'] = model_1_moderator_analysis(df)
        all_results['model_2'] = model_2_stratified_analysis(df)
        all_results['model_3'] = model_3_diabetes_centered_interactions(df)
        all_results['model_4'] = model_4_mediation_analysis(df)
        all_results['model_5'] = model_5_multiple_outcomes(df)
        all_results['model_6'] = model_6_diabetes_severity(df)
        
        # Create comparison plot
        create_comparison_plot(all_results)
        
        # Save results
        save_all_results(all_results)
        
        print("\n✅ All regression models completed successfully!")
        print("\n🎯 Summary:")
        print("• Model 1: Diabetes as moderator of privacy concerns")
        print("• Model 2: Stratified analysis by diabetes status")
        print("• Model 3: Diabetes-centered interaction model")
        print("• Model 4: Mediation analysis")
        print("• Model 5: Multiple outcomes analysis")
        print("• Model 6: Diabetes severity analysis")
        
    except Exception as e:
        print(f"❌ Error in comprehensive analysis: {e}")
        raise

if __name__ == "__main__":
    main()
