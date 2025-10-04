#!/usr/bin/env python3
"""
Privacy Index as Dependent Variable Analysis
重新分析：以privacy caution index作为因变量

Author: AI Assistant
Date: 2024-09-23
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from scipy import stats

def load_data() -> pd.DataFrame:
    """Load the privacy index data."""
    print("📊 Loading privacy index data...")
    
    try:
        # Load privacy index data
        privacy_df = pd.read_csv('analysis/privacy_caution_index_individual.csv')
        print(f"✅ Privacy index data loaded: {privacy_df.shape}")
        
        # Load cleaned ML data for additional features
        ml_df = pd.read_csv('analysis/ml_cleaned_data.csv')
        print(f"✅ ML data loaded: {ml_df.shape}")
        
        # Merge datasets
        df = pd.merge(privacy_df, ml_df, on='HHID', how='inner', suffixes=('', '_ml'))
        print(f"✅ Merged data: {df.shape}")
        
        return df
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        return pd.DataFrame()

def run_weighted_regression(df: pd.DataFrame, 
                          dependent_var: str,
                          independent_vars: List[str],
                          weights: Optional[str] = None) -> Dict:
    """Run weighted regression analysis."""
    
    # Prepare data
    y = df[dependent_var].values
    X = df[independent_vars].copy()
    
    # Add constant
    X['const'] = 1
    X = X[['const'] + independent_vars]
    
    # Handle weights
    if weights and weights in df.columns:
        w = df[weights].values
        # Check for negative or zero weights
        w = np.where(w <= 0, 1.0, w)  # Replace non-positive weights with 1
        w = w / w.sum() * len(w)  # Normalize weights
    else:
        w = np.ones(len(y))
    
    # Convert to numpy arrays
    X_matrix = X.values
    y_vector = y
    
    # Weighted least squares
    W = np.diag(w)
    try:
        # (X'WX)^(-1) X'Wy
        XtWX = X_matrix.T @ W @ X_matrix
        XtWy = X_matrix.T @ W @ y_vector
        coeffs = np.linalg.solve(XtWX, XtWy)
        
        # Calculate standard errors
        residuals = y_vector - X_matrix @ coeffs
        mse = np.sum(w * residuals**2) / (len(y) - len(coeffs))
        var_coeffs = mse * np.linalg.inv(XtWX)
        std_errors = np.sqrt(np.diag(var_coeffs))
        
        # Calculate t-statistics and p-values
        t_stats = coeffs / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - len(coeffs)))
        
        # Calculate R-squared
        y_mean = np.average(y_vector, weights=w)
        ss_tot = np.sum(w * (y_vector - y_mean)**2)
        ss_res = np.sum(w * residuals**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Organize results
        results = {
            'coefficients': dict(zip(X.columns, coeffs)),
            'std_errors': dict(zip(X.columns, std_errors)),
            't_statistics': dict(zip(X.columns, t_stats)),
            'p_values': dict(zip(X.columns, p_values)),
            'r_squared': r_squared,
            'n_obs': len(y),
            'mse': mse
        }
        
        return results
        
    except np.linalg.LinAlgError:
        print("⚠️ Singular matrix, using unweighted regression")
        # Fallback to unweighted regression
        coeffs = np.linalg.lstsq(X_matrix, y_vector, rcond=None)[0]
        residuals = y_vector - X_matrix @ coeffs
        mse = np.sum(residuals**2) / (len(y) - len(coeffs))
        
        # Simple standard errors (not weighted)
        var_coeffs = mse * np.linalg.inv(X_matrix.T @ X_matrix)
        std_errors = np.sqrt(np.diag(var_coeffs))
        t_stats = coeffs / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - len(coeffs)))
        
        y_mean = np.mean(y_vector)
        ss_tot = np.sum((y_vector - y_mean)**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        results = {
            'coefficients': dict(zip(X.columns, coeffs)),
            'std_errors': dict(zip(X.columns, std_errors)),
            't_statistics': dict(zip(X.columns, t_stats)),
            'p_values': dict(zip(X.columns, p_values)),
            'r_squared': r_squared,
            'n_obs': len(y),
            'mse': mse
        }
        
        return results

def run_main_privacy_regression(df: pd.DataFrame) -> Dict:
    """Run main regression with privacy index as dependent variable."""
    print("\n🔬 Running Main Privacy Regression Analysis")
    print("=" * 50)
    
    # Define variables
    dependent_var = 'privacy_caution_index'
    independent_vars = [
        'diabetic',
        'age_continuous',
        'education_numeric',
        'region_numeric',
        'urban',
        'has_insurance',
        'received_treatment',
        'stopped_treatment',
        'male',
        'race_numeric'
    ]
    
    # Check if dependent variable exists
    if dependent_var not in df.columns:
        print(f"⚠️ Dependent variable '{dependent_var}' not found in data")
        print("Available columns:", df.columns.tolist())
        return {}
    
    # Filter available independent variables
    available_vars = [var for var in independent_vars if var in df.columns]
    print(f"📊 Available independent variables: {available_vars}")
    
    # Run weighted regression
    results = run_weighted_regression(
        df, 
        dependent_var=dependent_var,
        independent_vars=available_vars,
        weights='weight'
    )
    
    # Print results
    print(f"📊 Privacy Regression Results (N={results['n_obs']})")
    print(f"R² = {results['r_squared']:.4f}")
    print("\nCoefficients:")
    for var in ['const'] + available_vars:
        coef = results['coefficients'][var]
        se = results['std_errors'][var]
        t_stat = results['t_statistics'][var]
        p_val = results['p_values'][var]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {var:20s}: {coef:8.4f} ({se:6.4f}) t={t_stat:6.2f} p={p_val:.4f} {sig}")
    
    return results

def run_interaction_privacy_regression(df: pd.DataFrame) -> Dict:
    """Run regression with interaction effects."""
    print("\n🔬 Running Privacy Regression with Interaction Effects")
    print("=" * 50)
    
    # Create interaction terms
    df_analysis = df.copy()
    
    # Diabetes × Age interaction
    if 'diabetic' in df_analysis.columns and 'age_continuous' in df_analysis.columns:
        df_analysis['diabetic_age'] = df_analysis['diabetic'] * df_analysis['age_continuous']
    
    # Diabetes × Education interaction
    if 'diabetic' in df_analysis.columns and 'education_numeric' in df_analysis.columns:
        df_analysis['diabetic_education'] = df_analysis['diabetic'] * df_analysis['education_numeric']
    
    # Define variables
    dependent_var = 'privacy_caution_index'
    independent_vars = [
        'diabetic',
        'age_continuous',
        'education_numeric',
        'region_numeric',
        'urban',
        'has_insurance',
        'received_treatment',
        'stopped_treatment',
        'male',
        'race_numeric',
        'diabetic_age',
        'diabetic_education'
    ]
    
    # Filter available variables
    available_vars = [var for var in independent_vars if var in df_analysis.columns]
    print(f"📊 Available variables with interactions: {available_vars}")
    
    # Run weighted regression
    results = run_weighted_regression(
        df_analysis, 
        dependent_var=dependent_var,
        independent_vars=available_vars,
        weights='weight'
    )
    
    # Print results
    print(f"📊 Privacy Interaction Regression Results (N={results['n_obs']})")
    print(f"R² = {results['r_squared']:.4f}")
    print("\nCoefficients:")
    for var in ['const'] + available_vars:
        coef = results['coefficients'][var]
        se = results['std_errors'][var]
        t_stat = results['t_statistics'][var]
        p_val = results['p_values'][var]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {var:20s}: {coef:8.4f} ({se:6.4f}) t={t_stat:6.2f} p={p_val:.4f} {sig}")
    
    return results

def run_stratified_privacy_analysis(df: pd.DataFrame) -> Dict:
    """Run stratified analysis by diabetes status."""
    print("\n🔬 Running Stratified Privacy Analysis")
    print("=" * 50)
    
    results = {}
    
    # Define variables
    dependent_var = 'privacy_caution_index'
    independent_vars = [
        'age_continuous',
        'education_numeric',
        'region_numeric',
        'urban',
        'has_insurance',
        'received_treatment',
        'stopped_treatment',
        'male',
        'race_numeric'
    ]
    
    # Filter available variables
    available_vars = [var for var in independent_vars if var in df.columns]
    
    # Diabetic group
    diabetic_df = df[df['diabetic'] == 1].copy()
    if len(diabetic_df) > 50:  # Minimum sample size
        print(f"\n📊 Diabetic Group Analysis (N={len(diabetic_df)})")
        diabetic_results = run_weighted_regression(
            diabetic_df,
            dependent_var=dependent_var,
            independent_vars=available_vars,
            weights='weight'
        )
        results['diabetic_group'] = diabetic_results
        
        print(f"R² = {diabetic_results['r_squared']:.4f}")
        print("Key coefficients:")
        for var in ['age_continuous', 'education_numeric', 'has_insurance']:
            if var in diabetic_results['coefficients']:
                coef = diabetic_results['coefficients'][var]
                p_val = diabetic_results['p_values'][var]
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                print(f"  {var}: {coef:.4f} (p={p_val:.4f}) {sig}")
    
    # Non-diabetic group
    non_diabetic_df = df[df['diabetic'] == 0].copy()
    if len(non_diabetic_df) > 50:  # Minimum sample size
        print(f"\n📊 Non-Diabetic Group Analysis (N={len(non_diabetic_df)})")
        non_diabetic_results = run_weighted_regression(
            non_diabetic_df,
            dependent_var=dependent_var,
            independent_vars=available_vars,
            weights='weight'
        )
        results['non_diabetic_group'] = non_diabetic_results
        
        print(f"R² = {non_diabetic_results['r_squared']:.4f}")
        print("Key coefficients:")
        for var in ['age_continuous', 'education_numeric', 'has_insurance']:
            if var in non_diabetic_results['coefficients']:
                coef = non_diabetic_results['coefficients'][var]
                p_val = non_diabetic_results['p_values'][var]
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                print(f"  {var}: {coef:.4f} (p={p_val:.4f}) {sig}")
    
    return results

def run_ml_privacy_analysis(df: pd.DataFrame) -> Dict:
    """Run ML analysis to find best model for privacy prediction."""
    print("\n🤖 Running ML Privacy Analysis")
    print("=" * 50)
    
    # Prepare features
    feature_columns = [
        'diabetic', 'age_continuous', 'education_numeric',
        'region_numeric', 'urban', 'has_insurance', 'received_treatment', 
        'stopped_treatment', 'male', 'race_numeric'
    ]
    
    # Filter available features
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"📊 Available features: {available_features}")
    
    # Prepare target variable
    target_column = 'privacy_caution_index'
    if target_column not in df.columns:
        print("❌ Target variable not found")
        return {'error': 'Target variable not found'}
    
    # Clean data - handle missing values properly
    analysis_df = df[available_features + [target_column, 'weight']].copy()
    
    # Fill missing values in features
    for col in available_features:
        if analysis_df[col].dtype in ['object', 'category']:
            analysis_df[col] = analysis_df[col].fillna('Unknown')
        else:
            analysis_df[col] = analysis_df[col].fillna(analysis_df[col].median())
    
    # Remove rows with missing target variable
    analysis_df = analysis_df.dropna(subset=[target_column])
    print(f"📊 Clean data shape: {analysis_df.shape}")
    
    if len(analysis_df) < 1000:
        return {'error': f'Insufficient data: {len(analysis_df)} observations'}
    
    # Split data
    X = analysis_df[available_features]
    y = analysis_df[target_column]
    weights = analysis_df['weight'] if 'weight' in analysis_df.columns else None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if weights is not None:
        weights_train, weights_test = train_test_split(weights, test_size=0.2, random_state=42)
    else:
        weights_train, weights_test = None, None
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    # Test each model
    results = {}
    for model_name, model in models.items():
        print(f"\n🔬 Testing {model_name}...")
        
        try:
            # Fit model
            if weights_train is not None:
                model.fit(X_train_scaled, y_train, sample_weight=weights_train)
            else:
                model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train, sample_weight=weights_train)
            test_r2 = r2_score(y_test, y_pred_test, sample_weight=weights_test)
            train_mse = mean_squared_error(y_train, y_pred_train, sample_weight=weights_train)
            test_mse = mean_squared_error(y_test, y_pred_test, sample_weight=weights_test)
            train_mae = mean_absolute_error(y_train, y_pred_train, sample_weight=weights_train)
            test_mae = mean_absolute_error(y_test, y_pred_test, sample_weight=weights_test)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(available_features, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(available_features, np.abs(model.coef_)))
            
            results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'feature_importance': feature_importance
            }
            
            print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
            
            if feature_importance:
                print("  Top 3 features:")
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                for feat, imp in sorted_features[:3]:
                    print(f"    {feat}: {imp:.4f}")
        
        except Exception as e:
            print(f"  Error: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

def create_privacy_visualizations(df: pd.DataFrame, results: Dict) -> None:
    """Create visualizations for privacy analysis."""
    print("\n📊 Creating Privacy Analysis Visualizations...")
    
    # Set up the plotting
    fig, axes = plt.subplots(2, 3, figsize=(20, 16))
    fig.suptitle('Privacy Index as Dependent Variable Analysis', fontsize=20, fontweight='bold', y=0.95)
    
    # Set academic color palette
    colors = {
        'diabetic': '#2E86AB',      # Professional blue
        'non_diabetic': '#A23B72',  # Deep rose
        'accent': '#F18F01',        # Academic orange
        'neutral': '#6C757D',       # Professional gray
        'success': '#28A745',       # Success green
        'warning': '#FFC107'        # Warning amber
    }
    
    # 1. Privacy Index Distribution by Diabetes Status
    ax1 = axes[0, 0]
    diabetic_privacy = df[df['diabetic'] == 1]['privacy_caution_index'].dropna()
    non_diabetic_privacy = df[df['diabetic'] == 0]['privacy_caution_index'].dropna()
    
    ax1.hist(diabetic_privacy, bins=30, alpha=0.7, color=colors['diabetic'], 
             label=f'Diabetic (n={len(diabetic_privacy)})', density=True)
    ax1.hist(non_diabetic_privacy, bins=30, alpha=0.7, color=colors['non_diabetic'], 
             label=f'Non-Diabetic (n={len(non_diabetic_privacy)})', density=True)
    ax1.set_xlabel('Privacy Caution Index', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Privacy Index Distribution by Diabetes Status', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Age vs Privacy Index
    ax2 = axes[0, 1]
    diabetic_data = df[df['diabetic'] == 1]
    non_diabetic_data = df[df['diabetic'] == 0]
    
    ax2.scatter(diabetic_data['age_continuous'], diabetic_data['privacy_caution_index'], 
                alpha=0.6, color=colors['diabetic'], label='Diabetic', s=20)
    ax2.scatter(non_diabetic_data['age_continuous'], non_diabetic_data['privacy_caution_index'], 
                alpha=0.6, color=colors['non_diabetic'], label='Non-Diabetic', s=20)
    ax2.set_xlabel('Age', fontsize=12)
    ax2.set_ylabel('Privacy Caution Index', fontsize=12)
    ax2.set_title('Age vs Privacy Index', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Education vs Privacy Index
    ax3 = axes[0, 2]
    education_groups = df.groupby('education_numeric')['privacy_caution_index'].agg(['mean', 'std', 'count']).reset_index()
    education_groups = education_groups[education_groups['count'] >= 10]  # Minimum sample size
    
    ax3.errorbar(education_groups['education_numeric'], education_groups['mean'], 
                 yerr=education_groups['std'], fmt='o-', color=colors['accent'], 
                 capsize=5, capthick=2, linewidth=2, markersize=8)
    ax3.set_xlabel('Education Level', fontsize=12)
    ax3.set_ylabel('Mean Privacy Caution Index', fontsize=12)
    ax3.set_title('Education vs Privacy Index', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Model Performance Comparison
    ax4 = axes[1, 0]
    if 'ml_results' in results:
        model_names = list(results['ml_results'].keys())
        test_r2_scores = [results['ml_results'][model].get('test_r2', 0) for model in model_names]
        
        bars = ax4.bar(model_names, test_r2_scores, color=[colors['diabetic'], colors['non_diabetic'], 
                                                           colors['accent'], colors['success'], colors['warning']])
        ax4.set_ylabel('Test R²', fontsize=12)
        ax4.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, test_r2_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Feature Importance (if available)
    ax5 = axes[1, 1]
    if 'ml_results' in results and 'Random Forest' in results['ml_results']:
        rf_results = results['ml_results']['Random Forest']
        if 'feature_importance' in rf_results and rf_results['feature_importance']:
            features = list(rf_results['feature_importance'].keys())
            importance = list(rf_results['feature_importance'].values())
            
            # Sort by importance
            sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
            features, importance = zip(*sorted_data)
            
            bars = ax5.barh(features, importance, color=colors['accent'])
            ax5.set_xlabel('Feature Importance', fontsize=12)
            ax5.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
    
    # 6. Diabetes Effect Summary
    ax6 = axes[1, 2]
    if 'main_results' in results:
        main_results = results['main_results']
        if 'diabetic' in main_results['coefficients']:
            diabetic_coef = main_results['coefficients']['diabetic']
            diabetic_p = main_results['p_values']['diabetic']
            
            # Create a simple bar chart showing diabetes effect
            effect_size = abs(diabetic_coef)
            significance = "***" if diabetic_p < 0.01 else "**" if diabetic_p < 0.05 else "*" if diabetic_p < 0.1 else "ns"
            
            bars = ax6.bar(['Diabetes Effect'], [effect_size], 
                           color=colors['success'] if diabetic_p < 0.05 else colors['warning'])
            ax6.set_ylabel('Effect Size', fontsize=12)
            ax6.set_title(f'Diabetes Effect on Privacy\n(p={diabetic_p:.4f} {significance})', 
                         fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # Add value label
            ax6.text(0, effect_size + 0.01, f'{diabetic_coef:.4f}', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plots
    output_path = Path(__file__).parent.parent / "figures" / "privacy_dependent_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    pdf_path = Path(__file__).parent.parent / "figures" / "privacy_dependent_analysis.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"✅ Privacy analysis visualizations saved to {output_path}")
    print(f"📄 PDF version saved to {pdf_path}")

def save_privacy_results(all_results: Dict) -> None:
    """Save all privacy analysis results."""
    output_path = Path(__file__).parent.parent / "analysis" / "privacy_dependent_results.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert results
    converted_results = convert_numpy_types(all_results)
    
    # Add metadata
    converted_results['analysis_date'] = '2024-09-23'
    converted_results['dependent_variable'] = 'privacy_caution_index'
    converted_results['analysis_type'] = 'privacy_as_dependent'
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Privacy analysis results saved to: {output_path}")

def main():
    """Main analysis function."""
    print("🔬 Privacy Index as Dependent Variable Analysis")
    print("=" * 60)
    
    # Load data
    df = load_data()
    if df.empty:
        print("❌ Failed to load data")
        return
    
    print(f"📊 Data loaded: {df.shape}")
    print(f"📊 Columns: {df.columns.tolist()}")
    
    # Check privacy index distribution
    privacy_stats = df['privacy_caution_index'].describe()
    print(f"\n📊 Privacy Index Statistics:")
    print(f"  Mean: {privacy_stats['mean']:.4f}")
    print(f"  Std: {privacy_stats['std']:.4f}")
    print(f"  Min: {privacy_stats['min']:.4f}")
    print(f"  Max: {privacy_stats['max']:.4f}")
    
    # Check diabetes distribution
    diabetes_counts = df['diabetic'].value_counts()
    print(f"\n📊 Diabetes Distribution:")
    print(f"  Non-Diabetic: {diabetes_counts[0]} ({diabetes_counts[0]/len(df)*100:.1f}%)")
    print(f"  Diabetic: {diabetes_counts[1]} ({diabetes_counts[1]/len(df)*100:.1f}%)")
    
    # Run analyses
    all_results = {}
    
    # 1. Main regression
    main_results = run_main_privacy_regression(df)
    all_results['main_results'] = main_results
    
    # 2. Interaction regression
    interaction_results = run_interaction_privacy_regression(df)
    all_results['interaction_results'] = interaction_results
    
    # 3. Stratified analysis
    stratified_results = run_stratified_privacy_analysis(df)
    all_results['stratified_results'] = stratified_results
    
    # 4. ML analysis
    ml_results = run_ml_privacy_analysis(df)
    all_results['ml_results'] = ml_results
    
    # 5. Create visualizations
    create_privacy_visualizations(df, all_results)
    
    # 6. Save results
    save_privacy_results(all_results)
    
    print("\n🎉 Privacy Analysis Complete!")
    print("=" * 60)
    
    # Summary
    if main_results and 'diabetic' in main_results['coefficients']:
        diabetic_coef = main_results['coefficients']['diabetic']
        diabetic_p = main_results['p_values']['diabetic']
        r_squared = main_results['r_squared']
        
        print(f"\n📊 Key Findings:")
        print(f"  Diabetes Effect on Privacy: {diabetic_coef:.4f} (p={diabetic_p:.4f})")
        print(f"  Model R²: {r_squared:.4f}")
        print(f"  Sample Size: {main_results['n_obs']}")
        
        if diabetic_p < 0.05:
            print(f"  ✅ Diabetes significantly affects privacy concerns!")
        else:
            print(f"  ⚠️ Diabetes effect on privacy is not statistically significant")
    
    # Best ML model
    if ml_results:
        best_model = max(ml_results.items(), key=lambda x: x[1].get('test_r2', -999))
        print(f"\n🤖 Best ML Model: {best_model[0]}")
        print(f"  Test R²: {best_model[1]['test_r2']:.4f}")
        print(f"  Test MSE: {best_model[1]['test_mse']:.4f}")

if __name__ == "__main__":
    main()
