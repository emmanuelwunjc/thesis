#!/usr/bin/env python3
"""
Comprehensive Privacy Analysis - Finding Best Model
以privacy index作为因变量，寻找最佳模型和变量组合

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
    """Load and merge data."""
    print("📊 Loading data...")
    
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

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data for analysis."""
    print("🧹 Cleaning data...")
    
    # Define all possible features
    all_features = [
        'diabetic', 'age_continuous', 'education_numeric', 'region_numeric', 
        'urban', 'has_insurance', 'received_treatment', 'stopped_treatment', 
        'male', 'race_numeric'
    ]
    
    # Filter available features
    available_features = [col for col in all_features if col in df.columns]
    print(f"📊 Available features: {available_features}")
    
    # Prepare analysis dataset
    analysis_df = df[available_features + ['privacy_caution_index']].copy()
    
    # Handle missing values
    for col in available_features:
        if analysis_df[col].dtype in ['object', 'category']:
            analysis_df[col] = analysis_df[col].fillna('Unknown')
        else:
            analysis_df[col] = analysis_df[col].fillna(analysis_df[col].median())
    
    # Remove rows with missing target variable
    analysis_df = analysis_df.dropna(subset=['privacy_caution_index'])
    
    print(f"✅ Clean data shape: {analysis_df.shape}")
    return analysis_df, available_features

def run_simple_regression(df: pd.DataFrame, features: List[str]) -> Dict:
    """Run simple linear regression."""
    print("\n🔬 Running Simple Linear Regression")
    print("=" * 50)
    
    X = df[features]
    y = df['privacy_caution_index']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Calculate p-values for coefficients
    n = len(y_train)
    p = len(features) + 1  # +1 for intercept
    residuals = y_train - y_pred_train
    mse = np.sum(residuals**2) / (n - p)
    
    # Standard errors
    X_with_const = np.column_stack([np.ones(len(X_train)), X_train])
    try:
        var_coeffs = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        std_errors = np.sqrt(np.diag(var_coeffs))
        t_stats = np.concatenate([[lr.intercept_], lr.coef_]) / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p))
    except:
        std_errors = np.full(len(features) + 1, np.nan)
        t_stats = np.full(len(features) + 1, np.nan)
        p_values = np.full(len(features) + 1, np.nan)
    
    # Organize results
    coefficients = {'intercept': lr.intercept_}
    coefficients.update(dict(zip(features, lr.coef_)))
    
    std_errors_dict = {'intercept': std_errors[0]}
    std_errors_dict.update(dict(zip(features, std_errors[1:])))
    
    t_stats_dict = {'intercept': t_stats[0]}
    t_stats_dict.update(dict(zip(features, t_stats[1:])))
    
    p_values_dict = {'intercept': p_values[0]}
    p_values_dict.update(dict(zip(features, p_values[1:])))
    
    results = {
        'model_type': 'Linear Regression',
        'features': features,
        'coefficients': coefficients,
        'std_errors': std_errors_dict,
        't_statistics': t_stats_dict,
        'p_values': p_values_dict,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'n_obs': len(df)
    }
    
    # Print results
    print(f"📊 Linear Regression Results (N={len(df)})")
    print(f"Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")
    print(f"Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")
    print("\nCoefficients:")
    for var in ['intercept'] + features:
        coef = coefficients[var]
        se = std_errors_dict[var]
        t_stat = t_stats_dict[var]
        p_val = p_values_dict[var]
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        print(f"  {var:20s}: {coef:8.4f} ({se:6.4f}) t={t_stat:6.2f} p={p_val:.4f} {sig}")
    
    return results

def run_ml_model_selection(df: pd.DataFrame, features: List[str]) -> Dict:
    """Run comprehensive ML model selection."""
    print("\n🤖 Running ML Model Selection")
    print("=" * 50)
    
    X = df[features]
    y = df['privacy_caution_index']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🔬 Testing {model_name}...")
        
        try:
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(features, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(features, np.abs(model.coef_)))
            
            results[model_name] = {
                'model_type': model_name,
                'features': features,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'feature_importance': feature_importance,
                'n_obs': len(df)
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

def find_best_feature_combination(df: pd.DataFrame, features: List[str]) -> Dict:
    """Find best feature combination using exhaustive search."""
    print("\n🔍 Finding Best Feature Combination")
    print("=" * 50)
    
    # Generate feature combinations (3 to 8 features)
    combinations_list = []
    for r in range(3, min(9, len(features) + 1)):
        for combo in combinations(features, r):
            combinations_list.append(list(combo))
    
    print(f"📊 Testing {len(combinations_list)} feature combinations...")
    
    best_combination = None
    best_r2 = -999
    best_results = None
    
    results_list = []
    
    for i, combo in enumerate(combinations_list):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(combinations_list)} combinations tested")
        
        try:
            # Run regression with this combination
            X = df[combo]
            y = df['privacy_caution_index']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Test both Linear Regression and Random Forest
            for model_name, model in [('Linear Regression', LinearRegression()), 
                                    ('Random Forest', RandomForestRegressor(n_estimators=50, random_state=42))]:
                
                model.fit(X_train, y_train)
                y_pred_test = model.predict(X_test)
                test_r2 = r2_score(y_test, y_pred_test)
                
                result = {
                    'features': combo,
                    'model': model_name,
                    'test_r2': test_r2,
                    'n_features': len(combo)
                }
                results_list.append(result)
                
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_combination = combo
                    best_results = result
        
        except Exception as e:
            continue
    
    # Sort results by R²
    results_list.sort(key=lambda x: x['test_r2'], reverse=True)
    
    print(f"\n🏆 Best Feature Combination:")
    print(f"  Features: {best_combination}")
    print(f"  Model: {best_results['model']}")
    print(f"  Test R²: {best_r2:.4f}")
    
    return {
        'best_combination': best_combination,
        'best_r2': best_r2,
        'best_results': best_results,
        'all_results': results_list[:20]  # Top 20 results
    }

def create_comprehensive_visualizations(df: pd.DataFrame, results: Dict) -> None:
    """Create comprehensive visualizations."""
    print("\n📊 Creating Comprehensive Visualizations...")
    
    # Set up the plotting
    fig, axes = plt.subplots(3, 3, figsize=(24, 20))
    fig.suptitle('Comprehensive Privacy Analysis: Best Model Selection', fontsize=24, fontweight='bold', y=0.95)
    
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
    diabetic_privacy = df[df['diabetic'] == 1]['privacy_caution_index']
    non_diabetic_privacy = df[df['diabetic'] == 0]['privacy_caution_index']
    
    ax1.hist(diabetic_privacy, bins=30, alpha=0.7, color=colors['diabetic'], 
             label=f'Diabetic (n={len(diabetic_privacy)})', density=True)
    ax1.hist(non_diabetic_privacy, bins=30, alpha=0.7, color=colors['non_diabetic'], 
             label=f'Non-Diabetic (n={len(non_diabetic_privacy)})', density=True)
    ax1.set_xlabel('Privacy Caution Index', fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.set_title('Privacy Index Distribution by Diabetes Status', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Age vs Privacy Index
    ax2 = axes[0, 1]
    diabetic_data = df[df['diabetic'] == 1]
    non_diabetic_data = df[df['diabetic'] == 0]
    
    ax2.scatter(diabetic_data['age_continuous'], diabetic_data['privacy_caution_index'], 
                alpha=0.6, color=colors['diabetic'], label='Diabetic', s=20)
    ax2.scatter(non_diabetic_data['age_continuous'], non_diabetic_data['privacy_caution_index'], 
                alpha=0.6, color=colors['non_diabetic'], label='Non-Diabetic', s=20)
    ax2.set_xlabel('Age', fontsize=14)
    ax2.set_ylabel('Privacy Caution Index', fontsize=14)
    ax2.set_title('Age vs Privacy Index', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Education vs Privacy Index
    ax3 = axes[0, 2]
    education_groups = df.groupby('education_numeric')['privacy_caution_index'].agg(['mean', 'std', 'count']).reset_index()
    education_groups = education_groups[education_groups['count'] >= 10]
    
    ax3.errorbar(education_groups['education_numeric'], education_groups['mean'], 
                 yerr=education_groups['std'], fmt='o-', color=colors['accent'], 
                 capsize=5, capthick=2, linewidth=2, markersize=8)
    ax3.set_xlabel('Education Level', fontsize=14)
    ax3.set_ylabel('Mean Privacy Caution Index', fontsize=14)
    ax3.set_title('Education vs Privacy Index', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Model Performance Comparison
    ax4 = axes[1, 0]
    if 'ml_results' in results:
        model_names = list(results['ml_results'].keys())
        test_r2_scores = [results['ml_results'][model].get('test_r2', 0) for model in model_names]
        
        bars = ax4.bar(model_names, test_r2_scores, color=[colors['diabetic'], colors['non_diabetic'], 
                                                           colors['accent'], colors['success'], colors['warning'], colors['neutral']])
        ax4.set_ylabel('Test R²', fontsize=14)
        ax4.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, test_r2_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=12)
    
    # 5. Feature Importance (Random Forest)
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
            ax5.set_xlabel('Feature Importance', fontsize=14)
            ax5.set_title('Random Forest Feature Importance', fontsize=16, fontweight='bold')
            ax5.grid(True, alpha=0.3)
    
    # 6. Diabetes Effect
    ax6 = axes[1, 2]
    if 'simple_regression' in results:
        simple_results = results['simple_regression']
        if 'diabetic' in simple_results['coefficients']:
            diabetic_coef = simple_results['coefficients']['diabetic']
            diabetic_p = simple_results['p_values']['diabetic']
            
            effect_size = abs(diabetic_coef)
            significance = "***" if diabetic_p < 0.01 else "**" if diabetic_p < 0.05 else "*" if diabetic_p < 0.1 else "ns"
            
            bars = ax6.bar(['Diabetes Effect'], [effect_size], 
                           color=colors['success'] if diabetic_p < 0.05 else colors['warning'])
            ax6.set_ylabel('Effect Size', fontsize=14)
            ax6.set_title(f'Diabetes Effect on Privacy\n(p={diabetic_p:.4f} {significance})', 
                         fontsize=16, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # Add value label
            ax6.text(0, effect_size + 0.001, f'{diabetic_coef:.4f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 7. Feature Combination Performance
    ax7 = axes[2, 0]
    if 'feature_combination' in results and 'all_results' in results['feature_combination']:
        combo_results = results['feature_combination']['all_results'][:10]  # Top 10
        combo_names = [f"{len(r['features'])} features" for r in combo_results]
        combo_r2s = [r['test_r2'] for r in combo_results]
        
        bars = ax7.bar(range(len(combo_names)), combo_r2s, color=colors['accent'])
        ax7.set_xlabel('Feature Combination Rank', fontsize=14)
        ax7.set_ylabel('Test R²', fontsize=14)
        ax7.set_title('Top 10 Feature Combinations', fontsize=16, fontweight='bold')
        ax7.set_xticks(range(len(combo_names)))
        ax7.set_xticklabels(combo_names, rotation=45)
        ax7.grid(True, alpha=0.3)
    
    # 8. Correlation Heatmap
    ax8 = axes[2, 1]
    numeric_features = ['diabetic', 'age_continuous', 'education_numeric', 'region_numeric', 
                       'urban', 'has_insurance', 'privacy_caution_index']
    corr_data = df[numeric_features].corr()
    
    im = ax8.imshow(corr_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax8.set_xticks(range(len(numeric_features)))
    ax8.set_yticks(range(len(numeric_features)))
    ax8.set_xticklabels(numeric_features, rotation=45, ha='right')
    ax8.set_yticklabels(numeric_features)
    ax8.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    
    # Add correlation values
    for i in range(len(numeric_features)):
        for j in range(len(numeric_features)):
            text = ax8.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax8)
    
    # 9. Best Model Summary
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    if 'feature_combination' in results:
        best_combo = results['feature_combination']['best_combination']
        best_r2 = results['feature_combination']['best_r2']
        best_model = results['feature_combination']['best_results']['model']
        
        summary_text = f"""
🏆 BEST MODEL SUMMARY

Model: {best_model}
Features: {len(best_combo)} variables
Test R²: {best_r2:.4f}

Key Features:
{chr(10).join([f'• {feat}' for feat in best_combo[:5]])}

Diabetes Effect:
{results['simple_regression']['coefficients']['diabetic']:.4f}
(p={results['simple_regression']['p_values']['diabetic']:.4f})
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['accent'], alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plots
    output_path = Path(__file__).parent.parent / "figures" / "comprehensive_privacy_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    pdf_path = Path(__file__).parent.parent / "figures" / "comprehensive_privacy_analysis.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"✅ Comprehensive analysis visualizations saved to {output_path}")
    print(f"📄 PDF version saved to {pdf_path}")

def save_comprehensive_results(all_results: Dict) -> None:
    """Save all comprehensive analysis results."""
    output_path = Path(__file__).parent.parent / "analysis" / "comprehensive_privacy_results.json"
    
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
    converted_results['analysis_type'] = 'comprehensive_privacy_analysis'
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Comprehensive analysis results saved to: {output_path}")

def main():
    """Main analysis function."""
    print("🔬 Comprehensive Privacy Analysis: Finding Best Model")
    print("=" * 60)
    
    # Load and clean data
    df = load_data()
    if df.empty:
        print("❌ Failed to load data")
        return
    
    df_clean, features = clean_data(df)
    
    print(f"\n📊 Analysis Dataset:")
    print(f"  Shape: {df_clean.shape}")
    print(f"  Features: {features}")
    print(f"  Privacy Index Range: {df_clean['privacy_caution_index'].min():.4f} to {df_clean['privacy_caution_index'].max():.4f}")
    
    # Check diabetes distribution
    diabetes_counts = df_clean['diabetic'].value_counts()
    print(f"\n📊 Diabetes Distribution:")
    print(f"  Non-Diabetic: {diabetes_counts[0]} ({diabetes_counts[0]/len(df_clean)*100:.1f}%)")
    print(f"  Diabetic: {diabetes_counts[1]} ({diabetes_counts[1]/len(df_clean)*100:.1f}%)")
    
    # Run analyses
    all_results = {}
    
    # 1. Simple regression
    simple_results = run_simple_regression(df_clean, features)
    all_results['simple_regression'] = simple_results
    
    # 2. ML model selection
    ml_results = run_ml_model_selection(df_clean, features)
    all_results['ml_results'] = ml_results
    
    # 3. Best feature combination
    feature_combo_results = find_best_feature_combination(df_clean, features)
    all_results['feature_combination'] = feature_combo_results
    
    # 4. Create visualizations
    create_comprehensive_visualizations(df_clean, all_results)
    
    # 5. Save results
    save_comprehensive_results(all_results)
    
    print("\n🎉 Comprehensive Privacy Analysis Complete!")
    print("=" * 60)
    
    # Summary
    print(f"\n📊 KEY FINDINGS:")
    
    # Diabetes effect
    if 'diabetic' in simple_results['coefficients']:
        diabetic_coef = simple_results['coefficients']['diabetic']
        diabetic_p = simple_results['p_values']['diabetic']
        print(f"  Diabetes Effect on Privacy: {diabetic_coef:.4f} (p={diabetic_p:.4f})")
        
        if diabetic_p < 0.05:
            print(f"  ✅ Diabetes significantly affects privacy concerns!")
        else:
            print(f"  ⚠️ Diabetes effect on privacy is not statistically significant")
    
    # Best model
    if 'feature_combination' in all_results:
        best_combo = all_results['feature_combination']['best_combination']
        best_r2 = all_results['feature_combination']['best_r2']
        best_model = all_results['feature_combination']['best_results']['model']
        
        print(f"\n🏆 BEST MODEL:")
        print(f"  Algorithm: {best_model}")
        print(f"  Features: {best_combo}")
        print(f"  Test R²: {best_r2:.4f}")
        print(f"  Number of Features: {len(best_combo)}")
    
    # Top ML model
    if ml_results:
        best_ml_model = max(ml_results.items(), key=lambda x: x[1].get('test_r2', -999))
        print(f"\n🤖 BEST ML MODEL:")
        print(f"  Algorithm: {best_ml_model[0]}")
        print(f"  Test R²: {best_ml_model[1]['test_r2']:.4f}")
        print(f"  Test MSE: {best_ml_model[1]['test_mse']:.4f}")
        
        if best_ml_model[1].get('feature_importance'):
            print(f"  Top Features:")
            sorted_features = sorted(best_ml_model[1]['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_features[:3]:
                print(f"    {feat}: {imp:.4f}")

if __name__ == "__main__":
    main()

