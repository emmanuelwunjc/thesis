#!/usr/bin/env python3
"""
Exhaustive Variable Search for Privacy Analysis
测试所有可能的变量组合，寻找能提高R²的变量

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
import time
import sys
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from itertools import combinations
from scipy import stats

def load_all_data() -> pd.DataFrame:
    """Load all HINTS 7 data."""
    print("📊 Loading all HINTS 7 data...")
    
    try:
        # Use R to load data
        import subprocess
        r_script = '''
        load("data/hints7_public copy.rda")
        data_name <- ls()[1]
        data <- get(data_name)
        
        # Convert to data frame and save as CSV
        write.csv(data, "temp_hints_data.csv", row.names=FALSE)
        
        # Print some info
        cat("Data shape:", nrow(data), "x", ncol(data), "\\n")
        cat("Columns with privacy-related keywords:\\n")
        privacy_cols <- colnames(data)[grepl("privacy|share|trust|secure|data|health|doctor|provider", colnames(data), ignore.case=TRUE)]
        writeLines(privacy_cols)
        
        cat("\\nColumns with demographic keywords:\\n")
        demo_cols <- colnames(data)[grepl("age|sex|gender|race|ethnic|education|income|region|urban|rural", colnames(data), ignore.case=TRUE)]
        writeLines(demo_cols)
        
        cat("\\nColumns with health keywords:\\n")
        health_cols <- colnames(data)[grepl("health|medical|condition|diabetes|cancer|heart|blood|pressure", colnames(data), ignore.case=TRUE)]
        writeLines(health_cols)
        
        cat("\\nColumns with technology keywords:\\n")
        tech_cols <- colnames(data)[grepl("computer|internet|phone|mobile|device|app|online", colnames(data), ignore.case=TRUE)]
        writeLines(tech_cols)
        '''
        
        with open('temp_r_script.R', 'w') as f:
            f.write(r_script)
        
        result = subprocess.run(['Rscript', 'temp_r_script.R'], 
                              capture_output=True, text=True, cwd='/Users/wuyiming/code/thesis')
        
        if result.returncode == 0:
            print("✅ R script executed successfully")
            print("R output:")
            print(result.stdout)
            
            # Load the CSV
            df = pd.read_csv('temp_hints_data.csv')
            print(f"✅ Data loaded: {df.shape}")
            return df
        else:
            print("❌ R script failed:")
            print(result.stderr)
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return pd.DataFrame()

def identify_potential_variables(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify all potential variables for analysis."""
    print("\n🔍 Identifying Potential Variables...")
    
    # Privacy-related variables
    privacy_keywords = ['privacy', 'share', 'trust', 'secure', 'data', 'health', 'doctor', 'provider']
    privacy_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in privacy_keywords):
            privacy_cols.append(col)
    
    # Demographic variables
    demo_keywords = ['age', 'sex', 'gender', 'race', 'ethnic', 'education', 'income', 'region', 'urban', 'rural', 'marital', 'birth']
    demo_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in demo_keywords):
            demo_cols.append(col)
    
    # Health variables
    health_keywords = ['health', 'medical', 'condition', 'diabetes', 'cancer', 'heart', 'blood', 'pressure', 'bmi', 'weight', 'height']
    health_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in health_keywords):
            health_cols.append(col)
    
    # Technology variables
    tech_keywords = ['computer', 'internet', 'phone', 'mobile', 'device', 'app', 'online', 'electronic', 'digital']
    tech_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in tech_keywords):
            tech_cols.append(col)
    
    # Lifestyle variables
    lifestyle_keywords = ['smoke', 'alcohol', 'exercise', 'sleep', 'diet', 'fruit', 'vegetable', 'marijuana']
    lifestyle_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in lifestyle_keywords):
            lifestyle_cols.append(col)
    
    # Mental health variables
    mental_keywords = ['depression', 'anxiety', 'hopeless', 'nervous', 'worry', 'isolated', 'phq']
    mental_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in mental_keywords):
            mental_cols.append(col)
    
    # Healthcare access variables
    healthcare_keywords = ['insurance', 'provider', 'telehealth', 'portal', 'access', 'care']
    healthcare_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in healthcare_keywords):
            healthcare_cols.append(col)
    
    variable_groups = {
        'privacy': privacy_cols,
        'demographic': demo_cols,
        'health': health_cols,
        'technology': tech_cols,
        'lifestyle': lifestyle_cols,
        'mental_health': mental_cols,
        'healthcare': healthcare_cols
    }
    
    print(f"📊 Variable Groups Found:")
    for group, cols in variable_groups.items():
        print(f"  {group}: {len(cols)} variables")
        if len(cols) <= 10:  # Show all if few
            for col in cols:
                print(f"    - {col}")
        else:  # Show first 10 if many
            for col in cols[:10]:
                print(f"    - {col}")
            print(f"    ... and {len(cols)-10} more")
    
    return variable_groups

def prepare_variable_for_analysis(df: pd.DataFrame, col: str) -> pd.Series:
    """Prepare a single variable for analysis."""
    series = df[col].copy()
    
    # Handle missing values
    if series.dtype == 'object' or series.dtype.name == 'category':
        # Categorical variable
        series = series.fillna('Unknown')
        # Convert to numeric if possible
        try:
            series = pd.to_numeric(series, errors='ignore')
        except:
            pass
    else:
        # Numeric variable
        series = series.fillna(series.median())
    
    return series

def test_variable_groups(df: pd.DataFrame, variable_groups: Dict[str, List[str]]) -> Dict:
    """Test different variable groups for R² improvement."""
    print("\n🧪 Testing Variable Groups...")
    
    # Load privacy index
    try:
        privacy_df = pd.read_csv('analysis/privacy_caution_index_individual.csv')
        df_analysis = pd.merge(df, privacy_df[['HHID', 'privacy_caution_index']], on='HHID', how='inner')
        print(f"✅ Privacy index merged: {df_analysis.shape}")
    except:
        print("❌ Privacy index not found")
        return {}
    
    results = {}
    
    # Calculate total tests for progress bar
    total_tests = 0
    for group_name, variables in variable_groups.items():
        prepared_vars = []
        for var in variables:
            if var in df_analysis.columns:
                try:
                    prepared_series = prepare_variable_for_analysis(df_analysis, var)
                    if prepared_series.nunique() > 1:
                        prepared_vars.append(var)
                except:
                    continue
        max_vars = min(10, len(prepared_vars))
        total_tests += max_vars
    
    print(f"📊 Total tests to run: {total_tests}")
    
    # Test each group individually with progress bar
    test_count = 0
    pbar = tqdm(total=total_tests, desc="Testing variable groups", unit="test")
    
    for group_name, variables in variable_groups.items():
        # Prepare variables
        prepared_vars = []
        for var in variables:
            if var in df_analysis.columns:
                try:
                    prepared_series = prepare_variable_for_analysis(df_analysis, var)
                    if prepared_series.nunique() > 1:
                        prepared_vars.append(var)
                except:
                    continue
        
        if len(prepared_vars) == 0:
            pbar.set_description(f"Skipping {group_name} (no usable variables)")
            continue
        
        pbar.set_description(f"Testing {group_name} ({len(prepared_vars)} vars)")
        
        # Test with different numbers of variables (up to 10)
        max_vars = min(10, len(prepared_vars))
        for n_vars in range(1, max_vars + 1):
            try:
                # Test all combinations of n_vars
                best_r2 = -999
                best_combo = None
                
                # Limit combinations for efficiency
                max_combinations = min(100, len(list(combinations(prepared_vars, n_vars))))
                combinations_list = list(combinations(prepared_vars, n_vars))[:max_combinations]
                
                for combo in combinations_list:
                    try:
                        # Prepare data
                        X_data = df_analysis[list(combo)].copy()
                        y_data = df_analysis['privacy_caution_index']
                        
                        # Handle missing values
                        for col in X_data.columns:
                            X_data[col] = prepare_variable_for_analysis(df_analysis, col)
                        
                        # Remove rows with missing target
                        valid_idx = ~y_data.isnull()
                        X_clean = X_data[valid_idx]
                        y_clean = y_data[valid_idx]
                        
                        if len(X_clean) < 100:  # Minimum sample size
                            continue
                        
                        # Convert categorical variables to numeric
                        for col in X_clean.columns:
                            if X_clean[col].dtype == 'object':
                                le = LabelEncoder()
                                X_clean[col] = le.fit_transform(X_clean[col].astype(str))
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_clean, y_clean, test_size=0.2, random_state=42
                        )
                        
                        # Test Linear Regression
                        lr = LinearRegression()
                        lr.fit(X_train, y_train)
                        y_pred = lr.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_combo = combo
                    
                    except Exception as e:
                        continue
                
                if best_r2 > -999:
                    if best_r2 > 0.1:  # Only report if R² > 0.1
                        pbar.set_description(f"{group_name}: R²={best_r2:.3f} ⭐")
                
                results[f"{group_name}_{n_vars}vars"] = {
                    'best_r2': best_r2,
                    'best_combo': best_combo,
                    'n_variables': n_vars
                }
                
                test_count += 1
                pbar.update(1)
            
            except Exception as e:
                test_count += 1
                pbar.update(1)
                continue
    
    pbar.close()
    return results

def test_combined_groups(df: pd.DataFrame, variable_groups: Dict[str, List[str]]) -> Dict:
    """Test combinations of different variable groups."""
    print("\n🔗 Testing Combined Variable Groups...")
    
    # Load privacy index
    try:
        privacy_df = pd.read_csv('analysis/privacy_caution_index_individual.csv')
        df_analysis = pd.merge(df, privacy_df[['HHID', 'privacy_caution_index']], on='HHID', how='inner')
    except:
        return {}
    
    results = {}
    
    # Test combinations of 2-3 groups
    group_names = list(variable_groups.keys())
    
    for n_groups in range(2, min(4, len(group_names) + 1)):
        print(f"\n🔬 Testing combinations of {n_groups} groups...")
        
        for combo in combinations(group_names, n_groups):
            combo_name = "_".join(combo)
            print(f"  Testing {combo_name}...")
            
            # Combine variables from selected groups
            all_vars = []
            for group in combo:
                all_vars.extend(variable_groups[group])
            
            # Remove duplicates
            all_vars = list(set(all_vars))
            
            # Test with different numbers of variables
            max_vars = min(15, len(all_vars))
            best_r2 = -999
            best_combo = None
            
            for n_vars in range(5, max_vars + 1, 5):  # Test every 5 variables
                try:
                    # Randomly sample variables (for efficiency)
                    if len(all_vars) > n_vars:
                        test_vars = np.random.choice(all_vars, n_vars, replace=False)
                    else:
                        test_vars = all_vars
                    
                    # Prepare data
                    X_data = df_analysis[test_vars].copy()
                    y_data = df_analysis['privacy_caution_index']
                    
                    # Handle missing values
                    for col in X_data.columns:
                        X_data[col] = prepare_variable_for_analysis(df_analysis, col)
                    
                    # Remove rows with missing target
                    valid_idx = ~y_data.isnull()
                    X_clean = X_data[valid_idx]
                    y_clean = y_data[valid_idx]
                    
                    if len(X_clean) < 100:
                        continue
                    
                    # Convert categorical variables to numeric
                    for col in X_clean.columns:
                        if X_clean[col].dtype == 'object':
                            le = LabelEncoder()
                            X_clean[col] = le.fit_transform(X_clean[col].astype(str))
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_clean, y_clean, test_size=0.2, random_state=42
                    )
                    
                    # Test Random Forest (better for many variables)
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_combo = test_vars
                
                except Exception as e:
                    continue
            
            if best_r2 > -999:
                print(f"    Best R² = {best_r2:.4f}")
                if best_r2 > 0.15:  # Only report if R² > 0.15
                    print(f"    Best variables: {best_combo[:5]}...")  # Show first 5
            
            results[combo_name] = {
                'best_r2': best_r2,
                'best_combo': best_combo,
                'groups': combo
            }
    
    return results

def find_best_overall_model(df: pd.DataFrame, variable_groups: Dict[str, List[str]]) -> Dict:
    """Find the overall best model by testing top variables from each group."""
    print("\n🏆 Finding Overall Best Model...")
    
    # Load privacy index
    try:
        privacy_df = pd.read_csv('analysis/privacy_caution_index_individual.csv')
        df_analysis = pd.merge(df, privacy_df[['HHID', 'privacy_caution_index']], on='HHID', how='inner')
    except:
        return {}
    
    # Select top variables from each group
    top_variables = []
    for group_name, variables in variable_groups.items():
        # Take up to 5 variables from each group
        top_variables.extend(variables[:5])
    
    print(f"📊 Testing {len(top_variables)} top variables...")
    
    best_r2 = -999
    best_model = None
    best_features = None
    
    # Calculate total trials
    n_feature_ranges = list(range(10, min(31, len(top_variables) + 1), 5))
    total_trials = len(n_feature_ranges) * 10
    
    # Test different numbers of features with progress bar
    pbar = tqdm(total=total_trials, desc="Finding best overall model", unit="trial")
    
    for n_features in n_feature_ranges:
        pbar.set_description(f"Testing {n_features} features")
        
        # Test multiple random combinations
        for trial in range(10):  # 10 trials per number of features
            try:
                # Randomly select features
                selected_features = np.random.choice(top_variables, n_features, replace=False)
                
                # Prepare data
                X_data = df_analysis[selected_features].copy()
                y_data = df_analysis['privacy_caution_index']
                
                # Handle missing values
                for col in X_data.columns:
                    X_data[col] = prepare_variable_for_analysis(df_analysis, col)
                
                # Remove rows with missing target
                valid_idx = ~y_data.isnull()
                X_clean = X_data[valid_idx]
                y_clean = y_data[valid_idx]
                
                if len(X_clean) < 100:
                    pbar.update(1)
                    continue
                
                # Convert categorical variables to numeric
                for col in X_clean.columns:
                    if X_clean[col].dtype == 'object':
                        le = LabelEncoder()
                        X_clean[col] = le.fit_transform(X_clean[col].astype(str))
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=0.2, random_state=42
                )
                
                # Test Random Forest
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_features = selected_features
                    best_model = rf
                    
                    pbar.set_description(f"New best R²={r2:.3f} ⭐")
            
            except Exception as e:
                pass
            
            pbar.update(1)
    
    pbar.close()
    
    return {
        'best_r2': best_r2,
        'best_features': best_features,
        'best_model': best_model,
        'n_features': len(best_features) if best_features is not None else 0
    }

def create_exhaustive_visualizations(results: Dict) -> None:
    """Create visualizations for exhaustive search results."""
    print("\n📊 Creating Exhaustive Search Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Exhaustive Variable Search Results', fontsize=20, fontweight='bold', y=0.95)
    
    # 1. R² by Variable Group
    ax1 = axes[0, 0]
    group_results = {}
    for key, value in results.items():
        if isinstance(value, dict) and 'best_r2' in value:
            if '_' in key and 'vars' in key:
                group_name = key.split('_')[0]
                if group_name not in group_results:
                    group_results[group_name] = []
                group_results[group_name].append(value['best_r2'])
    
    if group_results:
        group_names = list(group_results.keys())
        group_max_r2s = [max(r2s) for r2s in group_results.values()]
        
        bars = ax1.bar(group_names, group_max_r2s, color='steelblue')
        ax1.set_ylabel('Best R²', fontsize=12)
        ax1.set_title('Best R² by Variable Group', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, r2 in zip(bars, group_max_r2s):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Feature Count vs R²
    ax2 = axes[0, 1]
    feature_counts = []
    r2_scores = []
    
    for key, value in results.items():
        if isinstance(value, dict) and 'best_r2' in value and 'n_variables' in value:
            feature_counts.append(value['n_variables'])
            r2_scores.append(value['best_r2'])
    
    if feature_counts:
        ax2.scatter(feature_counts, r2_scores, alpha=0.6, color='red')
        ax2.set_xlabel('Number of Features', fontsize=12)
        ax2.set_ylabel('R² Score', fontsize=12)
        ax2.set_title('Feature Count vs R²', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # 3. Best Models Comparison
    ax3 = axes[1, 0]
    model_names = []
    model_r2s = []
    
    for key, value in results.items():
        if isinstance(value, dict) and 'best_r2' in value:
            if value['best_r2'] > 0.1:  # Only show models with R² > 0.1
                model_names.append(key)
                model_r2s.append(value['best_r2'])
    
    if model_names:
        # Sort by R²
        sorted_data = sorted(zip(model_names, model_r2s), key=lambda x: x[1], reverse=True)
        model_names, model_r2s = zip(*sorted_data)
        
        bars = ax3.bar(range(len(model_names)), model_r2s, color='green')
        ax3.set_ylabel('R² Score', fontsize=12)
        ax3.set_title('Best Models (R² > 0.1)', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    all_r2s = []
    for value in results.values():
        if isinstance(value, dict) and 'best_r2' in value:
            all_r2s.append(value['best_r2'])
    
    if all_r2s:
        summary_text = f"""
📊 EXHAUSTIVE SEARCH SUMMARY

Total Models Tested: {len(all_r2s)}
Best R² Found: {max(all_r2s):.4f}
Average R²: {np.mean(all_r2s):.4f}
Median R²: {np.median(all_r2s):.4f}

Models with R² > 0.1: {sum(1 for r2 in all_r2s if r2 > 0.1)}
Models with R² > 0.2: {sum(1 for r2 in all_r2s if r2 > 0.2)}
Models with R² > 0.3: {sum(1 for r2 in all_r2s if r2 > 0.3)}

Top 3 Models:
{chr(10).join([f'• {name}: R² = {r2:.4f}' for name, r2 in sorted(zip(model_names, model_r2s), key=lambda x: x[1], reverse=True)[:3]])}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plots
    output_path = Path(__file__).parent.parent / "figures" / "exhaustive_variable_search.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    pdf_path = Path(__file__).parent.parent / "figures" / "exhaustive_variable_search.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"✅ Exhaustive search visualizations saved to {output_path}")
    print(f"📄 PDF version saved to {pdf_path}")

def save_exhaustive_results(all_results: Dict) -> None:
    """Save exhaustive search results."""
    output_path = Path(__file__).parent.parent / "analysis" / "exhaustive_search_results.json"
    
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
    converted_results['analysis_type'] = 'exhaustive_variable_search'
    converted_results['total_variables_tested'] = len([k for k in converted_results.keys() if isinstance(converted_results[k], dict)])
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Exhaustive search results saved to: {output_path}")

def main():
    """Main exhaustive search function."""
    print("🔍 Exhaustive Variable Search for Privacy Analysis")
    print("=" * 60)
    
    # Load all data
    df = load_all_data()
    if df.empty:
        print("❌ Failed to load data")
        return
    
    print(f"📊 Full dataset: {df.shape}")
    
    # Identify potential variables
    variable_groups = identify_potential_variables(df)
    
    # Test individual groups
    group_results = test_variable_groups(df, variable_groups)
    
    # Test combined groups
    combined_results = test_combined_groups(df, variable_groups)
    
    # Find overall best model
    best_model_results = find_best_overall_model(df, variable_groups)
    
    # Combine all results
    all_results = {
        'group_results': group_results,
        'combined_results': combined_results,
        'best_model': best_model_results,
        'variable_groups': variable_groups
    }
    
    # Create visualizations
    create_exhaustive_visualizations(all_results)
    
    # Save results
    save_exhaustive_results(all_results)
    
    print("\n🎉 Exhaustive Variable Search Complete!")
    print("=" * 60)
    
    # Summary
    print(f"\n📊 SEARCH SUMMARY:")
    
    if best_model_results and best_model_results['best_r2'] > -999:
        print(f"🏆 Best Model Found:")
        print(f"  R² Score: {best_model_results['best_r2']:.4f}")
        print(f"  Number of Features: {best_model_results['n_features']}")
        print(f"  Features: {best_model_results['best_features'][:5]}...")  # Show first 5
    
    # Count successful models
    successful_models = 0
    for results_dict in [group_results, combined_results]:
        for value in results_dict.values():
            if isinstance(value, dict) and value.get('best_r2', -999) > 0.1:
                successful_models += 1
    
    print(f"\n📈 Models with R² > 0.1: {successful_models}")
    
    # Find best group
    best_group_r2 = -999
    best_group_name = None
    for key, value in group_results.items():
        if isinstance(value, dict) and value.get('best_r2', -999) > best_group_r2:
            best_group_r2 = value['best_r2']
            best_group_name = key
    
    if best_group_name:
        print(f"🏅 Best Variable Group: {best_group_name} (R² = {best_group_r2:.4f})")

if __name__ == "__main__":
    main()
