#!/usr/bin/env python3
"""
Multi-Chronic Disease Privacy Analysis

This script replicates the diabetes privacy analysis for other chronic diseases
that require day-to-day data tracking and logging:
- Hypertension (High BP) - blood pressure monitoring
- Heart Conditions - heart rate, medication tracking
- Depression - mood tracking, medication adherence
- Lung Disease - symptom tracking, medication

This analysis will:
1. Identify chronic diseases requiring daily tracking
2. Replicate regression models for each condition
3. Compare effects across conditions
4. Test for generalizability of diabetes findings
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
import sys
sys.path.append('/Users/wuyiming/code/thesis')
from wrangle import load_r_data, detect_weights

warnings.filterwarnings('ignore')

# Try to import statsmodels for regression
try:
    import statsmodels.api as sm
    from statsmodels.stats.weightstats import DescrStatsW
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️  statsmodels not available, will use basic correlations")

def load_data():
    """Load HINTS 7 data and privacy index"""
    print("📊 Loading data...")
    
    # Load main dataset
    base_path = Path(__file__).parent.parent
    possible_paths = [
        base_path / 'data' / 'hints7_public copy.rda',
        base_path / 'hints7_public copy.rda',
        Path('/Users/wuyiming/code/thesis/data/hints7_public copy.rda')
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Data file not found. Tried: {possible_paths}")
    
    df = load_r_data(data_path)
    weights = detect_weights(df)
    
    # Load privacy index
    privacy_index_path = base_path / 'analysis' / 'privacy_caution_index_individual.csv'
    if privacy_index_path.exists():
        privacy_df = pd.read_csv(privacy_index_path)
        if 'HHID' in privacy_df.columns and 'HHID' in df.columns:
            df = df.merge(privacy_df[['HHID', 'privacy_caution_index']], 
                         on='HHID', how='left', suffixes=('', '_privacy'))
        else:
            df['privacy_caution_index'] = privacy_df['privacy_caution_index'].values[:len(df)]
    else:
        print("⚠️  Privacy index not found. Building index...")
        from scripts.build_privacy_index import build_privacy_caution_index
        privacy_index, _ = build_privacy_caution_index(df, weights)
        df['privacy_caution_index'] = privacy_index
    
    return df, weights

def identify_chronic_diseases(df):
    """Identify chronic disease variables and categorize by tracking requirements"""
    
    # Find all MedConditions variables
    med_conditions = [col for col in df.columns 
                      if 'MedConditions' in col or 'medconditions' in col.lower()]
    
    # Categorize by daily tracking requirements
    daily_tracking_conditions = {
        'Diabetes': {
            'variable': 'MedConditions_Diabetes',
            'tracking_required': True,
            'tracking_type': 'Blood glucose, medication, diet, activity',
            'rationale': 'Requires continuous monitoring and data logging'
        },
        'Hypertension': {
            'variable': 'MedConditions_HighBP',
            'tracking_required': True,
            'tracking_type': 'Blood pressure, medication adherence',
            'rationale': 'Requires regular BP monitoring and medication tracking'
        },
        'Heart Condition': {
            'variable': 'MedConditions_HeartCondition',
            'tracking_required': True,
            'tracking_type': 'Heart rate, symptoms, medication, activity',
            'rationale': 'May require continuous monitoring depending on condition'
        },
        'Depression': {
            'variable': 'MedConditions_Depression',
            'tracking_required': True,
            'tracking_type': 'Mood tracking, medication adherence, symptoms',
            'rationale': 'Mood tracking apps and medication logging are common'
        },
        'Lung Disease': {
            'variable': 'MedConditions_LungDisease',
            'tracking_required': True,
            'tracking_type': 'Symptom tracking, medication, oxygen levels',
            'rationale': 'Symptom monitoring and medication tracking required'
        }
    }
    
    # Check which variables exist and get sample sizes
    available_conditions = {}
    for name, info in daily_tracking_conditions.items():
        var = info['variable']
        if var in df.columns:
            # Count cases
            condition_mask = df[var] == 'Yes'
            n_cases = condition_mask.sum()
            n_total = df[var].notna().sum()
            
            available_conditions[name] = {
                **info,
                'n_cases': int(n_cases),
                'n_total': int(n_total),
                'prevalence': float(n_cases / n_total) if n_total > 0 else 0.0,
                'available': True
            }
        else:
            available_conditions[name] = {
                **info,
                'available': False
            }
    
    return available_conditions

def prepare_analysis_data(df, condition_var):
    """Prepare data for analysis of a specific condition"""
    
    # Create condition dummy
    df_analysis = df.copy()
    df_analysis['has_condition'] = (df_analysis[condition_var] == 'Yes').astype(int)
    
    # Prepare willingness variable (binary)
    if 'WillingShareData_HCP2' in df_analysis.columns:
        df_analysis['willingness_binary'] = df_analysis['WillingShareData_HCP2'].map({
            'Yes': 1,
            'No': 0
        })
    else:
        df_analysis['willingness_binary'] = np.nan
    
    # Keep only valid cases
    valid_mask = (
        df_analysis['has_condition'].notna() &
        df_analysis['privacy_caution_index'].notna() &
        df_analysis['willingness_binary'].notna()
    )
    
    df_clean = df_analysis[valid_mask].copy()
    
    return df_clean

def run_regression_analysis(df, condition_name):
    """Run regression analysis for a specific condition"""
    
    results = {
        'condition': condition_name,
        'sample_size': len(df),
        'n_with_condition': int(df['has_condition'].sum()),
        'n_without_condition': int((df['has_condition'] == 0).sum())
    }
    
    # Correlation analysis
    privacy_index = df['privacy_caution_index']
    willingness = df['willingness_binary']
    condition = df['has_condition']
    
    # Point-biserial: condition vs privacy
    from scipy.stats import pointbiserialr, pearsonr
    r_cond_privacy, p_cond_privacy = pointbiserialr(condition, privacy_index)
    
    # Point-biserial: condition vs willingness
    r_cond_willing, p_cond_willing = pointbiserialr(condition, willingness)
    
    # Point-biserial: privacy vs willingness
    r_privacy_willing, p_privacy_willing = pointbiserialr(willingness, privacy_index)
    
    results['correlations'] = {
        'condition_privacy': {'r': float(r_cond_privacy), 'p': float(p_cond_privacy)},
        'condition_willingness': {'r': float(r_cond_willing), 'p': float(p_cond_willing)},
        'privacy_willingness': {'r': float(r_privacy_willing), 'p': float(p_privacy_willing)}
    }
    
    # Mean privacy index by condition
    mean_privacy_with = privacy_index[condition == 1].mean()
    mean_privacy_without = privacy_index[condition == 0].mean()
    
    # Mean willingness by condition
    mean_willing_with = willingness[condition == 1].mean()
    mean_willing_without = willingness[condition == 0].mean()
    
    results['mean_comparisons'] = {
        'privacy_index': {
            'with_condition': float(mean_privacy_with),
            'without_condition': float(mean_privacy_without),
            'difference': float(mean_privacy_with - mean_privacy_without)
        },
        'willingness': {
            'with_condition': float(mean_willing_with),
            'without_condition': float(mean_willing_without),
            'difference': float(mean_willing_with - mean_willing_without)
        }
    }
    
    # Regression analysis (if statsmodels available)
    if HAS_STATSMODELS:
        try:
            # Model 1: Condition → Privacy Index
            X1 = sm.add_constant(condition)
            y1 = privacy_index
            model1 = sm.OLS(y1, X1).fit()
            results['regression_privacy'] = {
                'coefficient': float(model1.params['has_condition']),
                'p_value': float(model1.pvalues['has_condition']),
                'r_squared': float(model1.rsquared)
            }
            
            # Model 2: Condition → Willingness
            X2 = sm.add_constant(condition)
            y2 = willingness
            model2 = sm.Logit(y2, X2).fit(disp=0)
            results['regression_willingness'] = {
                'coefficient': float(model2.params['has_condition']),
                'odds_ratio': float(np.exp(model2.params['has_condition'])),
                'p_value': float(model2.pvalues['has_condition']),
                'pseudo_r_squared': float(model2.prsquared)
            }
            
            # Model 3: Condition + Privacy → Willingness (interaction)
            X3 = sm.add_constant(pd.DataFrame({
                'has_condition': condition,
                'privacy_caution_index': privacy_index,
                'interaction': condition * privacy_index
            }))
            y3 = willingness
            model3 = sm.Logit(y3, X3).fit(disp=0)
            results['regression_interaction'] = {
                'condition_coef': float(model3.params['has_condition']),
                'privacy_coef': float(model3.params['privacy_caution_index']),
                'interaction_coef': float(model3.params['interaction']),
                'interaction_p': float(model3.pvalues['interaction']),
                'pseudo_r_squared': float(model3.prsquared)
            }
            
        except Exception as e:
            results['regression_error'] = str(e)
    
    return results

def compare_conditions(all_results):
    """Compare results across all conditions"""
    
    comparison = {
        'summary_table': [],
        'key_findings': {}
    }
    
    for condition_name, results in all_results.items():
        if 'correlations' in results:
            summary = {
                'condition': condition_name,
                'n_cases': results['n_with_condition'],
                'prevalence': f"{results['n_with_condition']/results['sample_size']*100:.1f}%",
                'privacy_diff': results['mean_comparisons']['privacy_index']['difference'],
                'willingness_diff': results['mean_comparisons']['willingness']['difference'],
                'privacy_corr_p': results['correlations']['condition_privacy']['p'],
                'willingness_corr_p': results['correlations']['condition_willingness']['p']
            }
            
            if 'regression_willingness' in results:
                summary['willingness_coef'] = results['regression_willingness']['coefficient']
                summary['willingness_p'] = results['regression_willingness']['p_value']
                summary['willingness_OR'] = results['regression_willingness']['odds_ratio']
            
            if 'regression_interaction' in results:
                summary['interaction_p'] = results['regression_interaction']['interaction_p']
            
            comparison['summary_table'].append(summary)
    
    # Identify patterns
    significant_conditions = []
    for condition_name, results in all_results.items():
        if 'regression_willingness' in results:
            if results['regression_willingness']['p_value'] < 0.05:
                significant_conditions.append(condition_name)
    
    comparison['key_findings'] = {
        'significant_conditions': significant_conditions,
        'total_conditions_tested': len(all_results),
        'generalizability': len(significant_conditions) > 1
    }
    
    return comparison

def main():
    """Main analysis function"""
    print("="*70)
    print("MULTI-CHRONIC DISEASE PRIVACY ANALYSIS")
    print("="*70)
    
    # Load data
    df, weights = load_data()
    
    # Identify chronic diseases
    print("\n📋 Identifying chronic diseases requiring daily tracking...")
    conditions = identify_chronic_diseases(df)
    
    print("\n✅ Available conditions:")
    for name, info in conditions.items():
        if info.get('available', False):
            print(f"\n  {name}:")
            print(f"    Variable: {info['variable']}")
            print(f"    Cases: {info['n_cases']:,} ({info['prevalence']*100:.1f}%)")
            print(f"    Tracking: {info['tracking_type']}")
            print(f"    Rationale: {info['rationale']}")
    
    # Run analysis for each condition
    print("\n" + "="*70)
    print("RUNNING ANALYSIS FOR EACH CONDITION")
    print("="*70)
    
    all_results = {}
    
    for condition_name, condition_info in conditions.items():
        if not condition_info.get('available', False):
            continue
        
        if condition_info['n_cases'] < 100:
            print(f"\n⚠️  Skipping {condition_name}: insufficient sample size ({condition_info['n_cases']} cases)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Analyzing: {condition_name}")
        print(f"{'='*70}")
        
        # Prepare data
        df_analysis = prepare_analysis_data(df, condition_info['variable'])
        
        if len(df_analysis) < 200:
            print(f"⚠️  Insufficient valid cases ({len(df_analysis)}), skipping...")
            continue
        
        # Run analysis
        results = run_regression_analysis(df_analysis, condition_name)
        all_results[condition_name] = results
        
        # Print summary
        print(f"\n📊 Sample: {results['sample_size']:,} (with condition: {results['n_with_condition']:,})")
        
        if 'correlations' in results:
            corr = results['correlations']
            print(f"\n📈 Correlations:")
            print(f"   Condition → Privacy: r = {corr['condition_privacy']['r']:.4f}, p = {corr['condition_privacy']['p']:.4f}")
            print(f"   Condition → Willingness: r = {corr['condition_willingness']['r']:.4f}, p = {corr['condition_willingness']['p']:.4f}")
        
        if 'mean_comparisons' in results:
            means = results['mean_comparisons']
            print(f"\n📊 Mean Comparisons:")
            print(f"   Privacy Index: With = {means['privacy_index']['with_condition']:.4f}, "
                  f"Without = {means['privacy_index']['without_condition']:.4f}, "
                  f"Diff = {means['privacy_index']['difference']:+.4f}")
            print(f"   Willingness: With = {means['willingness']['with_condition']:.4f}, "
                  f"Without = {means['willingness']['without_condition']:.4f}, "
                  f"Diff = {means['willingness']['difference']:+.4f}")
        
        if 'regression_willingness' in results:
            reg = results['regression_willingness']
            sig = "***" if reg['p_value'] < 0.001 else "**" if reg['p_value'] < 0.01 else "*" if reg['p_value'] < 0.05 else ""
            print(f"\n📊 Regression (Condition → Willingness):")
            print(f"   Coefficient: {reg['coefficient']:.4f} {sig}")
            print(f"   Odds Ratio: {reg['odds_ratio']:.4f}")
            print(f"   p-value: {reg['p_value']:.4f}")
        
        if 'regression_interaction' in results:
            inter = results['regression_interaction']
            sig = "*" if inter['interaction_p'] < 0.05 else ""
            print(f"\n📊 Interaction Effect:")
            print(f"   Interaction Coefficient: {inter['interaction_coef']:.4f} {sig}")
            print(f"   p-value: {inter['interaction_p']:.4f}")
    
    # Compare conditions
    print("\n" + "="*70)
    print("COMPARISON ACROSS CONDITIONS")
    print("="*70)
    
    comparison = compare_conditions(all_results)
    
    # Print summary table
    if comparison['summary_table']:
        print("\n📊 Summary Table:")
        print(f"\n{'Condition':<20} {'N Cases':<10} {'Prev':<8} {'Privacy Diff':<12} {'Willing Diff':<12} {'Willing P':<10} {'Interaction P':<12}")
        print("-" * 90)
        for row in comparison['summary_table']:
            privacy_diff = f"{row['privacy_diff']:+.4f}"
            willing_diff = f"{row['willingness_diff']:+.4f}"
            willing_p = f"{row.get('willingness_p', 1.0):.4f}"
            inter_p = f"{row.get('interaction_p', 1.0):.4f}"
            print(f"{row['condition']:<20} {row['n_cases']:<10} {row['prevalence']:<8} "
                  f"{privacy_diff:<12} {willing_diff:<12} {willing_p:<10} {inter_p:<12}")
    
    # Key findings
    print(f"\n🎯 Key Findings:")
    print(f"   Conditions tested: {comparison['key_findings']['total_conditions_tested']}")
    print(f"   Significant conditions: {len(comparison['key_findings']['significant_conditions'])}")
    if comparison['key_findings']['significant_conditions']:
        print(f"   Significant: {', '.join(comparison['key_findings']['significant_conditions'])}")
    print(f"   Generalizability: {'✅ YES' if comparison['key_findings']['generalizability'] else '❌ NO'}")
    
    # Save results
    output = {
        'conditions_info': conditions,
        'individual_results': all_results,
        'comparison': comparison
    }
    
    output_path = Path('analysis/multi_chronic_disease_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")
    print("="*70)

if __name__ == "__main__":
    main()

