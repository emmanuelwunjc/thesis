#!/usr/bin/env python3
"""
Privacy Index Correlation Analysis

This script:
1. Examines the dependent variable structure (binary vs multi-nominal)
2. Analyzes privacy index scale options (0-1, 1-5, 1-7)
3. Runs correlations within privacy index sub-dimensions
4. Runs correlation/regression of privacy index on willingness
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import sys
sys.path.append('/Users/wuyiming/code/thesis')
from wrangle import load_r_data, derive_diabetes_mask, detect_weights

warnings.filterwarnings('ignore')

def load_data():
    """Load data and privacy index"""
    print("📊 Loading data...")
    
    # Load main dataset - try multiple possible paths
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
    dia_mask = derive_diabetes_mask(df)
    weights = detect_weights(df)
    
    # Load privacy index
    privacy_index_path = Path('analysis/privacy_caution_index_individual.csv')
    if privacy_index_path.exists():
        privacy_df = pd.read_csv(privacy_index_path)
        # Merge with main dataset
        if 'HHID' in privacy_df.columns and 'HHID' in df.columns:
            df = df.merge(privacy_df, on='HHID', how='left', suffixes=('', '_privacy'))
        else:
            # Use index if HHID not available
            for col in privacy_df.columns:
                if col != 'HHID':
                    df[col] = privacy_df[col].values[:len(df)]
    else:
        print("⚠️  Privacy index file not found. Building index...")
        from build_privacy_index import build_privacy_caution_index
        privacy_index, subindices = build_privacy_caution_index(df, weights)
        df['privacy_caution_index'] = privacy_index
        for name, sub_idx in subindices.items():
            df[f'subindex_{name}'] = sub_idx
    
    return df, dia_mask, weights

def examine_dependent_variable(df):
    """Examine the structure of WillingShareData_HCP2"""
    print("\n" + "="*60)
    print("1. DEPENDENT VARIABLE STRUCTURE ANALYSIS")
    print("="*60)
    
    var = 'WillingShareData_HCP2'
    if var not in df.columns:
        print(f"❌ Variable {var} not found in dataset")
        return None
    
    # Count unique values
    value_counts = df[var].value_counts(dropna=False)
    print(f"\n📋 Value counts for {var}:")
    print(value_counts)
    print(f"\nTotal unique values: {value_counts.shape[0]}")
    
    # Check if binary
    non_missing = df[var].dropna()
    unique_values = non_missing.unique()
    
    result = {
        'variable': var,
        'total_observations': len(df),
        'missing_count': df[var].isna().sum(),
        'non_missing_count': len(non_missing),
        'unique_values': unique_values.tolist(),
        'value_counts': value_counts.to_dict(),
        'is_binary': len(unique_values) == 2,
        'is_multi_nominal': len(unique_values) > 2
    }
    
    print(f"\n✅ Is binary: {result['is_binary']}")
    print(f"✅ Is multi-nominal: {result['is_multi_nominal']}")
    
    # Suggest recoding options
    print("\n💡 Recoding Options:")
    print("   Option 1: Binary (Yes=1, No=0)")
    print("   Option 2: Multi-nominal (keep all categories)")
    
    # Check for common binary values
    if 'Yes' in unique_values and 'No' in unique_values:
        print("\n   ✅ Can be coded as binary: Yes=1, No=0")
        result['binary_recoding'] = {'Yes': 1, 'No': 0}
    
    return result

def examine_privacy_index_scale(df):
    """Examine privacy index scale and suggest alternatives"""
    print("\n" + "="*60)
    print("2. PRIVACY INDEX SCALE ANALYSIS")
    print("="*60)
    
    if 'privacy_caution_index' not in df.columns:
        print("❌ Privacy caution index not found")
        return None
    
    index = df['privacy_caution_index'].dropna()
    
    result = {
        'current_scale': '0-1',
        'min': float(index.min()),
        'max': float(index.max()),
        'mean': float(index.mean()),
        'std': float(index.std()),
        'median': float(index.median()),
        'q25': float(index.quantile(0.25)),
        'q75': float(index.quantile(0.75))
    }
    
    print(f"\n📊 Current Scale: 0-1")
    print(f"   Min: {result['min']:.4f}")
    print(f"   Max: {result['max']:.4f}")
    print(f"   Mean: {result['mean']:.4f}")
    print(f"   Std: {result['std']:.4f}")
    print(f"   Median: {result['median']:.4f}")
    
    # Create alternative scales
    print("\n💡 Alternative Scale Options:")
    
    # Option 1: 1-5 scale
    index_1_5 = (index * 4) + 1  # Transform 0-1 to 1-5
    result['scale_1_5'] = {
        'min': float(index_1_5.min()),
        'max': float(index_1_5.max()),
        'mean': float(index_1_5.mean()),
        'std': float(index_1_5.std())
    }
    print(f"\n   Option 1: 1-5 scale")
    print(f"      Min: {result['scale_1_5']['min']:.2f}")
    print(f"      Max: {result['scale_1_5']['max']:.2f}")
    print(f"      Mean: {result['scale_1_5']['mean']:.2f}")
    
    # Option 2: 1-7 scale
    index_1_7 = (index * 6) + 1  # Transform 0-1 to 1-7
    result['scale_1_7'] = {
        'min': float(index_1_7.min()),
        'max': float(index_1_7.max()),
        'mean': float(index_1_7.mean()),
        'std': float(index_1_7.std())
    }
    print(f"\n   Option 2: 1-7 scale")
    print(f"      Min: {result['scale_1_7']['min']:.2f}")
    print(f"      Max: {result['scale_1_7']['max']:.2f}")
    print(f"      Mean: {result['scale_1_7']['mean']:.2f}")
    
    print(f"\n   Option 3: Keep 0-1 scale (current)")
    print(f"      ✅ Recommended: 0-1 scale is standard and interpretable")
    
    return result

def correlation_within_index(df):
    """Run correlations within privacy index sub-dimensions"""
    print("\n" + "="*60)
    print("3. CORRELATION WITHIN PRIVACY INDEX SUB-DIMENSIONS")
    print("="*60)
    
    # Get all sub-index columns
    subindex_cols = [col for col in df.columns if col.startswith('subindex_')]
    
    if not subindex_cols:
        print("❌ No sub-index columns found")
        return None
    
    print(f"\n📊 Found {len(subindex_cols)} sub-dimensions:")
    for col in subindex_cols:
        print(f"   - {col}")
    
    # Create subset with only sub-indices
    subindex_df = df[subindex_cols].copy()
    
    # Remove rows with any missing values
    subindex_df_clean = subindex_df.dropna()
    
    print(f"\n📈 Sample size for correlation: {len(subindex_df_clean)}")
    
    # Calculate correlation matrix
    corr_matrix = subindex_df_clean.corr()
    
    print("\n📊 Correlation Matrix (Pearson):")
    print(corr_matrix.round(3))
    
    # Calculate pairwise correlations with p-values
    correlations = {}
    n = len(subindex_df_clean)
    
    for i, col1 in enumerate(subindex_cols):
        for col2 in subindex_cols[i+1:]:
            if col1 in subindex_df_clean.columns and col2 in subindex_df_clean.columns:
                data1 = subindex_df_clean[col1]
                data2 = subindex_df_clean[col2]
                
                # Pearson correlation
                r_pearson, p_pearson = pearsonr(data1, data2)
                
                # Spearman correlation
                r_spearman, p_spearman = spearmanr(data1, data2)
                
                correlations[f"{col1} vs {col2}"] = {
                    'pearson_r': float(r_pearson),
                    'pearson_p': float(p_pearson),
                    'spearman_r': float(r_spearman),
                    'spearman_p': float(p_spearman),
                    'n': int(n)
                }
    
    # Print significant correlations
    print("\n🔍 Significant Correlations (p < 0.05):")
    for pair, stats_dict in correlations.items():
        if stats_dict['pearson_p'] < 0.05:
            print(f"\n   {pair}:")
            print(f"      Pearson r = {stats_dict['pearson_r']:.3f}, p = {stats_dict['pearson_p']:.4f}")
            if stats_dict['spearman_p'] < 0.05:
                print(f"      Spearman ρ = {stats_dict['spearman_r']:.3f}, p = {stats_dict['spearman_p']:.4f}")
    
    result = {
        'correlation_matrix': corr_matrix.to_dict(),
        'pairwise_correlations': correlations,
        'n': int(n),
        'subdimensions': subindex_cols
    }
    
    return result

def index_on_willingness(df):
    """Run correlation/regression of privacy index on willingness"""
    print("\n" + "="*60)
    print("4. PRIVACY INDEX ON WILLINGNESS ANALYSIS")
    print("="*60)
    
    # Prepare willingness variable
    willingness_var = 'WillingShareData_HCP2'
    if willingness_var not in df.columns:
        print(f"❌ Variable {willingness_var} not found")
        return None
    
    # Create binary willingness variable
    df_analysis = df.copy()
    df_analysis['willingness_binary'] = df_analysis[willingness_var].map({
        'Yes': 1,
        'No': 0
    })
    
    # Keep only valid cases
    valid_mask = df_analysis['willingness_binary'].notna() & df_analysis['privacy_caution_index'].notna()
    df_clean = df_analysis[valid_mask].copy()
    
    print(f"\n📊 Sample size: {len(df_clean)}")
    print(f"   Willing (1): {df_clean['willingness_binary'].sum()}")
    print(f"   Not Willing (0): {(df_clean['willingness_binary'] == 0).sum()}")
    
    # Correlation analysis
    privacy_index = df_clean['privacy_caution_index']
    willingness = df_clean['willingness_binary']
    
    # Point-biserial correlation (for binary-continuous)
    r_pb, p_pb = stats.pointbiserialr(willingness, privacy_index)
    
    print(f"\n📈 Point-Biserial Correlation:")
    print(f"   r = {r_pb:.4f}, p = {p_pb:.4f}")
    
    # Pearson correlation (treating binary as continuous)
    r_pearson, p_pearson = pearsonr(willingness, privacy_index)
    
    print(f"\n📈 Pearson Correlation:")
    print(f"   r = {r_pearson:.4f}, p = {p_pearson:.4f}")
    
    # Spearman correlation
    r_spearman, p_spearman = spearmanr(willingness, privacy_index)
    
    print(f"\n📈 Spearman Correlation:")
    print(f"   ρ = {r_spearman:.4f}, p = {p_spearman:.4f}")
    
    # Mean privacy index by willingness
    print(f"\n📊 Mean Privacy Index by Willingness:")
    mean_willing = privacy_index[willingness == 1].mean()
    mean_not_willing = privacy_index[willingness == 0].mean()
    
    print(f"   Willing to share: {mean_willing:.4f}")
    print(f"   Not willing to share: {mean_not_willing:.4f}")
    print(f"   Difference: {mean_willing - mean_not_willing:.4f}")
    
    # Simple logistic regression (using statsmodels if available)
    try:
        import statsmodels.api as sm
        from statsmodels.stats.weightstats import DescrStatsW
        
        # Add constant
        X = sm.add_constant(privacy_index)
        y = willingness
        
        # Fit logistic regression
        logit_model = sm.Logit(y, X).fit(disp=0)
        
        print(f"\n📊 Logistic Regression Results:")
        print(logit_model.summary().tables[1])
        
        # Odds ratio
        coef = logit_model.params['privacy_caution_index']
        or_ratio = np.exp(coef)
        print(f"\n   Odds Ratio: {or_ratio:.4f}")
        print(f"   Interpretation: One unit increase in privacy index")
        print(f"                   changes odds by factor of {or_ratio:.4f}")
        
        regression_results = {
            'coefficient': float(coef),
            'odds_ratio': float(or_ratio),
            'p_value': float(logit_model.pvalues['privacy_caution_index']),
            'conf_int': logit_model.conf_int().loc['privacy_caution_index'].to_dict()
        }
    except ImportError:
        print("\n⚠️  statsmodels not available, skipping logistic regression")
        regression_results = None
    
    result = {
        'point_biserial': {'r': float(r_pb), 'p': float(p_pb)},
        'pearson': {'r': float(r_pearson), 'p': float(p_pearson)},
        'spearman': {'r': float(r_spearman), 'p': float(p_spearman)},
        'mean_by_group': {
            'willing': float(mean_willing),
            'not_willing': float(mean_not_willing),
            'difference': float(mean_willing - mean_not_willing)
        },
        'regression': regression_results,
        'n': int(len(df_clean))
    }
    
    return result

def main():
    """Main analysis function"""
    print("="*60)
    print("PRIVACY INDEX CORRELATION ANALYSIS")
    print("="*60)
    
    # Load data
    df, dia_mask, weights = load_data()
    
    # Run analyses
    results = {}
    
    # 1. Examine dependent variable
    dv_result = examine_dependent_variable(df)
    results['dependent_variable'] = dv_result
    
    # 2. Examine privacy index scale
    scale_result = examine_privacy_index_scale(df)
    results['privacy_index_scale'] = scale_result
    
    # 3. Correlation within index
    within_corr = correlation_within_index(df)
    results['correlation_within_index'] = within_corr
    
    # 4. Index on willingness
    willingness_result = index_on_willingness(df)
    results['index_on_willingness'] = willingness_result
    
    # Save results
    output_path = Path('analysis/privacy_index_correlation_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print(f"📁 Results saved to: {output_path}")
    print("="*60)
    
    # Print summary
    print("\n📋 SUMMARY:")
    print(f"   1. Dependent variable: {'Binary' if dv_result and dv_result['is_binary'] else 'Multi-nominal'}")
    print(f"   2. Privacy index scale: 0-1 (current)")
    print(f"   3. Sub-dimension correlations: {len(within_corr['pairwise_correlations']) if within_corr else 0} pairs")
    print(f"   4. Index-Willingness correlation: r = {willingness_result['point_biserial']['r']:.3f}, p = {willingness_result['point_biserial']['p']:.4f}")

if __name__ == "__main__":
    main()

