import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_data():
    # Load the main dataset
    import sys
    sys.path.append('/Users/wuyiming/code/thesis')
    from wrangle import load_r_data, derive_diabetes_mask, detect_weights
    
    df = load_r_data(Path('/Users/wuyiming/code/thesis/hints7_public copy.rda'))
    dia_mask = derive_diabetes_mask(df)
    weights = detect_weights(df)
    return df, dia_mask, weights

def build_privacy_caution_index(df: pd.DataFrame, weights: Dict) -> pd.Series:
    """
    构建隐私谨慎指数：越高 = 越谨慎（不分享、不信任、少数字参与）
    """
    
    # 定义变量映射：原始值 -> 谨慎程度分数 (0-1)
    def map_to_caution_score(series: pd.Series, mapping: Dict) -> pd.Series:
        s = series.fillna('NA').astype(str).str.strip()
        return s.map(mapping).fillna(0.5)  # 默认中等谨慎
    
    # 1. 数据分享意愿 (越高越不谨慎)
    share_vars = {
        'WillingShareData_HCP2': {'Yes': 0, 'No': 1, 'NA': 0.5},
        'SharedHealthDeviceInfo2': {'Yes': 0, 'No': 1, 'NA': 0.5},
        'SocMed_SharedPers': {'Yes': 0, 'No': 1, 'NA': 0.5},
        'SocMed_SharedGen': {'Yes': 0, 'No': 1, 'NA': 0.5},
    }
    
    # 2. 在线记录/门户使用 (越高越不谨慎)
    portal_vars = {
        'AccessOnlineRecord3': {'Yes': 0, 'No': 1, 'NA': 0.5},
        'OnlinePortal_PCP': {'Selected': 0, 'Not selected': 1, 'NA': 0.5},
        'OnlinePortal_OthHCP': {'Selected': 0, 'Not selected': 1, 'NA': 0.5},
        'OnlinePortal_Insurer': {'Selected': 0, 'Not selected': 1, 'NA': 0.5},
        'OnlinePortal_Lab': {'Selected': 0, 'Not selected': 1, 'NA': 0.5},
        'OnlinePortal_Pharmacy': {'Selected': 0, 'Not selected': 1, 'NA': 0.5},
        'OnlinePortal_Hospital': {'Selected': 0, 'Not selected': 1, 'NA': 0.5},
    }
    
    # 3. 设备使用 (越高越不谨慎)
    device_vars = {
        'UseDevice_Computer': {'Yes': 0, 'No': 1, 'NA': 0.5},
        'UseDevice_SmPhone': {'Yes': 0, 'No': 1, 'NA': 0.5},
        'UseDevice_Tablet': {'Yes': 0, 'No': 1, 'NA': 0.5},
        'UseDevice_SmWatch': {'Yes': 0, 'No': 1, 'NA': 0.5},
    }
    
    # 4. 信任度 (信任=不谨慎，不信任=谨慎)
    trust_vars = {
        'TrustHCSystem': {'A lot': 0, 'Some': 0.3, 'A little': 0.7, 'Not at all': 1, 'NA': 0.5},
        'CancerTrustDoctor': {'A lot': 0, 'Some': 0.3, 'A little': 0.7, 'Not at all': 1, 'NA': 0.5},
        'CancerTrustScientists': {'A lot': 0, 'Some': 0.3, 'A little': 0.7, 'Not at all': 1, 'NA': 0.5},
        'CancerTrustFamily': {'A lot': 0, 'Some': 0.3, 'A little': 0.7, 'Not at all': 1, 'NA': 0.5},
    }
    
    # 5. 社媒使用/误导信息 (使用社媒=不谨慎，不用=谨慎)
    social_vars = {
        'SocMed_Visited': {'Yes': 0, 'No': 1, 'NA': 0.5},
        'MisleadingHealthInfo': {'I do not use social media': 1, 'Yes': 0, 'No': 0.5, 'NA': 0.5},
    }
    
    # 6. 其他隐私相关
    other_vars = {
        'ConfidentMedForms': {'Very confident': 0, 'Somewhat confident': 0.3, 'A little confident': 0.7, 'Not confident at all': 1, 'NA': 0.5},
        'WillingUseTelehealth': {'Yes': 0, 'No': 1, 'NA': 0.5},
    }
    
    # 构建子维度指数
    subindices = {}
    
    # 分享意愿子指数
    share_scores = []
    for var, mapping in share_vars.items():
        if var in df.columns:
            score = map_to_caution_score(df[var], mapping)
            share_scores.append(score)
    if share_scores:
        subindices['sharing'] = pd.concat(share_scores, axis=1).mean(axis=1)
    
    # 门户使用子指数
    portal_scores = []
    for var, mapping in portal_vars.items():
        if var in df.columns:
            score = map_to_caution_score(df[var], mapping)
            portal_scores.append(score)
    if portal_scores:
        subindices['portals'] = pd.concat(portal_scores, axis=1).mean(axis=1)
    
    # 设备使用子指数
    device_scores = []
    for var, mapping in device_vars.items():
        if var in df.columns:
            score = map_to_caution_score(df[var], mapping)
            device_scores.append(score)
    if device_scores:
        subindices['devices'] = pd.concat(device_scores, axis=1).mean(axis=1)
    
    # 信任度子指数
    trust_scores = []
    for var, mapping in trust_vars.items():
        if var in df.columns:
            score = map_to_caution_score(df[var], mapping)
            trust_scores.append(score)
    if trust_scores:
        subindices['trust'] = pd.concat(trust_scores, axis=1).mean(axis=1)
    
    # 社媒子指数
    social_scores = []
    for var, mapping in social_vars.items():
        if var in df.columns:
            score = map_to_caution_score(df[var], mapping)
            social_scores.append(score)
    if social_scores:
        subindices['social'] = pd.concat(social_scores, axis=1).mean(axis=1)
    
    # 其他隐私子指数
    other_scores = []
    for var, mapping in other_vars.items():
        if var in df.columns:
            score = map_to_caution_score(df[var], mapping)
            other_scores.append(score)
    if other_scores:
        subindices['other'] = pd.concat(other_scores, axis=1).mean(axis=1)
    
    # 综合隐私谨慎指数
    if subindices:
        privacy_index = pd.concat(subindices.values(), axis=1).mean(axis=1)
    else:
        privacy_index = pd.Series(0.5, index=df.index)
    
    return privacy_index, subindices

def compute_weighted_stats(series: pd.Series, weights: pd.Series, group_mask: pd.Series) -> Dict:
    """计算加权统计"""
    w = weights if weights is not None else pd.Series(1, index=series.index)
    
    # 糖尿病组
    dia_mask = group_mask & series.notna() & w.notna()
    if dia_mask.sum() > 0:
        dia_mean = (series[dia_mask] * w[dia_mask]).sum() / w[dia_mask].sum()
        dia_std = np.sqrt(((series[dia_mask] - dia_mean) ** 2 * w[dia_mask]).sum() / w[dia_mask].sum())
    else:
        dia_mean = dia_std = np.nan
    
    # 非糖尿病组
    non_mask = ~group_mask & series.notna() & w.notna()
    if non_mask.sum() > 0:
        non_mean = (series[non_mask] * w[non_mask]).sum() / w[non_mask].sum()
        non_std = np.sqrt(((series[non_mask] - non_mean) ** 2 * w[non_mask]).sum() / w[non_mask].sum())
    else:
        non_mean = non_std = np.nan
    
    return {
        'diabetic_mean': dia_mean,
        'diabetic_std': dia_std,
        'non_diabetic_mean': non_mean,
        'non_diabetic_std': non_std,
        'difference': dia_mean - non_mean if not (np.isnan(dia_mean) or np.isnan(non_mean)) else np.nan
    }

def main():
    print("构建隐私谨慎指数...")
    df, dia_mask, weights = load_data()
    
    # 构建指数
    privacy_index, subindices = build_privacy_caution_index(df, weights)
    
    # 获取权重列
    main_weight = weights.get('main')
    w = df[main_weight] if main_weight and main_weight in df.columns else None
    
    # 计算统计
    stats = compute_weighted_stats(privacy_index, w, dia_mask)
    
    # 子维度统计
    sub_stats = {}
    for name, sub_idx in subindices.items():
        sub_stats[name] = compute_weighted_stats(sub_idx, w, dia_mask)
    
    # 保存结果
    result = {
        'privacy_caution_index': {
            'description': '隐私谨慎指数 (0-1, 越高越谨慎)',
            'statistics': stats,
            'subdimensions': sub_stats
        },
        'weights_used': weights,
        'sample_size': {
            'total': len(df),
            'diabetic': int(dia_mask.sum()),
            'non_diabetic': int((~dia_mask).sum())
        }
    }
    
    # 保存到JSON
    with open('privacy_caution_index.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 打印关键结果
    print(f"\n=== 隐私谨慎指数结果 ===")
    print(f"糖尿病组均值: {stats['diabetic_mean']:.3f} ± {stats['diabetic_std']:.3f}")
    print(f"非糖尿病组均值: {stats['non_diabetic_mean']:.3f} ± {stats['non_diabetic_std']:.3f}")
    print(f"组间差异: {stats['difference']:.3f}")
    
    print(f"\n=== 子维度差异 ===")
    for name, sub_stat in sub_stats.items():
        print(f"{name}: {sub_stat['difference']:+.3f}")
    
    # 保存个体指数到CSV
    output_df = df[['HHID']].copy() if 'HHID' in df.columns else pd.DataFrame(index=df.index)
    output_df['diabetic'] = dia_mask.astype(int)
    output_df['privacy_caution_index'] = privacy_index
    for name, sub_idx in subindices.items():
        output_df[f'subindex_{name}'] = sub_idx
    if w is not None:
        output_df['weight'] = w
    
    output_df.to_csv('privacy_caution_index_individual.csv', index=False)
    print(f"\n个体指数已保存到: privacy_caution_index_individual.csv")
    print(f"汇总结果已保存到: privacy_caution_index.json")

if __name__ == "__main__":
    main()
