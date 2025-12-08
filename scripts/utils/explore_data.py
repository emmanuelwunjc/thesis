#!/usr/bin/env python3
import json
import pandas as pd

def explore_json_data():
    print("🔍 数据探索工具")
    print("=" * 50)
    
    # 读取所有JSON文件
    with open('diabetes_summary.json', 'r') as f:
        summary = json.load(f)
    
    with open('diabetes_demographics_crosstabs.json', 'r') as f:
        demo = json.load(f)
    
    with open('diabetes_privacy_analysis.json', 'r') as f:
        privacy = json.load(f)
    
    # 1. 基本统计
    print("📊 基本统计信息:")
    print(f"  总样本数: {summary['info']['rows']:,}")
    print(f"  变量数量: {summary['info']['cols']}")
    print(f"  糖尿病人数: {summary['summary']['num_diabetic']:,}")
    print(f"  非糖尿病人数: {summary['summary']['num_non_diabetic']:,}")
    print(f"  患病率: {summary['summary']['num_diabetic']/summary['info']['rows']*100:.2f}%")
    
    # 2. 人口学变量
    print(f"\n👥 识别人口学变量 ({len(demo['demographics'])}个):")
    for key, var in demo['demographics'].items():
        print(f"  {key}: {var}")
    
    # 3. 隐私变量
    print(f"\n🔒 隐私相关变量 ({len(privacy['privacy_columns'])}个):")
    categories = {
        '信任度': [v for v in privacy['privacy_columns'] if 'Trust' in v],
        '数据分享': [v for v in privacy['privacy_columns'] if 'Share' in v or 'LabShare' in v],
        '在线记录': [v for v in privacy['privacy_columns'] if 'Record' in v or 'Portal' in v],
        '设备使用': [v for v in privacy['privacy_columns'] if 'Device' in v or 'UseDevice' in v]
    }
    
    for cat, vars_list in categories.items():
        if vars_list:
            print(f"  {cat}: {len(vars_list)}个变量")
            for var in vars_list[:3]:  # 显示前3个
                print(f"    - {var}")
            if len(vars_list) > 3:
                print(f"    ... 还有{len(vars_list)-3}个")
    
    # 4. 年龄分布分析
    print(f"\n📈 年龄分布分析:")
    age_data = demo['crosstabs']['age']['count']
    normal_ages = {k: v for k, v in age_data.items() if k.isdigit() and 18 <= int(k) <= 80}
    if normal_ages:
        ages = sorted([int(k) for k in normal_ages.keys()])
        print(f"  正常年龄范围: {min(ages)}-{max(ages)}岁")
        print(f"  年龄组数: {len(ages)}个")
        
        # 找出糖尿病最多的年龄组
        max_dia_age = max(normal_ages.items(), key=lambda x: x[1]['Diabetic'])
        print(f"  糖尿病最多年龄组: {max_dia_age[0]}岁 ({max_dia_age[1]['Diabetic']}人)")
    
    # 5. 隐私关注示例分析
    print(f"\n🔍 隐私关注示例 (CancerTrustDoctor):")
    if 'CancerTrustDoctor' in privacy['analysis']:
        trust_data = privacy['analysis']['CancerTrustDoctor']['count']
        total_dia = sum(v['Diabetic'] for v in trust_data.values())
        total_non = sum(v['Non-Diabetic'] for v in trust_data.values())
        
        print("  信任度分布:")
        for level, counts in trust_data.items():
            dia_pct = counts['Diabetic']/total_dia*100 if total_dia > 0 else 0
            non_pct = counts['Non-Diabetic']/total_non*100 if total_non > 0 else 0
            print(f"    {level}:")
            print(f"      糖尿病: {counts['Diabetic']}人 ({dia_pct:.1f}%)")
            print(f"      非糖尿病: {counts['Non-Diabetic']}人 ({non_pct:.1f}%)")

if __name__ == "__main__":
    explore_json_data()
