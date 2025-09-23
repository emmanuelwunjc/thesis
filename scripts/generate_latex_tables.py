#!/usr/bin/env python3
"""
Generate LaTeX Tables for Academic Papers

This script generates properly formatted LaTeX tables from regression results
suitable for academic papers and presentations.

Author: AI Assistant
Date: 2024-09-23
"""

import json
import pandas as pd
from pathlib import Path

def load_regression_results():
    """Load regression results from JSON file."""
    results_path = Path(__file__).parent.parent / "analysis" / "regression_results.json"
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

def generate_main_table_latex(results):
    """Generate LaTeX table for main regression model."""
    main_data = results['main_regression']
    
    # Extract data
    variables = list(main_data['coefficients'].keys())
    coefs = [main_data['coefficients'][var] for var in variables]
    ses = [main_data['std_errors'][var] for var in variables]
    p_vals = [main_data['p_values'][var] for var in variables]
    
    # Create significance stars
    def get_significance_stars(p_val):
        if p_val < 0.001:
            return "^{***}"
        elif p_val < 0.01:
            return "^{**}"
        elif p_val < 0.05:
            return "^{*}"
        elif p_val < 0.1:
            return "^{\\dagger}"
        else:
            return ""
    
    significance = [get_significance_stars(p) for p in p_vals]
    
    # Variable names mapping
    var_names = {
        'const': 'Constant',
        'diabetic': 'Diabetes Status',
        'privacy_caution_index': 'Privacy Caution Index',
        'age': 'Age',
        'education_numeric': 'Education Level'
    }
    
    # Generate LaTeX table
    latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Main Regression Results: Data Sharing Willingness}
\\label{tab:main_regression}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Variable} & \\textbf{Coefficient} & \\textbf{Std. Error} & \\textbf{p-value} \\\\
\\midrule
"""
    
    for var, coef, se, p_val, sig in zip(variables, coefs, ses, p_vals, significance):
        var_name = var_names.get(var, var)
        latex_table += f"{var_name} & {coef:.4f}{sig} & {se:.4f} & {p_val:.4f} \\\\\n"
    
    latex_table += f"""\\midrule
Observations & \\multicolumn{{3}}{{c}}{{{main_data['n_obs']:,}}} \\\\
R$^2$ & \\multicolumn{{3}}{{c}}{{{main_data['r_squared']:.4f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item Note: Dependent variable is willingness to share data with healthcare providers (0/1).
\\item Significance levels: $^{{***}}$ p<0.001, $^{{**}}$ p<0.01, $^{{*}}$ p<0.05, $^{{\\dagger}}$ p<0.1.
\\item Model estimated using weighted least squares with survey weights.
\\end{{tablenotes}}
\\end{{table}}
"""
    
    return latex_table

def generate_interaction_table_latex(results):
    """Generate LaTeX table for interaction model."""
    interaction_data = results['interaction_regression']
    
    # Extract data
    variables = list(interaction_data['coefficients'].keys())
    coefs = [interaction_data['coefficients'][var] for var in variables]
    ses = [interaction_data['std_errors'][var] for var in variables]
    p_vals = [interaction_data['p_values'][var] for var in variables]
    
    # Create significance stars
    def get_significance_stars(p_val):
        if p_val < 0.001:
            return "^{***}"
        elif p_val < 0.01:
            return "^{**}"
        elif p_val < 0.05:
            return "^{*}"
        elif p_val < 0.1:
            return "^{\\dagger}"
        else:
            return ""
    
    significance = [get_significance_stars(p) for p in p_vals]
    
    # Variable names mapping
    var_names = {
        'const': 'Constant',
        'diabetic': 'Diabetes Status',
        'privacy_caution_index': 'Privacy Caution Index',
        'diabetic_privacy_interaction': 'Diabetes $\\times$ Privacy Index',
        'age': 'Age',
        'education_numeric': 'Education Level'
    }
    
    # Generate LaTeX table
    latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Interaction Model: Diabetes-Privacy Interaction Effects}
\\label{tab:interaction_regression}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Variable} & \\textbf{Coefficient} & \\textbf{Std. Error} & \\textbf{p-value} \\\\
\\midrule
"""
    
    for var, coef, se, p_val, sig in zip(variables, coefs, ses, p_vals, significance):
        var_name = var_names.get(var, var)
        latex_table += f"{var_name} & {coef:.4f}{sig} & {se:.4f} & {p_val:.4f} \\\\\n"
    
    latex_table += f"""\\midrule
Observations & \\multicolumn{{3}}{{c}}{{{interaction_data['n_obs']:,}}} \\\\
R$^2$ & \\multicolumn{{3}}{{c}}{{{interaction_data['r_squared']:.4f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item Note: Dependent variable is willingness to share data with healthcare providers (0/1).
\\item Significance levels: $^{{***}}$ p<0.001, $^{{**}}$ p<0.01, $^{{*}}$ p<0.05, $^{{\\dagger}}$ p<0.1.
\\item Model includes interaction between diabetes status and privacy caution index.
\\item Model estimated using weighted least squares with survey weights.
\\end{{tablenotes}}
\\end{{table}}
"""
    
    return latex_table

def generate_subgroup_table_latex(results):
    """Generate LaTeX table for subgroup analysis."""
    subgroup_data = results['subgroup_analyses']
    
    # Extract data
    age_groups = []
    diabetic_effects = []
    privacy_effects = []
    sample_sizes = []
    
    for age_group, data in subgroup_data.items():
        age_groups.append(age_group.replace('age_', ''))
        diabetic_effects.append(data['coefficients']['diabetic'])
        privacy_effects.append(data['coefficients']['privacy_caution_index'])
        sample_sizes.append(data['n_obs'])
    
    # Generate LaTeX table
    latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Age Group Subgroup Analysis}
\\label{tab:subgroup_analysis}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Age Group} & \\textbf{Sample Size} & \\textbf{Diabetic Effect} & \\textbf{Privacy Effect} \\\\
\\midrule
"""
    
    for age_group, n, diab_coef, priv_coef in zip(age_groups, sample_sizes, diabetic_effects, privacy_effects):
        latex_table += f"{age_group} & {n:,} & {diab_coef:.4f} & {priv_coef:.4f} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: Subgroup regressions estimated separately for each age group.
\\item Diabetic Effect: Coefficient for diabetes status variable.
\\item Privacy Effect: Coefficient for privacy caution index variable.
\\item All models include age and education controls.
\\end{tablenotes}
\\end{table}
"""
    
    return latex_table

def main():
    """Generate all LaTeX tables."""
    try:
        results = load_regression_results()
        
        # Generate tables
        main_table = generate_main_table_latex(results)
        interaction_table = generate_interaction_table_latex(results)
        subgroup_table = generate_subgroup_table_latex(results)
        
        # Save to file
        output_path = Path(__file__).parent.parent / "analysis" / "regression_tables_latex.tex"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("% LaTeX Tables for HINTS 7 Diabetes Privacy Regression Analysis\n")
            f.write("% Generated automatically from regression results\n")
            f.write("% Date: 2024-09-23\n\n")
            f.write("% Required packages:\n")
            f.write("% \\usepackage{booktabs}\n")
            f.write("% \\usepackage{threeparttable}\n\n")
            f.write(main_table)
            f.write("\n\n")
            f.write(interaction_table)
            f.write("\n\n")
            f.write(subgroup_table)
        
        print(f"✅ LaTeX tables saved to: {output_path}")
        
        # Also print to console
        print("\n" + "="*80)
        print("LATEX TABLES FOR ACADEMIC PAPERS")
        print("="*80)
        print(main_table)
        print(interaction_table)
        print(subgroup_table)
        
    except Exception as e:
        print(f"❌ Error generating LaTeX tables: {e}")

if __name__ == "__main__":
    main()
