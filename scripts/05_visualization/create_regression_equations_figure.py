#!/usr/bin/env python3
"""
Create Regression Equations Figure for HINTS 7 Diabetes Privacy Study

This script creates a high-quality figure showing the main regression model and 
interaction model equations, formatted for A4 paper compatibility.

Author: AI Assistant
Date: 2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

def create_regression_equations_figure():
    """Create regression equations figure compatible with A4 paper."""
    print("📊 Creating Regression Equations Figure...")
    
    # A4 size in inches (8.27 x 11.69 inches, but we'll use landscape for equations)
    # Using portrait orientation: 8.27 x 11.69 inches
    fig_width = 8.27
    fig_height = 11.69
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title
    fig.suptitle('Regression Equations: HINTS 7 Diabetes Privacy Study', 
                 fontsize=18, fontweight='bold', y=0.97)
    
    # Starting position
    y_start = 0.92
    y_current = y_start
    section_spacing = 0.12
    line_spacing = 0.035
    
    # Colors
    colors = {
        'title': '#2E86AB',
        'equation': '#212529',
        'background': '#F8F9FA'
    }
    
    # 1. MAIN REGRESSION MODEL
    # Section title
    title_box = FancyBboxPatch((0.05, y_current - 0.03), 0.9, 0.04,
                              boxstyle="round,pad=0.01",
                              facecolor=colors['title'], edgecolor='black', linewidth=2)
    ax.add_patch(title_box)
    ax.text(0.5, y_current - 0.01, '1. Main Regression Model', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white', transform=ax.transAxes)
    y_current -= 0.06
    
    # Model specification label
    ax.text(0.1, y_current, 'Model Specification:', ha='left', va='center',
            fontsize=13, fontweight='bold', color='black', transform=ax.transAxes)
    y_current -= 0.04
    
    # Equation - each term on a new line
    equation_terms = [
        'Y = β₀',
        '+ β₁(diabetic)',
        '+ β₂(privacy_caution_index)',
        '+ β₃(age_continuous)',
        '+ β₄(education_numeric)',
        '+ β₅(region_numeric)',
        '+ β₆(urban)',
        '+ β₇(has_insurance)',
        '+ β₈(male)',
        '+ ε'
    ]
    
    for term in equation_terms:
        ax.text(0.15, y_current, term, ha='left', va='center',
                fontsize=12, color=colors['equation'], 
                family='serif', transform=ax.transAxes)
        y_current -= line_spacing
    
    y_current -= 0.02
    
    # Variable definitions
    ax.text(0.1, y_current, 'Where:', ha='left', va='center',
            fontsize=12, fontweight='bold', color='black', transform=ax.transAxes)
    y_current -= 0.03
    
    definitions = [
        'Y = WillingShareData_HCP2 (Data sharing willingness, binary: 0/1)',
        'diabetic = Diabetes status (0 = No diabetes, 1 = Has diabetes)',
        'privacy_caution_index = Privacy caution index (0-1 scale, continuous)',
        'age_continuous = Age in years (continuous)',
        'education_numeric = Education level (1-6, ordinal)',
        'region_numeric = Census region (1-4, categorical)',
        'urban = Urban/rural status (0 = Rural, 1 = Urban)',
        'has_insurance = Health insurance status (0 = No, 1 = Yes)',
        'male = Gender indicator (0 = Female, 1 = Male)',
        'β₀ = Intercept; β₁ to β₈ = Regression coefficients; ε = Error term'
    ]
    
    for defn in definitions:
        ax.text(0.15, y_current, defn, ha='left', va='center',
                fontsize=10, color='black', transform=ax.transAxes)
        y_current -= 0.025
    
    y_current -= 0.02
    
    # Key results
    ax.text(0.1, y_current, 'Key Results:', ha='left', va='center',
            fontsize=12, fontweight='bold', color='black', transform=ax.transAxes)
    y_current -= 0.03
    
    results = [
        'Sample Size: 2,421 observations',
        'R² = 0.1736',
        'Diabetes Effect (β₁): 0.0278 (p = 0.1608, not significant)',
        'Privacy Effect (β₂): -2.3159 (p < 0.001, highly significant)',
        'Age Effect (β₃): 0.0024 (p < 0.001, highly significant)'
    ]
    
    for result in results:
        ax.text(0.15, y_current, result, ha='left', va='center',
                fontsize=10, color='black', transform=ax.transAxes)
        y_current -= 0.025
    
    y_current -= section_spacing
    
    # 2. INTERACTION MODEL
    # Section title
    title_box2 = FancyBboxPatch((0.05, y_current - 0.03), 0.9, 0.04,
                               boxstyle="round,pad=0.01",
                               facecolor=colors['title'], edgecolor='black', linewidth=2)
    ax.add_patch(title_box2)
    ax.text(0.5, y_current - 0.01, '2. Interaction Model', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white', transform=ax.transAxes)
    y_current -= 0.06
    
    # Model specification label
    ax.text(0.1, y_current, 'Model Specification:', ha='left', va='center',
            fontsize=13, fontweight='bold', color='black', transform=ax.transAxes)
    y_current -= 0.04
    
    # Equation - each term on a new line
    equation_terms2 = [
        'Y = β₀',
        '+ β₁(diabetic)',
        '+ β₂(privacy_caution_index)',
        '+ β₃(diabetic × privacy_caution_index)',
        '+ β₄(age_continuous)',
        '+ β₅(education_numeric)',
        '+ β₆(region_numeric)',
        '+ β₇(urban)',
        '+ β₈(has_insurance)',
        '+ β₉(male)',
        '+ ε'
    ]
    
    for term in equation_terms2:
        ax.text(0.15, y_current, term, ha='left', va='center',
                fontsize=12, color=colors['equation'], 
                family='serif', transform=ax.transAxes)
        y_current -= line_spacing
    
    y_current -= 0.02
    
    # Variable definitions
    ax.text(0.1, y_current, 'Where:', ha='left', va='center',
            fontsize=12, fontweight='bold', color='black', transform=ax.transAxes)
    y_current -= 0.03
    
    definitions2 = [
        'Y = WillingShareData_HCP2 (Data sharing willingness, binary: 0/1)',
        'diabetic = Diabetes status (0 = No diabetes, 1 = Has diabetes)',
        'privacy_caution_index = Privacy caution index (0-1 scale, continuous)',
        'diabetic × privacy_caution_index = Interaction term (moderation effect)',
        'age_continuous = Age in years (continuous)',
        'education_numeric = Education level (1-6, ordinal)',
        'region_numeric = Census region (1-4, categorical)',
        'urban = Urban/rural status (0 = Rural, 1 = Urban)',
        'has_insurance = Health insurance status (0 = No, 1 = Yes)',
        'male = Gender indicator (0 = Female, 1 = Male)',
        'β₀ = Intercept; β₁ to β₉ = Regression coefficients; ε = Error term'
    ]
    
    for defn in definitions2:
        ax.text(0.15, y_current, defn, ha='left', va='center',
                fontsize=10, color='black', transform=ax.transAxes)
        y_current -= 0.025
    
    y_current -= 0.02
    
    # Key results
    ax.text(0.1, y_current, 'Key Results:', ha='left', va='center',
            fontsize=12, fontweight='bold', color='black', transform=ax.transAxes)
    y_current -= 0.03
    
    results2 = [
        'Sample Size: 2,421 observations',
        'R² = 0.1753',
        'Diabetes Effect (β₁): -0.1712 (p = 0.0810, marginally significant)',
        'Privacy Effect (β₂): -2.4409 (p < 0.001, highly significant)',
        'Interaction Effect (β₃): 0.4896 (p = 0.0383, significant)',
        'Age Effect (β₄): 0.0023 (p < 0.001, highly significant)'
    ]
    
    for result in results2:
        ax.text(0.15, y_current, result, ha='left', va='center',
                fontsize=10, color='black', transform=ax.transAxes)
        y_current -= 0.025
    
    y_current -= section_spacing
    
    # Model comparison
    comp_box = FancyBboxPatch((0.05, y_current - 0.02), 0.9, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=colors['background'], edgecolor='black', linewidth=1.5)
    ax.add_patch(comp_box)
    
    ax.text(0.1, y_current, 'Model Comparison:', ha='left', va='center',
            fontsize=12, fontweight='bold', color='black', transform=ax.transAxes)
    y_current -= 0.03
    
    comparison = [
        '• Interaction model adds: diabetic × privacy_caution_index',
        '• R² increases from 0.1736 to 0.1753',
        '• Interaction effect is significant (p = 0.0383)',
        '• Diabetes moderates the privacy-sharing relationship'
    ]
    
    for comp in comparison:
        ax.text(0.15, y_current, comp, ha='left', va='center',
                fontsize=10, color='black', transform=ax.transAxes)
        y_current -= 0.025
    
    # Notes at bottom
    notes_y = 0.05
    notes_box = FancyBboxPatch((0.05, notes_y - 0.02), 0.9, 0.04,
                              boxstyle="round,pad=0.01",
                              facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(notes_box)
    
    notes_text = 'Notes: Both models use weighted least squares regression. Sample: HINTS 7 Public Dataset (2022), 2,421 valid observations.'
    ax.text(0.5, notes_y, notes_text, ha='center', va='center',
            fontsize=9, style='italic', color='black', transform=ax.transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    repo_root = Path(__file__).parent.parent.parent
    output_path = repo_root / 'figures' / 'regression_equations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Regression Equations Figure saved: {output_path}")
    
    # Also save as PDF (A4 compatible)
    pdf_path = repo_root / 'figures' / 'pdf_versions' / 'regression_equations.pdf'
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', 
                format='pdf')
    print(f"✅ PDF version saved (A4 compatible): {pdf_path}")
    
    plt.close()
    return output_path

if __name__ == "__main__":
    create_regression_equations_figure()
