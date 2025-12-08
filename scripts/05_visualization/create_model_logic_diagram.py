#!/usr/bin/env python3
"""
Create Model Logic Diagram for HINTS 7 Diabetes Privacy Study

This script creates a comprehensive diagram showing the variable relationships
and influence logic for all regression and causal inference models.

Author: AI Assistant
Date: 2024-09-23
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path

def create_model_logic_diagram():
    """Create a comprehensive model logic diagram."""
    print("📊 Creating Model Logic Diagram...")
    
    # Set up the figure with much larger size and better spacing
    fig, ax = plt.subplots(1, 1, figsize=(32, 20))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Set title
    fig.suptitle('HINTS 7 Diabetes Privacy Study: Model Logic and Variable Relationships', 
                 fontsize=24, fontweight='bold', y=0.95)
    
    # Define colors
    colors = {
        'diabetic': '#2E86AB',      # Blue
        'privacy': '#A23B72',       # Rose
        'outcome': '#F18F01',       # Orange
        'demographics': '#28A745',   # Green
        'instrument': '#FFC107',    # Yellow
        'time': '#17A2B8',         # Cyan
        'background': '#F8F9FA',    # Light gray
        'text': '#212529'           # Dark gray
    }
    
    # Model 1: Main Regression - Top Left
    ax.text(1, 15, '1. Main Regression Model', fontsize=18, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['background'], alpha=0.8))
    
    # Variables with better spacing
    diabetic_box = FancyBboxPatch((0.5, 13.5), 1.2, 1, boxstyle="round,pad=0.2", 
                                 facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(diabetic_box)
    ax.text(1.1, 14, 'Diabetic\n(0/1)', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    privacy_box = FancyBboxPatch((2.2, 13.5), 1.2, 1, boxstyle="round,pad=0.2", 
                                facecolor=colors['privacy'], alpha=0.7)
    ax.add_patch(privacy_box)
    ax.text(2.8, 14, 'Privacy\nIndex', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    demo_box = FancyBboxPatch((3.9, 13.5), 1.2, 1, boxstyle="round,pad=0.2", 
                              facecolor=colors['demographics'], alpha=0.7)
    ax.add_patch(demo_box)
    ax.text(4.5, 14, 'Demographics', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    outcome_box = FancyBboxPatch((5.6, 13.5), 1.8, 1, boxstyle="round,pad=0.2", 
                                facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box)
    ax.text(6.5, 14, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Arrows with better spacing
    ax.arrow(1.7, 14, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    ax.arrow(3.4, 14, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    ax.arrow(5.1, 14, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    
    # Model 2: Interaction Model - Second Row Left
    ax.text(1, 12, '2. Interaction Model', fontsize=18, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['background'], alpha=0.8))
    
    # Variables with better spacing
    diabetic_box2 = FancyBboxPatch((0.5, 10.5), 1.2, 1, boxstyle="round,pad=0.2", 
                                  facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(diabetic_box2)
    ax.text(1.1, 11, 'Diabetic\n(0/1)', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    privacy_box2 = FancyBboxPatch((2.2, 10.5), 1.2, 1, boxstyle="round,pad=0.2", 
                                 facecolor=colors['privacy'], alpha=0.7)
    ax.add_patch(privacy_box2)
    ax.text(2.8, 11, 'Privacy\nIndex', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    interaction_box = FancyBboxPatch((3.9, 10.5), 1.8, 1, boxstyle="round,pad=0.2", 
                                    facecolor='#6C757D', alpha=0.7)
    ax.add_patch(interaction_box)
    ax.text(4.8, 11, 'Diabetic ×\nPrivacy', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    outcome_box2 = FancyBboxPatch((6.2, 10.5), 1.8, 1, boxstyle="round,pad=0.2", 
                                  facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box2)
    ax.text(7.1, 11, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Arrows with better spacing
    ax.arrow(1.7, 11, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    ax.arrow(3.4, 11, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    ax.arrow(5.7, 11, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    
    # Model 3: Mediation Model - Third Row Left
    ax.text(1, 9, '3. Mediation Model', fontsize=18, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['background'], alpha=0.8))
    
    # Step 1
    diabetic_box3 = FancyBboxPatch((0.5, 7.5), 1.2, 1, boxstyle="round,pad=0.2", 
                                  facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(diabetic_box3)
    ax.text(1.1, 8, 'Diabetic\n(0/1)', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    privacy_box3 = FancyBboxPatch((2.2, 7.5), 1.2, 1, boxstyle="round,pad=0.2", 
                                 facecolor=colors['privacy'], alpha=0.7)
    ax.add_patch(privacy_box3)
    ax.text(2.8, 8, 'Privacy\nIndex', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    ax.arrow(1.7, 8, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    ax.text(1.9, 8.3, 'Step 1', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Step 2
    outcome_box3 = FancyBboxPatch((3.9, 7.5), 1.8, 1, boxstyle="round,pad=0.2", 
                                  facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box3)
    ax.text(4.8, 8, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    ax.arrow(3.4, 8, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    ax.text(3.6, 8.3, 'Step 2', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Direct effect
    ax.arrow(1.1, 7.2, 3.2, 0, head_width=0.08, head_length=0.15, fc='red', ec='red', linestyle='--', linewidth=2)
    ax.text(2.6, 7, 'Direct Effect', ha='center', va='center', fontsize=12, fontweight='bold', color='red')
    
    # Model 4: Instrumental Variables - Fourth Row Left
    ax.text(1, 6, '4. Instrumental Variables', fontsize=18, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['background'], alpha=0.8))
    
    # Instrument
    instrument_box = FancyBboxPatch((0.5, 4.5), 1.2, 1, boxstyle="round,pad=0.2", 
                                   facecolor=colors['instrument'], alpha=0.7)
    ax.add_patch(instrument_box)
    ax.text(1.1, 5, 'Age > 65\n(Instrument)', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # First stage
    diabetic_box4 = FancyBboxPatch((2.2, 4.5), 1.2, 1, boxstyle="round,pad=0.2", 
                                  facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(diabetic_box4)
    ax.text(2.8, 5, 'Diabetic\n(Predicted)', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Second stage
    outcome_box4 = FancyBboxPatch((3.9, 4.5), 1.8, 1, boxstyle="round,pad=0.2", 
                                  facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box4)
    ax.text(4.8, 5, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Arrows
    ax.arrow(1.7, 5, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    ax.text(1.9, 5.3, 'First Stage', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.arrow(3.4, 5, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    ax.text(3.6, 5.3, 'Second Stage', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Model 5: Difference-in-Differences - Top Right
    ax.text(8, 15, '5. Difference-in-Differences', fontsize=18, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['background'], alpha=0.8))
    
    # Variables with better spacing
    diabetic_box5 = FancyBboxPatch((7.5, 13.5), 1.2, 1, boxstyle="round,pad=0.2", 
                                  facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(diabetic_box5)
    ax.text(8.1, 14, 'Diabetic\n(Treatment)', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    time_box = FancyBboxPatch((9.2, 13.5), 1.2, 1, boxstyle="round,pad=0.2", 
                              facecolor=colors['time'], alpha=0.7)
    ax.add_patch(time_box)
    ax.text(9.8, 14, 'Time\n(Age/Region)', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    interaction_box5 = FancyBboxPatch((10.9, 13.5), 1.8, 1, boxstyle="round,pad=0.2", 
                                    facecolor='#6C757D', alpha=0.7)
    ax.add_patch(interaction_box5)
    ax.text(11.8, 14, 'Diabetic ×\nTime (DiD)', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Arrows with better spacing
    ax.arrow(8.7, 14, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    ax.arrow(10.4, 14, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    
    # Model 6: Regression Discontinuity - Second Row Right
    ax.text(8, 12, '6. Regression Discontinuity', fontsize=18, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['background'], alpha=0.8))
    
    # Variables with better spacing
    age_box = FancyBboxPatch((7.5, 10.5), 1.2, 1, boxstyle="round,pad=0.2", 
                             facecolor=colors['time'], alpha=0.7)
    ax.add_patch(age_box)
    ax.text(8.1, 11, 'Age - 65\n(Running Var)', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    treatment_box = FancyBboxPatch((9.2, 10.5), 1.2, 1, boxstyle="round,pad=0.2", 
                                   facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(treatment_box)
    ax.text(9.8, 11, 'Age ≥ 65\n(Treatment)', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    outcome_box6 = FancyBboxPatch((10.9, 10.5), 1.8, 1, boxstyle="round,pad=0.2", 
                                  facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box6)
    ax.text(11.8, 11, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Arrows with better spacing
    ax.arrow(8.7, 11, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    ax.arrow(10.4, 11, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    
    # Model 7: Propensity Score Matching - Third Row Right
    ax.text(8, 9, '7. Propensity Score Matching', fontsize=18, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors['background'], alpha=0.8))
    
    # Variables with better spacing
    demo_box7 = FancyBboxPatch((7.5, 7.5), 1.2, 1, boxstyle="round,pad=0.2", 
                               facecolor=colors['demographics'], alpha=0.7)
    ax.add_patch(demo_box7)
    ax.text(8.1, 8, 'Demographics\n(Covariates)', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    ps_box = FancyBboxPatch((9.2, 7.5), 1.2, 1, boxstyle="round,pad=0.2", 
                            facecolor=colors['privacy'], alpha=0.7)
    ax.add_patch(ps_box)
    ax.text(9.8, 8, 'Propensity\nScore', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    outcome_box7 = FancyBboxPatch((10.9, 7.5), 1.8, 1, boxstyle="round,pad=0.2", 
                                  facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box7)
    ax.text(11.8, 8, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Arrows with better spacing
    ax.arrow(8.7, 8, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    ax.arrow(10.4, 8, 0.3, 0, head_width=0.15, head_length=0.15, fc=colors['text'], ec=colors['text'], linewidth=2)
    
    # Legend - Bottom right
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['diabetic'], alpha=0.7, label='Diabetes Status'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['privacy'], alpha=0.7, label='Privacy Index'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['outcome'], alpha=0.7, label='Data Sharing Willingness'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['demographics'], alpha=0.7, label='Demographics'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['instrument'], alpha=0.7, label='Instrument'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['time'], alpha=0.7, label='Time/Running Variable'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#6C757D', alpha=0.7, label='Interaction Term')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02), 
              fontsize=14, frameon=True, fancybox=True, shadow=True)
    
    # Add explanatory text - Bottom left
    ax.text(0.5, 3, 'Model Logic Summary:', fontsize=16, fontweight='bold')
    ax.text(0.5, 2.5, '• Direct Effect: Diabetes → Data Sharing Willingness', fontsize=14)
    ax.text(0.5, 2.1, '• Indirect Effect: Diabetes → Privacy Index → Data Sharing Willingness', fontsize=14)
    ax.text(0.5, 1.7, '• Interaction Effect: Diabetes × Privacy Index → Data Sharing Willingness', fontsize=14)
    ax.text(0.5, 1.3, '• Causal Inference: Controls for selection bias and endogeneity', fontsize=14)
    ax.text(0.5, 0.9, '• Heterogeneity: Effects vary across groups and contexts', fontsize=14)
    ax.text(0.5, 0.5, '• Multiple Methods: Robustness checks across different approaches', fontsize=14)
    
    # Save the plot
    output_path = Path(__file__).parent.parent / "figures" / "model_logic_diagram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', 
                pad_inches=0.5, format='png')
    pdf_path = Path(__file__).parent.parent / "figures" / "model_logic_diagram.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none',
                pad_inches=0.5, format='pdf')
    
    plt.close(fig)
    
    print(f"✅ Model logic diagram saved to {output_path}")
    print(f"📄 PDF version saved to {pdf_path}")

if __name__ == "__main__":
    create_model_logic_diagram()
