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
    
    # Set up the figure with large size
    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
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
    
    # Model 1: Main Regression
    ax.text(1, 11, '1. Main Regression Model', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))
    
    # Variables
    diabetic_box = FancyBboxPatch((0.5, 10), 1, 0.8, boxstyle="round,pad=0.1", 
                                 facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(diabetic_box)
    ax.text(1, 10.4, 'Diabetic\n(0/1)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    privacy_box = FancyBboxPatch((2, 10), 1, 0.8, boxstyle="round,pad=0.1", 
                                facecolor=colors['privacy'], alpha=0.7)
    ax.add_patch(privacy_box)
    ax.text(2.5, 10.4, 'Privacy\nIndex', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    demo_box = FancyBboxPatch((3.5, 10), 1, 0.8, boxstyle="round,pad=0.1", 
                              facecolor=colors['demographics'], alpha=0.7)
    ax.add_patch(demo_box)
    ax.text(4, 10.4, 'Demographics', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    outcome_box = FancyBboxPatch((5.5, 10), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box)
    ax.text(6.25, 10.4, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Arrows
    ax.arrow(1.5, 10.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    ax.arrow(3, 10.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    ax.arrow(4.5, 10.4, 0.8, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    
    # Model 2: Interaction Model
    ax.text(1, 9, '2. Interaction Model', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))
    
    # Variables
    diabetic_box2 = FancyBboxPatch((0.5, 8), 1, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(diabetic_box2)
    ax.text(1, 8.4, 'Diabetic\n(0/1)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    privacy_box2 = FancyBboxPatch((2, 8), 1, 0.8, boxstyle="round,pad=0.1", 
                                 facecolor=colors['privacy'], alpha=0.7)
    ax.add_patch(privacy_box2)
    ax.text(2.5, 8.4, 'Privacy\nIndex', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    interaction_box = FancyBboxPatch((3.5, 8), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                    facecolor='#6C757D', alpha=0.7)
    ax.add_patch(interaction_box)
    ax.text(4.25, 8.4, 'Diabetic ×\nPrivacy', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    outcome_box2 = FancyBboxPatch((5.5, 8), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box2)
    ax.text(6.25, 8.4, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Arrows
    ax.arrow(1.5, 8.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    ax.arrow(3, 8.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    ax.arrow(5, 8.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    
    # Model 3: Mediation Model
    ax.text(1, 7, '3. Mediation Model', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))
    
    # Step 1
    diabetic_box3 = FancyBboxPatch((0.5, 6), 1, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(diabetic_box3)
    ax.text(1, 6.4, 'Diabetic\n(0/1)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    privacy_box3 = FancyBboxPatch((2, 6), 1, 0.8, boxstyle="round,pad=0.1", 
                                 facecolor=colors['privacy'], alpha=0.7)
    ax.add_patch(privacy_box3)
    ax.text(2.5, 6.4, 'Privacy\nIndex', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    ax.arrow(1.5, 6.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    ax.text(1.75, 6.7, 'Step 1', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Step 2
    outcome_box3 = FancyBboxPatch((3.5, 6), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box3)
    ax.text(4.25, 6.4, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    ax.arrow(3, 6.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    ax.text(3.25, 6.7, 'Step 2', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Direct effect
    ax.arrow(1, 5.8, 2.8, 0, head_width=0.05, head_length=0.1, fc='red', ec='red', linestyle='--')
    ax.text(2.4, 5.6, 'Direct Effect', ha='center', va='center', fontsize=10, fontweight='bold', color='red')
    
    # Model 4: Instrumental Variables
    ax.text(1, 4.5, '4. Instrumental Variables', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))
    
    # Instrument
    instrument_box = FancyBboxPatch((0.5, 3.5), 1, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor=colors['instrument'], alpha=0.7)
    ax.add_patch(instrument_box)
    ax.text(1, 3.9, 'Age > 65\n(Instrument)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # First stage
    diabetic_box4 = FancyBboxPatch((2, 3.5), 1, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(diabetic_box4)
    ax.text(2.5, 3.9, 'Diabetic\n(Predicted)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Second stage
    outcome_box4 = FancyBboxPatch((3.5, 3.5), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box4)
    ax.text(4.25, 3.9, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Arrows
    ax.arrow(1.5, 3.9, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    ax.text(1.75, 4.2, 'First Stage', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.arrow(3, 3.9, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    ax.text(3.25, 4.2, 'Second Stage', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Model 5: Difference-in-Differences
    ax.text(6, 11, '5. Difference-in-Differences', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))
    
    # Variables
    diabetic_box5 = FancyBboxPatch((5.5, 10), 1, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(diabetic_box5)
    ax.text(6, 10.4, 'Diabetic\n(Treatment)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    time_box = FancyBboxPatch((7, 10), 1, 0.8, boxstyle="round,pad=0.1", 
                              facecolor=colors['time'], alpha=0.7)
    ax.add_patch(time_box)
    ax.text(7.5, 10.4, 'Time\n(Age/Region)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    interaction_box5 = FancyBboxPatch((8.5, 10), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                    facecolor='#6C757D', alpha=0.7)
    ax.add_patch(interaction_box5)
    ax.text(9.25, 10.4, 'Diabetic ×\nTime (DiD)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Arrows
    ax.arrow(6.5, 10.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    ax.arrow(8, 10.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    
    # Model 6: Regression Discontinuity
    ax.text(6, 9, '6. Regression Discontinuity', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))
    
    # Variables
    age_box = FancyBboxPatch((5.5, 8), 1, 0.8, boxstyle="round,pad=0.1", 
                             facecolor=colors['time'], alpha=0.7)
    ax.add_patch(age_box)
    ax.text(6, 8.4, 'Age - 65\n(Running Var)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    treatment_box = FancyBboxPatch((7, 8), 1, 0.8, boxstyle="round,pad=0.1", 
                                   facecolor=colors['diabetic'], alpha=0.7)
    ax.add_patch(treatment_box)
    ax.text(7.5, 8.4, 'Age ≥ 65\n(Treatment)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    outcome_box6 = FancyBboxPatch((8.5, 8), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box6)
    ax.text(9.25, 8.4, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Arrows
    ax.arrow(6.5, 8.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    ax.arrow(8, 8.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    
    # Model 7: Propensity Score Matching
    ax.text(6, 7, '7. Propensity Score Matching', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.8))
    
    # Variables
    demo_box7 = FancyBboxPatch((5.5, 6), 1, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=colors['demographics'], alpha=0.7)
    ax.add_patch(demo_box7)
    ax.text(6, 6.4, 'Demographics\n(Covariates)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    ps_box = FancyBboxPatch((7, 6), 1, 0.8, boxstyle="round,pad=0.1", 
                            facecolor=colors['privacy'], alpha=0.7)
    ax.add_patch(ps_box)
    ax.text(7.5, 6.4, 'Propensity\nScore', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    outcome_box7 = FancyBboxPatch((8.5, 6), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=colors['outcome'], alpha=0.7)
    ax.add_patch(outcome_box7)
    ax.text(9.25, 6.4, 'Data Sharing\nWillingness', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Arrows
    ax.arrow(6.5, 6.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    ax.arrow(8, 6.4, 0.3, 0, head_width=0.1, head_length=0.1, fc=colors['text'], ec=colors['text'])
    
    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['diabetic'], alpha=0.7, label='Diabetes Status'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['privacy'], alpha=0.7, label='Privacy Index'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['outcome'], alpha=0.7, label='Data Sharing Willingness'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['demographics'], alpha=0.7, label='Demographics'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['instrument'], alpha=0.7, label='Instrument'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['time'], alpha=0.7, label='Time/Running Variable'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#6C757D', alpha=0.7, label='Interaction Term')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
              fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Add explanatory text
    ax.text(0.5, 2, 'Model Logic Summary:', fontsize=14, fontweight='bold')
    ax.text(0.5, 1.5, '• Direct Effect: Diabetes → Data Sharing Willingness', fontsize=12)
    ax.text(0.5, 1.2, '• Indirect Effect: Diabetes → Privacy Index → Data Sharing Willingness', fontsize=12)
    ax.text(0.5, 0.9, '• Interaction Effect: Diabetes × Privacy Index → Data Sharing Willingness', fontsize=12)
    ax.text(0.5, 0.6, '• Causal Inference: Controls for selection bias and endogeneity', fontsize=12)
    ax.text(0.5, 0.3, '• Heterogeneity: Effects vary across groups and contexts', fontsize=12)
    
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
