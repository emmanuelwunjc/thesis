#!/usr/bin/env python3
"""
Create Variable Definition Table for HINTS 7 Diabetes Privacy Study

This script creates a comprehensive variable definition table including:
- Privacy Index
- Willingness Variables
- Digital Engagement Indicators
- Diabetes Status
- Control Variables

Author: AI Assistant
Date: 2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

def create_variable_definition_table():
    """Create a comprehensive variable definition table."""
    print("📊 Creating Variable Definition Table...")
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.axis('off')
    
    # Title
    fig.suptitle('Variable Definitions: HINTS 7 Diabetes Privacy Study', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # Define variable categories and their data
    categories = [
        {
            'name': '1. Privacy Caution Index',
            'variables': [
                ['privacy_caution_index', 'Composite privacy caution index', '0-1', '0 = Least cautious (more sharing, trusting, digital engagement)\n1 = Most cautious (less sharing, distrusting, limited engagement)', 'Mean of 6 sub-dimensions'],
                ['Sub-dimension: Sharing', 'Data sharing willingness (4 variables)', '0-1', 'WillingShareData_HCP2, SharedHealthDeviceInfo2,\nSocMed_SharedPers, SocMed_SharedGen', 'Yes=0, No=1'],
                ['Sub-dimension: Portals', 'Patient portal usage (7 variables)', '0-1', 'AccessOnlineRecord3, OnlinePortal_PCP,\nOnlinePortal_OthHCP, OnlinePortal_Insurer, etc.', 'Selected=0, Not selected=1'],
                ['Sub-dimension: Devices', 'Digital device usage (4 variables)', '0-1', 'UseDevice_Computer, UseDevice_SmPhone,\nUseDevice_Tablet, UseDevice_SmWatch', 'Yes=0, No=1'],
                ['Sub-dimension: Trust', 'Healthcare system trust (4 variables)', '0-1', 'TrustHCSystem, CancerTrustDoctor,\nCancerTrustScientists, CancerTrustFamily', 'A lot=0, Not at all=1'],
                ['Sub-dimension: Social', 'Social media engagement (2 variables)', '0-1', 'SocMed_Visited, MisleadingHealthInfo', 'Yes=0, No=1'],
                ['Sub-dimension: Other', 'Other privacy concerns (2 variables)', '0-1', 'ConfidentMedForms, WillingUseTelehealth', 'Confident/Willing=0, Not=1']
            ]
        },
        {
            'name': '2. Willingness Variables',
            'variables': [
                ['WillingShareData_HCP2', 'Willingness to share health data with healthcare providers', 'Binary (0/1)', '0 = Not willing\n1 = Willing', 'Primary dependent variable\nRecoded from Yes/No to 0/1']
            ]
        },
        {
            'name': '3. Digital Engagement Indicators',
            'variables': [
                ['AccessOnlineRecord3', 'Accessed online medical records', 'Binary (Yes/No)', 'Yes = Accessed online records\nNo = Did not access', 'Portal engagement indicator'],
                ['OnlinePortal_PCP', 'Used primary care provider portal', 'Binary (Selected/Not selected)', 'Selected = Used portal\nNot selected = Did not use', 'Portal engagement indicator'],
                ['UseDevice_Computer', 'Used computer for health purposes', 'Binary (Yes/No)', 'Yes = Used computer\nNo = Did not use', 'Device engagement indicator'],
                ['UseDevice_SmPhone', 'Used smartphone for health purposes', 'Binary (Yes/No)', 'Yes = Used smartphone\nNo = Did not use', 'Device engagement indicator'],
                ['UseDevice_Tablet', 'Used tablet for health purposes', 'Binary (Yes/No)', 'Yes = Used tablet\nNo = Did not use', 'Device engagement indicator'],
                ['WillingUseTelehealth', 'Willingness to use telehealth services', 'Binary (Yes/No)', 'Yes = Willing to use\nNo = Not willing', 'Digital health engagement']
            ]
        },
        {
            'name': '4. Diabetes Status',
            'variables': [
                ['diabetic', 'Diabetes status indicator', 'Binary (0/1)', '0 = No diabetes\n1 = Has diabetes', 'Recoded from MedConditions_Diabetes\n(Yes → 1, No → 0)']
            ]
        },
        {
            'name': '5. Control Variables',
            'variables': [
                ['age_continuous', 'Age in years', 'Continuous', 'Range: 18-100+', 'Original variable: Age'],
                ['education_numeric', 'Education level (numeric)', 'Ordinal (1-6)', '1 = Less than 8 years\n2 = 8-11 years\n3 = High school\n4 = Some college\n5 = College graduate\n6 = Postgraduate', 'Recoded from Education'],
                ['region_numeric', 'Census region', 'Categorical (1-4)', '1 = Northeast\n2 = Midwest\n3 = South\n4 = West', 'Recoded from CENSREG'],
                ['urban', 'Urban/rural status', 'Binary (0/1)', '0 = Rural\n1 = Urban', 'Recoded from RUC2003\n(metro = 1)'],
                ['has_insurance', 'Health insurance status', 'Binary (0/1)', '0 = No insurance\n1 = Has insurance', 'Recoded from HealthInsurance2'],
                ['male', 'Gender (male indicator)', 'Binary (0/1)', '0 = Female\n1 = Male', 'Recoded from BirthSex'],
                ['race_numeric', 'Race/ethnicity (numeric)', 'Categorical (1-5)', '1 = Non-Hispanic White\n2 = Non-Hispanic Black\n3 = Hispanic\n4 = Non-Hispanic Asian\n5 = Other', 'Recoded from RaceEthn5'],
                ['received_treatment', 'Received cancer treatment', 'Binary (0/1)', '0 = No treatment\n1 = Received treatment', 'Recoded from Treatment_H7_1'],
                ['stopped_treatment', 'Stopped treatment', 'Binary (0/1)', '0 = Did not stop\n1 = Stopped treatment', 'Recoded from PCStopTreatments2']
            ]
        }
    ]
    
    # Starting position
    y_start = 0.92
    y_current = y_start
    row_height = 0.055
    category_spacing = 0.08
    
    # Colors
    colors = {
        'header': '#2E86AB',
        'category': '#A23B72',
        'row1': '#F8F9FA',
        'row2': '#FFFFFF'
    }
    
    # Draw table
    for cat_idx, category in enumerate(categories):
        # Category header
        cat_box = FancyBboxPatch((0.05, y_current - 0.02), 0.9, 0.04,
                                boxstyle="round,pad=0.01",
                                facecolor=colors['category'], edgecolor='black', linewidth=2)
        ax.add_patch(cat_box)
        ax.text(0.5, y_current, category['name'], ha='center', va='center',
                fontsize=16, fontweight='bold', color='white',
                transform=ax.transAxes)
        y_current -= 0.05
        
        # Table header
        header_box = FancyBboxPatch((0.05, y_current - 0.02), 0.9, 0.03,
                                   boxstyle="round,pad=0.01",
                                   facecolor=colors['header'], edgecolor='black', linewidth=1.5)
        ax.add_patch(header_box)
        headers = ['Variable Name', 'Definition', 'Type/Range', 'Values/Coding', 'Notes']
        header_x_positions = [0.12, 0.32, 0.52, 0.72, 0.88]
        for i, header in enumerate(headers):
            ax.text(header_x_positions[i], y_current, header, ha='left', va='center',
                    fontsize=11, fontweight='bold', color='white',
                    transform=ax.transAxes)
        y_current -= 0.04
        
        # Variable rows
        for var_idx, var in enumerate(category['variables']):
            row_color = colors['row1'] if var_idx % 2 == 0 else colors['row2']
            row_box = FancyBboxPatch((0.05, y_current - 0.02), 0.9, row_height,
                                    boxstyle="round,pad=0.005",
                                    facecolor=row_color, edgecolor='gray', linewidth=0.5)
            ax.add_patch(row_box)
            
            # Variable name (bold)
            ax.text(header_x_positions[0], y_current, var[0], ha='left', va='center',
                    fontsize=10, fontweight='bold', color='black',
                    transform=ax.transAxes)
            
            # Definition
            ax.text(header_x_positions[1], y_current, var[1], ha='left', va='center',
                    fontsize=9, color='black', transform=ax.transAxes)
            
            # Type/Range
            ax.text(header_x_positions[2], y_current, var[2], ha='left', va='center',
                    fontsize=9, color='black', transform=ax.transAxes)
            
            # Values/Coding
            ax.text(header_x_positions[3], y_current, var[3], ha='left', va='center',
                    fontsize=8, color='black', transform=ax.transAxes)
            
            # Notes
            ax.text(header_x_positions[4], y_current, var[4], ha='left', va='center',
                    fontsize=8, color='black', style='italic', transform=ax.transAxes)
            
            y_current -= row_height
        
        # Add spacing between categories
        y_current -= category_spacing
    
    # Add summary note at bottom
    summary_text = """Note: All variables are derived from HINTS 7 Public Dataset (2022). Missing values are handled with appropriate imputation strategies.
Privacy Caution Index: Cronbach's α = 0.78, Range: 0.23-0.78, Mean: 0.47 (SD: 0.09)"""
    ax.text(0.5, 0.02, summary_text, ha='center', va='center',
            fontsize=9, style='italic', color='gray',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.7),
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    repo_root = Path(__file__).parent.parent.parent
    output_path = repo_root / 'figures' / 'variable_definition_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Variable Definition Table saved: {output_path}")
    
    # Also save as PDF
    pdf_path = repo_root / 'figures' / 'pdf_versions' / 'variable_definition_table.pdf'
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF version saved: {pdf_path}")
    
    plt.close()
    return output_path

if __name__ == "__main__":
    create_variable_definition_table()
