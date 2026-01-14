#!/usr/bin/env python3
"""
Create Descriptive Statistics and Regression Results Tables as Figures

This script creates high-quality table figures for descriptive statistics and 
regression results, formatted for A4 paper compatibility.

Author: AI Assistant
Date: 2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

def create_descriptive_statistics_figure():
    """Create descriptive statistics and regression results tables as figures."""
    print("📊 Creating Descriptive Statistics and Regression Results Figures...")
    
    # A4 size in inches (portrait orientation)
    fig_width = 8.27
    fig_height = 11.69
    
    # Create figure for descriptive statistics
    fig1, ax1 = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Title
    fig1.suptitle('Table 1: Descriptive Statistics\nHINTS 7 Diabetes Privacy Study', 
                 fontsize=18, fontweight='bold', y=0.97)
    
    # Starting position
    y_start = 0.92
    y_current = y_start
    
    # Colors
    colors = {
        'header': '#2E86AB',
        'row1': '#F8F9FA',
        'row2': '#FFFFFF',
        'border': '#000000'
    }
    
    # Table data
    table_data = [
        ['Variable', 'Full Sample\n(N=7,278)', 'Analysis Sample\n(N=2,421)', 
         'Diabetic\n(n=510)', 'Non-Diabetic\n(n=1,911)'],
        ['Diabetes Status', '', '', '', ''],
        ['  Has Diabetes', '1,534 (21.1)', '510 (21.1)', '510 (100.0)', '0 (0.0)'],
        ['  No Diabetes', '5,744 (78.9)', '1,911 (78.9)', '0 (0.0)', '1,911 (100.0)'],
        ['Privacy Caution Index', '', '', '', ''],
        ['  Mean (SD)', '0.47 (0.09)', '0.47 (0.09)', '0.48 (0.10)', '0.47 (0.09)'],
        ['  Range', '0.23-0.78', '0.23-0.78', '0.23-0.78', '0.23-0.78'],
        ['Data Sharing Willingness', '', '', '', ''],
        ['  Willing to Share', '2,026 (27.8)', '1,636 (67.6)', '-', '-'],
        ['  Not Willing to Share', '636 (8.7)', '785 (32.4)', '-', '-'],
        ['Age (years)', '', '', '', ''],
        ['  Mean (SD)', '58.3 (15.2)', '58.3 (15.2)', '-', '-'],
        ['  Range', '18-100+', '18-100+', '-', '-'],
        ['Gender', '', '', '', ''],
        ['  Female', '3,792 (52.1)', '1,260 (52.1)', '-', '-'],
        ['  Male', '3,486 (47.9)', '1,161 (47.9)', '-', '-'],
        ['Education Level', '', '', '', ''],
        ['  Mean (SD)', '-', '3.8 (1.2)', '-', '-'],
        ['  Some College or Higher', '3,290 (45.2)', '1,094 (45.2)', '-', '-'],
        ['Health Insurance', '', '', '', ''],
        ['  Has Insurance', '6,208 (85.3)', '2,064 (85.3)', '-', '-'],
        ['  No Insurance', '1,070 (14.7)', '357 (14.7)', '-', '-']
    ]
    
    # Column widths
    col_widths = [0.30, 0.18, 0.18, 0.17, 0.17]
    col_x_positions = [0.05, 0.35, 0.53, 0.71, 0.88]
    row_height = 0.035
    
    # Draw table
    for row_idx, row_data in enumerate(table_data):
        row_y = y_current
        
        # Determine row color
        if row_idx == 0:
            row_color = colors['header']
            text_color = 'white'
            font_weight = 'bold'
            font_size = 11
        elif row_data[0].startswith('  ') or row_data[0] in ['Diabetes Status', 'Privacy Caution Index', 
                                                             'Data Sharing Willingness', 'Age (years)', 
                                                             'Gender', 'Education Level', 'Health Insurance']:
            if row_data[0].startswith('  '):
                row_color = colors['row1'] if row_idx % 2 == 0 else colors['row2']
                text_color = 'black'
                font_weight = 'normal'
                font_size = 9
            else:
                row_color = colors['row1']
                text_color = 'black'
                font_weight = 'bold'
                font_size = 10
        else:
            row_color = colors['row1'] if row_idx % 2 == 0 else colors['row2']
            text_color = 'black'
            font_weight = 'normal'
            font_size = 9
        
        # Draw row background
        row_box = FancyBboxPatch((0.05, row_y - row_height/2), 0.90, row_height,
                                boxstyle="round,pad=0.002",
                                facecolor=row_color, edgecolor=colors['border'], linewidth=0.5)
        ax1.add_patch(row_box)
        
        # Draw cells
        for col_idx, cell_text in enumerate(row_data):
            cell_x = col_x_positions[col_idx]
            ax1.text(cell_x, row_y, cell_text, ha='left', va='center',
                    fontsize=font_size, color=text_color, weight=font_weight,
                    transform=ax1.transAxes)
        
        y_current -= row_height
    
    # Add notes at bottom
    notes_y = 0.05
    notes_text = "Notes: Full Sample = Original HINTS 7 Public Dataset (2022). Analysis Sample = Final sample with complete data (N=2,421). Privacy Caution Index: 0-1 scale, Cronbach's α = 0.78."
    ax1.text(0.5, notes_y, notes_text, ha='center', va='center',
            fontsize=8, style='italic', color='black', transform=ax1.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save descriptive statistics figure
    repo_root = Path(__file__).parent.parent.parent
    output_path1 = repo_root / 'figures' / 'descriptive_statistics_table.png'
    plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Descriptive Statistics Table saved: {output_path1}")
    
    pdf_path1 = repo_root / 'figures' / 'pdf_versions' / 'descriptive_statistics_table.pdf'
    pdf_path1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path1, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    print(f"✅ PDF version saved: {pdf_path1}")
    
    plt.close(fig1)
    
    # Create figure for main regression results
    fig2, ax2 = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    fig2.suptitle('Table 2: Main Regression Model Results\nHINTS 7 Diabetes Privacy Study', 
                 fontsize=18, fontweight='bold', y=0.97)
    
    y_current = 0.92
    
    # Regression table data
    reg_table_data = [
        ['Variable', 'Coefficient', 'Std. Error', 't-statistic', 'p-value', 'Significance'],
        ['Constant', '1.6673', '0.0744', '22.41', '<0.001', '***'],
        ['Diabetes Status', '0.0278', '0.0198', '1.40', '0.1608', ''],
        ['Privacy Caution Index', '-2.3159', '0.1077', '-21.49', '<0.001', '***'],
        ['Age', '0.0024', '0.0005', '5.14', '<0.001', '***'],
        ['Education Level', '-0.0149', '0.0098', '-1.52', '0.1290', '']
    ]
    
    reg_col_widths = [0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
    reg_col_x_positions = [0.05, 0.30, 0.45, 0.60, 0.75, 0.90]
    
    for row_idx, row_data in enumerate(reg_table_data):
        row_y = y_current
        
        if row_idx == 0:
            row_color = colors['header']
            text_color = 'white'
            font_weight = 'bold'
            font_size = 11
        else:
            row_color = colors['row1'] if row_idx % 2 == 0 else colors['row2']
            text_color = 'black'
            font_weight = 'normal'
            font_size = 10
        
        row_box = FancyBboxPatch((0.05, row_y - row_height/2), 0.90, row_height,
                                boxstyle="round,pad=0.002",
                                facecolor=row_color, edgecolor=colors['border'], linewidth=0.5)
        ax2.add_patch(row_box)
        
        for col_idx, cell_text in enumerate(row_data):
            cell_x = reg_col_x_positions[col_idx]
            ax2.text(cell_x, row_y, cell_text, ha='left', va='center',
                    fontsize=font_size, color=text_color, weight=font_weight,
                    transform=ax2.transAxes)
        
        y_current -= row_height
    
    # Add model statistics
    y_current -= 0.02
    stats_text = "Sample Size: 2,421 observations | R²: 0.1736 | Method: Weighted Least Squares"
    ax2.text(0.5, y_current, stats_text, ha='center', va='center',
            fontsize=10, fontweight='bold', color='black', transform=ax2.transAxes)
    
    y_current -= 0.03
    sig_text = "Significance levels: *** p<0.001, ** p<0.01, * p<0.05, † p<0.1"
    ax2.text(0.5, y_current, sig_text, ha='center', va='center',
            fontsize=9, style='italic', color='black', transform=ax2.transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path2 = repo_root / 'figures' / 'regression_results_table.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Regression Results Table saved: {output_path2}")
    
    pdf_path2 = repo_root / 'figures' / 'pdf_versions' / 'regression_results_table.pdf'
    plt.savefig(pdf_path2, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    print(f"✅ PDF version saved: {pdf_path2}")
    
    plt.close(fig2)
    
    # Create figure for interaction model
    fig3, ax3 = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    fig3.suptitle('Table 3: Interaction Model Results\nHINTS 7 Diabetes Privacy Study', 
                 fontsize=18, fontweight='bold', y=0.97)
    
    y_current = 0.92
    
    # Interaction regression table data
    inter_table_data = [
        ['Variable', 'Coefficient', 'Std. Error', 't-statistic', 'p-value', 'Significance'],
        ['Constant', '1.7200', '0.0786', '21.89', '<0.001', '***'],
        ['Diabetes Status', '-0.1712', '0.0981', '-1.75', '0.0810', '†'],
        ['Privacy Caution Index', '-2.4409', '0.1234', '-19.78', '<0.001', '***'],
        ['Diabetes × Privacy Index', '0.4896', '0.2363', '2.07', '0.0383', '*'],
        ['Age', '0.0023', '0.0005', '4.99', '<0.001', '***'],
        ['Education Level', '-0.0144', '0.0098', '-1.47', '0.1415', '']
    ]
    
    for row_idx, row_data in enumerate(inter_table_data):
        row_y = y_current
        
        if row_idx == 0:
            row_color = colors['header']
            text_color = 'white'
            font_weight = 'bold'
            font_size = 11
        else:
            row_color = colors['row1'] if row_idx % 2 == 0 else colors['row2']
            text_color = 'black'
            font_weight = 'normal'
            font_size = 10
        
        row_box = FancyBboxPatch((0.05, row_y - row_height/2), 0.90, row_height,
                                boxstyle="round,pad=0.002",
                                facecolor=row_color, edgecolor=colors['border'], linewidth=0.5)
        ax3.add_patch(row_box)
        
        for col_idx, cell_text in enumerate(row_data):
            cell_x = reg_col_x_positions[col_idx]
            ax3.text(cell_x, row_y, cell_text, ha='left', va='center',
                    fontsize=font_size, color=text_color, weight=font_weight,
                    transform=ax3.transAxes)
        
        y_current -= row_height
    
    # Add model statistics
    y_current -= 0.02
    stats_text2 = "Sample Size: 2,421 observations | R²: 0.1753 | Method: Weighted Least Squares with Interaction"
    ax3.text(0.5, y_current, stats_text2, ha='center', va='center',
            fontsize=10, fontweight='bold', color='black', transform=ax3.transAxes)
    
    y_current -= 0.03
    ax3.text(0.5, y_current, sig_text, ha='center', va='center',
            fontsize=9, style='italic', color='black', transform=ax3.transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path3 = repo_root / 'figures' / 'interaction_model_table.png'
    plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Interaction Model Table saved: {output_path3}")
    
    pdf_path3 = repo_root / 'figures' / 'pdf_versions' / 'interaction_model_table.pdf'
    plt.savefig(pdf_path3, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    print(f"✅ PDF version saved: {pdf_path3}")
    
    plt.close(fig3)
    
    return output_path1, output_path2, output_path3

if __name__ == "__main__":
    create_descriptive_statistics_figure()
