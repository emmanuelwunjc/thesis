#!/usr/bin/env python3
"""
Create Conceptual Framework Diagram for HINTS 7 Diabetes Privacy Study

This script creates a four-box conceptual model diagram showing:
- Top: Diabetes
- Right: Privacy Caution
- Bottom: Willingness to Share
- Left: Demographics/External Factors

Author: AI Assistant
Date: 2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

def create_conceptual_framework_diagram():
    """Create a four-box conceptual framework diagram with no overlaps and prominent connections."""
    print("📊 Creating Conceptual Framework Diagram...")
    
    # Set up the figure with larger size
    fig, ax = plt.subplots(1, 1, figsize=(22, 22))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Set title
    fig.suptitle('Conceptual Framework: Privacy Caution, Digital Engagement, and Diabetes\nImpact on Willingness to Share Health Information', 
                 fontsize=22, fontweight='bold', y=0.97)
    
    # Define colors
    colors = {
        'diabetes': '#2E86AB',        # Blue
        'privacy': '#A23B72',         # Rose/Purple
        'willingness': '#F18F01',     # Orange
        'demographics': '#28A745',    # Green
        'arrow': '#212529',           # Dark gray - much more visible
        'text': '#212529'             # Dark gray
    }
    
    # Box dimensions - larger boxes to accommodate bigger text
    box_width = 3.4
    box_height = 3.0
    
    # Center point of the diagram
    center_x = 7
    center_y = 7
    
    # Distance from center to box centers - INCREASED to prevent any overlap
    distance_from_center = 4.0
    
    # Calculate positions for four boxes in perfect cross formation
    # All boxes centered on their respective positions
    # Top: Diabetes
    diabetes_center_x = center_x
    diabetes_center_y = center_y + distance_from_center
    diabetes_x = diabetes_center_x - box_width / 2
    diabetes_y = diabetes_center_y - box_height / 2
    
    # Right: Privacy Caution
    privacy_center_x = center_x + distance_from_center
    privacy_center_y = center_y
    privacy_x = privacy_center_x - box_width / 2
    privacy_y = privacy_center_y - box_height / 2
    
    # Bottom: Willingness to Share
    willingness_center_x = center_x
    willingness_center_y = center_y - distance_from_center
    willingness_x = willingness_center_x - box_width / 2
    willingness_y = willingness_center_y - box_height / 2
    
    # Left: Demographics/External Factors
    demographics_center_x = center_x - distance_from_center
    demographics_center_y = center_y
    demographics_x = demographics_center_x - box_width / 2
    demographics_y = demographics_center_y - box_height / 2
    
    # Create boxes with rounded corners
    boxes = [
        # Top: Diabetes
        {
            'box': FancyBboxPatch((diabetes_x, diabetes_y), box_width, box_height, 
                                 boxstyle="round,pad=0.2", 
                                 facecolor=colors['diabetes'], 
                                 edgecolor='black', linewidth=3),
            'title': 'Diabetes',
            'content': [
                'Chronic Disease Status',
                '• Frequent digital tool usage',
                '• Continuous monitoring needs',
                '• Sensitive health information',
                '• Benefits vs. risks calculation',
                '• Altered privacy calculus'
            ],
            'center_x': diabetes_center_x,
            'center_y': diabetes_center_y
        },
        # Right: Privacy Caution
        {
            'box': FancyBboxPatch((privacy_x, privacy_y), box_width, box_height,
                                 boxstyle="round,pad=0.2",
                                 facecolor=colors['privacy'],
                                 edgecolor='black', linewidth=3),
            'title': 'Privacy Caution',
            'content': [
                'Risk Perception',
                '• Misuse concerns',
                '• Monitoring fears',
                '• Secondary use risks',
                '• Threat identification',
                '• Susceptibility feelings'
            ],
            'center_x': privacy_center_x,
            'center_y': privacy_center_y
        },
        # Bottom: Willingness to Share
        {
            'box': FancyBboxPatch((willingness_x, willingness_y), box_width, box_height,
                                 boxstyle="round,pad=0.2",
                                 facecolor=colors['willingness'],
                                 edgecolor='black', linewidth=3),
            'title': 'Willingness to Share\nHealth Information',
            'content': [
                'Outcome Variable',
                '• Data sharing decisions',
                '• Risk-benefit assessment',
                '• Trust in organizations',
                '• Information disclosure'
            ],
            'center_x': willingness_center_x,
            'center_y': willingness_center_y
        },
        # Left: Demographics/External Factors
        {
            'box': FancyBboxPatch((demographics_x, demographics_y), box_width, box_height,
                                 boxstyle="round,pad=0.2",
                                 facecolor=colors['demographics'],
                                 edgecolor='black', linewidth=3),
            'title': 'Demographics &\nExternal Factors',
            'content': [
                'Contextual Variables',
                '• Age, Education, Income',
                '• Race/Ethnicity',
                '• Governance',
                '• Risk environment',
                '• Regulation'
            ],
            'center_x': demographics_center_x,
            'center_y': demographics_center_y
        }
    ]
    
    # Draw boxes with MUCH LARGER text (2x size) and reduced line spacing
    for box_info in boxes:
        ax.add_patch(box_info['box'])
        
        # Title - 2x larger font (34 instead of 17)
        ax.text(box_info['center_x'], box_info['center_y'] + 1.0, box_info['title'],
                ha='center', va='center', fontsize=34, fontweight='bold', color='white')
        
        # Content - 2x larger font (24 instead of 12), reduced line spacing
        content_text = '\n'.join(box_info['content'])
        ax.text(box_info['center_x'], box_info['center_y'] - 0.3, content_text,
                ha='center', va='center', fontsize=24, color='white', linespacing=1.1)
    
    # Draw arrows with DIVERSE line styles
    # From Diabetes to Willingness to Share (direct effect) - thick solid line
    ax.annotate('', xy=(willingness_center_x, willingness_y + box_height),
                xytext=(diabetes_center_x, diabetes_y),
                arrowprops=dict(arrowstyle='->', lw=5.5, color=colors['arrow'],
                               mutation_scale=30))
    
    # From Privacy Caution to Willingness to Share (negative effect) - medium solid line
    ax.annotate('', xy=(willingness_x + box_width, willingness_center_y),
                xytext=(privacy_x, privacy_center_y),
                arrowprops=dict(arrowstyle='->', lw=4.0, color=colors['arrow'],
                               mutation_scale=25))
    
    # From Demographics to Privacy Caution - thin solid line
    ax.annotate('', xy=(privacy_x - 0.1, privacy_center_y),
                xytext=(demographics_x + box_width, demographics_center_y),
                arrowprops=dict(arrowstyle='->', lw=3.0, color=colors['arrow'],
                               mutation_scale=20))
    
    # From Demographics to Diabetes - dotted line
    ax.annotate('', xy=(diabetes_x - 0.1, diabetes_y + box_height / 2),
                xytext=(demographics_x + box_width, demographics_center_y + 0.3),
                arrowprops=dict(arrowstyle='->', lw=3.5, color=colors['arrow'],
                               linestyle=':', mutation_scale=20))
    
    # From Demographics to Willingness to Share - dotted line
    ax.annotate('', xy=(willingness_x - 0.1, willingness_center_y),
                xytext=(demographics_x + box_width, demographics_center_y - 0.3),
                arrowprops=dict(arrowstyle='->', lw=3.5, color=colors['arrow'],
                               linestyle=':', mutation_scale=20))
    
    # From Diabetes to Privacy Caution (moderation/interaction) - dashed line
    ax.annotate('', xy=(privacy_x - 0.1, privacy_y + box_height),
                xytext=(diabetes_x + box_width, diabetes_center_y),
                arrowprops=dict(arrowstyle='->', lw=4.0, color=colors['arrow'],
                               linestyle='--', mutation_scale=25))
    
    # Add Legend in top right corner - explaining line styles
    # Smaller box with better padding
    legend_box_x = 10.5
    legend_box_y = 10.8
    legend_box_width = 2.6
    legend_box_height = 2.2
    legend_box = FancyBboxPatch((legend_box_x, legend_box_y), legend_box_width, legend_box_height, 
                               boxstyle="round,pad=0.08",
                               facecolor='white', edgecolor='black', linewidth=2.5)
    ax.add_patch(legend_box)
    
    # Title - positioned with proper padding from top
    legend_title_x = legend_box_x + legend_box_width / 2
    legend_title_y = legend_box_y + legend_box_height - 0.25
    ax.text(legend_title_x, legend_title_y, 'Line Style Legend', ha='center', va='center',
            fontsize=18, fontweight='bold', color='black')
    
    # Draw legend lines with labels - adjusted positions with proper padding
    legend_line_x_start = legend_box_x + 0.2
    legend_line_x_end = legend_box_x + 0.9
    legend_text_x = legend_box_x + 1.05
    legend_y_start = legend_box_y + legend_box_height - 0.55
    legend_spacing = 0.32
    
    # Thick solid line - Direct effect
    ax.plot([legend_line_x_start, legend_line_x_end], [legend_y_start, legend_y_start], 
            lw=5.5, color=colors['arrow'], zorder=10)
    ax.annotate('', xy=(legend_line_x_end, legend_y_start), xytext=(legend_line_x_start, legend_y_start),
                arrowprops=dict(arrowstyle='->', lw=5.5, color=colors['arrow'],
                               mutation_scale=18))
    ax.text(legend_text_x, legend_y_start, 'Direct Effect', ha='left', va='center',
            fontsize=15, color='black')
    
    # Medium solid line - Negative effect
    legend_y_start -= legend_spacing
    ax.plot([legend_line_x_start, legend_line_x_end], [legend_y_start, legend_y_start], 
            lw=4.0, color=colors['arrow'], zorder=10)
    ax.annotate('', xy=(legend_line_x_end, legend_y_start), xytext=(legend_line_x_start, legend_y_start),
                arrowprops=dict(arrowstyle='->', lw=4.0, color=colors['arrow'],
                               mutation_scale=14))
    ax.text(legend_text_x, legend_y_start, 'Negative Effect', ha='left', va='center',
            fontsize=15, color='black')
    
    # Thin solid line - Influences
    legend_y_start -= legend_spacing
    ax.plot([legend_line_x_start, legend_line_x_end], [legend_y_start, legend_y_start], 
            lw=3.0, color=colors['arrow'], zorder=10)
    ax.annotate('', xy=(legend_line_x_end, legend_y_start), xytext=(legend_line_x_start, legend_y_start),
                arrowprops=dict(arrowstyle='->', lw=3.0, color=colors['arrow'],
                               mutation_scale=11))
    ax.text(legend_text_x, legend_y_start, 'Influences', ha='left', va='center',
            fontsize=15, color='black')
    
    # Dotted line - Contextual factors
    legend_y_start -= legend_spacing
    ax.plot([legend_line_x_start, legend_line_x_end], [legend_y_start, legend_y_start], 
            lw=3.5, color=colors['arrow'], linestyle=':', zorder=10)
    ax.annotate('', xy=(legend_line_x_end, legend_y_start), xytext=(legend_line_x_start, legend_y_start),
                arrowprops=dict(arrowstyle='->', lw=3.5, color=colors['arrow'],
                               linestyle=':', mutation_scale=11))
    ax.text(legend_text_x, legend_y_start, 'Contextual Factors', ha='left', va='center',
            fontsize=15, color='black')
    
    # Dashed line - Moderation
    legend_y_start -= legend_spacing
    ax.plot([legend_line_x_start, legend_line_x_end], [legend_y_start, legend_y_start], 
            lw=4.0, color=colors['arrow'], linestyle='--', zorder=10)
    ax.annotate('', xy=(legend_line_x_end, legend_y_start), xytext=(legend_line_x_start, legend_y_start),
                arrowprops=dict(arrowstyle='->', lw=4.0, color=colors['arrow'],
                               linestyle='--', mutation_scale=14))
    ax.text(legend_text_x, legend_y_start, 'Moderation', ha='left', va='center',
            fontsize=15, color='black')
    
    # Add Digital Engagement as a note - positioned in bottom right corner, well separated
    # MUCH LARGER text (2x)
    engagement_note = FancyBboxPatch((10.2, 1.0), 3.0, 2.0, boxstyle="round,pad=0.15",
                                    facecolor='lightyellow', edgecolor='black', linewidth=2.5)
    ax.add_patch(engagement_note)
    ax.text(11.7, 2.5, 'Digital Engagement\n(Moderator)', ha='center', va='center',
            fontsize=26, fontweight='bold', color='black')  # 2x from 13
    engagement_content = '• Portal usage\n• Telehealth\n• Mobile health\n• May moderate\n  privacy effects'
    ax.text(11.7, 1.6, engagement_content,
            ha='center', va='center', fontsize=20, color='black', linespacing=1.2)  # 2x from 10
    
    # Add arrow from Digital Engagement to Privacy Caution relationship
    ax.annotate('', xy=(privacy_center_x, privacy_y),
                xytext=(11.7, 3.0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='gray', linestyle='--', alpha=0.8))
    
    # Add key relationships note - positioned in bottom left corner, well separated
    # MUCH LARGER text (2x)
    note_box = FancyBboxPatch((0.8, 1.0), 3.0, 2.0, boxstyle="round,pad=0.15",
                             facecolor='lightblue', edgecolor='black', linewidth=2.5)
    ax.add_patch(note_box)
    ax.text(2.3, 2.5, 'Key Relationships', ha='center', va='center',
            fontsize=28, fontweight='bold', color='black')  # 2x from 14
    relationships_text = '• Privacy Caution → ↓ Sharing\n• Diabetes → Altered calculus\n• Demographics → Context\n• Digital Engagement → Moderation'
    ax.text(2.3, 1.6, relationships_text, ha='center', va='center',
            fontsize=20, color='black', linespacing=1.3)  # 2x from 10
    
    plt.tight_layout()
    
    # Save figure
    repo_root = Path(__file__).parent.parent.parent
    output_path = repo_root / 'figures' / 'conceptual_framework_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Conceptual Framework Diagram saved: {output_path}")
    
    # Also save as PDF
    pdf_path = repo_root / 'figures' / 'pdf_versions' / 'conceptual_framework_diagram.pdf'
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF version saved: {pdf_path}")
    
    plt.close()
    return output_path

if __name__ == "__main__":
    create_conceptual_framework_diagram()
