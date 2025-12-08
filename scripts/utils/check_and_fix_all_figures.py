#!/usr/bin/env python3
"""
Check and Fix All Figures

This script:
1. Checks all figure files for issues (missing, outdated, etc.)
2. Identifies which scripts need to be run to regenerate figures
3. Runs scripts to regenerate figures with English labels
4. Verifies all figures are updated
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

# Get repository root
repo_root = Path(__file__).parent.parent.parent
figures_dir = repo_root / 'figures'

# Expected figures from visualization scripts
expected_figures = {
    'plot_privacy_index.py': ['privacy_caution_index_analysis.png'],
    'plot_privacy_diffs.py': [
        'privacy_top10_diffs.png',
        'privacy_shared_device.png',
        'privacy_use_computer.png',
        'privacy_use_watch.png',
        'privacy_trust_hcsystem.png',
        'privacy_trust_scientists.png',
        'privacy_portal_pharmacy.png'
    ],
    'age_group_analysis.py': ['age_group_comparison.png'],
    'age_distribution_plot.py': ['age_distribution_comparison.png'],
    'create_optimized_index_diagram.py': ['privacy_index_construction_diagram_optimized.png'],
    'create_optimized_detailed_table.py': ['privacy_index_detailed_table_optimized.png'],
    'create_best_model_visualizations.py': [
        'best_ml_model_detailed_analysis.png',
        'best_model_architecture.png'
    ],
    'create_model_logic_diagram.py': ['model_logic_diagram.png'],
}

def check_figure_exists(figure_name):
    """Check if a figure exists."""
    png_path = figures_dir / figure_name
    return png_path.exists()

def get_figure_mod_time(figure_name):
    """Get figure modification time."""
    png_path = figures_dir / figure_name
    if png_path.exists():
        return datetime.fromtimestamp(png_path.stat().st_mtime)
    return None

def run_script(script_name):
    """Run a visualization script."""
    script_path = repo_root / 'scripts' / '05_visualization' / script_name
    
    if not script_path.exists():
        return False, f"Script not found: {script_name}"
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, result.stderr or result.stdout
    except Exception as e:
        return False, str(e)

def main():
    """Main function to check and fix all figures."""
    print("="*70)
    print("CHECKING AND FIXING ALL FIGURES")
    print("="*70)
    
    # Check current figures
    print("\n📊 Checking existing figures...")
    missing_figures = []
    outdated_figures = []
    
    for script, figures in expected_figures.items():
        for fig in figures:
            if not check_figure_exists(fig):
                missing_figures.append((script, fig))
                print(f"  ❌ Missing: {fig} (from {script})")
            else:
                mod_time = get_figure_mod_time(fig)
                print(f"  ✅ Exists: {fig} (modified: {mod_time})")
    
    # Regenerate all figures
    print("\n" + "="*70)
    print("REGENERATING ALL FIGURES")
    print("="*70)
    
    scripts_to_run = list(expected_figures.keys())
    results = {}
    
    for script in scripts_to_run:
        print(f"\n🔄 Running: {script}")
        success, message = run_script(script)
        results[script] = success
        if success:
            print(f"  ✅ Success")
        else:
            print(f"  ❌ Failed: {message}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    # Verify figures
    print("\n📋 Verifying figures...")
    all_exist = True
    for script, figures in expected_figures.items():
        for fig in figures:
            if check_figure_exists(fig):
                print(f"  ✅ {fig}")
            else:
                print(f"  ❌ {fig} - STILL MISSING")
                all_exist = False
    
    if all_exist and successful == total:
        print("\n🎉 All figures checked and regenerated successfully!")
    else:
        print("\n⚠️  Some figures may need manual attention.")
    
    print(f"\n📁 Figures directory: {figures_dir}")

if __name__ == "__main__":
    main()

