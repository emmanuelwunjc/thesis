#!/usr/bin/env python3
"""
Regenerate All Figures

This script regenerates all figures with updated English labels and correct file paths.
Run this after fixing visualization scripts to update all figures.
"""

import subprocess
import sys
from pathlib import Path

# Get repository root
repo_root = Path(__file__).parent.parent.parent
figures_dir = repo_root / 'figures'

# List of scripts to run (in order)
scripts = [
    'plot_privacy_index.py',
    'plot_privacy_diffs.py',
    'age_group_analysis.py',
    'age_distribution_plot.py',
    # Add other visualization scripts here as they are fixed
]

def run_script(script_name):
    """Run a visualization script and report results."""
    script_path = repo_root / 'scripts' / '05_visualization' / script_name
    
    if not script_path.exists():
        print(f"⚠️  Script not found: {script_name}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  {script_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Main function to regenerate all figures."""
    print("="*60)
    print("REGENERATING ALL FIGURES")
    print("="*60)
    print(f"Repository root: {repo_root}")
    print(f"Figures directory: {figures_dir}")
    
    # Check if figures directory exists
    if not figures_dir.exists():
        figures_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created figures directory: {figures_dir}")
    
    # Run each script
    results = {}
    for script in scripts:
        results[script] = run_script(script)
    
    # Summary
    print("\n" + "="*60)
    print("REGENERATION SUMMARY")
    print("="*60)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    if successful == total:
        print("\n🎉 All figures regenerated successfully!")
    else:
        print("\n⚠️  Some figures failed to regenerate. Check errors above.")
    
    print(f"\n📁 Figures saved to: {figures_dir}")

if __name__ == "__main__":
    main()

