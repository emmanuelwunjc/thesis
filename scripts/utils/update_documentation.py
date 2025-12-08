#!/usr/bin/env python3
"""
Auto-Update Documentation Script

This script automatically updates README.md and NAVIGATION_GUIDE.md
whenever the repository structure changes.

Run this script after making changes to keep documentation in sync.
"""

import os
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple

def get_repo_root():
    """Get repository root directory."""
    return Path(__file__).parent.parent.parent

def count_files_by_type(directory: Path, extensions: List[str]) -> int:
    """Count files with given extensions in directory."""
    count = 0
    for ext in extensions:
        count += len(list(directory.rglob(f"*.{ext}")))
    return count

def get_directory_structure(base_path: Path, max_depth: int = 3) -> Dict:
    """Get directory structure as dictionary."""
    structure = {}
    
    for item in sorted(base_path.iterdir()):
        if item.name.startswith('.') or item.name in ['__pycache__', 'node_modules']:
            continue
        
        if item.is_dir():
            structure[item.name] = {
                'type': 'directory',
                'path': str(item.relative_to(base_path)),
                'files': []
            }
            
            # Get files in directory
            for file in sorted(item.iterdir()):
                if file.is_file() and not file.name.startswith('.'):
                    structure[item.name]['files'].append(file.name)
        elif item.is_file():
            if 'files' not in structure:
                structure['files'] = []
            structure['files'].append(item.name)
    
    return structure

def count_scripts_by_category() -> Dict[str, int]:
    """Count scripts in each category."""
    repo_root = get_repo_root()
    scripts_dir = repo_root / 'scripts'
    
    counts = {}
    if scripts_dir.exists():
        for subdir in sorted(scripts_dir.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith('_'):
                py_files = list(subdir.rglob('*.py'))
                counts[subdir.name] = len(py_files)
    
    return counts

def count_analysis_files() -> Dict[str, int]:
    """Count analysis output files."""
    repo_root = get_repo_root()
    analysis_dir = repo_root / 'analysis'
    
    counts = {
        'results': 0,
        'summaries_english': 0,
        'summaries_chinese': 0,
        'data': 0
    }
    
    if analysis_dir.exists():
        results_dir = analysis_dir / 'results'
        if results_dir.exists():
            counts['results'] = len(list(results_dir.glob('*.json'))) + len(list(results_dir.glob('*.tex')))
        
        summaries_dir = analysis_dir / 'summaries'
        if summaries_dir.exists():
            english_dir = summaries_dir / 'english'
            chinese_dir = summaries_dir / 'chinese'
            if english_dir.exists():
                counts['summaries_english'] = len(list(english_dir.glob('*.md')))
            if chinese_dir.exists():
                counts['summaries_chinese'] = len(list(chinese_dir.glob('*.md')))
        
        data_dir = analysis_dir / 'data'
        if data_dir.exists():
            counts['data'] = len(list(data_dir.glob('*.csv')))
    
    return counts

def count_figures() -> Tuple[int, int]:
    """Count figure files."""
    repo_root = get_repo_root()
    figures_dir = repo_root / 'figures'
    
    png_count = 0
    pdf_count = 0
    
    if figures_dir.exists():
        png_count = len(list(figures_dir.rglob('*.png')))
        pdf_dir = figures_dir / 'pdf_versions'
        if pdf_dir.exists():
            pdf_count = len(list(pdf_dir.glob('*.pdf')))
    
    return png_count, pdf_count

def generate_readme_stats() -> str:
    """Generate statistics section for README."""
    repo_root = get_repo_root()
    
    script_counts = count_scripts_by_category()
    analysis_counts = count_analysis_files()
    png_count, pdf_count = count_figures()
    
    total_scripts = sum(script_counts.values())
    
    stats = f"""## 📊 Repository Statistics

### Scripts ({total_scripts} total)
"""
    for category, count in script_counts.items():
        category_name = category.replace('_', ' ').title()
        stats += f"- **{category_name}**: {count} scripts\n"
    
    stats += f"""
### Analysis Outputs
- **Results**: {analysis_counts['results']} JSON/TEX files
- **Summaries (English)**: {analysis_counts['summaries_english']} documents
- **Summaries (Chinese)**: {analysis_counts['summaries_chinese']} documents
- **Data Files**: {analysis_counts['data']} CSV files

### Figures
- **PNG Files**: {png_count} files
- **PDF Files**: {pdf_count} files (in pdf_versions/)
- **Total**: {png_count + pdf_count} visualizations

*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    return stats

def update_readme():
    """Update README.md with current statistics."""
    repo_root = get_repo_root()
    readme_path = repo_root / 'README.md'
    
    if not readme_path.exists():
        print("⚠️  README.md not found")
        return
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace statistics section
    stats_section = generate_readme_stats()
    
    # Check if statistics section exists
    if '## 📊 Repository Statistics' in content:
        # Replace existing section
        import re
        pattern = r'## 📊 Repository Statistics.*?\*Last Updated:.*?\*'
        content = re.sub(pattern, stats_section.strip(), content, flags=re.DOTALL)
    else:
        # Add before the last section
        if '## ✅ Project Status' in content:
            content = content.replace('## ✅ Project Status', stats_section + '\n\n## ✅ Project Status')
        else:
            content += '\n\n' + stats_section
    
    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ README.md updated with current statistics")

def generate_navigation_file_listing() -> str:
    """Generate file listing for navigation guide."""
    repo_root = get_repo_root()
    
    listing = "## 📁 Current File Structure\n\n"
    
    # Scripts
    scripts_dir = repo_root / 'scripts'
    if scripts_dir.exists():
        listing += "### `/scripts/` - Analysis Scripts\n\n"
        for subdir in sorted(scripts_dir.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith('_'):
                py_files = list(subdir.rglob('*.py'))
                if py_files:
                    listing += f"#### `{subdir.name}/` ({len(py_files)} scripts)\n"
                    for py_file in sorted(py_files)[:10]:  # Show first 10
                        listing += f"- `{py_file.name}`\n"
                    if len(py_files) > 10:
                        listing += f"- ... and {len(py_files) - 10} more\n"
                    listing += "\n"
    
    # Analysis
    analysis_dir = repo_root / 'analysis'
    if analysis_dir.exists():
        listing += "### `/analysis/` - Analysis Outputs\n\n"
        
        results_dir = analysis_dir / 'results'
        if results_dir.exists():
            json_files = list(results_dir.glob('*.json'))
            listing += f"#### `results/` ({len(json_files)} JSON files)\n"
            for json_file in sorted(json_files)[:5]:
                listing += f"- `{json_file.name}`\n"
            if len(json_files) > 5:
                listing += f"- ... and {len(json_files) - 5} more\n"
            listing += "\n"
        
        summaries_dir = analysis_dir / 'summaries'
        if summaries_dir.exists():
            english_dir = summaries_dir / 'english'
            chinese_dir = summaries_dir / 'chinese'
            if english_dir.exists():
                md_files = list(english_dir.glob('*.md'))
                listing += f"#### `summaries/english/` ({len(md_files)} summaries)\n"
                for md_file in sorted(md_files)[:5]:
                    listing += f"- `{md_file.name}`\n"
                if len(md_files) > 5:
                    listing += f"- ... and {len(md_files) - 5} more\n"
                listing += "\n"
    
    listing += f"\n*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    return listing

def update_navigation_guide():
    """Update NAVIGATION_GUIDE.md with current file structure."""
    repo_root = get_repo_root()
    nav_path = repo_root / 'NAVIGATION_GUIDE.md'
    
    if not nav_path.exists():
        print("⚠️  NAVIGATION_GUIDE.md not found")
        return
    
    # Read current guide
    with open(nav_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace file structure section
    file_listing = generate_navigation_file_listing()
    
    # Check if file structure section exists
    if '## 📁 Current File Structure' in content:
        # Replace existing section
        import re
        pattern = r'## 📁 Current File Structure.*?\*Last Updated:.*?\*'
        content = re.sub(pattern, file_listing.strip(), content, flags=re.DOTALL)
    else:
        # Add at the end
        content += '\n\n' + file_listing
    
    # Write updated guide
    with open(nav_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ NAVIGATION_GUIDE.md updated with current file structure")

def main():
    """Main function to update all documentation."""
    print("🔄 Updating documentation...")
    print("=" * 60)
    
    update_readme()
    update_navigation_guide()
    
    print("=" * 60)
    print("✅ Documentation update complete!")
    print("\n💡 Tip: Run this script after making changes to keep docs in sync")

if __name__ == "__main__":
    main()

