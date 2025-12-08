# Auto-Update Documentation Guide

## Overview

The repository includes an automatic documentation update system that keeps README.md and NAVIGATION_GUIDE.md in sync with the current repository structure.

## How It Works

### Automatic Updates

**Git Hook (Post-Commit)**: After each commit, the documentation is automatically updated to reflect:
- Current file counts (scripts, results, summaries, figures)
- Updated file structure
- Latest statistics

The hook runs silently in the background and updates documentation files.

### Manual Updates

If you want to update documentation manually:

```bash
# Option 1: Run Python script directly
python3 scripts/utils/update_documentation.py

# Option 2: Use the shell script
./scripts/utils/update_docs.sh
```

## What Gets Updated

### README.md
- **Repository Statistics Section**: 
  - Script counts by category
  - Analysis output file counts
  - Figure counts (PNG and PDF)
  - Last updated timestamp

### NAVIGATION_GUIDE.md
- **Current File Structure Section**:
  - Script listings by category
  - Analysis file listings
  - Last updated timestamp

## When to Update

Documentation is automatically updated:
- ✅ After each commit (via git hook)
- ✅ When you add/remove scripts
- ✅ When you add/remove analysis outputs
- ✅ When you add/remove figures

You can also manually update anytime by running the script.

## Disabling Auto-Update

If you want to disable the automatic post-commit hook:

```bash
# Remove the hook
rm .git/hooks/post-commit

# Or make it non-executable
chmod -x .git/hooks/post-commit
```

## Troubleshooting

### Hook Not Running
- Check if hook is executable: `ls -l .git/hooks/post-commit`
- Make executable: `chmod +x .git/hooks/post-commit`
- Check Python path: Ensure `python3` is in your PATH

### Documentation Not Updating
- Run manually: `python3 scripts/utils/update_documentation.py`
- Check for errors in script output
- Verify file permissions

### Statistics Look Wrong
- Run script manually to see output
- Check if files are in expected locations
- Verify directory structure matches expected format

## Script Details

**Location**: `scripts/utils/update_documentation.py`

**Functions**:
- `count_scripts_by_category()` - Counts scripts in each folder
- `count_analysis_files()` - Counts analysis outputs
- `count_figures()` - Counts PNG and PDF files
- `update_readme()` - Updates README.md
- `update_navigation_guide()` - Updates NAVIGATION_GUIDE.md

**Dependencies**: None (uses only standard library)

---

*Last Updated: 2024*  
*Auto-Update System: Active*

