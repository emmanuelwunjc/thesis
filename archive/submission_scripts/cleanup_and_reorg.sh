#!/bin/bash
# Comprehensive cleanup and reorganization of thesis folder

cd /Users/wuyiming/code/thesis

echo "🧹 Starting comprehensive cleanup..."

# ============================================
# 1. CREATE ARCHIVE FOLDER FOR OLD FILES
# ============================================
echo "📦 Creating archive folder..."
mkdir -p archive/old_docs
mkdir -p archive/submission_scripts

# ============================================
# 2. MOVE OLD/REDUNDANT ROOT FILES TO ARCHIVE
# ============================================
echo "📁 Archiving old root files..."
mv CLEANUP_SUMMARY.md archive/old_docs/ 2>/dev/null
mv FIGURES_FIX_SUMMARY.md archive/old_docs/ 2>/dev/null
mv REORGANIZATION_COMPLETE.md archive/old_docs/ 2>/dev/null
mv JOURNAL_SUBMISSION_EDITS.md archive/old_docs/ 2>/dev/null

# ============================================
# 3. CLEAN UP REDUNDANT FILES
# ============================================
echo "🗑️  Removing redundant files..."
rm -f .Rhistory
rm -f "diabetes_privacy_lit copy.bib"  # Remove copy file

# ============================================
# 4. ORGANIZE SCRIPTS - MOVE SUBMISSION HELPERS
# ============================================
echo "📂 Organizing scripts..."
mv scripts/add_all_refs.py archive/submission_scripts/ 2>/dev/null
mv scripts/add_intext_refs.py archive/submission_scripts/ 2>/dev/null
mv scripts/add_intext_refs_v2.py archive/submission_scripts/ 2>/dev/null
mv scripts/add_tables_figures.py archive/submission_scripts/ 2>/dev/null
mv scripts/add_variable_table.py archive/submission_scripts/ 2>/dev/null
mv scripts/create_blind_copy.py archive/submission_scripts/ 2>/dev/null
mv scripts/create_submission_docs.py archive/submission_scripts/ 2>/dev/null
mv scripts/embed_images.py archive/submission_scripts/ 2>/dev/null
mv scripts/enhance_paper.py archive/submission_scripts/ 2>/dev/null
mv scripts/finalize_document.py archive/submission_scripts/ 2>/dev/null
mv scripts/finalize_document_v2.py archive/submission_scripts/ 2>/dev/null
mv scripts/fix_table_refs.py archive/submission_scripts/ 2>/dev/null
mv scripts/format_academic_docx.py archive/submission_scripts/ 2>/dev/null

# Remove duplicate csv in scripts folder
rm -f scripts/analysis/ml_cleaned_data.csv
rmdir scripts/analysis 2>/dev/null

# ============================================
# 5. ORGANIZE FIGURES
# ============================================
echo "🖼️  Organizing figures..."
cd figures
mkdir -p tables
mv *.txt tables/ 2>/dev/null
cd ..

# ============================================
# 6. ORGANIZE DOCS - MOVE HINTS PDFs
# ============================================
echo "📚 Organizing documentation..."
mkdir -p docs/hints_documentation
mv "docs/HINTS 7 Annotated English.pdf" docs/hints_documentation/ 2>/dev/null
mv "docs/HINTS 7 Annotated Spanish.pdf" docs/hints_documentation/ 2>/dev/null
mv "docs/HINTS 7 History Document.pdf" docs/hints_documentation/ 2>/dev/null
mv "docs/HINTS 7 Methodology Report.pdf" docs/hints_documentation/ 2>/dev/null
mv "docs/HINTS 7 Public Codebook.pdf" docs/hints_documentation/ 2>/dev/null
mv "docs/HINTS 7 Survey Overview Data Analysis Recommendations.pdf" docs/hints_documentation/ 2>/dev/null

# Move chinese docs to guides
mv docs/QUICK_START_chinese.md docs/guides/ 2>/dev/null
mv docs/PROJECT_LOG_chinese.md docs/guides/ 2>/dev/null

# ============================================
# 7. MOVE ORIGINAL THESIS TO ARCHIVE
# ============================================
echo "📄 Archiving original thesis draft..."
mv Yiming_Thesis_Final_New.docx archive/ 2>/dev/null

echo "✅ Cleanup complete!"
