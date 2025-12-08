#!/bin/bash
#
# Setup auto-push git hook
# This script sets up automatic pushing after each commit

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_FILE="$REPO_ROOT/.git/hooks/post-commit"

cat > "$HOOK_FILE" << 'EOF'
#!/bin/sh
#
# Post-commit hook to automatically push to remote
# Also updates documentation

cd "$(git rev-parse --show-toplevel)"

# Update documentation first
python3 scripts/utils/update_documentation.py 2>/dev/null || true

# Auto-push to remote (suppress "Everything up-to-date" message)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push origin "$BRANCH" 2>&1 | grep -v "Everything up-to-date" || true
EOF

chmod +x "$HOOK_FILE"

echo "✅ Auto-push hook installed at: $HOOK_FILE"
echo "📝 This hook will:"
echo "   1. Update documentation (README.md, NAVIGATION_GUIDE.md)"
echo "   2. Automatically push to remote after each commit"
echo ""
echo "💡 To test, make a commit and it will auto-push!"

