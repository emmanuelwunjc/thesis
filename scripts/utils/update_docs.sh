#!/bin/bash
# Quick script to manually update documentation
# Usage: ./scripts/utils/update_docs.sh

cd "$(dirname "$0")/../.."
python3 scripts/utils/update_documentation.py

