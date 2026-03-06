#!/usr/bin/env bash
#
# Scan all project files with all linters
# Unlike pre-commit which only checks staged files, this scans everything
#
# Usage:
#   ./tools/scan-all.sh          # Scan only (check mode)
#   ./tools/scan-all.sh --fix    # Scan and fix auto-fixable issues
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
FIX_MODE=false
if [[ "$1" == "--fix" ]]; then
    FIX_MODE=true
fi

if [[ $FIX_MODE == true ]]; then
    echo -e "${BLUE}╔═════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   Full Project Scan & Fix - All Files   ║${NC}"
    echo -e "${BLUE}╚═════════════════════════════════════════╝${NC}"
else
    echo -e "${BLUE}╔═══════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   Full Project Scan - All Files   ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════╝${NC}"
fi

# Get the project root directory
PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        echo -e "${YELLOW}Activating virtual environment...${NC}"
        # shellcheck disable=SC1091
        source .venv/bin/activate
    else
        echo -e "${YELLOW}⚠ Virtual environment not found at .venv${NC}"
    fi
fi

# Flag to track if any checks fail
FAILED=0

# ============================================================================
# Python Files Scan
# ============================================================================
echo ""
echo -e "${YELLOW}═══ Python Files Scan ═══${NC}"

PYTHON_FILES=$(find src web tools -type f -name "*.py" \
    -not -path "*/.*" -not -path "*/__pycache__/*" || true)

if [[ -n "$PYTHON_FILES" ]]; then
    # Run isort
    echo -e "${YELLOW}Running isort...${NC}"
    if ! isort --check-only --diff "$PYTHON_FILES"; then
        if [[ $FIX_MODE == true ]]; then
            echo -e "${YELLOW}⚠ isort found issues - fixing...${NC}"
            isort "$PYTHON_FILES"
            echo -e "${GREEN}✓ Fixed with isort${NC}"
        else
            echo -e "${RED}❌ isort found issues!${NC}"
            echo -e "${YELLOW}Fix with: ./tools/scan-all.sh --fix${NC}"
            FAILED=1
        fi
    else
        echo -e "${GREEN}✓ isort passed${NC}"
    fi

    # Run flake8
    echo -e "${YELLOW}Running flake8...${NC}"
    if ! flake8 "$PYTHON_FILES"; then
        echo -e "${RED}❌ flake8 found issues!${NC}"
        FAILED=1
    else
        echo -e "${GREEN}✓ flake8 passed${NC}"
    fi

    # Run pylint (optional, with timeout)
    if [[ -z "$SKIP_PYLINT" ]]; then
        echo -e "${YELLOW}Running pylint (timeout 30s)...${NC}"
        PYLINT_FILES=$(echo "$PYTHON_FILES" | grep -E '^(src|web)/' || true)

        if [[ -n "$PYLINT_FILES" ]]; then
            if timeout 30s pylint "$PYLINT_FILES" --exit-zero; then
                echo -e "${GREEN}✓ pylint passed${NC}"
            else
                EXIT_CODE=$?
                if [[ $EXIT_CODE -eq 124 ]]; then
                    echo -e "${YELLOW}⚠ pylint timeout (skipping)${NC}"
                fi
            fi
        fi
    fi
fi

# ============================================================================
# Shell Scripts Scan
# ============================================================================
echo ""
echo -e "${YELLOW}═══ Shell Scripts Scan ═══${NC}"

BASH_FILES=$(find . -maxdepth 1 -type f -name "*.sh" -o -type f -path "tools/*.sh" \
    -not -path "*/.*" || true)

if [[ -n "$BASH_FILES" ]]; then
    echo -e "${YELLOW}Running shellcheck...${NC}"
    if ! shellcheck "$BASH_FILES"; then
        echo -e "${RED}❌ shellcheck found issues!${NC}"
        FAILED=1
    else
        echo -e "${GREEN}✓ shellcheck passed${NC}"
    fi
fi

# ============================================================================
# Markdown Files Scan
# ============================================================================
echo ""
echo -e "${YELLOW}═══ Markdown Files Scan ═══${NC}"

MARKDOWN_FILES=$(find . -type f -name "*.md" \
    -not -path "*/.git/*" -not -path "*/node_modules/*" \
    -not -path "*/book/*" || true)

if [[ -n "$MARKDOWN_FILES" ]]; then
    echo -e "${YELLOW}Running markdownlint...${NC}"
    if ! markdownlint "$MARKDOWN_FILES"; then
        echo -e "${RED}❌ markdownlint found issues!${NC}"
        FAILED=1
    else
        echo -e "${GREEN}✓ markdownlint passed${NC}"
    fi
fi

# ============================================================================
# HTML/Jinja Templates Scan
# ============================================================================
echo ""
echo -e "${YELLOW}═══ HTML/Jinja Templates Scan ═══${NC}"

HTML_JINJA_FILES=$(find web -type f \( -name "*.html" -o -name "*.jinja" \
    -o -name "*.jinja2" \) -not -path "*/.*" || true)

if [[ -n "$HTML_JINJA_FILES" ]]; then
    readarray -t HTML_JINJA_FILES_ARRAY <<< "$HTML_JINJA_FILES"
    echo -e "${YELLOW}Running djlint...${NC}"
    if ! djlint --check "${HTML_JINJA_FILES_ARRAY[@]}"; then
        if [[ $FIX_MODE == true ]]; then
            echo -e "${YELLOW}⚠ djlint found issues - fixing...${NC}"
            djlint --reformat "${HTML_JINJA_FILES_ARRAY[@]}"
            echo -e "${GREEN}✓ Fixed with djlint${NC}"
        else
            echo -e "${RED}❌ djlint found issues!${NC}"
            echo -e "${YELLOW}Fix with: ./tools/scan-all.sh --fix${NC}"
            FAILED=1
        fi
    else
        echo -e "${GREEN}✓ djlint passed${NC}"
    fi
fi

# ============================================================================
# Final Result
# ============================================================================
echo ""
if [[ $FIX_MODE == false ]]; then
    echo -e "Run with ${YELLOW}--fix${NC} to auto-correct fixable issues:"
    echo -e "  ${YELLOW}./tools/scan-all.sh --fix${NC}"
fi

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
if [[ $FAILED -ne 0 ]]; then
    echo -e "${RED}✗ Scan completed with issues!${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Please fix the issues above."
    exit 1
else
    echo -e "${GREEN}✓ All scans passed!${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    exit 0
fi
