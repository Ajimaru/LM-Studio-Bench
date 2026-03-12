#!/usr/bin/env bash
#
# Install git hooks for LM-Studio-Bench
#
# Usage: ./tools/install-hooks.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Installing git hooks...${NC}"

# Get the project root directory
PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        echo -e "${YELLOW}Activating virtual environment...${NC}"
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
fi

# Check if we're in a git repository
if [[ ! -d ".git" ]]; then
    echo -e "${RED}Error: Not a git repository${NC}"
    exit 1
fi

# Create .git/hooks directory if it doesn't exist
mkdir -p .git/hooks

# Copy pre-commit hook
echo -e "${YELLOW}Installing pre-commit hook...${NC}"
cp tools/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

echo -e "${GREEN}✓ pre-commit hook installed${NC}"

# Check if required tools are installed
echo ""
echo -e "${YELLOW}Checking for required tools...${NC}"

MISSING_TOOLS=()
MISSING_SYSTEM_TOOLS=()

# Python tools (pip)
for tool in flake8 isort pylint djlint markdownlint; do
    if ! command -v "$tool" &> /dev/null; then
        MISSING_TOOLS+=("$tool")
    fi
done

# System tools (package manager)
if ! command -v shellcheck &> /dev/null; then
    MISSING_SYSTEM_TOOLS+=("shellcheck")
fi

# Report missing tools
if [[ ${#MISSING_TOOLS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}Missing Python tools: ${MISSING_TOOLS[*]}${NC}"
    echo -e "${YELLOW}Install with: pip install ${MISSING_TOOLS[*]}${NC}"
    echo ""
fi

# Check if pylint is available
echo ""
if command -v pylint &> /dev/null; then
    echo -e "${GREEN}✓ pylint is installed${NC}"
    echo -e "${YELLOW}Pylint will run for src/ and web/ directories during pre-commit${NC}"
    echo -e "${YELLOW}To skip pylint: ${YELLOW}SKIP_PYLINT=1 git commit${NC}"
else
    echo -e "${YELLOW}Note: pylint is not installed but is optional${NC}"
    echo -e "${YELLOW}It can be skipped with SKIP_PYLINT=1 git commit${NC}"
fi

if [[ ${#MISSING_SYSTEM_TOOLS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}Missing system tools: ${MISSING_SYSTEM_TOOLS[*]}${NC}"
    echo -e "${YELLOW}Install on Ubuntu/Debian: sudo apt-get install ${MISSING_SYSTEM_TOOLS[*]}${NC}"
    echo -e "${YELLOW}Install on macOS: brew install ${MISSING_SYSTEM_TOOLS[*]}${NC}"
    echo ""
fi

echo -e "${GREEN}Git hooks installed successfully!${NC}"
echo ""
echo -e "The pre-commit hook will now run automatically before each commit."
echo -e "To skip the hook, use: ${YELLOW}git commit --no-verify${NC}"
echo ""
