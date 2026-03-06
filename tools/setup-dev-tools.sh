#!/usr/bin/env bash
#
# Setup development tools for LM-Studio-Bench
# Installs Python, Node.js, and system tools for pre-commit hooks
#
# Usage: ./tools/setup-dev-tools.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   LM-Studio-Bench Development Tools Setup   ║${NC}"
echo -e "${BLUE}╚═════════════════════════════════════════════╝${NC}"

# Get the project root directory
PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT"

# Check if virtual environment is activated
echo ""
echo -e "${YELLOW}Checking Python environment...${NC}"
if [[ -z "${VIRTUAL_ENV}" ]]; then
    if [[ -f ".venv/bin/activate" ]]; then
        echo -e "${YELLOW}Activating virtual environment...${NC}"
        source .venv/bin/activate
    else
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv .venv
        source .venv/bin/activate
    fi
fi
echo -e "${GREEN}✓ Python environment: $VIRTUAL_ENV${NC}"

# Install Python development tools
echo ""
echo -e "${YELLOW}Installing Python development tools...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
echo -e "${GREEN}✓ Python tools installed${NC}"

# Check for Node.js / npm
echo ""
echo -e "${YELLOW}Checking for Node.js...${NC}"
if command -v npm &> /dev/null; then
    echo -e "${GREEN}✓ npm found: $(npm --version)${NC}"

    # Install markdownlint-cli2
    echo ""
    echo -e "${YELLOW}Installing markdownlint-cli2 (npm)...${NC}"
    npm install -g markdownlint-cli2 2>/dev/null || {
        echo -e "${YELLOW}Note: Could not install markdownlint-cli2 globally${NC}"
        echo -e "${YELLOW}You can install it locally in the project:${NC}"
        echo -e "${YELLOW}  npm install --save-dev markdownlint-cli2${NC}"
    }
else
    echo -e "${YELLOW}⚠ Node.js/npm not found${NC}"
    echo -e "${YELLOW}To install markdownlint-cli2, install Node.js first:${NC}"
    echo -e "${YELLOW}  Ubuntu/Debian: sudo apt-get install nodejs npm${NC}"
    echo -e "${YELLOW}  macOS: brew install node${NC}"
fi

# Check for system tools
echo ""
echo -e "${YELLOW}Checking for system tools...${NC}"

if ! command -v shellcheck &> /dev/null; then
    echo -e "${YELLOW}⚠ shellcheck is not installed${NC}"
    read -p "Would you like to install shellcheck? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Installing shellcheck...${NC}"

        # Detect OS
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux (Ubuntu/Debian)
            sudo apt-get update
            sudo apt-get install -y shellcheck
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install shellcheck
            else
                echo -e "${RED}Error: Homebrew not found${NC}"
                echo -e "${YELLOW}Please install Homebrew first: https://brew.sh${NC}"
                exit 1
            fi
        else
            echo -e "${RED}Error: Unsupported OS ($OSTYPE)${NC}"
            echo -e "${YELLOW}Please install shellcheck manually${NC}"
            exit 1
        fi

        if command -v shellcheck &> /dev/null; then
            echo -e "${GREEN}✓ shellcheck installed: $(shellcheck --version | head -1)${NC}"
        else
            echo -e "${RED}Error: Failed to install shellcheck${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Skipping shellcheck installation${NC}"
        echo -e "${YELLOW}You can install it later with:${NC}"
        echo -e "${YELLOW}  Ubuntu/Debian: sudo apt-get install shellcheck${NC}"
        echo -e "${YELLOW}  macOS: brew install shellcheck${NC}"
    fi
else
    echo -e "${GREEN}✓ shellcheck found: $(shellcheck --version | head -1)${NC}"
fi

# Install git hooks
echo ""
echo -e "${YELLOW}Installing git hooks...${NC}"
./tools/install-hooks.sh

echo ""
echo -e "${BLUE}╔═════════════════════╗${NC}"
echo -e "${BLUE}║   Setup Complete!   ║${NC}"
echo -e "${BLUE}╚═════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Development tools are ready to use.${NC}"
echo -e "Your first commit will run the pre-commit checks automatically!"
echo ""
