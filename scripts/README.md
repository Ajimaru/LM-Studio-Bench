# Scripts Directory

This directory contains non-Python development and build scripts for LM-Studio-Bench.

## AppImage Build

Build a local AppImage artifact using one of two methods:

### Method 1: Docker Build (Recommended)

Uses Ubuntu 24.04 Docker container for reproducible builds:

```bash
# Make script executable once
chmod +x ./scripts/build_appimage_docker.sh

# Build dist/LM-Studio-Bench-<VERSION>-x86_64.AppImage
./scripts/build_appimage_docker.sh
```

Requirements:

- `docker` installed and running
- No other dependencies needed (all bundled in container)

### Method 2: Native Build

Build directly on Ubuntu 24.04 (or compatible):

```bash
# Make script executable once
chmod +x ./scripts/build_appimage.sh

# Build dist/LM-Studio-Bench-<VERSION>-x86_64.AppImage
./scripts/build_appimage.sh
```

Requirements:

- `python3`
- `appimagetool` in `PATH`
  - Download from: [AppImageTool continuous releases](https://github.com/AppImage/appimagetool/releases/tag/continuous)
  - Install: `chmod +x appimagetool-*.AppImage && sudo mv appimagetool-*.AppImage /usr/local/bin/appimagetool`
- LM Studio CLI (`lms`) is **not** bundled and must exist on host

Output:

- `dist/LM-Studio-Bench-<VERSION>-x86_64.AppImage`

### Docker Details

The Docker build:

- Uses `ubuntu:24.04` base image
- Installs Python 3, appimagetool, and all dependencies
- Runs appimagetool in extracted mode (no FUSE required)
- Copies a minimal build context (incl. runtime `scripts/`) via `docker cp`
- Works even when Docker Desktop path sharing for `/mnt` is disabled
- Reuses a persistent build container by default (`lmstudio-bench-appimage-build-cache`)
- Preserves file permissions for the generated AppImage
- Dockerfile: `scripts/Dockerfile.AppImage`
- Image name: `lmstudio-bench-appimage-builder:ubuntu24.04`

Optional environment variables:

- `REUSE_CONTAINER=1` (default): Reuse existing build container and caches
- `REUSE_CONTAINER=0`: Remove and recreate build container for a clean run
- `FAST_MODE=1`: Skip `docker build` if builder image already exists
- `CONTAINER_NAME=<name>`: Use a custom container name

AppStream metadata notes:

- AppImage builds include desktop/appdata metadata to satisfy
  `appimagetool` validation.
- Metadata sources:
  - `scripts/io.github.Ajimaru.LMStudioBench.desktop`
  - `scripts/io.github.Ajimaru.LMStudioBench.appdata.xml`
- If the builder image changes, the Docker script recreates the cached
  build container automatically to avoid stale dependency issues.

## Git Hooks

### Pre-commit Hook

The pre-commit hook runs automatic code quality checks before each commit.
Fixable issues are corrected automatically and re-staged.

**Checks performed:**

- **Python**: `isort` (import sorting with auto-fix), `flake8` (linting), `pylint` (code quality, optional)
- **Shell**: `shellcheck` (bash/shell linting)
- **Markdown**: `markdownlint` (markdown linting)
- **HTML/Jinja**: `djlint` (template linting with auto-fix formatting)

### Installation

**Easy Setup (recommended):**

```bash
# 1. Run the complete setup script
./scripts/setup-dev-tools.sh

# This will:
# - Create/activate Python virtual environment
# - Install Python dev tools (flake8, isort, pylint, djlint)
# - Install Node.js tools (markdownlint-cli2)
# - Ask to install system tools (shellcheck)
# - Install git hooks
```

**Manual Setup:**

```bash
# 1. Install git hooks
./scripts/install-hooks.sh

# 2. Install required Python tools
pip install -r requirements-dev.txt

# 3. Install system tools
# On Ubuntu/Debian:
sudo apt-get install shellcheck

# On macOS:
brew install shellcheck

# 4. Install Node.js tools (if you have npm)
npm install -g markdownlint-cli2
```

### Skip pylint (optional)

Pylint runs by default on `core/`, `cli/`, `agents/` and `web/` files with a 10-second timeout. To skip it:

```bash
SKIP_PYLINT=1 git commit -m "Your message"
```

Or to disable only on timeout:

- Timeout takes 10 seconds max, then continues
- If pylint is very slow, you can skip it entirely

### Usage

The pre-commit hook runs automatically when you commit:

```bash
git commit -m "Your commit message"
```

**If checks fail:**

1. The hook auto-fixes `isort` and `djlint` issues and re-stages changes
2. Fix remaining reported issues manually
3. Commit again: `git commit -m "Your message"`

**To skip the hook (not recommended):**

```bash
git commit --no-verify -m "Your message"
```

**To skip only pylint (optional):**

```bash
SKIP_PYLINT=1 git commit -m "Your message"
```

---

## Code Quality Tools

### Python Tools

#### isort - Import Sorting

Sort and organize Python imports automatically.

```bash
# Check imports without modifying files
isort --check-only --diff core/ cli/ agents/ web/ tools/

# Apply changes
isort core/ cli/ agents/ web/ tools/

# Check specific file
isort --check-only cli/benchmark.py
isort cli/benchmark.py  # Apply changes
```

**Configuration:** `pyproject.toml` → `[tool.isort]`

#### flake8 - Python Linter

Check Python code for style and errors (PEP 8, errors, complexity).

```bash
# Check all Python files
flake8 core/ cli/ agents/ web/ tools/

# Check specific file
flake8 cli/benchmark.py
```

**Configuration:** `.flake8`

**Common errors:**

- `E501`: Line too long (max 88 chars in config)
- `F401`: Import not used
- `W293`: Blank line contains whitespace
- `E302`: Expected 2 blank lines

#### pylint - Code Quality Checker

Advanced Python code analysis (complexity, maintainability, conventions).

```bash
# Check specific file
pylint cli/benchmark.py

# Check with specific options
pylint --disable=fixme core/ cli/ agents/

# Check core/, cli/, agents/ and web/ directories
pylint core/ cli/ agents/ web/
```

**Configuration:** `.pylintrc`

**In pre-commit:**

- Runs on `core/`, `cli/`, `agents/` and `web/` directories (not on `scripts/`)
- Has 10-second timeout (prevents slowdown)
- Optional: `SKIP_PYLINT=1 git commit`

**Common issues:**

- `C0103`: Invalid name
- `C0114`: Missing module docstring
- `R0903`: Too few public methods
- `R0913`: Too many arguments

### Shell Tools

#### shellcheck - Bash/Shell Linter

Lint shell scripts for errors and best practices.

```bash
# Check specific script
shellcheck scripts/install-hooks.sh

# Check multiple scripts
shellcheck setup.sh scripts/*.sh

# Fix issues (some can be auto-fixed)
shellcheck -format=json scripts/install-hooks.sh
```

**Common issues:**

- `SC2086`: Quote variables
- `SC2048`: Use "$@" instead of $*
- `SC2046`: Quote to prevent word splitting

### Markdown Tools

#### markdownlint - Markdown Linter

Check Markdown files for style consistency.

```bash
# Check specific file
markdownlint docs/README.md

# Check all markdown files
markdownlint docs/**/*.md

# Check with detailed output
markdownlint -d relaxed README.md
```

**Configuration:** `.markdownlintrc.json`

**Common rules:**

- `MD001`: Heading levels should increase by one
- `MD003`: Heading style consistency
- `MD013`: Line length (100 chars in config)

### HTML/Jinja Tools

#### djlint - HTML/Jinja Template Linter

Lint and format HTML and Jinja2 templates.

```bash
# Check templates
djlint --check web/templates/

# Check specific file
djlint --check web/templates/dashboard.html.jinja

# Auto-format templates
djlint --reformat web/templates/

# Check with detailed output
djlint --check --verbose web/templates/

# Format specific file
djlint --reformat web/templates/dashboard.html.jinja
```

**Configuration:** `pyproject.toml` → `[tool.djlint]`

**Options:**

- `--check`: Check only, don't modify
- `--reformat`: Auto-format files
- `--profile jinja`: Use Jinja profile
- `--format`: Output format (default is text)

---

## Configuration Files

| File | Purpose |
| --- | --- |
| `.flake8` | Flake8 style configuration |
| `.pylintrc` | Pylint configuration |
| `.markdownlintrc.json` | Markdown lint rules |
| `.shellcheckrc` | Shellcheck configuration |
| `pyproject.toml` | isort and djlint configuration |
| `requirements-dev.txt` | Python development dependencies |

## Setup & Installation Scripts

| Script | Purpose |
| --- | --- |
| `scripts/setup-dev-tools.sh` | Full setup (recommended) - installs all tools |
| `scripts/install-hooks.sh` | Install git hooks only |
| `scripts/scan-all.sh` | Scan all project files (with optional `--fix`) |
| `scripts/pre-commit` | Git pre-commit hook (runs linters) |

---

## Usage Examples

### Before committing

```bash
# Run all checks manually
isort --check-only core/ cli/ agents/ web/ tools/
flake8 core/ cli/ agents/ web/ tools/
pylint core/ cli/ agents/ web/      # Code quality (optional)
shellcheck scripts/*.sh
markdownlint docs/**/*.md
djlint --check web/templates/

# Or use the hook automatically in git commit
```

### Scan all files

Scan the entire project (not just staged files):

```bash
# Full scan
./scripts/scan-all.sh

# Skip pylint (faster)
SKIP_PYLINT=1 ./scripts/scan-all.sh
```

### Auto-fix issues

Automatically fix issues that can be auto-corrected:

```bash
# Interactive auto-fix (fixes isort and djlint issues)
./scripts/scan-all.sh --fix
```

**What gets auto-fixed:**

- ✅ isort - import order (automatic)
- ✅ djlint - HTML/Jinja formatting (automatic)

**What needs manual fixes:**

- ⚠️ flake8 - style issues
- ⚠️ pylint - code quality
- ⚠️ markdownlint - markdown style
- ⚠️ shellcheck - shell scripts

### Fix all files

```bash
# Fix import order
isort core/ cli/ agents/ web/ tools/

# Auto-format templates
djlint --reformat web/templates/

# Fix flake8/shellcheck/markdownlint issues manually
```

### Disable specific checks inline

**Python:**

```python
import os, sys  # noqa: E401 (flake8)
result = function()  # noqa: E501 (flake8)

# pylint: disable=line-too-long  (entire section)
def very_long_function_name_that_exceeds_line_length():
    pass
# pylint: enable=line-too-long

# Single line pylint disable
value = some_long_value  # pylint: disable=unused-variable
```

**Shell:**

```bash
# shellcheck disable=SC2086
result=$var
```

**Markdown:**

```markdown
<!-- markdownlint-disable MD013 -->
This is a very long line that exceeds the normal limit for documentation purposes
<!-- markdownlint-enable MD013 -->
```

**HTML/Jinja:**

```jinja
{# djlint-disable #}
<div>
  Some unformatted HTML
</div>
{# djlint-enable #}
```

---

## Troubleshooting

### Virtual environment not activated

The pre-commit hook tries to auto-activate `.venv/bin/activate`.
If using a different environment:

```bash
source /path/to/your/venv/bin/activate
```

### Missing tools

Use the setup script for easy installation:

```bash
./scripts/setup-dev-tools.sh
```

Or install manually:

```bash
# Python tools
pip install -r requirements-dev.txt

# System tools
sudo apt-get install shellcheck     # Ubuntu/Debian
brew install shellcheck              # macOS

# Node.js tool (requires npm)
npm install -g markdownlint-cli2
```

### Pylint is slow

Pylint has a 10-second timeout in the pre-commit hook. If it's still timing out:

```bash
# Skip pylint for this commit
SKIP_PYLINT=1 git commit -m "Your message"

# Run pylint manually later on specific files
pylint cli/benchmark.py
```

### Hook doesn't run

Make sure it's executable:

```bash
chmod +x .git/hooks/pre-commit
```

### Remove the hook

```bash
rm .git/hooks/pre-commit
```

---
