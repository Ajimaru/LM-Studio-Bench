---
description: Code standards and best practices for LM-Studio-Tray-Manager
applyTo: '**'
---

# Code Standards & Best Practices

## File Naming Conventions

❌ **NEVER use long file names** (max 50 chars including extension)
- Use lowercase, underscores, descriptive but concise names
- Example: `lmstudio_tray.py` ✅ vs `lmstudio_tray_manager_with_gtk3_integration_module.py` ❌

## Line Length

✅ **Maximum line length: 80 characters**
- Python: 88 chars (Black formatter standard, acceptable)
- Bash/Shell: 100 chars (acceptable for long paths)
- Break long lines using parentheses, backslashes, or line continuation
- **Linter codes to avoid:**
  - `E501` (pycodestyle): Line too long
  - `W505` (pycodestyle): Doc line too long

## Assertions & Testing

❌ **NEVER use `assert` in production code** (disabled with `python -O`)
- Use explicit error handling: `if not x: raise ValueError(...)`
- Assertions only in test files (`test_*.py`)
- **Linter codes to avoid:**
  - `B101` (Bandit): Use of assert detected
  - `S101` (Ruff): Use of assert detected
  - `reportAssertAlwaysTrue` (Pylance): Assert is always true

## Indentation & Whitespace

✅ **ALWAYS use spaces, NEVER tabs**
- Python/Bash: 4 spaces
- YAML: 2 spaces
- HTML/CSS/JS: 2 or 4 spaces (consistent per file)
- **Linter codes to avoid:**
  - `W191` (pycodestyle): Indentation contains tabs
  - `E101` (pycodestyle): Indentation contains mixed spaces and tabs
  - `reportTabsNotSpaces` (Pylance): Use of tabs instead of spaces

❌ **NEVER leave trailing whitespace on blank lines**
- Keep blank lines empty (no spaces/tabs)
- **Linter codes to avoid:**
  - `W293` (pycodestyle): Blank line contains whitespace

## Variable Quoting (Bash/Shell)

✅ **ALWAYS quote variable expansions** in shell scripts
- Use `"$variable"` not `$variable`
- Prevents word splitting and globbing issues
- **Linter codes to avoid:**
  - `SC2086` (shellcheck): Double quote to prevent globbing and word splitting
  - `SC2046` (shellcheck): Quote this to prevent word splitting
  - `SC2248` (shellcheck): Prefer explicit -n to check for output

## Docstrings & Documentation

✅ **ALWAYS document public functions, classes, modules**
- Use triple double-quotes: `"""`
- Include: brief description, Args, Returns, Raises
- Follow Google-style format
- **Linter codes to avoid:**
  - `D100-D107` (pydocstyle): Missing docstrings
  - `D200-D215` (pydocstyle): Docstring formatting issues
  - `C0115` (Pylint): Missing class docstring
  - `C0116` (Pylint): Missing function docstring

## Protected Members Access

❌ **AVOID accessing protected members** (starting with `_`) from outside class
- Use public methods/properties or helper functions
- In tests: Use helper functions like `_call_member(obj, "method_name")` instead of `obj._method()`
- **Linter codes to avoid:**
  - `W0212` (Pylint): Access to a protected member of a client class

## Function Arguments

✅ **ALWAYS use all declared function arguments** or prefix with `_`
- Unused arguments should be prefixed: `def func(_unused_arg):`
- Or use `*args, **kwargs` if truly variable
- Remove completely if not needed
- **Linter codes to avoid:**
  - `W0613` (Pylint): Unused argument
  - `ARG001` (Ruff): Unused function argument
