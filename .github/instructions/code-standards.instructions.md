---
description: Code standards and best practices for LM-Studio-Bench
applyTo: '**'
---

# Code Standards and Best Practices

## File Naming Conventions

### NEVER use long file names (max 50 chars including extension)

- Use lowercase, underscores, descriptive but concise names

### Maximum line length: 80 characters

- Python: 88 chars (Black formatter standard, acceptable)
- Bash/Shell: 100 chars (acceptable for long paths)
- Break long lines using parentheses, backslashes, or line continuation

- **Linter codes to avoid:**

  - `E501` (pycodestyle): Line too long
  - `W505` (pycodestyle): Doc line too long

### NEVER use `assert` in production code (disabled with `python -O`)

- Use explicit error handling: `if not x: raise ValueError(...)`
- Assertions only in test files (`test_*.py`)

- **Linter codes to avoid:**

  - `B101` (Bandit): Use of assert detected
  - `S101` (Ruff): Use of assert detected
  - `reportAssertAlwaysTrue` (Pylance): Assert is always true

### ALWAYS use spaces, NEVER tabs

- Python/Bash: 4 spaces
- YAML: 2 spaces
- HTML/CSS/JS: 2 or 4 spaces (consistent per file)

- **Linter codes to avoid:**

  - `W191` (pycodestyle): Indentation contains tabs
  - `E101` (pycodestyle): Indentation contains mixed spaces and tabs
  - `reportTabsNotSpaces` (Pylance): Use of tabs instead of spaces

### NEVER leave trailing whitespace on blank lines

- Keep blank lines empty (no spaces/tabs)

- **Linter codes to avoid:**

  - `W293` (pycodestyle): Blank line contains whitespace

### ALWAYS quote variable expansions in shell scripts

- Use `"$variable"` not `$variable`
- Prevents word splitting and globbing issues

- **Linter codes to avoid:**

  - `SC2086` (shellcheck): Double quote to prevent globbing and word splitting
  - `SC2046` (shellcheck): Quote this to prevent word splitting
  - `SC2248` (shellcheck): Prefer explicit -n to check for output

### ALWAYS document public functions, classes, modules

- Use triple double-quotes: `"""`
- Include: brief description, Args, Returns, Raises
- Follow Google-style format

- **Linter codes to avoid:**

  - `D100-D107` (pydocstyle): Missing docstrings
  - `D200-D215` (pydocstyle): Docstring formatting issues
  - `C0115` (Pylint): Missing class docstring
  - `C0116` (Pylint): Missing function docstring

### AVOID accessing protected members (starting with `_`) from outside class

- Use public methods/properties or helper functions
- In tests: Use helper functions like `_call_member(obj, "method_name")`
  instead of `obj._method()`

- **Linter codes to avoid:**

  - `W0212` (Pylint): Access to a protected member of a client class

### ALWAYS use all declared function arguments or prefix with `_`

- Unused arguments should be prefixed: `def func(_unused_arg):`
- Or use `*args, **kwargs` if truly variable
- Remove completely if not needed

- **Linter codes to avoid:**

  - `W0613` (Pylint): Unused argument
  - `ARG001` (Ruff): Unused function argument

### Python Best Practices

- Follow PEP 8 style guidelines strictly
- Use type hints for all function parameters and return values
- Prefer f-strings for string formatting over older methods
- Use virtual environments for dependency management

### Clean Code Principles

- Keep functions small and focused on a single responsibility
- Avoid deep nesting and complex conditional statements
- Write code that tells a story and is easy to understand
- Refactor ruthlessly to eliminate code smells

### Testing Guidelines

- Write comprehensive unit tests for all business logic
- Follow the AAA pattern: Arrange, Act, Assert
- Maintain good test coverage (aim for 80%+ for critical paths)
- Write descriptive test names that explain the expected behavior
- Use test doubles (mocks, stubs, spies) appropriately
- Implement integration tests for API endpoints and user flows
- Keep tests fast, isolated, and deterministic

### When generating code, please

- Generate complete, working code examples with proper imports
- Include inline comments for complex logic and business rules
- Follow the established patterns and conventions in this project
- Suggest improvements and alternative approaches when relevant
- Consider performance, security, and maintainability
- Follow accessibility best practices for UI components
- Use semantic HTML and proper ARIA attributes when applicable

---
