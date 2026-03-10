# Contributing

Thanks for contributing to LM-Studio-Bench.

## Quick Start

**1.** Fork and clone the repository.
**2.** Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate.bat
```

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For testing and development tools
```

**3.** Run the benchmark or web dashboard:

```bash
python run.py --help
```

```bash
python run.py --webapp
```

**4.** Run the test suite to ensure everything works:

```bash
pytest
```

## Testing Requirements

LM-Studio-Bench maintains a comprehensive test suite with 520+ tests and 51% code coverage. When contributing:

### Running Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test module
pytest tests/test_benchmark.py

# Run tests with coverage report
pytest --cov=src --cov=web --cov-report=html

# Run tests matching a pattern
pytest -k "test_gpu_detection"
```

### Writing Tests

- Add tests for new features and bug fixes
- Place tests in the appropriate test file under `tests/`
- Follow existing test patterns and naming conventions
- Use fixtures from `tests/conftest.py` for common setup
- Mock external dependencies (LM Studio API, system commands, file I/O)
- Ensure tests are isolated and can run in any order

### Test Organization

The test suite is organized by module:

- `test_benchmark.py` - Core benchmark logic and caching
- `test_app.py` - Web dashboard backend
- `test_api_endpoints.py` - REST API endpoint testing
- `test_rest_client.py` - LM Studio REST API client
- `test_tray.py` - Linux tray control functionality
- `test_preset_manager.py` - Configuration preset management
- `test_config_loader.py` - Configuration loading and validation
- `test_user_paths.py` - XDG Base Directory compliance
- `test_version_checker.py` - Version checking logic
- `test_scrape_metadata.py` - Model metadata extraction
- `test_run.py` - Main entry point and CLI argument handling

### Continuous Integration

All pull requests are automatically tested via GitHub Actions:

- Unit tests must pass
- Code quality checks (flake8, pylint)
- Security scans (Bandit, CodeQL, Snyk)
- Markdown linting

Check the CI status before requesting review.

## Development Guidelines

- Keep changes focused and minimal.
- Preserve existing behavior unless the change explicitly targets behavior
  updates.
- Follow the current style of each file (Python/Markdown/HTML).
- Use type hints and docstrings for public functions where applicable.
- Use logging for notable events; avoid print for application logging.
- Do not abort the full benchmark on a single model failure.
- Update documentation when behavior or CLI options change.

## Typical Contribution Areas

- Benchmark runner and caching (`src/benchmark.py`)
- Web dashboard backend (`web/app.py`)
- Dashboard UI (`web/templates/dashboard.html.jinja`)
- Metadata tooling (`tools/scrape_metadata.py`)
- Documentation (`docs/` and `README.md`)

## Commit and PR Guidance

- Use clear commit messages in imperative style.
  - Example: `Add cache cleanup for failed models`
- Reference related issues in PR descriptions.
- Run tests before submitting:
  - `pytest` - Ensure all tests pass
  - `pytest --cov=src --cov=web` - Check coverage impact
- Include a short test or verification section in each PR, e.g.:
  - `pytest tests/test_benchmark.py -v`
  - `python run.py --help`
  - `python run.py --export-only`
  - manual web dashboard check

## Documentation Expectations

If you change benchmark or web behavior, update at least:

- `README.md`
- `docs/QUICKSTART.md`
- `docs/CONFIGURATION.md`

If you contribute code or documentation, add yourself to `AUTHORS`.

## Code of Conduct and Security

Please follow:

- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
