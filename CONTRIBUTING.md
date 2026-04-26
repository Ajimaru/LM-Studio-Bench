# Contributing

Thanks for contributing to LM-Studio-Bench.

## Quick Start

**1.** Fork and clone the repository.
**2.** Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

```bash
pip install -r requirements-dev.txt
```

`requirements-dev.txt` is the correct environment for contributors and test
runs. It includes the runtime dependencies from `requirements.txt` plus the
extra tooling needed for `pytest`, linting, and local development.

**3.** Run the benchmark or web dashboard:

```bash
python run.py --help
```

```bash
python run.py --webapp
```

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

- Benchmark runner and caching (`cli/benchmark.py`, `agents/cache.py`)
- Web dashboard backend (`web/app.py`)
- Dashboard UI (`web/templates/dashboard.html.jinja`)
- Metadata tooling (`tools/scrape_metadata.py`)
- Documentation (`docs/` and `README.md`)

## Commit and PR Guidance

- Use clear commit messages in imperative style.
  - Example: `Add cache cleanup for failed models`
- Reference related issues in PR descriptions.
- Include a short test or verification section in each PR, e.g.:
  - `pytest`
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
