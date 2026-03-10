# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive test suite with 520+ tests covering all Python modules
- Test coverage of 51% across the entire codebase
- Unit tests for all core components:
  - `test_benchmark.py` - Benchmark runner and caching (55+ tests)
  - `test_app.py` - Web dashboard backend (23+ tests)
  - `test_api_endpoints.py` - REST API endpoints (32+ tests)
  - `test_rest_client.py` - REST API client (22+ tests)
  - `test_tray.py` - Linux tray control (26+ tests)
  - `test_preset_manager.py` - Preset management (19+ tests)
  - `test_config_loader.py` - Configuration loading (9+ tests)
  - `test_user_paths.py` - XDG path handling (4+ tests)
  - `test_version_checker.py` - Version checking (7+ tests)
  - `test_scrape_metadata.py` - Metadata scraping (24+ tests)
  - `test_run.py` - Main entry point (10+ tests)
- Test fixtures and utilities in `conftest.py`
- Pytest configuration in `pytest.ini`
- Development requirements in `requirements-dev.txt`

### Changed

- Improved code quality and maintainability through comprehensive testing
- Enhanced reliability of benchmark execution
- Better error handling validated through unit tests

---

For detailed release history, see the [repository tags](https://github.com/Ajimaru/LM-Studio-Bench/tags) and [pull requests](https://github.com/Ajimaru/LM-Studio-Bench/pulls).
