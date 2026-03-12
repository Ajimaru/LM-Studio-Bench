# Source Code

This directory contains the main Python application code.

## Files

- `benchmark.py`: Core benchmark runner, cache handling, exports, and CLI.
- `config_loader.py`: Loads project and user configuration defaults.
- `preset_manager.py`: Built-in and user preset management.
- `rest_client.py`: LM Studio REST API integration helpers.
- `tray.py`: Linux tray controller for the web dashboard.
- `user_paths.py`: XDG/user data path handling.
- `version_checker.py`: Release/version lookup helpers.

## Notes

- Runtime data is not stored here.
- Results and logs are written to `~/.local/share/lm-studio-bench/`.
