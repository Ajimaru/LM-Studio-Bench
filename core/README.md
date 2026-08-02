# Core Utilities

This directory contains shared core modules used by CLI, agents, and web app.

## Files

- `config.py`: Loads project and user configuration defaults.
- `presets.py`: Built-in and user preset management.
- `client.py`: LM Studio REST API integration helpers.
- `code_exec.py`: Sandboxed helper for generated-code capability tests.
- `tray.py`: Linux tray controller for the web dashboard.
- `paths.py`: XDG/user data path handling.
- `platform_info.py`: OS/platform detection helpers.
- `prompts.py`: Project and user prompt file lookup.
- `version.py`: Release/version lookup helpers.

## Notes

- Runtime data is not stored here.
- Results and logs are written to `~/.local/share/lm-studio-bench/`.
