# Web Dashboard

This directory contains the FastAPI web application and dashboard UI.

## Files

- `app.py`: FastAPI backend, WebSocket streaming, benchmark process
  control, preset APIs, result APIs and shutdown/status endpoints.
- `templates/`: Jinja templates for the dashboard UI.

## Notes

- Classic and capability benchmark runs are launched as sanitized
  subprocesses from the web app.
- Web app logs are written to `~/.local/share/lm-studio-bench/logs/`.
- Cached results are read from
  `~/.local/share/lm-studio-bench/results/benchmark_cache.db`.
