# Web Dashboard

This directory contains the FastAPI web application.

## Files

- `app.py`: FastAPI backend, WebSocket streaming, and API routes.
- `templates/`: Jinja templates for the dashboard UI.

## Notes

- Benchmark runs are launched as subprocesses from the web app.
- Web app logs are written to `~/.local/share/lm-studio-bench/logs/`.
