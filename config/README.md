# Configuration Defaults

This directory contains version-controlled default configuration.

## Files

- `defaults.json`: Project-wide default settings for classic benchmarks.
- `bench.yaml`: Default settings for capability-driven benchmark runs.

## Override Order

1. CLI arguments
2. User config in `~/.config/lm-studio-bench/defaults.json`
3. Project defaults in this directory
4. Hard-coded fallback values

User-created benchmark presets are stored in
`~/.config/lm-studio-bench/presets/`.
