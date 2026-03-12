# Security Policy

## Supported Versions

This project currently supports security fixes on the `main` branch.

| Version | Supported |
| --- | --- |
| `main` | Yes |
| older commits/tags | No |

## Reporting a Vulnerability

Please do **not** open public issues for sensitive security reports.

Use one of these channels:

- GitHub Security Advisories (preferred):
  - Repository > Security > Advisories > Report a vulnerability
- Direct contact with the maintainer (`@Ajimaru`) if advisories are unavailable

Please include:

- A clear description of the issue
- Reproduction steps or proof of concept
- Potential impact
- Suggested mitigation (if available)

## Response Process

- Initial triage target: within **days**
- If confirmed, a fix will be prepared on `main`
- Coordinated disclosure is preferred until a fix is available

## Scope Notes

Security reports are especially relevant for:

- Command execution and process control in `src/benchmark.py`
- Web endpoints and WebSocket handling in `web/app.py`
- Metadata scraping and subprocess calls in `tools/scrape_metadata.py`
- Cache and results handling under `~/.local/share/lm-studio-bench/`
- Configuration handling in `~/.config/lm-studio-bench/`
- CI/CD and GitHub Actions workflow security
