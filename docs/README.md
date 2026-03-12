# LM Studio Benchmark Documentation

Welcome to the LM Studio Benchmark documentation! This tool helps you measure and compare token/s performance across all your locally installed LLM models and their quantizations.

## What is this?

A Python benchmark tool for LM Studio with a modern web dashboard that:

- **Automatically tests** all local LLM models and quantizations
- **Measures token/s speeds** with warmup and multiple runs
- **Exports results** in JSON, CSV, PDF, and interactive HTML formats
- **Detects GPU capabilities** (NVIDIA, AMD, Intel) and monitors VRAM usage
- **Provides a web dashboard** with live charts and filtering options
- **Includes Linux tray controls** with live status icons and quick actions

## Quick Links

- [Quickstart Guide](QUICKSTART.md) — Get started in 5 minutes
- [Configuration Reference](CONFIGURATION.md) — All CLI arguments and config file options
- [Architecture Documentation](ARCHITECTURE.md) — System architecture with Mermaid diagrams, including testing architecture
- [REST API Integration](REST_API_FEATURES.md) — Advanced features with LM Studio API v1
- [Hardware Monitoring](HARDWARE_MONITORING_GUIDE.md) — GPU, CPU, RAM tracking
- [LLM Metadata Guide](LLM_METADATA_GUIDE.md) — Model capabilities and metadata
- [User Data & Configuration](USER_DATA.md) — XDG directory structure and config management

## Features at a Glance

✅ Multi-model benchmarking with intelligent GPU offload
✅ Vision & tool-calling model detection
✅ Progressive VRAM management (automatic fallback)
✅ Caching system (skip already-tested models)
✅ Filter by quantization, architecture, params, context length
✅ Live web dashboard with 27 themes
✅ Linux tray controller with dynamic benchmark status icons
✅ REST API mode with parallel inference support
✅ Download progress tracking, MCP integration, stateful chats
✅ Response caching with 10,000x+ speedup for repeated prompts
✅ **520+ comprehensive tests with 51% code coverage**
✅ **Automated CI/CD with quality checks and security scans**  

## Getting Started

Check out the [Quickstart Guide](QUICKSTART.md) to begin benchmarking your models!
