#!/usr/bin/env python3
"""Version checking utilities for LM Studio Benchmark.

Provides functions to read the current version, fetch the latest
release from GitHub, compare versions, and format release URLs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import httpx

LOGGER = logging.getLogger(__name__)


def get_current_version() -> str:
    """Read the current version from VERSION file.

    The VERSION file is located at project root and contains a single
    line with the version string (e.g., "v0.1.0").

    Returns:
        Version string (e.g., "v0.1.0"), or "unknown" if file not found.

    Raises:
        ValueError: If VERSION file is empty or unreadable.
    """
    project_root = Path(__file__).resolve().parent.parent
    version_file = project_root / "VERSION"

    if not version_file.exists():
        LOGGER.warning("VERSION file not found at %s", version_file)
        return "unknown"

    try:
        current = version_file.read_text(encoding="utf-8").strip()
        if not current:
            raise ValueError("VERSION file is empty")
        LOGGER.debug("Current version: %s", current)
        return current
    except (OSError, ValueError) as exc:
        LOGGER.error("Failed to read VERSION file: %s", exc)
        raise ValueError(f"Cannot read VERSION: {exc}") from exc


def fetch_latest_release() -> Optional[dict]:
    """Fetch latest release info from GitHub API.

    Queries https://api.github.com/repos/Ajimaru/LM-Studio-Bench/
    releases/latest and returns the JSON response.

    Returns:
        Dict with keys 'tag_name', 'html_url' on success, None on
        failure.

    Raises:
        None - Errors are logged and None is returned (graceful
        degradation).
    """
    url = (
        "https://api.github.com/repos/Ajimaru/"
        "LM-Studio-Bench/releases/latest"
    )

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            LOGGER.debug(
                "GitHub latest release: %s", data.get("tag_name")
            )
            return data
    except httpx.HTTPStatusError as exc:
        LOGGER.warning(
            "GitHub API error (%s): %s",
            exc.response.status_code,
            exc
        )
        return None
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        LOGGER.warning("Failed to fetch latest release: %s", exc)
        return None
    except (ValueError, KeyError) as exc:
        LOGGER.error("Failed to parse GitHub API response: %s", exc)
        return None


def compare_versions(
    current: str, latest: str
) -> bool:
    """Check if a newer version is available.

    Compares two semantic version strings (e.g., "v0.1.0" vs
    "v0.2.0"). Returns True if latest > current.

    Args:
        current: Current version string (e.g., "v0.1.0").
        latest: Latest version string (e.g., "v0.2.0").

    Returns:
        True if latest is newer than current, False otherwise.
    """
    def parse_version(v: str) -> tuple:
        """Parse version string to tuple (major, minor, patch)."""
        v_clean = v.lstrip("v").split("-")[0]  # Remove "v" prefix, ignore pre-release
        try:
            parts = v_clean.split(".")
            return tuple(int(p) for p in parts[:3])
        except (ValueError, IndexError):
            LOGGER.warning(
                "Invalid version format: %s, assuming older", v
            )
            return (0, 0, 0)

    current_tuple = parse_version(current)
    latest_tuple = parse_version(latest)

    is_update_available = latest_tuple > current_tuple
    LOGGER.debug(
        "Version comparison: %s (%s) < %s (%s) = %s",
        current,
        current_tuple,
        latest,
        latest_tuple,
        is_update_available,
    )
    return is_update_available


def format_release_url(tag_name: str) -> str:
    """Format a GitHub release URL from tag name.

    Args:
        tag_name: Tag name from GitHub (e.g., "v0.2.0").

    Returns:
        Full GitHub release URL.
    """
    url = (
        f"https://github.com/Ajimaru/LM-Studio-Bench/"
        f"releases/tag/{tag_name}"
    )
    LOGGER.debug("Release URL: %s", url)
    return url
