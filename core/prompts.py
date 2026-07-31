"""Prompt files for benchmark runs.

Long benchmark prompts (a file of source code, a multi-turn transcript) are
impractical as CLI strings or textarea contents, so they live as files in two
places: ``prompts/`` inside the project for the shipped ones, and a
user-writable directory for anything added locally.

Callers pass a bare file name, never a path. Resolution is confined to those
two directories because the name can originate from an HTTP request, and a
raw path there would be a traversal waiting to happen.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from core.paths import USER_CONFIG_DIR

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_PROMPTS_DIR = PROJECT_ROOT / "prompts"
USER_PROMPTS_DIR = USER_CONFIG_DIR / "prompts"

PROMPT_SUFFIXES = (".md", ".txt")

# Guards against loading something that is not a prompt at all, and keeps a
# stray multi-megabyte file from being pushed through the API as an argument.
MAX_PROMPT_BYTES = 512 * 1024


def prompt_dirs() -> List[Path]:
    """Directories searched for prompt files, project first."""
    return [PROJECT_PROMPTS_DIR, USER_PROMPTS_DIR]


def list_prompt_files() -> List[str]:
    """Return available prompt file names, sorted and de-duplicated."""
    names: list[str] = []
    seen: set[str] = set()
    for directory in prompt_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in PROMPT_SUFFIXES:
                continue
            if path.name in seen:
                continue
            names.append(path.name)
            seen.add(path.name)
    return names


def resolve_prompt_file(name: str) -> Path:
    """Resolve a prompt file name to a path inside a known prompt directory.

    Args:
        name: Bare file name, e.g. ``coding_assistant.md``.

    Returns:
        Absolute path to an existing prompt file.

    Raises:
        ValueError: If the name is empty, carries path separators, uses an
            unsupported suffix, escapes the prompt directories, or is too big.
        FileNotFoundError: If no prompt directory holds that file.
    """
    candidate_name = (name or "").strip()
    if not candidate_name:
        raise ValueError("Prompt file name must not be empty")

    if Path(candidate_name).name != candidate_name:
        raise ValueError(
            f"Prompt file must be a bare file name, got: {name}"
        )

    if Path(candidate_name).suffix.lower() not in PROMPT_SUFFIXES:
        supported = ", ".join(PROMPT_SUFFIXES)
        raise ValueError(
            f"Unsupported prompt file type: {name} (allowed: {supported})"
        )

    for directory in prompt_dirs():
        candidate = (directory / candidate_name).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError:
            # Symlink pointing outside the prompt directory.
            continue
        if not candidate.is_file():
            continue
        if candidate.stat().st_size > MAX_PROMPT_BYTES:
            raise ValueError(
                f"Prompt file exceeds {MAX_PROMPT_BYTES} bytes: {name}"
            )
        return candidate

    searched = ", ".join(str(directory) for directory in prompt_dirs())
    raise FileNotFoundError(
        f"Prompt file not found: {name} (searched: {searched})"
    )


def load_prompt_file(name: str) -> str:
    """Read a prompt file by name and return its text.

    Raises:
        ValueError: On an invalid name or unreadable content.
        FileNotFoundError: If the file does not exist.
    """
    path = resolve_prompt_file(name)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read prompt file {name}: {exc}") from exc

    if not text:
        raise ValueError(f"Prompt file is empty: {name}")

    logger.info("📄 Loaded prompt file: %s (%s chars)", path.name, len(text))
    return text
