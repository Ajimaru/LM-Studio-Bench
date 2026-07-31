"""Tests for core/prompts.py - prompt file resolution and loading."""

from pathlib import Path
from unittest.mock import patch

import pytest

from core import prompts


@pytest.fixture(name="prompt_dir")
def prompt_dir_fixture(tmp_path: Path):
    """Point prompt resolution at a temporary directory."""
    project_dir = tmp_path / "prompts"
    project_dir.mkdir()
    user_dir = tmp_path / "user_prompts"
    user_dir.mkdir()
    with patch.object(prompts, "PROJECT_PROMPTS_DIR", project_dir), \
            patch.object(prompts, "USER_PROMPTS_DIR", user_dir):
        yield project_dir, user_dir


class TestListPromptFiles:
    """Tests for list_prompt_files()."""

    def test_lists_supported_suffixes_only(self, prompt_dir):
        """Only .md and .txt files are offered."""
        project_dir, _ = prompt_dir
        (project_dir / "a.md").write_text("a", encoding="utf-8")
        (project_dir / "b.txt").write_text("b", encoding="utf-8")
        (project_dir / "c.py").write_text("c", encoding="utf-8")
        assert prompts.list_prompt_files() == ["a.md", "b.txt"]

    def test_merges_both_directories(self, prompt_dir):
        """User prompts show up next to the shipped ones."""
        project_dir, user_dir = prompt_dir
        (project_dir / "shipped.md").write_text("x", encoding="utf-8")
        (user_dir / "mine.md").write_text("y", encoding="utf-8")
        assert prompts.list_prompt_files() == ["shipped.md", "mine.md"]

    def test_project_file_wins_over_user_file(self, prompt_dir):
        """A duplicate name is listed once, resolved from the project dir."""
        project_dir, user_dir = prompt_dir
        (project_dir / "dup.md").write_text("project", encoding="utf-8")
        (user_dir / "dup.md").write_text("user", encoding="utf-8")
        assert prompts.list_prompt_files() == ["dup.md"]
        assert prompts.load_prompt_file("dup.md") == "project"


class TestResolvePromptFile:
    """Tests for resolve_prompt_file() path confinement."""

    @pytest.mark.parametrize("name", [
        "../secrets.md",
        "sub/dir.md",
        "/etc/passwd",
        "prompt.md/../../setup.sh",
    ])
    def test_rejects_paths(self, prompt_dir, name):
        """Anything that is not a bare file name is refused."""
        with pytest.raises(ValueError):
            prompts.resolve_prompt_file(name)

    def test_rejects_unsupported_suffix(self, prompt_dir):
        """Executable-looking files are not prompt files."""
        project_dir, _ = prompt_dir
        (project_dir / "evil.py").write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            prompts.resolve_prompt_file("evil.py")

    def test_rejects_empty_name(self, prompt_dir):
        """An empty name is an error, not a silent no-op."""
        with pytest.raises(ValueError):
            prompts.resolve_prompt_file("   ")

    def test_missing_file_raises(self, prompt_dir):
        """A valid but absent name reports not-found."""
        with pytest.raises(FileNotFoundError):
            prompts.resolve_prompt_file("nope.md")

    def test_symlink_escaping_dir_is_ignored(self, prompt_dir, tmp_path):
        """A symlink pointing outside the prompt dirs does not resolve."""
        project_dir, _ = prompt_dir
        outside = tmp_path / "outside.md"
        outside.write_text("secret", encoding="utf-8")
        (project_dir / "link.md").symlink_to(outside)
        with pytest.raises(FileNotFoundError):
            prompts.resolve_prompt_file("link.md")

    def test_rejects_oversized_file(self, prompt_dir):
        """Files beyond the size cap are refused."""
        project_dir, _ = prompt_dir
        big = project_dir / "big.md"
        big.write_text("x" * (prompts.MAX_PROMPT_BYTES + 1), encoding="utf-8")
        with pytest.raises(ValueError):
            prompts.resolve_prompt_file("big.md")


class TestLoadPromptFile:
    """Tests for load_prompt_file()."""

    def test_returns_stripped_text(self, prompt_dir):
        """Content is returned with surrounding whitespace removed."""
        project_dir, _ = prompt_dir
        (project_dir / "p.md").write_text("\n  hello  \n", encoding="utf-8")
        assert prompts.load_prompt_file("p.md") == "hello"

    def test_preserves_inner_newlines(self, prompt_dir):
        """Code snippets keep their line structure."""
        project_dir, _ = prompt_dir
        (project_dir / "p.md").write_text("def f():\n    return 1\n",
                                          encoding="utf-8")
        assert prompts.load_prompt_file("p.md") == "def f():\n    return 1"

    def test_empty_file_raises(self, prompt_dir):
        """An empty prompt would silently weaken the benchmark."""
        project_dir, _ = prompt_dir
        (project_dir / "empty.md").write_text("   \n", encoding="utf-8")
        with pytest.raises(ValueError):
            prompts.load_prompt_file("empty.md")


class TestShippedPrompt:
    """The prompt shipped with the project must stay usable."""

    def test_coding_assistant_prompt_exists(self):
        """The coding_assistant preset references a file that resolves."""
        assert "coding_assistant.md" in prompts.list_prompt_files()
        text = prompts.load_prompt_file("coding_assistant.md")
        assert len(text) > 1000
