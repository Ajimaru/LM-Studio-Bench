"""Tests for src/version_checker.py."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from version_checker import (
    compare_versions,
    fetch_latest_release,
    format_release_url,
    get_current_version,
)


class TestGetCurrentVersion:
    """Tests for get_current_version()."""

    def test_reads_version_from_file(self):
        """Returns a non-empty version string from the real VERSION file."""
        result = get_current_version()
        assert isinstance(result, str)
        assert result

    def test_returns_unknown_when_file_missing(self):
        """Returns 'unknown' when VERSION file does not exist."""
        import version_checker as vc

        with patch("version_checker.Path") as mock_path:
            proj_root = MagicMock()
            version_f = MagicMock()
            version_f.exists.return_value = False
            proj_root.__truediv__ = MagicMock(return_value=version_f)
            mock_path.return_value.resolve.return_value.parent.parent = proj_root

            result = vc.get_current_version()
            assert result == "unknown"

    def test_raises_on_empty_file(self):
        """Raises ValueError when VERSION file is empty."""
        with patch("version_checker.Path") as mock_path:
            proj_root = MagicMock()
            version_f = MagicMock()
            version_f.exists.return_value = True
            version_f.read_text.return_value = ""
            proj_root.__truediv__ = MagicMock(return_value=version_f)
            mock_path.return_value.resolve.return_value.parent.parent = proj_root

            with pytest.raises(ValueError):
                import version_checker as vc
                vc.get_current_version()

    def test_returns_non_empty_string_from_version_file(self):
        """Returns string with version data when VERSION exists."""
        with patch("version_checker.Path") as mock_path:
            proj_root = MagicMock()
            version_f = MagicMock()
            version_f.exists.return_value = True
            version_f.read_text.return_value = "v1.2.3\n"
            proj_root.__truediv__ = MagicMock(return_value=version_f)
            mock_path.return_value.resolve.return_value.parent.parent = proj_root

            import version_checker as vc
            result = vc.get_current_version()
            assert result == "v1.2.3"

    def test_raises_on_os_error(self):
        """Raises ValueError on OSError reading version file."""
        with patch("version_checker.Path") as mock_path:
            proj_root = MagicMock()
            version_f = MagicMock()
            version_f.exists.return_value = True
            version_f.read_text.side_effect = OSError("permission denied")
            proj_root.__truediv__ = MagicMock(return_value=version_f)
            mock_path.return_value.resolve.return_value.parent.parent = proj_root

            with pytest.raises(ValueError):
                import version_checker as vc
                vc.get_current_version()


class TestFetchLatestRelease:
    """Tests for fetch_latest_release()."""

    def test_returns_dict_on_success(self):
        """Returns dict with tag_name when GitHub API succeeds."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tag_name": "v1.0.0",
            "html_url": "https://github.com/example",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("version_checker.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client
            result = fetch_latest_release()

        assert result is not None
        assert "tag_name" in result

    def test_returns_none_on_http_error(self):
        """Returns None on HTTP status error."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("version_checker.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            error = httpx.HTTPStatusError(
                "Not Found",
                request=MagicMock(),
                response=mock_response,
            )
            mock_client.get.side_effect = error
            mock_client_cls.return_value = mock_client
            result = fetch_latest_release()

        assert result is None

    def test_returns_none_on_connection_error(self):
        """Returns None on network connection error."""
        with patch("version_checker.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("unreachable")
            mock_client_cls.return_value = mock_client
            result = fetch_latest_release()

        assert result is None

    def test_returns_none_on_timeout(self):
        """Returns None on request timeout."""
        with patch("version_checker.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("timeout")
            mock_client_cls.return_value = mock_client
            result = fetch_latest_release()

        assert result is None


class TestCompareVersions:
    """Tests for compare_versions()."""

    def test_newer_available(self):
        """Returns True when latest version is newer."""
        assert compare_versions("v0.1.0", "v0.2.0") is True

    def test_same_version(self):
        """Returns False when versions are equal."""
        assert compare_versions("v1.0.0", "v1.0.0") is False

    def test_current_is_newer(self):
        """Returns False when current is newer than latest."""
        assert compare_versions("v2.0.0", "v1.9.9") is False

    def test_minor_version_bump(self):
        """Returns True for minor version increment."""
        assert compare_versions("v0.1.0", "v0.1.1") is True

    def test_major_version_bump(self):
        """Returns True for major version increment."""
        assert compare_versions("v1.0.0", "v2.0.0") is True

    def test_invalid_version_treated_as_zero(self):
        """Invalid version strings are treated as (0,0,0)."""
        result = compare_versions("invalid", "v1.0.0")
        assert result is True

    def test_without_v_prefix(self):
        """Versions without 'v' prefix are compared correctly."""
        assert compare_versions("1.0.0", "2.0.0") is True

    def test_pre_release_stripped(self):
        """Pre-release suffixes are stripped before comparison."""
        assert compare_versions("v1.0.0-alpha", "v1.0.0") is False

    def test_both_invalid(self):
        """Both invalid returns False (0,0,0 vs 0,0,0)."""
        assert compare_versions("bad", "alsoBad") is False


class TestFormatReleaseUrl:
    """Tests for format_release_url()."""

    def test_returns_correct_url(self):
        """Returns GitHub release URL for given tag."""
        url = format_release_url("v1.2.3")
        assert "v1.2.3" in url
        assert "github.com" in url
        assert "Ajimaru/LM-Studio-Bench" in url

    def test_url_structure(self):
        """URL contains /releases/tag/ path segment."""
        url = format_release_url("v0.5.0")
        assert "/releases/tag/" in url
