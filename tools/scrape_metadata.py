#!/usr/bin/env python3
"""
Scrape LM Studio model metadata into a dedicated SQLite database.
- Reads models from `lms ls --json`
- Adds heuristic capabilities (reasoning, coding, chat, creative, math)
- Optionally enriches with Hugging Face tags/description (best-effort)
- Persists into results/model_metadata.db, keyed by model_key
"""

import argparse
from datetime import datetime, timezone
import html as htmllib
from http.client import IncompleteRead, RemoteDisconnected
import json
import logging
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.user_paths import USER_LOGS_DIR, USER_RESULTS_DIR
except ModuleNotFoundError:
    from user_paths import USER_LOGS_DIR, USER_RESULTS_DIR

RESULTS_DIR = USER_RESULTS_DIR
METADATA_DB = RESULTS_DIR / "model_metadata.db"
LOGS_DIR = USER_LOGS_DIR
LOGS_DIR.mkdir(parents=True, exist_ok=True)
BACKUPS_DIR = RESULTS_DIR / "backups"
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

logger: logging.Logger = logging.getLogger("metadata_scraper")


def setup_logger() -> logging.Logger:
    """Set up and configure a logger for the metadata scraper.

    Creates a timestamped log file in the LOGS_DIR directory and sets up both
    file and console logging handlers. Also creates a symlink named
    'metadata_scraper_latest.log' pointing to the current log file.

    Returns:
        logging.Logger: Configured logger instance named 'metadata_scraper' with
            INFO level logging to both file and stdout.

    Side Effects:
        - Creates a new log file in LOGS_DIR with timestamp
        - Removes and recreates the 'metadata_scraper_latest.log' symlink
        - Configures the root logging settings
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"metadata_scraper_{timestamp}.log"
    latest_link = LOGS_DIR / "metadata_scraper_latest.log"
    latest_link.unlink(missing_ok=True)
    latest_link.symlink_to(log_file.name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("metadata_scraper")
    log.info("Logging to %s", log_file)
    return log


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the model_metadata table if it does not exist.

    Args:
        conn: SQLite database connection.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_metadata (
            model_key TEXT PRIMARY KEY,
            display_name TEXT,
            publisher TEXT,
            architecture TEXT,
            params TEXT,
            size_bytes INTEGER,
            max_context_length INTEGER,
            vision INTEGER,
            tool_use INTEGER,
            capabilities TEXT,
            source_url TEXT,
            hf_tags TEXT,
            description TEXT,
            scraped_at TEXT
        )
        """)
    conn.commit()


def _strip_tags(html: str) -> str:
    script_style_pattern = r"<(script|style)[^>]*>.*?</\1>"
    flags = re.DOTALL | re.IGNORECASE
    html = re.sub(
        script_style_pattern,
        " ",
        html,
        flags=flags,
    )
    text = re.sub(r"<[^>]+>", " ", html)
    text = htmllib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_lmstudio_readme(model_key: str, timeout: int = 5) -> str:
    """Fetch and extract a model description from the LM Studio model page.

    Given a model key (optionally including a version suffix like ``@...``), this
    function requests the corresponding LM Studio model URL, then attempts to
    derive readable metadata from the HTML:

    1. Prefer the ``<meta name="description" ...>`` content when present and
        sufficiently informative.
    2. Otherwise, scan common content containers (for example ``article``,
        ``section``, or markdown/prose/readme-styled ``div`` blocks).
    3. As a fallback, concatenate text from the first few paragraph tags.

    The extracted text is HTML-unescaped, stripped of tags, and truncated to a
    bounded length. If the page cannot be fetched or parsed, an empty string is
    returned.

    Args:
         model_key: Model identifier used by LM Studio. If it includes a version
              suffix (e.g., ``"org/model@q4"``), only the base portion before ``@``
              is used for URL construction.
         timeout: Network timeout in seconds for the HTTP request.

    Returns:
         A best-effort plain-text summary/readme snippet for the model, or an empty
         string if retrieval/parsing fails.
    """
    base_key = model_key.split("@")[0]
    url = f"https://lmstudio.ai/models/{base_key}"
    req = Request(url, headers={"User-Agent": "lmstudio-metadata-scraper"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            try:
                html_bytes = resp.read()
            except IncompleteRead as exc:
                html_bytes = exc.partial or b""
            html = html_bytes.decode("utf-8", errors="ignore")
    except (
        HTTPError,
        URLError,
        TimeoutError,
        ValueError,
        RemoteDisconnected,
        IncompleteRead,
    ):
        return ""

    meta_desc_pattern = r"<meta\s+name=\"description\"\s+content=\"(.*?)\""
    m = re.search(
        meta_desc_pattern,
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        text = htmllib.unescape(m.group(1)).strip()
        if len(text) >= 40:
            return text[:1500]

    candidates: List[str] = []
    for pattern in [
        r"<article[^>]*>(.*?)</article>",
        r"<section[^>]*>(.*?)</section>",
        r"<div[^>]*class=\"[^\"]*(prose|markdown|readme)[^\"]*\"[^>]*>(.*?)</div>",
    ]:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE | re.DOTALL):
            group_content = match.group(match.lastindex or 1)
            candidates.append(group_content)

    if not candidates:
        ps = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
        if ps:
            candidates.append(" ".join(ps[:5]))

    best = ""
    for c in candidates:
        text = _strip_tags(c)
        if len(text) > len(best):
            best = text

    return best[:2000].strip()


def infer_caps_from_description(description: str) -> List[str]:
    """Infer capability tags from a free-form model description using keyword matching.

    The function lowercases the input text and appends predefined capability labels
    when related keywords are found. Matching is heuristic and based on simple
    substring checks.

    Args:
        description: Natural-language description of a model.

    Returns:
        A list of inferred capability labels in detection order. Possible values are:
        ``"coding"``, ``"math"``, ``"reasoning"``, ``"chat"``, ``"vision"``,
        ``"tool_use"``, and ``"creative"``. Returns an empty list if
        ``description`` is empty or falsy.
    """
    if not description:
        return []
    text = description.lower()
    caps: List[str] = []
    if any(
        keyword in text
        for keyword in [
            "coding",
            "code",
            "programming",
            "developer",
            "software engineering",
            "source code",
        ]
    ):
        caps.append("coding")
    if any(
        keyword in text
        for keyword in [
            "math",
            "mathematics",
            "arithmetic",
            "algebra",
            "calculus",
            "geometry",
        ]
    ):
        caps.append("math")
    if any(
        keyword in text
        for keyword in [
            "reasoning",
            "chain-of-thought",
            "logical reasoning",
            "logic",
        ]
    ):
        caps.append("reasoning")
    if any(
        keyword in text
        for keyword in [
            "chat",
            "assistant",
            "conversational",
            "dialogue",
            "instruct",
        ]
    ):
        caps.append("chat")
    if any(
        keyword in text
        for keyword in [
            "vision",
            "image",
            "multimodal",
            "vision-language",
            "vl",
        ]
    ):
        caps.append("vision")
    if any(
        keyword in text
        for keyword in [
            "tool use",
            "tool-use",
            "tools",
            "function calling",
            "function-calling",
            "api calling",
        ]
    ):
        caps.append("tool_use")
    if any(
        keyword in text
        for keyword in [
            "creative",
            "story",
            "writer",
            "poem",
            "poetry",
            "narrative",
        ]
    ):
        caps.append("creative")
    return caps


def backup_metadata_db() -> None:
    """Create a timestamped backup of model_metadata.db if it exists."""
    if not METADATA_DB.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUPS_DIR / f"model_metadata_{timestamp}_backup.db"
    try:
        shutil.copy2(METADATA_DB, backup_path)
        logger.info("Backup created: %s", backup_path)
    except (OSError, shutil.Error) as copy_err:
        logger.warning("Could not create backup: %s", copy_err)


def infer_capabilities(model_key: str, display_name: str) -> List[str]:
    """Infer likely capability tags from a model's key and display name.

    This function performs simple keyword-based matching on the lowercase
    combination of `model_key` and `display_name`, returning a list of
    capability labels in detection order.

    Args:
        model_key: Canonical identifier for the model.
        display_name: Human-readable model name.

    Returns:
        A list of inferred capability tags, which may include:
        `reasoning`, `coding`, `chat`, `creative`, and `math`.
    """
    text = f"{model_key} {display_name}".lower()
    caps: List[str] = []
    if any(tok in text for tok in ["deepseek-r1", "qwq", "o1", "reason", "logic"]):
        caps.append("reasoning")
    if any(tok in text for tok in ["coder", "code", "codestral", "deepseek-coder"]):
        caps.append("coding")
    if "instruct" in text or "chat" in text:
        caps.append("chat")
    if any(tok in text for tok in ["creative", "writer", "story"]):
        caps.append("creative")
    if "math" in text:
        caps.append("math")
    return caps


def fetch_hf_metadata(model_key: str, timeout: int = 3) -> Dict:
    """Fetch selected metadata for a Hugging Face model.

    Parses ``model_key`` (optionally containing a ``@revision`` suffix), queries the
    Hugging Face model API, and returns a normalized metadata dictionary.

    Args:
        model_key: Model identifier in ``publisher/name`` form, optionally with
            ``@revision`` (for example, ``org/model@main``).
        timeout: Request timeout in seconds.

    Returns:
        A dictionary containing:
            - ``hf_tags`` (list): Model tags from Hugging Face.
            - ``hf_pipeline_tag`` (str): Primary pipeline/task tag.
            - ``hf_description`` (str): Description from model card data.

        Returns an empty dictionary if the key format is invalid or if the request,
        parsing, or decoding fails.
    """
    base_key = model_key.split("@")[0]
    if "/" not in base_key:
        return {}
    publisher, name = base_key.split("/", 1)
    url = f"https://huggingface.co/api/models/{publisher}/{name}"
    req = Request(url, headers={"User-Agent": "lmstudio-metadata-scraper"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            try:
                payload_bytes = resp.read()
            except IncompleteRead as exc:
                payload_bytes = exc.partial or b""
            if not payload_bytes:
                return {}
            data = json.loads(payload_bytes.decode("utf-8", errors="ignore"))
            card = data.get("cardData") or {}
            return {
                "hf_tags": data.get("tags", []),
                "hf_pipeline_tag": data.get("pipeline_tag", ""),
                "hf_description": card.get("description", ""),
            }
    except (HTTPError, URLError, TimeoutError, ValueError, IncompleteRead):
        return {}


def load_lms_models(timeout: int = 30) -> List[Dict]:
    """Load installed LM Studio models via the `lms ls --json` CLI command.

    Args:
        timeout: Maximum number of seconds to wait for the subprocess to finish.

    Returns:
        A list of model metadata dictionaries parsed from the CLI JSON output.

    Raises:
        RuntimeError: If the `lms ls --json` command exits with a non-zero status
            or if its output cannot be parsed as valid JSON.
        ValueError: If the parsed JSON is not a list as expected.
    """
    proc = subprocess.run(
        ["lms", "ls", "--json"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"lms ls failed: {proc.stderr.strip()}")
    try:
        data = json.loads(proc.stdout)
        if not isinstance(data, list):
            raise ValueError("Unexpected JSON structure from lms ls --json")
        return data
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse lms ls output: {exc}") from exc


def upsert_metadata(conn: sqlite3.Connection, rows: List[Dict]) -> None:
    """Insert or replace metadata rows in the model_metadata table.

    Args:
        conn: SQLite database connection.
        rows: List of row dictionaries to upsert.
    """
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO model_metadata (
            model_key, display_name, publisher, architecture, params, size_bytes,
            max_context_length, vision, tool_use, capabilities, source_url,
            hf_tags, description, scraped_at
        ) VALUES (
            :model_key, :display_name, :publisher, :architecture, :params, :size_bytes,
            :max_context_length, :vision, :tool_use, :capabilities, :source_url,
            :hf_tags, :description, :scraped_at
        )
        """,
        rows,
    )
    conn.commit()


def update_rows(conn: sqlite3.Connection, updates: List[Dict]) -> None:
    """Update existing rows in ``model_metadata`` using a batch of updates.

    Each update entry must include ``model_key`` and ``scraped_at``. Optional
    fields (``description``, ``capabilities``, ``source_url``) are defaulted to
    ``None`` when missing and only overwrite existing database values when they
    are non-``None``.

    Args:
        conn: Open SQLite connection used to execute the updates.
        updates: List of dictionaries containing named SQL parameters for each row.

    Notes:
        - Returns immediately when ``updates`` is empty.
        - Persists all changes with a single ``commit()`` after ``executemany()``.
    """
    if not updates:
        return
    for u in updates:
        u.setdefault("description", None)
        u.setdefault("capabilities", None)
        u.setdefault("source_url", None)
    cur = conn.cursor()
    cur.executemany(
        """
        UPDATE model_metadata
        SET
            description =
                CASE
                    WHEN :description IS NOT NULL
                    THEN :description
                    ELSE description
                END,
            capabilities =
                CASE
                    WHEN :capabilities IS NOT NULL
                    THEN :capabilities
                    ELSE capabilities
                END,
            source_url =
                CASE
                    WHEN :source_url IS NOT NULL
                    THEN :source_url
                    ELSE source_url
                END,
            scraped_at = :scraped_at
        WHERE model_key = :model_key
        """,
        updates,
    )
    conn.commit()


def scrape(only_missing: bool = True, enable_hf: bool = True) -> None:
    """Scrape model metadata and persist it to the local SQLite metadata database.

    This routine loads LM Studio model entries, enriches each LLM record with
    capabilities and descriptions (optionally including Hugging Face metadata),
    and writes results into `model_metadata` using upsert/update flows.

    Behavior:
        - Creates the results directory if needed.
        - Backs up the metadata database and ensures schema exists.
        - Reads existing rows to support incremental updates.
        - Skips non-LLM entries and entries without a model key.
        - If `only_missing` is True, only fills missing description/capabilities for
          existing rows; otherwise prepares full insert/upsert rows.
        - Merges inferred capabilities from model identifiers, descriptions, and
          explicit vision/tool-use flags.
        - Updates timestamps (`scraped_at`) for changed records.

    Args:
        only_missing (bool, optional): When True, only update missing/changed fields
            for models already present in the database. Defaults to True.
        enable_hf (bool, optional): When True, fetch Hugging Face metadata for
            enrichment. Defaults to True.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    backup_metadata_db()
    conn = sqlite3.connect(METADATA_DB)
    ensure_schema(conn)

    existing_map: Dict[str, Dict] = {}
    for row in conn.execute(
        (
            "SELECT model_key, "
            "description, "
            "source_url, "
            "capabilities "
            "FROM model_metadata"
        )
    ):
        existing_map[row[0]] = {
            "description": row[1] or "",
            "source_url": row[2] or "",
            "capabilities": (
                json.loads(row[3]) if (row[3] and str(row[3]).strip()) else []
            ),
        }
    existing_keys = set(existing_map.keys())

    models = load_lms_models()
    to_insert: List[Dict] = []
    to_update: List[Dict] = []

    for entry in models:
        if entry.get("type") != "llm":
            continue
        model_key = entry.get("modelKey") or entry.get("model_key")
        if not model_key:
            continue
        base_key = model_key.split("@")[0]
        display_name = entry.get("displayName", model_key)
        architecture = entry.get("architecture", "unknown")
        params = entry.get("paramsString", "")
        size_bytes = entry.get("sizeBytes") or 0
        max_ctx = entry.get("maxContextLength") or 0
        vision = 1 if entry.get("vision") else 0
        tool_use = 1 if entry.get("trainedForToolUse") else 0

        lms_desc = fetch_lmstudio_readme(model_key) or ""
        if only_missing and model_key in existing_keys:
            row_update: Dict = {
                "model_key": model_key,
                "scraped_at": datetime.now(
                    timezone.utc,
                ).isoformat(),
            }
            changed = False
            if lms_desc and not existing_map[model_key]["description"]:
                row_update["description"] = lms_desc
                row_update["source_url"] = f"https://lmstudio.ai/models/{base_key}"
                changed = True
            desc_caps = infer_caps_from_description(lms_desc)
            existing_caps = existing_map[model_key].get("capabilities", [])
            new_caps_set = set(existing_caps)
            if vision:
                new_caps_set.add("vision")
            if tool_use:
                new_caps_set.add("tool_use")
            for c in desc_caps:
                new_caps_set.add(c)
            if new_caps_set != set(existing_caps):
                row_update["capabilities"] = json.dumps(sorted(new_caps_set))
                changed = True
            if changed:
                to_update.append(row_update)
            continue

        caps = infer_capabilities(model_key, display_name)
        caps += [c for c in infer_caps_from_description(lms_desc) if c not in caps]
        if vision:
            caps.append("vision")
        if tool_use:
            caps.append("tool_use")
        caps = sorted(list(dict.fromkeys(caps)))
        existing_caps = existing_map.get(model_key, {}).get("capabilities", [])
        if existing_caps:
            caps = sorted(set(existing_caps) | set(caps))

        hf_meta = fetch_hf_metadata(model_key) if enable_hf else {}
        description = lms_desc or hf_meta.get("hf_description", "")

        row = {
            "model_key": model_key,
            "display_name": display_name,
            "publisher": model_key.split("/")[0] if "/" in model_key else "",
            "architecture": architecture,
            "params": params,
            "size_bytes": int(size_bytes) if size_bytes else 0,
            "max_context_length": int(max_ctx) if max_ctx else 0,
            "vision": vision,
            "tool_use": tool_use,
            "capabilities": json.dumps(caps),
            "source_url": f"https://lmstudio.ai/models/{base_key}",
            "hf_tags": json.dumps(hf_meta.get("hf_tags", [])),
            "description": description,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }
        to_insert.append(row)

    if to_insert:
        upsert_metadata(conn, to_insert)
        logger.info(
            "Inserted/updated %d models into model_metadata.db",
            len(to_insert),
        )
    else:
        logger.info("No new models to insert")

    if to_update:
        update_rows(conn, to_update)
        logger.info(
            "Updated metadata for %d existing models (description/capabilities)",
            len(to_update),
        )

    conn.close()


def main() -> None:
    """Run the metadata scraping CLI entry point.

    Initializes logging, parses command-line flags, and executes the scrape
    workflow with optional refresh and Hugging Face enrichment controls.

    Flags:
        --refresh: Rescrape metadata even when existing records are present.
        --no-hf: Disable Hugging Face enrichment during scraping.

    Raises:
        SystemExit: Exits with status code 1 if scraping fails due to an
            unhandled exception.
    """

    def _expand_short_flag_clusters(cli_args: List[str]) -> List[str]:
        """Expand combined short flags like ``-rn`` to ``-r -n``."""
        combinable = {"r", "n", "h"}
        normalized: List[str] = []
        for arg in cli_args:
            if arg.startswith("--") or not arg.startswith("-"):
                normalized.append(arg)
                continue
            if len(arg) <= 2:
                normalized.append(arg)
                continue

            cluster = arg[1:]
            if all(flag in combinable for flag in cluster):
                normalized.extend(f"-{flag}" for flag in cluster)
            else:
                normalized.append(arg)
        return normalized

    setup_logger()
    parser = argparse.ArgumentParser(description="Scrape LM Studio model metadata")
    parser.add_argument(
        "--refresh",
        "-r",
        action="store_true",
        help=("Rescrape even " "if already present"),
    )
    parser.add_argument(
        "--no-hf",
        "-n",
        action="store_true",
        help=("Disable Hugging Face " "enrichment"),
    )
    normalized_args = _expand_short_flag_clusters(sys.argv[1:])
    args = parser.parse_args(args=normalized_args)

    try:
        scrape(only_missing=not args.refresh, enable_hf=not args.no_hf)
        logger.info("Scrape completed")
    except (
        RuntimeError,
        ValueError,
        sqlite3.Error,
        OSError,
        subprocess.SubprocessError,
    ):
        logger.exception("Error during scraping")
        sys.exit(1)


if __name__ == "__main__":
    main()
