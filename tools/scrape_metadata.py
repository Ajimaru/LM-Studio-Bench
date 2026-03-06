#!/usr/bin/env python3
"""
Scrape LM Studio model metadata into a dedicated SQLite database.
- Reads models from `lms ls --json`
- Adds heuristic capabilities (reasoning, coding, chat, creative, math)
- Optionally enriches with Hugging Face tags/description (best-effort)
- Persists into results/model_metadata.db, keyed by model_key
"""

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
import shutil
import re
import html as htmllib
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
METADATA_DB = RESULTS_DIR / "model_metadata.db"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
BACKUPS_DIR = RESULTS_DIR / "backups"
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger() -> logging.Logger:
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
    logger = logging.getLogger("metadata_scraper")
    logger.info(f"Logging to {log_file}")
    return logger


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
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
        """
    )
    conn.commit()


def _strip_tags(html: str) -> str:
    # Remove script/style
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Unescape entities and collapse whitespace
    text = htmllib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_lmstudio_readme(model_key: str, timeout: int = 5) -> str:
    base_key = model_key.split("@")[0]
    url = f"https://lmstudio.ai/models/{base_key}"
    req = Request(url, headers={"User-Agent": "lmstudio-metadata-scraper"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError, ValueError):
        return ""

    # Try meta description first
    m = re.search(r"<meta\\s+name=\"description\"\\s+content=\"(.*?)\"", html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text = htmllib.unescape(m.group(1)).strip()
        if len(text) >= 40:
            return text[:1500]

    # Heuristics: README or main content candidates
    candidates: List[str] = []
    # Common containers for markdown/README-like blocks
    for pattern in [
        r"<article[^>]*>(.*?)</article>",
        r"<section[^>]*>(.*?)</section>",
        r"<div[^>]*class=\"[^\"]*(prose|markdown|readme)[^\"]*\"[^>]*>(.*?)</div>",
    ]:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE | re.DOTALL):
            # last group may be the content group (handle 1 or 2 capture groups)
            group_content = match.group(match.lastindex or 1)
            candidates.append(group_content)

    # Fallback: gather first paragraphs
    if not candidates:
        ps = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
        if ps:
            candidates.append(" ".join(ps[:5]))

    # Sanitize and pick the longest meaningful candidate
    best = ""
    for c in candidates:
        text = _strip_tags(c)
        if len(text) > len(best):
            best = text

    return best[:2000].strip()


def infer_caps_from_description(description: str) -> List[str]:
    if not description:
        return []
    text = description.lower()
    caps: List[str] = []
    if any(k in text for k in ["coding", "code", "programming", "developer", "software engineering", "source code"]):
        caps.append("coding")
    if any(k in text for k in ["math", "mathematics", "arithmetic", "algebra", "calculus", "geometry"]):
        caps.append("math")
    if any(k in text for k in ["reasoning", "chain-of-thought", "logical reasoning", "logic"]):
        caps.append("reasoning")
    if any(k in text for k in ["chat", "assistant", "conversational", "dialogue", "instruct"]):
        caps.append("chat")
    if any(k in text for k in ["vision", "image", "multimodal", "vision-language", "vl"]):
        caps.append("vision")
    if any(k in text for k in ["tool use", "tool-use", "tools", "function calling", "function-calling", "api calling"]):
        caps.append("tool_use")
    if any(k in text for k in ["creative", "story", "writer", "poem", "poetry", "narrative"]):
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
        logger.info(f"Backup created: {backup_path}")
    except Exception as copy_err:
        logger.warning(f"Could not create backup: {copy_err}")


def infer_capabilities(model_key: str, display_name: str) -> List[str]:
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
    # model_key typically "publisher/name". If quant variant contains '@', strip it.
    base_key = model_key.split("@")[0]
    if "/" not in base_key:
        return {}
    publisher, name = base_key.split("/", 1)
    url = f"https://huggingface.co/api/models/{publisher}/{name}"
    req = Request(url, headers={"User-Agent": "lmstudio-metadata-scraper"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            card = data.get("cardData") or {}
            return {
                "hf_tags": data.get("tags", []),
                "hf_pipeline_tag": data.get("pipeline_tag", ""),
                "hf_description": card.get("description", ""),
            }
    except (HTTPError, URLError, TimeoutError, ValueError):
        return {}


def load_lms_models(timeout: int = 30) -> List[Dict]:
    proc = subprocess.run(
        ["lms", "ls", "--json"],
        capture_output=True,
        text=True,
        timeout=timeout,
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
    if not updates:
        return
    # ensure optional keys present
    for u in updates:
        u.setdefault("description", None)
        u.setdefault("capabilities", None)
        u.setdefault("source_url", None)
    cur = conn.cursor()
    cur.executemany(
        """
        UPDATE model_metadata
        SET 
            description = CASE WHEN :description IS NOT NULL THEN :description ELSE description END,
            capabilities = CASE WHEN :capabilities IS NOT NULL THEN :capabilities ELSE capabilities END,
            source_url = CASE WHEN :source_url IS NOT NULL THEN :source_url ELSE source_url END,
            scraped_at = :scraped_at
        WHERE model_key = :model_key
        """,
        updates,
    )
    conn.commit()


def scrape(only_missing: bool = True, enable_hf: bool = True) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    backup_metadata_db()
    conn = sqlite3.connect(METADATA_DB)
    ensure_schema(conn)

    existing_map: Dict[str, Dict] = {}
    for row in conn.execute("SELECT model_key, description, source_url, capabilities FROM model_metadata"):
        existing_map[row[0]] = {
            "description": row[1] or "",
            "source_url": row[2] or "",
            "capabilities": json.loads(row[3]) if (row[3] and str(row[3]).strip()) else []
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
        # Read entry fields
        display_name = entry.get("displayName", model_key)
        architecture = entry.get("architecture", "unknown")
        params = entry.get("paramsString", "")
        size_bytes = entry.get("sizeBytes") or 0
        max_ctx = entry.get("maxContextLength") or 0
        vision = 1 if entry.get("vision") else 0
        tool_use = 1 if entry.get("trainedForToolUse") else 0

        # Always best-effort fetch LM Studio README for description enhancement
        lms_desc = fetch_lmstudio_readme(model_key) or ""
        if only_missing and model_key in existing_keys:
            # Conditional updates: description backfill + capability enrichment from description
            row_update: Dict = {"model_key": model_key, "scraped_at": datetime.utcnow().isoformat()}
            changed = False
            if lms_desc and not existing_map[model_key]["description"]:
                row_update["description"] = lms_desc
                row_update["source_url"] = f"https://lmstudio.ai/models/{base_key}"
                changed = True
            # Enrich capabilities from description and flags
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
        # From description
        caps += [c for c in infer_caps_from_description(lms_desc) if c not in caps]
        if vision:
            caps.append("vision")
        if tool_use:
            caps.append("tool_use")
        # Deduplicate & sort
        caps = sorted(list(dict.fromkeys(caps)))
        # Only-add policy: if row exists, union with existing capabilities
        existing_caps = existing_map.get(model_key, {}).get("capabilities", [])
        if existing_caps:
            caps = sorted(set(existing_caps) | set(caps))

        hf_meta = fetch_hf_metadata(model_key) if enable_hf else {}
        # Prefer LM Studio README; fallback to HF description
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
            "scraped_at": datetime.utcnow().isoformat(),
        }
        to_insert.append(row)

    if to_insert:
        upsert_metadata(conn, to_insert)
        logger.info(f"Inserted/updated {len(to_insert)} models into model_metadata.db")
    else:
        logger.info("No new models to insert")

    if to_update:
        update_rows(conn, to_update)
        logger.info(f"Updated metadata for {len(to_update)} existing models (description/capabilities)")

    conn.close()


def main() -> None:
    global logger
    logger = setup_logger()
    parser = argparse.ArgumentParser(description="Scrape LM Studio model metadata")
    parser.add_argument("--refresh", action="store_true", help="Rescrape even if already present")
    parser.add_argument("--no-hf", action="store_true", help="Disable Hugging Face enrichment")
    args = parser.parse_args()

    try:
        scrape(only_missing=not args.refresh, enable_hf=not args.no_hf)
        logger.info("Scrape completed")
    except Exception as exc:  # pragma: no cover - best-effort script
        logger.error(f"Error during scraping: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
