"""
Document Loader — reads .md / .txt / .json files into a uniform structure.

Output contract:
    {
        "title": str,
        "content": str,
        "metadata": dict   # Chroma-compatible (str, int, float, bool only)
    }

Supported formats: .md, .txt, .json
NOT supported: .pdf, .docx (extra dependencies not justified for MVP)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


# ── Public API ──────────────────────────────────────────────────────

def load_document(file_path: str) -> dict:
    """Load a single document file and return a uniform dict.

    Args:
        file_path: Absolute or relative path to .md / .txt / .json file.

    Returns:
        {"title": str, "content": str, "metadata": {...}}
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(
            f"Unsupported format: {suffix}. Supported: {sorted(_LOADERS.keys())}"
        )

    raw = path.read_text(encoding="utf-8")
    return _LOADERS[suffix](raw, path)


def load_documents_from_dir(dir_path: str, extensions: tuple = (".md", ".txt", ".json")) -> list[dict]:
    """Recursively load all supported documents from a directory.

    Args:
        dir_path: Path to directory.
        extensions: File extensions to include.

    Returns:
        List of document dicts.
    """
    results: list[dict] = []
    skipped: list[str] = []

    for root, _dirs, files in os.walk(dir_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = Path(fname).suffix.lower()
            if ext not in extensions:
                if ext:  # skip hidden files silently
                    skipped.append(fpath)
                continue
            try:
                results.append(load_document(fpath))
            except Exception as exc:
                skipped.append(f"{fpath} ({exc})")

    return results


# ── Per-format loaders ──────────────────────────────────────────────

def _load_markdown(raw: str, path: Path) -> dict:
    """Load .md: extract first heading as title, rest as content."""
    title = path.stem
    # Try to extract the first # heading as title
    heading_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    if heading_match:
        title = heading_match.group(1).strip()
    return _build_doc(title, raw, path)


def _load_text(raw: str, path: Path) -> dict:
    """Load .txt: use filename as title."""
    title = path.stem
    # If first line looks like a title, use it
    first_line = raw.strip().split("\n")[0] if raw.strip() else ""
    if first_line and len(first_line) < 80 and not first_line.endswith((".", "。", "；")):
        title = first_line.strip()
    return _build_doc(title, raw, path)


def _load_json(raw: str, path: Path) -> dict:
    """Load .json: expect {"title": "...", "content": "..."} or list of paragraphs."""
    data = json.loads(raw)
    title = path.stem
    content = raw

    if isinstance(data, dict):
        title = data.get("title", data.get("name", title))
        # Priority: content > text > body > description
        content = str(
            data.get("content") or
            data.get("text") or
            data.get("body") or
            data.get("description") or
            raw
        )
    elif isinstance(data, list):
        # Array of strings or objects — concatenate
        parts: list[str] = []
        for item in data:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("content", item.get("text", str(item))))
        content = "\n\n".join(parts) if parts else raw

    return _build_doc(title, content, path)


# ── Helpers ─────────────────────────────────────────────────────────

def _build_doc(title: str, content: str, path: Path) -> dict:
    return {
        "title": title.strip(),
        "content": content.strip(),
        "metadata": _extract_metadata(path),
    }


def _extract_metadata(path: Path) -> dict[str, Any]:
    """Build Chroma-safe metadata dict (only str/int/float/bool values)."""
    stat = path.stat()
    meta: dict[str, Any] = {
        "file_name": path.name,
        "file_path": str(path.absolute()),
        "format": path.suffix.lower().lstrip("."),
        "size_bytes": stat.st_size,
    }
    # Ensure all values are Chroma-compatible
    return {k: _sanitize_metadata_value(v) for k, v in meta.items()}


def _sanitize_metadata_value(value: Any) -> str | int | float | bool:
    """Coerce a value to a Chroma-compatible type."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# ── Registry ────────────────────────────────────────────────────────

_LOADERS: dict[str, Any] = {
    ".md":   _load_markdown,
    ".txt":  _load_text,
    ".json": _load_json,
}
