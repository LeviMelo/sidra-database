from __future__ import annotations

import csv
import sqlite3
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Set, Tuple

SynonymMap = Dict[tuple[str, str], Set[str]]


def normalize_basic(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    cleaned = []
    last_space = False
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch)
            last_space = False
        else:
            if not last_space:
                cleaned.append(" ")
                last_space = True
    normalized = "".join(cleaned).strip()
    return " ".join(normalized.split())


def load_synonyms_into_memory(connection: sqlite3.Connection) -> SynonymMap:
    cur = connection.execute("SELECT kind, key, alt FROM synonyms")
    result: SynonymMap = {}
    for kind, key, alt in cur.fetchall():
        nk = normalize_basic(key)
        na = normalize_basic(alt)
        key_tuple = (kind, nk)
        result.setdefault(key_tuple, set()).add(na)
        # make symmetric for convenience
        key_tuple_alt = (kind, na)
        result.setdefault(key_tuple_alt, set()).add(nk)
    return result


def normalize_token(
    text: str,
    *,
    apply_synonyms: bool = True,
    synonyms: SynonymMap | None = None,
    kind: str | None = None,
) -> str:
    base = normalize_basic(text)
    if not apply_synonyms or not base:
        return base
    if synonyms is None:
        return base
    if kind is None:
        # aggregate synonyms across kinds
        for (_, key), values in synonyms.items():
            if key == base:
                if values:
                    return sorted(values)[0]
        return base
    key = (kind, base)
    alts = synonyms.get(key)
    if not alts:
        return base
    return sorted(alts)[0]


def import_synonyms_csv(path: str | Path, connection: sqlite3.Connection) -> int:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [
            (row.get("kind", ""), row.get("key", ""), row.get("alt", ""))
            for row in reader
        ]
    inserted = 0
    with connection:
        for kind, key, alt in rows:
            if not kind or not key or not alt:
                continue
            connection.execute(
                "INSERT OR IGNORE INTO synonyms(kind, key, alt) VALUES(?,?,?)",
                (kind.strip().lower(), normalize_basic(key), normalize_basic(alt)),
            )
            inserted += 1
    return inserted


def export_synonyms_csv(path: str | Path, connection: sqlite3.Connection) -> int:
    path = Path(path)
    cur = connection.execute("SELECT kind, key, alt FROM synonyms ORDER BY kind, key, alt")
    rows = cur.fetchall()
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "key", "alt"])
        writer.writerows(rows)
    return len(rows)


__all__ = [
    "normalize_basic",
    "normalize_token",
    "load_synonyms_into_memory",
    "import_synonyms_csv",
    "export_synonyms_csv",
]
