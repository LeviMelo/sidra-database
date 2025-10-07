from __future__ import annotations

import asyncio
import hashlib
from array import array
from datetime import datetime, timezone
from typing import Any, Sequence

import orjson

from ..config import get_settings
from ..db.session import sqlite_session
from ..db.migrations import apply_search_schema
from ..net.api_client import SidraApiClient
from ..net.embedding_client import EmbeddingClient
from .links import build_links_for_table


ISO = "%Y-%m-%dT%H:%M:%SZ"

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime(ISO)

def _hash_text(*parts: str) -> str:
    joined = "||".join(part or "" for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

def _json_dump(obj: Any) -> bytes:
    return orjson.dumps(obj)

def _json_text(obj: Any) -> str:
    return orjson.dumps(obj).decode("utf-8")

def _line(prefix: str, value: Any) -> str | None:
    if value is None: return None
    s = str(value).strip()
    if not s: return None
    return f"{prefix}{s}"

def _period_ord_kind(pid: Any) -> tuple[int | None, str]:
    s = "".join(ch for ch in str(pid or "").strip() if ch.isdigit())
    if len(s) == 4: return int(s + "00"), "Y"
    if len(s) == 6: return int(s), "YM"
    if len(s) == 8: return int(s), "YMD"
    return None, "UNK"

def _agregado_text(md: dict[str, Any]) -> str:
    periodicidade = md.get("periodicidade") or {}
    freq = periodicidade.get("frequencia")
    inicio = periodicidade.get("inicio"); fim = periodicidade.get("fim")
    if inicio and fim and inicio != fim: period_line = f"Period: {inicio} - {fim}"
    elif inicio or fim: period_line = f"Period: {inicio or fim}"
    else: period_line = None
    nivel = md.get("nivelTerritorial") or {}
    level_parts = []
    for level_type in sorted(nivel):
        codes = nivel.get(level_type) or []
        if codes: level_parts.append(f"{level_type}: {', '.join(codes)}")
    lines = [
        f"Table {md.get('id')}: {str(md.get('nome') or '').strip()}".strip(),
        _line("Survey: ", md.get("pesquisa")),
        _line("Subject: ", md.get("assunto")),
        _line("Frequency: ", freq),
        period_line,
        f"Territorial levels: {'; '.join(level_parts)}" if level_parts else None,
        _line("URL: ", md.get("URL")),
    ]
    return "\n".join(l for l in lines if l)

def _vec_blob(vec: Sequence[float]) -> bytes:
    arr = array("f", (float(v) for v in vec))
    return arr.tobytes()

async def _persist_embeddings(conn, table_id: int, text: str, emb: EmbeddingClient, ts: str) -> None:
    model = emb.model
    text_hash = _hash_text("agregado", str(table_id), text)
    row = conn.execute(
        "SELECT text_hash FROM embeddings WHERE entity_type='agregado' AND entity_id=? AND model=?",
        (str(table_id), model),
    ).fetchone()
    if row and row["text_hash"] == text_hash:
        return
    vec = await asyncio.to_thread(emb.embed_text, text, model=model)
    with conn:
        conn.execute(
