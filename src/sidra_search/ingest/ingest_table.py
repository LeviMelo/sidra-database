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
from ..db.base_schema import apply_base_schema
from ..ingest.links import build_links_for_table
from ..net.api_client import SidraApiClient
from ..net.embedding_client import EmbeddingClient
from ..search.fuzzy3gram import reset_cache

ISO = "%Y-%m-%dT%H:%M:%SZ"


def _hash_fields(*vals: object) -> str:
    # Robust: stringify, replace None with "", and hash a stable joined string
    return _sha256_text("||".join("" if v is None else str(v) for v in vals))

def _now() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def _json(obj: Any) -> bytes:
    return orjson.dumps(obj)


def _json_text(obj: Any) -> str:
    return orjson.dumps(obj).decode("utf-8")


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _period_to_ord_kind(periodo_id: Any) -> tuple[int | None, str]:
    """
    YYYY -> (YYYY00, 'Y')
    YYYYMM -> (YYYYMM, 'YM')
    YYYYMMDD -> (YYYYMMDD, 'YMD')
    otherwise -> (None, 'UNK')
    """
    s = str(periodo_id or "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 4:
        return int(digits + "00"), "Y"
    if len(digits) == 6:
        return int(digits), "YM"
    if len(digits) == 8:
        return int(digits), "YMD"
    return None, "UNK"


def _canonical_table_text(md: dict[str, Any]) -> str:
    periodicidade = md.get("periodicidade") or {}
    freq = periodicidade.get("frequencia")
    inicio = periodicidade.get("inicio")
    fim = periodicidade.get("fim")

    if inicio and fim:
        period_line = f"Period: {inicio} - {fim}" if inicio != fim else f"Period: {inicio}"
    elif inicio or fim:
        period_line = f"Period: {inicio or fim}"
    else:
        period_line = None

    nivel = md.get("nivelTerritorial") or {}
    level_parts: list[str] = []
    if isinstance(nivel, dict):
        for level_type in sorted(nivel):
            codes = nivel.get(level_type) or []
            if codes:
                level_parts.append(f"{level_type}: {', '.join(str(c) for c in codes)}")

    lines = [
        f"Table {str(md.get('id'))}: {str(md.get('nome') or '').strip()}".strip(),
        f"Survey: {str(md.get('pesquisa') or '')}" if md.get("pesquisa") else None,
        f"Subject: {str(md.get('assunto') or '')}" if md.get("assunto") else None,
        f"Frequency: {str(freq)}" if freq else None,
        period_line,
        f"Territorial levels: {'; '.join(level_parts)}" if level_parts else None,
        f"URL: {str(md.get('URL') or '')}" if md.get("URL") else None,
    ]
    # every yielded piece is a str; ignore falsy items
    return "\n".join([s for s in lines if isinstance(s, str) and s])



def _vec_to_blob(vec: Sequence[float]) -> bytes:
    arr = array("f", (float(x) for x in vec))
    return arr.tobytes()


async def _ensure_embeddings(table_id: int, text: str, *, embedding_client: EmbeddingClient) -> None:
    if not text.strip():
        return
    text_hash = _sha256_text(text)
    model = embedding_client.model

    def _read_existing() -> str | None:
        with sqlite_session() as conn:
            cur = conn.execute(
                "SELECT text_hash FROM embeddings "
                "WHERE entity_type='agregado' AND entity_id=? AND model=?",
                (str(table_id), model),
            ).fetchone()
            return cur["text_hash"] if cur else None

    existing_hash = await asyncio.to_thread(_read_existing)
    if existing_hash == text_hash:
        return

    # compute vector
    vector = await asyncio.to_thread(embedding_client.embed_text, text, model=model)
    dim = len(vector)

    def _write():
        with sqlite_session() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings(
                      entity_type, entity_id, agregado_id, text_hash, model, dimension, vector, created_at
                    ) VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        "agregado",
                        str(table_id),
                        table_id,
                        text_hash,
                        model,
                        dim,
                        _vec_to_blob(vector),
                        _now(),
                    ),
                )

    await asyncio.to_thread(_write)


async def ingest_table(
    table_id: int,
    *,
    client: SidraApiClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    build_links: bool = True,
    prefetched_localities: dict[str, list] | None = None,  # optional reuse from probe
) -> None:
    """
    Fetch metadata from SIDRA and persist into base + search schemas.
    Also refresh the table title FTS row and (optionally) a title embedding.
    """
    # ensure schema
    with sqlite_session() as conn:
        apply_base_schema(conn)
        apply_search_schema(conn)
        conn.commit()

    own_client = False
    if client is None:
        client = SidraApiClient()
        own_client = True

    try:
        # --- fetch payloads
        try:
            md = await client.fetch_metadata(table_id)
        except Exception:
            _log_ingestion(table_id, "error", "api:metadata")
            raise
        try:
            periods = await client.fetch_periods(table_id)
        except Exception:
            _log_ingestion(table_id, "error", "api:periods")
            raise

        # --- derive levels + localities (count + names)
        nivel_groups = md.get("nivelTerritorial") or {}
        if not isinstance(nivel_groups, dict):
            nivel_groups = {}

        settings = get_settings()
        municipality_count = 0
        level_rows: list[tuple[int, str, str | None, str, int]] = []
        locality_rows: list[tuple[int, str, str | None, str | None]] = []

        # NOTE:
        # - `prefetched_localities` is a dict like {"N3": [...], "N6": [...]} if passed by the caller.
        # - Keys we store/reuse are UPPERCASE to avoid case mismatches.
        for level_type, codes in (nivel_groups or {}).items():
            if not codes:
                continue
            # Be defensive: ensure we can iterate even if API returned a scalar.
            try:
                code_list = list(codes)
            except TypeError:
                code_list = [codes]

            for code in code_list:
                code_s = str(code)
                code_key = code_s.upper()

                # Prefer the probe's cached payload for this level (e.g., N3/N6).
                payload: list | None = None
                if prefetched_localities and isinstance(prefetched_localities.get(code_key), list):
                    payload = prefetched_localities[code_key]
                else:
                    try:
                        payload = await client.fetch_localities(table_id, code_s)
                    except Exception:
                        payload = []
                    if not isinstance(payload, list):
                        try:
                            payload = list(payload)
                        except Exception:
                            payload = []

                # Count first (coverage uses counts only)
                count = len(payload)
                if code_key == "N6":
                    municipality_count = max(municipality_count, count)

                # Try to extract a representative level_name (same as before)
                level_name = None
                if payload:
                    try:
                        node = payload[0].get("nivel") if isinstance(payload[0], dict) else None
                        if isinstance(node, dict):
                            level_name = node.get("nome")
                    except Exception:
                        level_name = None

                # Record per-level counts
                level_rows.append((table_id, code_s, level_name, str(level_type), count))

                # Persist exact membership (unchanged behavior)
                if payload:
                    for loc in payload:
                        if isinstance(loc, dict):
                            lid = loc.get("id")
                            lname = loc.get("nome")
                        else:
                            lid = None
                            lname = None
                        locality_rows.append((table_id, code_s, lid, lname))


        covers_nat = (
            1
            if (municipality_count >= max(0, int(settings.municipality_national_threshold)))
            else 0
        )

        # variables
        variables = md.get("variaveis") or []
        var_rows: list[tuple[Any, int, Any, Any, str, str]] = []
        for v in variables:
            var_rows.append(
                (
                    v.get("id"),
                    table_id,
                    v.get("nome"),
                    v.get("unidade"),
                    _json_text(v.get("sumarizacao", [])),
                    _hash_fields(v.get("id"), v.get("nome"), v.get("unidade")),   # <— here
                )
            )

        # classifications + categories
        class_rows: list[tuple[int, int, str | None, int, str]] = []
        cat_rows: list[tuple[int, int, int, str | None, str | None, Any, str]] = []
        for cl in md.get("classificacoes", []) or []:
            cid = cl.get("id")
            class_rows.append(
                (
                    cid,
                    table_id,
                    cl.get("nome"),
                    1 if (cl.get("sumarizacao", {}).get("status")) else 0,
                    _json_text(cl.get("sumarizacao", {}).get("excecao", [])),
                )
            )
            for cat in cl.get("categorias", []) or []:
                cat_rows.append(
                    (
                        table_id,
                        cid,
                        cat.get("id"),
                        cat.get("nome"),
                        cat.get("unidade"),
                        cat.get("nivel"),
                        _hash_fields(cid, cat.get("id"), cat.get("nome"), cat.get("unidade")),  # <— here
                    )
                )

        # periods
        period_rows: list[tuple[int, str, str, Any, int | None, str]] = []
        for p in periods or []:
            pid = p.get("id") if isinstance(p, dict) else p
            literals = p.get("literals", [pid]) if isinstance(p, dict) else [pid]
            modificacao = p.get("modificacao") if isinstance(p, dict) else None
            ord_val, kind = _period_to_ord_kind(pid)
            period_rows.append((table_id, str(pid), _json_text(literals), modificacao, ord_val, kind))

        # --- write to DB (replace child rows)
        fetched_at = _now()

        with sqlite_session() as conn:
            apply_base_schema(conn)
            apply_search_schema(conn)

            conn.execute("BEGIN")
            try:
                # parent: agregados header
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agregados(
                      id, nome, pesquisa, assunto, url, freq, periodo_inicio, periodo_fim,
                      raw_json, fetched_at, municipality_locality_count, covers_national_municipalities
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        md.get("id"),
                        md.get("nome"),
                        md.get("pesquisa"),
                        md.get("assunto"),
                        md.get("URL"),
                        (md.get("periodicidade") or {}).get("frequencia"),
                        (md.get("periodicidade") or {}).get("inicio"),
                        (md.get("periodicidade") or {}).get("fim"),
                        _json(md),
                        fetched_at,
                        municipality_count,
                        covers_nat,
                    ),
                )

                # purge children → insert fresh
                conn.execute("DELETE FROM localities WHERE agregado_id=?", (table_id,))
                conn.execute("DELETE FROM agregados_levels WHERE agregado_id=?", (table_id,))
                conn.execute("DELETE FROM categories WHERE agregado_id=?", (table_id,))
                conn.execute("DELETE FROM classifications WHERE agregado_id=?", (table_id,))
                conn.execute("DELETE FROM variables WHERE agregado_id=?", (table_id,))
                conn.execute("DELETE FROM periods WHERE agregado_id=?", (table_id,))

                if level_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO agregados_levels(
                          agregado_id, level_id, level_name, level_type, locality_count
                        ) VALUES(?,?,?,?,?)
                        """,
                        level_rows,
                    )
                if var_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO variables(
                          id, agregado_id, nome, unidade, sumarizacao, text_hash
                        ) VALUES(?,?,?,?,?,?)
                        """,
                        var_rows,
                    )
                if class_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO classifications(
                          id, agregado_id, nome, sumarizacao_status, sumarizacao_excecao
                        ) VALUES(?,?,?,?,?)
                        """,
                        class_rows,
                    )
                if cat_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO categories(
                          agregado_id, classification_id, categoria_id, nome, unidade, nivel, text_hash
                        ) VALUES(?,?,?,?,?,?,?)
                        """,
                        cat_rows,
                    )
                if period_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO periods(
                          agregado_id, periodo_id, literals, modificacao, periodo_ord, periodo_kind
                        ) VALUES(?,?,?,?,?,?)
                        """,
                        period_rows,
                    )
                if locality_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO localities(
                          agregado_id, level_id, locality_id, nome
                        ) VALUES(?,?,?,?)
                        """,
                        locality_rows,
                    )

                # refresh FTS for title/survey/subject
                conn.execute("DELETE FROM table_titles_fts WHERE table_id=?", (table_id,))
                conn.execute(
                    "INSERT INTO table_titles_fts(table_id, title, survey, subject) VALUES(?,?,?,?)",
                    (
                        table_id,
                        md.get("nome") or "",
                        md.get("pesquisa") or "",
                        md.get("assunto") or "",
                    ),
                )

                # log
                conn.execute(
                    "INSERT INTO ingestion_log(agregado_id, stage, status, detail, run_at) VALUES(?,?,?,?,?)",
                    (table_id, "metadata", "success", None, fetched_at),
                )

                conn.commit()
            except Exception:
                conn.rollback()
                _log_ingestion(table_id, "error", "db:write")
                raise

        # build name→table link indexes
        if build_links:
            await asyncio.to_thread(build_links_for_table, table_id)
            # make new names visible to in-process searches
            reset_cache()  # ADD

        # optional embeddings for title text
        if get_settings().enable_title_embeddings:
            try:
                embedding_client = embedding_client or EmbeddingClient()
                text = _canonical_table_text(md)
                await _ensure_embeddings(table_id, text, embedding_client=embedding_client)
            except Exception:
                # don't break ingestion if embeddings fail
                pass

    finally:
        if own_client:
            await client.close()


def _log_ingestion(table_id: int, status: str, detail: str) -> None:
    with sqlite_session() as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO ingestion_log(agregado_id, stage, status, detail, run_at)
                VALUES(?,?,?,?,?)
                """,
                (table_id, "metadata", status, detail, _now()),
            )
