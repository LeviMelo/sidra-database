# src/sidra_va/links.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .db import create_connection
from .schema_migrations import apply_va_schema
from .synonyms import normalize_basic
from .utils import run_with_retries


@dataclass
class LinkCounts:
    vars: int = 0
    classes: int = 0
    cats: int = 0
    var_class: int = 0


def _delete_links_for_table(conn, table_id: int) -> None:
    def _do():
        with conn:
            conn.execute("DELETE FROM link_var WHERE table_id = ?", (table_id,))
            conn.execute("DELETE FROM link_class WHERE table_id = ?", (table_id,))
            conn.execute("DELETE FROM link_cat WHERE table_id = ?", (table_id,))
            conn.execute("DELETE FROM link_var_class WHERE table_id = ?", (table_id,))
    run_with_retries(_do)


def build_links_for_agregado(agregado_id: int) -> LinkCounts:
    """
    Build (or rebuild) link indexes for a single table.
    """
    conn = create_connection()
    try:
        apply_va_schema(conn)

        # Load variables
        var_rows = conn.execute(
            "SELECT id, nome FROM variables WHERE agregado_id = ? ORDER BY id",
            (agregado_id,),
        ).fetchall()
        if not var_rows:
            return LinkCounts()

        # Load classes and categories
        class_rows = conn.execute(
            "SELECT id, nome FROM classifications WHERE agregado_id = ? ORDER BY id",
            (agregado_id,),
        ).fetchall()
        cat_rows = conn.execute(
            """
            SELECT classification_id, categoria_id, nome
            FROM categories
            WHERE agregado_id = ?
            ORDER BY classification_id, categoria_id
            """,
            (agregado_id,),
        ).fetchall()

        # Map categories by class_id
        cats_by_class: Dict[int, List[Tuple[int, str]]] = {}
        for cr in cat_rows:
            cats_by_class.setdefault(int(cr["classification_id"]), []).append((int(cr["categoria_id"]), cr["nome"]))

        _delete_links_for_table(conn, agregado_id)

        var_count = 0
        class_count = 0
        cat_count = 0
        var_class_count = 0

        def _insert():
            with conn:
                # name_keys + link_var
                for vr in var_rows:
                    v_id = int(vr["id"])
                    v_raw = str(vr["nome"] or "")
                    v_key = normalize_basic(v_raw)
                    if not v_key:
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO name_keys(kind, key, raw) VALUES(?,?,?)",
                        ("var", v_key, v_raw),
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO link_var(var_key, table_id, variable_id) VALUES(?,?,?)",
                        (v_key, agregado_id, v_id),
                    )
                # name_keys + link_class + link_cat (+ var_class)
                for cl in class_rows:
                    c_id = int(cl["id"])
                    c_raw = str(cl["nome"] or "")
                    c_key = normalize_basic(c_raw)
                    if not c_key:
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO name_keys(kind, key, raw) VALUES(?,?,?)",
                        ("class", c_key, c_raw),
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO link_class(class_key, table_id, class_id) VALUES(?,?,?)",
                        (c_key, agregado_id, c_id),
                    )
                    # categories for this class
                    for cat_id, cat_raw in cats_by_class.get(c_id, []):
                        cat_key = normalize_basic(str(cat_raw or ""))
                        if not cat_key:
                            continue
                        conn.execute(
                            "INSERT OR IGNORE INTO name_keys(kind, key, raw) VALUES(?,?,?)",
                            ("cat", cat_key, cat_raw),
                        )
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO link_cat(class_key, cat_key, table_id, class_id, category_id)
                            VALUES(?,?,?,?,?)
                            """,
                            (c_key, cat_key, agregado_id, c_id, int(cat_id)),
                        )
                # var_class pairs (cross all vars × all classes present in table)
                for vr in var_rows:
                    v_id = int(vr["id"])
                    v_key = normalize_basic(str(vr["nome"] or ""))
                    if not v_key:
                        continue
                    for cl in class_rows:
                        c_id = int(cl["id"])
                        c_key = normalize_basic(str(cl["nome"] or ""))
                        if not c_key:
                            continue
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO link_var_class(var_key, class_key, table_id, variable_id, class_id)
                            VALUES(?,?,?,?,?)
                            """,
                            (v_key, c_key, agregado_id, v_id, c_id),
                        )

        run_with_retries(_insert)

        # quick counts
        var_count = conn.execute("SELECT COUNT(*) FROM link_var WHERE table_id = ?", (agregado_id,)).fetchone()[0]
        class_count = conn.execute("SELECT COUNT(*) FROM link_class WHERE table_id = ?", (agregado_id,)).fetchone()[0]
        cat_count = conn.execute("SELECT COUNT(*) FROM link_cat WHERE table_id = ?", (agregado_id,)).fetchone()[0]
        var_class_count = conn.execute("SELECT COUNT(*) FROM link_var_class WHERE table_id = ?", (agregado_id,)).fetchone()[0]

        return LinkCounts(var_count, class_count, cat_count, var_class_count)
    finally:
        conn.close()


async def build_links_for_all(concurrency: int = 4) -> dict[int, LinkCounts]:
    """
    Build link indexes for all tables that have variables (like build_va_index_for_all).
    """
    conn = create_connection()
    try:
        apply_va_schema(conn)
        cur = conn.execute("SELECT DISTINCT agregado_id FROM variables ORDER BY agregado_id")
        ids = [int(r[0]) for r in cur.fetchall()]
    finally:
        conn.close()

    if not ids:
        return {}

    sem = asyncio.Semaphore(max(1, concurrency))
    results: dict[int, LinkCounts] = {}

    async def worker(table_id: int) -> None:
        async with sem:
            counts = await asyncio.to_thread(build_links_for_agregado, table_id)
            results[table_id] = counts

    await asyncio.gather(*(worker(tid) for tid in ids))
    return results
