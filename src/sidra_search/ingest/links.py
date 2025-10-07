from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..db.session import create_connection
from ..db.migrations import apply_search_schema
from ..search.normalize import normalize_basic


@dataclass
class LinkCounts:
    vars: int = 0
    classes: int = 0
    cats: int = 0
    var_class: int = 0


def _delete_links_for_table(conn, table_id: int) -> None:
    with conn:
        conn.execute("DELETE FROM link_var WHERE table_id = ?", (table_id,))
        conn.execute("DELETE FROM link_class WHERE table_id = ?", (table_id,))
        conn.execute("DELETE FROM link_cat WHERE table_id = ?", (table_id,))
        conn.execute("DELETE FROM link_var_class WHERE table_id = ?", (table_id,))


def _infer_var_class_pairs(metadata_json: dict) -> List[Tuple[int, int]]:
    """
    Optional hook: try to infer actual applicable {variable_id, class_id} pairs from metadata.
    For now, return empty (fallback to cross-product).
    """
    return []


def build_links_for_table(table_id: int) -> LinkCounts:
    conn = create_connection()
    try:
        apply_search_schema(conn)

        # load rows
        var_rows = conn.execute(
            "SELECT id, nome FROM variables WHERE agregado_id = ? ORDER BY id", (table_id,)
        ).fetchall()
        class_rows = conn.execute(
            "SELECT id, nome FROM classifications WHERE agregado_id = ? ORDER BY id", (table_id,)
        ).fetchall()
        cat_rows = conn.execute(
            """
            SELECT classification_id, categoria_id, nome
            FROM categories
            WHERE agregado_id = ?
            ORDER BY classification_id, categoria_id
            """,
            (table_id,),
        ).fetchall()
        if not var_rows and not class_rows:
            return LinkCounts()

        cats_by_class: Dict[int, List[Tuple[int, str]]] = {}
        for cr in cat_rows:
            cats_by_class.setdefault(int(cr["classification_id"]), []).append((int(cr["categoria_id"]), cr["nome"]))

        _delete_links_for_table(conn, table_id)

        with conn:
            # variables
            for vr in var_rows:
                v_id = int(vr["id"]); raw = str(vr["nome"] or ""); key = normalize_basic(raw)
                if not key: continue
                conn.execute("INSERT OR IGNORE INTO name_keys(kind,key,raw) VALUES(?,?,?)", ("var", key, raw))
                conn.execute(
                    "INSERT OR IGNORE INTO link_var(var_key, table_id, variable_id) VALUES(?,?,?)",
                    (key, table_id, v_id),
                )

            # classes & cats
            for cl in class_rows:
                c_id = int(cl["id"]); raw = str(cl["nome"] or ""); key = normalize_basic(raw)
                if not key: continue
                conn.execute("INSERT OR IGNORE INTO name_keys(kind,key,raw) VALUES(?,?,?)", ("class", key, raw))
                conn.execute(
                    "INSERT OR IGNORE INTO link_class(class_key, table_id, class_id) VALUES(?,?,?)",
                    (key, table_id, c_id),
                )
                for cat_id, cat_raw in cats_by_class.get(c_id, []):
                    cat_key = normalize_basic(str(cat_raw or ""))
                    if not cat_key: continue
                    conn.execute("INSERT OR IGNORE INTO name_keys(kind,key,raw) VALUES(?,?,?)", ("cat", cat_key, cat_raw))
                    conn.execute(
                        "INSERT OR IGNORE INTO link_cat(class_key, cat_key, table_id, class_id, category_id) VALUES(?,?,?,?,?)",
                        (key, cat_key, table_id, c_id, int(cat_id)),
                    )

            # var×class pairs — cross-product fallback
            inferred = set(_infer_var_class_pairs({}))  # reserved for future use
            if inferred:
                to_pairs = inferred
            else:
                to_pairs = {(int(vr["id"]), int(cl["id"])) for vr in var_rows for cl in class_rows}

            for v_id, c_id in to_pairs:
                v_key = normalize_basic(str(next((vr["nome"] for vr in var_rows if int(vr["id"])==v_id), "")))
                c_key = normalize_basic(str(next((cr["nome"] for cr in class_rows if int(cr["id"])==c_id), "")))
                if not v_key or not c_key: continue
                conn.execute(
                    "INSERT OR IGNORE INTO link_var_class(var_key, class_key, table_id, variable_id, class_id) VALUES(?,?,?,?,?)",
                    (v_key, c_key, table_id, v_id, c_id),
                )

        c_var = conn.execute("SELECT COUNT(*) FROM link_var WHERE table_id=?", (table_id,)).fetchone()[0]
        c_cls = conn.execute("SELECT COUNT(*) FROM link_class WHERE table_id=?", (table_id,)).fetchone()[0]
        c_cat = conn.execute("SELECT COUNT(*) FROM link_cat WHERE table_id=?", (table_id,)).fetchone()[0]
        c_vc  = conn.execute("SELECT COUNT(*) FROM link_var_class WHERE table_id=?", (table_id,)).fetchone()[0]
        return LinkCounts(c_var, c_cls, c_cat, c_vc)
    finally:
        conn.close()
