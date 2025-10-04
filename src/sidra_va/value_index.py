from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict

from .db import create_connection
from .fingerprints import variable_fingerprint
from .schema_migrations import apply_va_schema
from .synonyms import SynonymMap, load_synonyms_into_memory
from .utils import json_dumps, run_with_retries, utcnow_iso

TWO_DIM_ALLOWLIST: set[tuple[int, int]] = set()


@dataclass
class _Variable:
    id: int
    name: str
    unit: str | None


@dataclass
class _Category:
    classification_id: int
    id: int
    name: str
    unit: str | None


@dataclass
class _Classification:
    id: int
    name: str
    categories: list[_Category]


async def build_va_index_for_agregado(
    agregado_id: int,
    *,
    allow_two_dim_combos: bool = False,
    upsert: bool = True,
) -> int:
    return await asyncio.to_thread(
        _build_va_index_for_agregado_sync,
        agregado_id,
        allow_two_dim_combos,
        upsert,
    )


def _build_va_index_for_agregado_sync(
    agregado_id: int,
    allow_two_dim_combos: bool,
    upsert: bool,
) -> int:
    conn = create_connection()
    try:
        apply_va_schema(conn)
        conn.row_factory = dict_row_factory
        cursor = conn.execute(
            "SELECT id, nome, pesquisa, assunto, periodo_inicio, periodo_fim, raw_json FROM agregados WHERE id = ?",
            (agregado_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Agregado {agregado_id} not found")
        variables = _load_variables(conn, agregado_id)
        if not variables:
            return 0
        classifications = _load_classifications(conn, agregado_id)
        levels = _load_levels(conn, agregado_id)
        synonyms = load_synonyms_into_memory(conn)

        created_at = utcnow_iso()
        total = 0
        for variable in variables:
            base_va_id = f"{agregado_id}::v{variable.id}"
            dims_json: list[dict[str, Any]] = []
            va_text = _build_va_text(
                variable,
                dims_json,
                levels,
                row,
            )
            total += _upsert_va(
                conn,
                base_va_id,
                agregado_id,
                variable,
                dims_json,
                va_text,
                levels,
                row,
                created_at,
            )

            for classification in classifications:
                for category in classification.categories:
                    dims_json = [
                        {
                            "classification_id": classification.id,
                            "classification_name": classification.name,
                            "category_id": category.id,
                            "category_name": category.name,
                        }
                    ]
                    suffix = f"c{classification.id}:{category.id}"
                    va_id = f"{base_va_id}::{suffix}"
                    va_text = _build_va_text(
                        variable,
                        dims_json,
                        levels,
                        row,
                    )
                    total += _upsert_va(
                        conn,
                        va_id,
                        agregado_id,
                        variable,
                        dims_json,
                        va_text,
                        levels,
                        row,
                        created_at,
                    )

            _upsert_fingerprint(conn, variable, synonyms)
        return total
    finally:
        conn.close()


def _build_va_text(
    variable: _Variable,
    dims_json: list[dict[str, Any]],
    levels: Dict[str, str],
    agreg_row: Dict[str, Any],
) -> str:
    lines: list[str] = []
    var_line = f"VAR: {variable.name}"
    if variable.unit:
        var_line += f" | UNIT: {variable.unit}"
    lines.append(var_line)
    for dim in dims_json:
        lines.append(
            f"CLASS: {dim['classification_name']} = {dim['category_name']}"
        )
    if levels:
        level_codes = ",".join(sorted(levels))
        lines.append(f"LEVELS: {level_codes}")
    period_start = agreg_row.get("periodo_inicio")
    period_end = agreg_row.get("periodo_fim")
    if period_start or period_end:
        if period_start and period_end and period_start != period_end:
            period_line = f"PERIOD: {period_start}–{period_end}"
        else:
            period_line = f"PERIOD: {period_start or period_end}"
        lines.append(period_line)
    if agreg_row.get("pesquisa"):
        lines.append(f"SURVEY: {agreg_row['pesquisa']}")
    if agreg_row.get("assunto"):
        lines.append(f"SUBJECT: {agreg_row['assunto']}")
    if agreg_row.get("nome"):
        lines.append(f"TABLE: {agreg_row['nome']}")
    return "\n".join(lines)


def _upsert_va(
    conn,
    va_id: str,
    agregado_id: int,
    variable: _Variable,
    dims_json: list[dict[str, Any]],
    va_text: str,
    levels: Dict[str, str],
    agreg_row: Dict[str, Any],
    created_at: str,
) -> int:
    has_n1 = 1 if "N1" in levels else 0
    has_n2 = 1 if "N2" in levels else 0
    has_n3 = 1 if "N3" in levels else 0
    has_n6 = 1 if "N6" in levels else 0
    def _write() -> None:
        with conn:
            conn.execute(
                """
                INSERT INTO value_atoms (
                    va_id, agregado_id, variable_id, unit, text, dims_json,
                    has_n1, has_n2, has_n3, has_n6, period_start, period_end,
                    survey, subject, table_title, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(va_id) DO UPDATE SET
                    agregado_id=excluded.agregado_id,
                    variable_id=excluded.variable_id,
                    unit=excluded.unit,
                    text=excluded.text,
                    dims_json=excluded.dims_json,
                    has_n1=excluded.has_n1,
                    has_n2=excluded.has_n2,
                    has_n3=excluded.has_n3,
                    has_n6=excluded.has_n6,
                    period_start=excluded.period_start,
                    period_end=excluded.period_end,
                    survey=excluded.survey,
                    subject=excluded.subject,
                    table_title=excluded.table_title
                """,
                (
                    va_id,
                    agregado_id,
                    variable.id,
                    variable.unit,
                    va_text,
                    json_dumps(dims_json),
                    has_n1,
                    has_n2,
                    has_n3,
                    has_n6,
                    agreg_row.get("periodo_inicio"),
                    agreg_row.get("periodo_fim"),
                    agreg_row.get("pesquisa"),
                    agreg_row.get("assunto"),
                    agreg_row.get("nome"),
                    created_at,
                ),
            )

            conn.execute("DELETE FROM value_atoms_fts WHERE va_id = ?", (va_id,))
            conn.execute(
                "INSERT INTO value_atoms_fts(va_id, text, table_title, survey, subject) VALUES(?,?,?,?,?)",
                (
                    va_id,
                    va_text,
                    agreg_row.get("nome"),
                    agreg_row.get("pesquisa"),
                    agreg_row.get("assunto"),
                ),
            )

            conn.execute("DELETE FROM value_atom_dims WHERE va_id = ?", (va_id,))
            if dims_json:
                conn.executemany(
                    """
                    INSERT INTO value_atom_dims(
                        va_id, classification_id, classification_name, category_id, category_name
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            va_id,
                            dim["classification_id"],
                            dim["classification_name"],
                            dim["category_id"],
                            dim["category_name"],
                        )
                        for dim in dims_json
                    ],
                )

    run_with_retries(_write)
    return 1


def _upsert_fingerprint(conn, variable: _Variable, synonyms: SynonymMap) -> None:
    fingerprint = variable_fingerprint(variable.name, variable.unit, synonyms)
    run_with_retries(
        lambda: _write_fingerprint(conn, variable.id, fingerprint)
    )


def _write_fingerprint(conn, variable_id: int, fingerprint: str) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO variable_fingerprints(variable_id, fingerprint) VALUES(?, ?)",
            (variable_id, fingerprint),
        )


def _load_variables(conn, agregado_id: int) -> list[_Variable]:
    cursor = conn.execute(
        "SELECT id, nome, unidade FROM variables WHERE agregado_id = ? ORDER BY id",
        (agregado_id,),
    )
    return [
        _Variable(id=row["id"], name=row["nome"], unit=row["unidade"]) for row in cursor.fetchall()
    ]


def _load_classifications(conn, agregado_id: int) -> list[_Classification]:
    classifications_cur = conn.execute(
        "SELECT id, nome FROM classifications WHERE agregado_id = ? ORDER BY id",
        (agregado_id,),
    )
    classifications: list[_Classification] = []
    for row in classifications_cur.fetchall():
        cat_cursor = conn.execute(
            """
            SELECT categoria_id, nome, unidade
            FROM categories
            WHERE agregado_id = ? AND classification_id = ?
            ORDER BY categoria_id
            """,
            (agregado_id, row["id"]),
        )
        categories = [
            _Category(
                classification_id=row["id"],
                id=cat_row["categoria_id"],
                name=cat_row["nome"],
                unit=cat_row["unidade"],
            )
            for cat_row in cat_cursor.fetchall()
        ]
        classifications.append(
            _Classification(id=row["id"], name=row["nome"], categories=categories)
        )
    return classifications


def _load_levels(conn, agregado_id: int) -> Dict[str, str]:
    cursor = conn.execute(
        "SELECT level_id, level_name FROM agregados_levels WHERE agregado_id = ?",
        (agregado_id,),
    )
    return {row["level_id"]: row["level_name"] for row in cursor.fetchall()}


def _decode_raw_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return dict(raw or {})


def dict_row_factory(cursor, row):
    return {cursor.description[idx][0]: value for idx, value in enumerate(row)}


async def build_va_index_for_all(
    *,
    concurrency: int = 1,
    allow_two_dim_combos: bool = False,
) -> dict[str, int]:
    conn = create_connection()
    try:
        apply_va_schema(conn)
        cursor = conn.execute(
            "SELECT DISTINCT agregado_id FROM variables ORDER BY agregado_id"
        )
        ids = [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()

    if not ids:
        return {}

    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, int] = {}

    async def _run(agregado_id: int) -> None:
        async with semaphore:
            count = await build_va_index_for_agregado(
                agregado_id,
                allow_two_dim_combos=allow_two_dim_combos,
            )
            results[str(agregado_id)] = count

    await asyncio.gather(*[_run(ag_id) for ag_id in ids])
    return results


__all__ = [
    "build_va_index_for_agregado",
    "build_va_index_for_all",
]
