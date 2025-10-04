from __future__ import annotations

import argparse
import asyncio
import json
from typing import Iterable

from sidra_database.db import create_connection, ensure_schema

from .embed import embed_vas_for_agregados
from .neighbors import find_neighbors_for_va
from .schema_migrations import apply_va_schema, get_schema_version
from .search_va import VaResult, VaSearchFilters, search_value_atoms
from .synonyms import export_synonyms_csv, import_synonyms_csv
from .value_index import build_va_index_for_agregado, build_va_index_for_all


def _ensure_base_schema() -> None:
    conn = create_connection()
    try:
        ensure_schema(conn)
        apply_va_schema(conn)
        conn.commit()
    finally:
        conn.close()


def cmd_db_migrate(args: argparse.Namespace) -> None:
    _ensure_base_schema()
    conn = create_connection()
    try:
        version = get_schema_version(conn)
    finally:
        conn.close()
    print(f"VA schema version: {version}")


def cmd_db_stats(args: argparse.Namespace) -> None:
    _ensure_base_schema()
    conn = create_connection()
    try:
        tables = [
            "value_atoms",
            "value_atom_dims",
            "value_atoms_fts",
            "variable_fingerprints",
            "synonyms",
        ]
        stats = {}
        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count
    finally:
        conn.close()
    for table, count in stats.items():
        print(f"{table}: {count}")


def cmd_index_build(args: argparse.Namespace) -> None:
    _ensure_base_schema()
    if args.all:
        result = asyncio.run(
            build_va_index_for_all(
                concurrency=args.concurrent,
                allow_two_dim_combos=args.allow_two_dim_combos,
            )
        )
        if not result:
            print("No agregados found. Please ingest metadata first.")
        else:
            for ag, count in sorted(result.items()):
                print(f"agregado {ag}: {count} VAs")
        return

    if not args.ids:
        raise SystemExit("Provide --ids or --all")
    for ag_id in args.ids:
        count = asyncio.run(
            build_va_index_for_agregado(
                ag_id,
                allow_two_dim_combos=args.allow_two_dim_combos,
            )
        )
        print(f"agregado {ag_id}: {count} VAs")


def cmd_index_embed(args: argparse.Namespace) -> None:
    _ensure_base_schema()
    ids = args.ids if not args.all else None
    if ids is None and not args.all:
        raise SystemExit("Provide --ids or --all")
    stats = asyncio.run(
        embed_vas_for_agregados(
            ids,
            concurrency=args.concurrent,
            model=args.model,
        )
    )
    if not any(stats.values()):
        print("No VAs to embed. Build VAs first.")
    else:
        print(json.dumps(stats, indent=2))


def cmd_index_rebuild_fts(args: argparse.Namespace) -> None:
    _ensure_base_schema()
    conn = create_connection()
    try:
        rows = conn.execute("SELECT va_id, text, table_title, survey, subject FROM value_atoms").fetchall()
        with conn:
            conn.execute("DELETE FROM value_atoms_fts")
            conn.executemany(
                "INSERT INTO value_atoms_fts(va_id, text, table_title, survey, subject) VALUES(?,?,?,?,?)",
                rows,
            )
    finally:
        conn.close()
    print(f"Rebuilt FTS index for {len(rows)} VAs")


def cmd_synonyms_import(args: argparse.Namespace) -> None:
    _ensure_base_schema()
    conn = create_connection()
    try:
        count = import_synonyms_csv(args.path, conn)
    finally:
        conn.close()
    print(f"Imported {count} synonym rows")


def cmd_synonyms_export(args: argparse.Namespace) -> None:
    _ensure_base_schema()
    conn = create_connection()
    try:
        count = export_synonyms_csv(args.path, conn)
    finally:
        conn.close()
    print(f"Exported {count} synonym rows to {args.path}")


def _print_results(results: list[VaResult], json_output: bool) -> None:
    if json_output:
        payload = [
            {
                "va_id": item.va_id,
                "agregado_id": item.agregado_id,
                "variable_id": item.variable_id,
                "title": item.title,
                "score": item.score,
                "rrf_score": item.rrf_score,
                "struct_score": item.struct_score,
                "metadata": dict(item.metadata),
                "why": item.why,
            }
            for item in results
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    for item in results:
        print(f"score={item.score:.3f} (struct={item.struct_score:.3f}, rrf={item.rrf_score:.3f}) {item.title}")
        print(f"  va_id={item.va_id} agregado={item.agregado_id}")
        print(f"  why: {item.why}")


def _parse_period(period: str | None) -> tuple[int | None, int | None]:
    if not period:
        return None, None
    if "-" in period:
        start, end = period.split("-", 1)
        return int(start), int(end)
    year = int(period)
    return year, year


def cmd_search_va(args: argparse.Namespace) -> None:
    _ensure_base_schema()
    period_start, period_end = _parse_period(args.period)
    filters = VaSearchFilters(
        require_levels=tuple(args.require_level or []),
        period_start=period_start,
        period_end=period_end,
        must_variable_ids=tuple(args.must_variable_id or []),
        must_variable_names=tuple(args.must_variable or []),
        must_classification_names=tuple(args.must_class or []),
        must_category_names=tuple(args.must_category or []),
        min_municipalities=args.min_municipalities,
        requires_national_munis=args.requires_national_munis,
    )
    results = asyncio.run(
        search_value_atoms(
            args.query,
            filters=filters,
            limit=args.limit,
        )
    )
    if not results:
        print("VA index empty. Run 'index build-va' first.")
        return
    _print_results(results, args.json)


def cmd_show_table(args: argparse.Namespace) -> None:
    _ensure_base_schema()
    conn = create_connection()
    try:
        row = conn.execute(
            "SELECT id, nome, pesquisa, assunto, periodo_inicio, periodo_fim FROM agregados WHERE id = ?",
            (args.agregado_id,),
        ).fetchone()
        if not row:
            print("Agregado not found")
            return
        print(f"Tabela {row['id']}: {row['nome']}")
        if row["pesquisa"]:
            print(f"Pesquisa: {row['pesquisa']}")
        if row["assunto"]:
            print(f"Assunto: {row['assunto']}")
        if row["periodo_inicio"] or row["periodo_fim"]:
            print(f"Período: {row['periodo_inicio']} - {row['periodo_fim']}")
        print("Variáveis:")
        cursor = conn.execute(
            "SELECT id, nome, unidade FROM variables WHERE agregado_id = ? ORDER BY id",
            (args.agregado_id,),
        )
        for vid, nome, unidade in cursor.fetchall():
            print(f"  {vid}: {nome} ({unidade or 'sem unidade'})")
        print("Classificações (top 10 categorias):")
        cursor = conn.execute(
            "SELECT id, nome FROM classifications WHERE agregado_id = ? ORDER BY id",
            (args.agregado_id,),
        )
        for cid, nome in cursor.fetchall():
            print(f"  {cid}: {nome}")
            cat_cursor = conn.execute(
                """
                SELECT categoria_id, nome FROM categories
                WHERE agregado_id = ? AND classification_id = ?
                ORDER BY categoria_id LIMIT 10
                """,
                (args.agregado_id, cid),
            )
            for cat_id, cat_nome in cat_cursor.fetchall():
                print(f"    - {cat_id}: {cat_nome}")
    finally:
        conn.close()


def cmd_show_va(args: argparse.Namespace) -> None:
    _ensure_base_schema()
    conn = create_connection()
    try:
        row = conn.execute(
            "SELECT va_id, text FROM value_atoms WHERE va_id = ?",
            (args.va_id,),
        ).fetchone()
        if not row:
            print("VA not found")
            return
        print(row["text"])
    finally:
        conn.close()


def cmd_link_neighbors(args: argparse.Namespace) -> None:
    _ensure_base_schema()
    neighbors = find_neighbors_for_va(
        args.va_id,
        top_k=args.top_k,
        require_same_unit=not args.allow_unit_mismatch,
    )
    if not neighbors:
        print("No compatible VAs found")
        return
    for result, score in neighbors:
        print(f"compat={score:.3f} {result.title} [{result.va_id}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sidra-va")
    subparsers = parser.add_subparsers(dest="command")

    db_parser = subparsers.add_parser("db")
    db_sub = db_parser.add_subparsers(dest="db_command")

    migrate = db_sub.add_parser("migrate")
    migrate.set_defaults(func=cmd_db_migrate)

    stats = db_sub.add_parser("stats")
    stats.set_defaults(func=cmd_db_stats)

    index_parser = subparsers.add_parser("index")
    index_sub = index_parser.add_subparsers(dest="index_command")

    build_va = index_sub.add_parser("build-va")
    build_va.add_argument("--ids", nargs="*", type=int)
    build_va.add_argument("--all", action="store_true")
    build_va.add_argument("--allow-two-dim-combos", action="store_true")
    build_va.add_argument("--concurrent", type=int, default=1)
    build_va.set_defaults(func=cmd_index_build)

    embed_va = index_sub.add_parser("embed-va")
    embed_va.add_argument("--ids", nargs="*", type=int)
    embed_va.add_argument("--all", action="store_true")
    embed_va.add_argument("--concurrent", type=int, default=6)
    embed_va.add_argument("--model")
    embed_va.set_defaults(func=cmd_index_embed)

    rebuild_fts = index_sub.add_parser("rebuild-fts")
    rebuild_fts.set_defaults(func=cmd_index_rebuild_fts)

    synonyms_parser = index_sub.add_parser("synonyms")
    syn_sub = synonyms_parser.add_subparsers(dest="syn_command")
    syn_import = syn_sub.add_parser("import")
    syn_import.add_argument("path")
    syn_import.set_defaults(func=cmd_synonyms_import)
    syn_export = syn_sub.add_parser("export")
    syn_export.add_argument("path")
    syn_export.set_defaults(func=cmd_synonyms_export)

    search_parser = subparsers.add_parser("search")
    search_sub = search_parser.add_subparsers(dest="search_command")
    search_va = search_sub.add_parser("va")
    search_va.add_argument("query")
    search_va.add_argument("--limit", type=int, default=20)
    search_va.add_argument("--require-level", action="append")
    search_va.add_argument("--period")
    search_va.add_argument("--must-variable-id", action="append", type=int)
    search_va.add_argument("--must-variable", action="append")
    search_va.add_argument("--must-class", action="append")
    search_va.add_argument("--must-category", action="append")
    search_va.add_argument("--min-municipalities", type=int)
    search_va.add_argument("--requires-national-munis", action="store_true")
    search_va.add_argument("--json", action="store_true")
    search_va.set_defaults(func=cmd_search_va)

    show_parser = subparsers.add_parser("show")
    show_sub = show_parser.add_subparsers(dest="show_command")
    show_table = show_sub.add_parser("table")
    show_table.add_argument("agregado_id", type=int)
    show_table.set_defaults(func=cmd_show_table)
    show_va_cmd = show_sub.add_parser("va")
    show_va_cmd.add_argument("va_id")
    show_va_cmd.set_defaults(func=cmd_show_va)

    link_parser = subparsers.add_parser("link")
    link_sub = link_parser.add_subparsers(dest="link_command")
    neighbors_cmd = link_sub.add_parser("neighbors")
    neighbors_cmd.add_argument("va_id")
    neighbors_cmd.add_argument("--top-k", type=int, default=50)
    neighbors_cmd.add_argument("--allow-unit-mismatch", action="store_true")
    neighbors_cmd.set_defaults(func=cmd_link_neighbors)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
