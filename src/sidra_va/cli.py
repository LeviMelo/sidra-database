from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from dataclasses import asdict
from typing import Iterable, Sequence

from .api_client import SidraApiClient
from .bulk_ingest import ingest_by_coverage
from .catalog import list_agregados
from .db import create_connection, ensure_full_schema
from .diagnostics_base import (
    api_vs_db_spot_check,
    global_health_report,
    repair_missing_variables,
)
from .embed import embed_vas_for_agregados
from .embedding_client import EmbeddingClient
from .ingest_base import ingest_agregado
from .neighbors import find_neighbors_for_va
from .schema_migrations import get_schema_version
from .search_va import VaResult, VaSearchFilters, search_value_atoms
from .synonyms import export_synonyms_csv, import_synonyms_csv
from .value_index import build_va_index_for_agregado, build_va_index_for_all


def _ensure_va_schema() -> None:
    ensure_full_schema()


def _base_counts(conn) -> tuple[int, int]:
    try:
        total = conn.execute("SELECT COUNT(*) FROM agregados").fetchone()[0]
    except sqlite3.OperationalError:
        return 0, 0
    try:
        with_vars = conn.execute(
            "SELECT COUNT(DISTINCT agregado_id) FROM variables"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        with_vars = 0
    return int(total or 0), int(with_vars or 0)


def _require_base_metadata(
    conn,
    ids: Iterable[int] | None = None,
) -> tuple[bool, list[int]]:
    for table in ("agregados", "variables"):
        try:
            conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
        except sqlite3.OperationalError:
            print(
                f"Base table '{table}' is missing. Run 'sidra_va.cli ingest' before using sidra-va.",
            )
            return False, []

    total, with_vars = _base_counts(conn)
    if total == 0:
        print("Database contains no agregados. Run 'sidra_va.cli ingest' first.")
        return False, []
    if with_vars == 0:
        print(
            "No agregados with variables found. Re-ingest metadata or run 'sidra_va.cli repair-missing'."
        )
        return False, []

    zero_ids: list[int] = []
    if ids:
        for raw_id in ids:
            agregado_id = int(raw_id)
            count = conn.execute(
                "SELECT COUNT(*) FROM variables WHERE agregado_id = ?",
                (agregado_id,),
            ).fetchone()[0]
            if int(count or 0) == 0:
                zero_ids.append(agregado_id)
    return True, zero_ids


def _count_rows(conn, table: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0] or 0)


async def _ingest_ids(
    agregado_ids: Sequence[int],
    *,
    concurrency: int,
    generate_embeddings: bool,
) -> tuple[int, list[tuple[int, str]]]:
    """Ingest the provided agregados concurrently."""

    if not agregado_ids:
        return 0, []

    concurrency = max(1, concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    db_lock = asyncio.Lock() if concurrency > 1 else None
    failures: list[tuple[int, str]] = []
    succeeded = 0

    async with SidraApiClient() as client:
        embedding_client = EmbeddingClient() if generate_embeddings else None

        async def worker(agregado_id: int) -> None:
            nonlocal succeeded
            async with semaphore:
                try:
                    await ingest_agregado(
                        agregado_id,
                        client=client,
                        embedding_client=embedding_client,
                        generate_embeddings=generate_embeddings,
                        db_lock=db_lock,
                    )
                except Exception as exc:  # noqa: BLE001
                    message = str(exc)[:200]
                    failures.append((agregado_id, message))
                    print(f"Failed agregado {agregado_id}: {message}")
                else:
                    succeeded += 1

        await asyncio.gather(*(worker(agregado_id) for agregado_id in agregado_ids))

    return succeeded, failures


def cmd_ingest(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    agregado_ids = [int(value) for value in args.agregado_ids]
    succeeded, failures = asyncio.run(
        _ingest_ids(
            agregado_ids,
            concurrency=args.concurrent,
            generate_embeddings=not args.skip_embeddings,
        )
    )
    total = len(agregado_ids)
    print(f"Ingested {succeeded}/{total} agregados")
    if failures:
        print("Failures:")
        for agregado_id, message in failures[:10]:
            print(f"  {agregado_id}: {message}")
        remaining = len(failures) - 10
        if remaining > 0:
            print(f"  ... ({remaining} more)")


def cmd_ingest_coverage(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    report = asyncio.run(
        ingest_by_coverage(
            require_any_levels=args.any_levels,
            require_all_levels=args.all_levels,
            exclude_levels=args.exclude_levels,
            subject_contains=args.subject_contains,
            survey_contains=args.survey_contains,
            limit=args.limit,
            concurrency=max(1, args.concurrent),
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            generate_embeddings=not args.skip_embeddings,
        )
    )
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))


def cmd_repair_missing(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    result = asyncio.run(
        repair_missing_variables(
            chunk_size=max(1, args.chunk),
            concurrency=max(1, args.concurrent),
            limit=args.limit,
            max_retries=max(1, args.retries),
        )
    )
    payload = asdict(result)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_diagnostics_health(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    conn = create_connection()
    try:
        report = global_health_report(conn, sample_limit=max(0, args.health_sample))
    finally:
        conn.close()
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_diagnostics_spot_check(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    conn = create_connection()
    try:
        report = asyncio.run(
            api_vs_db_spot_check(
                conn,
                sample_size=max(0, args.spot_sample),
            )
        )
    finally:
        conn.close()
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_list_agregados(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    rows = list_agregados(
        requires_national_munis=args.requires_national_munis,
        min_municipality_count=args.min_municipalities,
        limit=args.limit,
        order_by=args.order,
    )
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False))
        return
    for row in rows:
        coverage = f"municipalities={row.municipality_locality_count}"
        national = "national" if row.covers_national_municipalities else "partial"
        print(
            f"{row.id}: {row.nome} | assunto={row.assunto} | pesquisa={row.pesquisa} | "
            f"{coverage} ({national})"
        )


def cmd_db_migrate(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    conn = create_connection()
    try:
        version = get_schema_version(conn)
    finally:
        conn.close()
    print(f"VA schema version: {version}")


def cmd_db_stats(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    conn = create_connection()
    try:
        base_total, base_with_vars = _base_counts(conn)
        tables = [
            "value_atoms",
            "value_atom_dims",
            "value_atoms_fts",
            "variable_fingerprints",
            "synonyms",
        ]
        stats = {}
        for table in tables:
            stats[table] = _count_rows(conn, table)
    finally:
        conn.close()
    print(f"agregados_with_variables: {base_with_vars}/{base_total}")
    for table, count in stats.items():
        print(f"{table}: {count}")


def cmd_index_build(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    conn = create_connection()
    try:
        target_ids = args.ids if not args.all else None
        ready, missing_ids = _require_base_metadata(conn, target_ids)
    finally:
        conn.close()

    if not ready:
        return

    if args.all:
        result = asyncio.run(
            build_va_index_for_all(
                concurrency=args.concurrent,
                allow_two_dim_combos=args.allow_two_dim_combos,
            )
        )
        if not result:
            print(
                "No agregados with variables available. Ingest metadata before building the VA index."
            )
        else:
            for ag, count in sorted(result.items()):
                print(f"agregado {ag}: {count} VAs")
        return

    if not args.ids:
        raise SystemExit("Provide --ids or --all")

    ready_ids = [ag for ag in args.ids if ag not in missing_ids]
    for missing in missing_ids:
        print(f"No variables for table {missing}; run 'sidra_va.cli ingest {missing}' first.")
    if not ready_ids:
        return

    for ag_id in ready_ids:
        count = asyncio.run(
            build_va_index_for_agregado(
                ag_id,
                allow_two_dim_combos=args.allow_two_dim_combos,
            )
        )
        print(f"agregado {ag_id}: {count} VAs")


def cmd_index_embed(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    requested_ids = args.ids if not args.all else None
    conn = create_connection()
    try:
        ready, missing_ids = _require_base_metadata(conn, requested_ids)
    finally:
        conn.close()

    if not ready:
        return

    if requested_ids is None and not args.all:
        raise SystemExit("Provide --ids or --all")

    if requested_ids:
        valid_ids = [ag for ag in requested_ids if ag not in missing_ids]
        for missing in missing_ids:
            print(f"No variables for table {missing}; run 'sidra_va.cli ingest {missing}' first.")
    else:
        valid_ids = None

    if valid_ids is not None and not valid_ids:
        return

    stats = asyncio.run(
        embed_vas_for_agregados(
            valid_ids,
            concurrency=args.concurrent,
            model=args.model,
        )
    )
    if not any(stats.values()):
        print("No VAs to embed. Build VAs first.")
    else:
        print(json.dumps(stats, indent=2))


def cmd_index_rebuild_fts(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    conn = create_connection()
    try:
        ready, _ = _require_base_metadata(conn)
        if not ready:
            return
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
    _ensure_va_schema()
    conn = create_connection()
    try:
        ready, _ = _require_base_metadata(conn)
        if not ready:
            return
        count = import_synonyms_csv(args.path, conn)
    finally:
        conn.close()
    print(f"Imported {count} synonym rows")


def cmd_synonyms_export(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    conn = create_connection()
    try:
        ready, _ = _require_base_metadata(conn)
        if not ready:
            return
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
    _ensure_va_schema()
    conn = create_connection()
    try:
        ready, _ = _require_base_metadata(conn)
    finally:
        conn.close()
    if not ready:
        return
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
    _ensure_va_schema()
    conn = create_connection()
    try:
        ready, missing_ids = _require_base_metadata(conn, [args.agregado_id])
        if not ready:
            return
        if missing_ids:
            print(
                f"No variables for table {args.agregado_id}; run 'sidra_va.cli ingest {args.agregado_id}' first."
            )
            return
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
    _ensure_va_schema()
    conn = create_connection()
    try:
        ready, _ = _require_base_metadata(conn)
        if not ready:
            return
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
    _ensure_va_schema()
    conn = create_connection()
    try:
        ready, _ = _require_base_metadata(conn)
    finally:
        conn.close()
    if not ready:
        return
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


def cmd_diagnostics(args: argparse.Namespace) -> None:
    _ensure_va_schema()
    conn = create_connection()
    try:
        ready, _ = _require_base_metadata(conn)
        base_total, base_with_vars = _base_counts(conn)
        stats = {
            "value_atoms": _count_rows(conn, "value_atoms"),
            "value_atom_dims": _count_rows(conn, "value_atom_dims"),
            "value_atoms_fts": _count_rows(conn, "value_atoms_fts"),
            "variable_fingerprints": _count_rows(conn, "variable_fingerprints"),
            "synonyms": _count_rows(conn, "synonyms"),
        }
        sample: list[dict[str, object]] = []
        sample_limit = min(max(args.sample, 0), stats["value_atoms"])
        if ready and sample_limit > 0:
            cursor = conn.execute(
                "SELECT va_id FROM value_atoms ORDER BY RANDOM() LIMIT ?",
                (sample_limit,),
            )
            for row in cursor.fetchall():
                dims_count = conn.execute(
                    "SELECT COUNT(*) FROM value_atom_dims WHERE va_id = ?",
                    (row["va_id"],),
                ).fetchone()[0]
                sample.append({"va_id": row["va_id"], "has_dims": bool(dims_count)})
    finally:
        conn.close()

    diagnostics = {
        "base": {
            "ready": ready,
            "agregados": base_total,
            "agregados_with_variables": base_with_vars,
        },
        "tables": stats,
        "fts_consistent": stats["value_atoms"] == stats["value_atoms_fts"],
        "sample": sample,
    }

    smoke_query = args.smoke_query
    smoke_results: list[str] = []
    smoke_ok = False
    if ready and stats["value_atoms"] > 0 and smoke_query:
        results = asyncio.run(
            search_value_atoms(
                smoke_query,
                filters=VaSearchFilters(),
                limit=max(1, args.smoke_limit),
            )
        )
        smoke_results = [item.va_id for item in results]
        smoke_ok = bool(results)

    diagnostics["smoke_test"] = {
        "query": smoke_query,
        "ok": smoke_ok,
        "result_ids": smoke_results,
    }

    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sidra-va")
    subparsers = parser.add_subparsers(dest="command")

    db_parser = subparsers.add_parser("db")
    db_sub = db_parser.add_subparsers(dest="db_command")

    migrate = db_sub.add_parser("migrate")
    migrate.set_defaults(func=cmd_db_migrate)

    stats = db_sub.add_parser("stats")
    stats.set_defaults(func=cmd_db_stats)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest one or more agregados")
    ingest_parser.add_argument("agregado_ids", metavar="AGREGADO", type=int, nargs="+")
    ingest_parser.add_argument("--concurrent", type=int, default=4)
    ingest_parser.add_argument("--skip-embeddings", action="store_true")
    ingest_parser.set_defaults(func=cmd_ingest)

    coverage_parser = subparsers.add_parser(
        "ingest-coverage",
        help="Discover agregados by territorial coverage and ingest them",
    )
    coverage_parser.add_argument("--any-level", dest="any_levels", nargs="+", default=None)
    coverage_parser.add_argument("--all-level", dest="all_levels", nargs="+", default=None)
    coverage_parser.add_argument("--exclude-level", dest="exclude_levels", nargs="+", default=None)
    coverage_parser.add_argument("--subject-contains", dest="subject_contains", default=None)
    coverage_parser.add_argument("--survey-contains", dest="survey_contains", default=None)
    coverage_parser.add_argument("--limit", type=int, default=None)
    coverage_parser.add_argument("--concurrent", type=int, default=8)
    coverage_parser.add_argument("--skip-embeddings", action="store_true")
    coverage_parser.add_argument("--dry-run", action="store_true")
    coverage_parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    coverage_parser.set_defaults(skip_existing=True)
    coverage_parser.set_defaults(func=cmd_ingest_coverage)

    repair_parser = subparsers.add_parser(
        "repair-missing",
        help="Re-ingest agregados that currently have zero variables",
    )
    repair_parser.add_argument("--chunk", type=int, default=50)
    repair_parser.add_argument("--concurrent", type=int, default=6)
    repair_parser.add_argument("--limit", type=int, default=None)
    repair_parser.add_argument("--retries", type=int, default=3)
    repair_parser.set_defaults(func=cmd_repair_missing)

    list_parser = subparsers.add_parser("list", help="List stored agregados")
    list_parser.add_argument("--requires-national-munis", action="store_true")
    list_parser.add_argument("--min-municipalities", type=int, default=None)
    list_parser.add_argument("--limit", type=int, default=None)
    list_parser.add_argument(
        "--order",
        choices=["municipalities", "id", "name", "fetched"],
        default="municipalities",
    )
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=cmd_list_agregados)

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

    diag_parser = subparsers.add_parser("diagnostics", help="Run diagnostics")
    diag_parser.add_argument("--sample", type=int, default=5)
    diag_parser.add_argument(
        "--smoke-query",
        default="população",
        help="Query used for the search smoke test (default: população)",
    )
    diag_parser.add_argument("--smoke-limit", type=int, default=5)
    diag_parser.set_defaults(func=cmd_diagnostics)
    diag_sub = diag_parser.add_subparsers(dest="diagnostics_command")

    diag_health = diag_sub.add_parser("health", help="Report base ingestion health")
    diag_health.add_argument("--sample", dest="health_sample", type=int, default=50)
    diag_health.set_defaults(func=cmd_diagnostics_health)

    diag_spot = diag_sub.add_parser(
        "spot-check",
        help="Sample agregados missing variables and compare with live API",
    )
    diag_spot.add_argument("--sample", dest="spot_sample", type=int, default=10)
    diag_spot.set_defaults(func=cmd_diagnostics_spot_check)

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
