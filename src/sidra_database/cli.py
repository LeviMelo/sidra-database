"""Command-line interface for sidra-database."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Mapping, Sequence

from .config import get_settings
from .embedding import EmbeddingClient
from .ingest import ingest_agregado, generate_embeddings_for_agregado
from .bulk_ingest import ingest_by_coverage
from .search import hybrid_search, SearchFilters
from .catalog import list_agregados
from .db import sqlite_session, ensure_schema


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SIDRA metadata management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest one or more agregados")
    ingest_parser.add_argument(
        "agregado_ids",
        metavar="AGREGADO",
        type=int,
        nargs="+",
        help="One or more agregados IDs to ingest",
    )
    ingest_parser.add_argument(
        "--concurrent",
        type=int,
        default=4,
        help="Number of concurrent ingestion tasks (default: 4)",
    )
    ingest_parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation during ingestion",
    )

    embed_parser = subparsers.add_parser(
        "embed",
        help="Regenerate embeddings for stored agregados",
    )
    embed_parser.add_argument(
        "agregado_ids",
        metavar="AGREGADO",
        type=int,
        nargs="*",
        help="Optional list of agregados IDs to embed (default: all stored)",
    )
    embed_parser.add_argument(
        "--concurrent",
        type=int,
        default=6,
        help="Number of concurrent embedding tasks (default: 6)",
    )
    embed_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override embedding model identifier for LM Studio",
    )

    bulk_parser = subparsers.add_parser(
        "ingest-coverage",
        help="Discover agregados by territorial coverage and ingest them",
    )
    bulk_parser.add_argument(
        "--any-level",
        dest="any_levels",
        metavar="LEVEL",
        nargs="+",
        default=None,
        help="Require at least one of these territorial level codes (default: N3 N6)",
    )
    bulk_parser.add_argument(
        "--all-level",
        dest="all_levels",
        metavar="LEVEL",
        nargs="+",
        default=None,
        help="Require all of these territorial level codes",
    )
    bulk_parser.add_argument(
        "--exclude-level",
        dest="exclude_levels",
        metavar="LEVEL",
        nargs="+",
        default=None,
        help="Exclude agregados containing any of these level codes",
    )
    bulk_parser.add_argument(
        "--subject-contains",
        dest="subject_contains",
        default=None,
        help="Case-insensitive substring filter for the subject description",
    )
    bulk_parser.add_argument(
        "--survey-contains",
        dest="survey_contains",
        default=None,
        help="Case-insensitive substring filter for the survey description",
    )
    bulk_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of discovered agregados to ingest",
    )
    bulk_parser.add_argument(
        "--concurrent",
        type=int,
        default=8,
        help="Number of concurrent ingestion tasks (default: 8)",
    )
    bulk_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover matching agregados without ingesting them",
    )
    bulk_parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Reingest agregados already present in the database",
    )
    bulk_parser.set_defaults(skip_existing=True)
    bulk_parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation during ingestion",
    )

    search_parser = subparsers.add_parser("search", help="Run semantic search over stored embeddings")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument(
        "--types",
        nargs="*",
        choices=["agregado", "variable", "classification", "category"],
        help="Child entity types that may contribute lexical evidence",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to display (default: 5)",
    )
    search_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override embedding model identifier",
    )
    search_parser.add_argument(
        "--expand",
        type=int,
        default=6,
        help="Maximum number of child matches to show per table (default: 6)",
    )
    search_parser.add_argument(
        "--flat",
        action="store_true",
        help="Print child matches as a flat list instead of per table",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Return results as JSON",
    )
    search_parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Override scoring weights, e.g. sem=0.6,lex_table=0.2,lex_children=0.2",
    )
    search_parser.add_argument(
        "--min-municipalities",
        type=int,
        default=None,
        help="Only return agregados covering at least this many municipalities",
    )
    search_parser.add_argument(
        "--requires-national-munis",
        action="store_true",
        help="Only return agregados flagged with national municipal coverage",
    )
    search_parser.add_argument(
        "--subject-contains",
        type=str,
        default=None,
        help="Filter agregados whose subject contains this text",
    )
    search_parser.add_argument(
        "--survey-contains",
        type=str,
        default=None,
        help="Filter agregados whose survey contains this text",
    )
    search_parser.add_argument(
        "--period-start",
        type=int,
        default=None,
        help="Require tables with period end >= this year",
    )
    search_parser.add_argument(
        "--period-end",
        type=int,
        default=None,
        help="Require tables with period start <= this year",
    )
    search_parser.add_argument(
        "--has-variable",
        dest="has_variables",
        type=int,
        action="append",
        default=None,
        help="Only return tables containing this variable ID (repeatable)",
    )
    search_parser.add_argument(
        "--has-classification",
        dest="has_classifications",
        type=int,
        action="append",
        default=None,
        help="Only return tables containing this classification ID (repeatable)",
    )
    search_parser.add_argument(
        "--has-category",
        dest="has_categories",
        type=str,
        action="append",
        default=None,
        help="Only return tables containing classification:category pairs",
    )

    list_parser = subparsers.add_parser("list", help="List stored agregados with optional coverage filters")
    list_parser.add_argument(
        "--requires-national-munis",
        action="store_true",
        help="Only show agregados flagged with national municipal coverage",
    )
    list_parser.add_argument(
        "--min-municipalities",
        type=int,
        default=None,
        help="Only show agregados covering at least this many municipalities",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of rows to display (default: 20)",
    )
    list_parser.add_argument(
        "--order-by",
        choices=["municipalities", "id", "name", "fetched"],
        default="municipalities",
        help="Sort order for results (default: municipalities)",
    )

    return parser


async def _run_ids(
    agregado_ids: Sequence[int],
    concurrency: int,
    *,
    generate_embeddings: bool,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(agregado_id: int) -> None:
        async with semaphore:
            await ingest_agregado(agregado_id, generate_embeddings=generate_embeddings)

    await asyncio.gather(*(worker(agregado_id) for agregado_id in agregado_ids))


async def _embed_ids(
    agregado_ids: Sequence[int],
    concurrency: int,
    *,
    model: str | None = None,
) -> None:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    semaphore = asyncio.Semaphore(concurrency)
    progress_lock = asyncio.Lock()
    db_lock = asyncio.Lock()

    embed_client = EmbeddingClient(model=model) if model else EmbeddingClient()

    total = len(agregado_ids)
    completed = 0
    failed: list[tuple[int, str]] = []
    progress_step = max(1, total // 100) if total else 1
    last_emit = time.monotonic()
    emit_interval = 15.0

    async def worker(agregado_id: int) -> None:
        nonlocal completed, last_emit
        async with semaphore:
            try:
                await generate_embeddings_for_agregado(
                    agregado_id,
                    embedding_client=embed_client,
                    db_lock=db_lock,
                )
            except Exception as exc:  # noqa: BLE001
                async with progress_lock:
                    failed.append((agregado_id, str(exc)))
            finally:
                async with progress_lock:
                    completed += 1
                    now = time.monotonic()
                    should_emit = (
                        completed <= 10
                        or completed == total
                        or completed % progress_step == 0
                        or (now - last_emit) >= emit_interval
                    )
                    if should_emit:
                        last_emit = now
                        print(
                            f"Embedding progress {completed}/{total}: failed={len(failed)}",
                            flush=True,
                        )

    await asyncio.gather(*(worker(agregado_id) for agregado_id in agregado_ids))

    if failed:
        print("Embedding failures:")
        for agregado_id, message in failed[:10]:
            print(f"   {agregado_id}: {message[:180]}")
        if len(failed) > 10:
            print("   ...")


def _parse_weight_overrides(spec: str | None) -> Mapping[str, float] | None:
    if spec is None:
        return None
    overrides: dict[str, float] = {}
    for part in spec.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"Invalid weight override '{chunk}'. Expected format key=value, e.g. sem=0.6."
            )
        key, value = chunk.split("=", 1)
        key = key.strip()
        try:
            overrides[key] = float(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric weight for '{key}': {value.strip()}"
            ) from exc
    return overrides


def _parse_category_filters(raw: Sequence[str] | None) -> list[tuple[int, int]]:
    if not raw:
        return []
    parsed: list[tuple[int, int]] = []
    for value in raw:
        chunk = value.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Invalid category filter '{value}'. Expected CLASSIFICATION:CATEGORY format."
            )
        left, right = chunk.split(":", 1)
        try:
            parsed.append((int(left), int(right)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid category filter '{value}'. Both parts must be integers."
            ) from exc
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    print(f"Using database at {settings.database_path}")

    if args.command == "ingest":
        if args.skip_embeddings:
            print("Embedding generation skipped; metadata only.")
        asyncio.run(
            _run_ids(
                args.agregado_ids,
                args.concurrent,
                generate_embeddings=not args.skip_embeddings,
            )
        )
        return

    if args.command == "ingest-coverage":
        any_levels = args.any_levels or ["N3", "N6"]

        if args.skip_embeddings:
            print("Embedding generation skipped; metadata only.")

        def _progress(message: str) -> None:
            print(message, flush=True)

        report = asyncio.run(
            ingest_by_coverage(
                require_any_levels=any_levels,
                require_all_levels=args.all_levels,
                exclude_levels=args.exclude_levels,
                subject_contains=args.subject_contains,
                survey_contains=args.survey_contains,
                limit=args.limit,
                concurrency=args.concurrent,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                generate_embeddings=not args.skip_embeddings,
                progress_callback=_progress,
            )
        )
        discovered = len(report.discovered_ids)
        print(f"Discovered {discovered} agregados matching the provided filters.")
        if report.discovered_ids:
            preview = ", ".join(str(tid) for tid in report.discovered_ids[:10])
            suffix = " ..." if discovered > 10 else ""
            print(f"   IDs: {preview}{suffix}")
        if report.skipped_existing:
            print(
                f"Skipped {len(report.skipped_existing)} already ingested agregados"
            )
        if args.dry_run:
            scheduled = len(report.scheduled_ids)
            print(f"Dry run complete: {scheduled} agregados would be ingested.")
            return
        ingested = len(report.ingested_ids)
        print(f"Ingested {ingested} agregados.")
        if report.failed:
            print(f"Failed to ingest {len(report.failed)} agregados:")
            for agregado_id, message in report.failed[:10]:
                print(f"   {agregado_id}: {message[:180]}")
            if len(report.failed) > 10:
                print("   ...")
        return

    if args.command == "embed":
        ensure_schema()
        if args.agregado_ids:
            ids = list(dict.fromkeys(args.agregado_ids))
        else:
            with sqlite_session() as conn:
                rows = conn.execute("SELECT id FROM agregados ORDER BY id").fetchall()
                ids = [int(row["id"]) for row in rows]

        if not ids:
            print("No agregados available for embedding.")
            return

        print(f"Embedding {len(ids)} agregados with concurrency={args.concurrent}")
        asyncio.run(
            _embed_ids(
                ids,
                args.concurrent,
                model=args.model,
            )
        )
        return

    if args.command == "search":
        try:
            weight_overrides = _parse_weight_overrides(args.weights)
        except ValueError as exc:
            parser.error(str(exc))

        try:
            category_filters = _parse_category_filters(args.has_categories)
        except ValueError as exc:
            parser.error(str(exc))

        has_variables = args.has_variables or None
        has_classifications = args.has_classifications or None
        has_categories = category_filters or None

        filters = SearchFilters(
            min_municipalities=args.min_municipalities,
            requires_national_munis=args.requires_national_munis,
            subject_contains=args.subject_contains,
            survey_contains=args.survey_contains,
            period_start=args.period_start,
            period_end=args.period_end,
            has_variables=has_variables,
            has_classifications=has_classifications,
            has_categories=has_categories,
        )

        allowed_children = ["variable", "classification", "category"]
        if args.types:
            requested = {item.lower() for item in args.types}
            child_types = [child for child in allowed_children if child in requested]
        else:
            child_types = allowed_children

        expand = max(0, args.expand)

        client = EmbeddingClient(model=args.model) if args.model else EmbeddingClient()
        results = hybrid_search(
            args.query,
            limit=args.limit,
            filters=filters,
            embedding_client=client,
            model=args.model,
            child_types=child_types,
            max_child_matches=expand,
            weights=weight_overrides,
        )
        if not results:
            print("No results found.")
            return
        if args.json:
            payload = []
            for item in results:
                metadata = dict(item.metadata)
                children: list[dict[str, object]] = []
                for child in item.child_matches:
                    child_meta = dict(child.metadata)
                    children.append(
                        {
                            "type": child.entity_type,
                            "id": child.entity_id,
                            "title": child.title,
                            "lexical_score": child.lexical_score,
                            "metadata": child_meta,
                        }
                    )
                payload.append(
                    {
                        "table_id": item.agregado_id,
                        "title": item.title,
                        "description": item.description,
                        "scores": {
                            "combined": item.combined_score,
                            "semantic": item.score,
                            "lexical_table": item.lexical_table_score,
                            "lexical_children": item.lexical_children_score,
                        },
                        "metadata": metadata,
                        "children": children,
                    }
                )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return

        if args.flat:
            for item in results:
                base = f"{item.agregado_id}: {item.title}"
                if not item.child_matches:
                    print(f"{base} [score={item.combined_score:.3f}]")
                    continue
                for child in item.child_matches:
                    child_meta = dict(child.metadata)
                    details: list[str] = []
                    unit = child_meta.get("unit")
                    if unit:
                        details.append(f"unit={unit}")
                    classification_id = child_meta.get("classification_id")
                    if classification_id:
                        details.append(f"classification={classification_id}")
                    detail_text = f" ({', '.join(details)})" if details else ""
                    print(
                        f"{item.agregado_id}: [{child.entity_type}] {child.title}{detail_text} "
                        f"lex={child.lexical_score:.3f}"
                    )
            return

        for index, item in enumerate(results, start=1):
            metadata = dict(item.metadata)
            score_line = (
                f"{index}. table={item.agregado_id} "
                f"score={item.combined_score:.3f} "
                f"(sem={item.score:.3f} | lex_table={item.lexical_table_score:.3f} | "
                f"lex_children={item.lexical_children_score:.3f})"
            )
            print(score_line)
            print(f"   {item.title}")
            if item.description:
                print(f"   {item.description}")

            subject = metadata.get("subject")
            survey = metadata.get("survey")
            period_start = metadata.get("period_start")
            period_end = metadata.get("period_end")
            muni_count = metadata.get("municipality_locality_count")
            covers_national = metadata.get("covers_national_municipalities")
            url = metadata.get("url")
            levels_sample = metadata.get("levels_sample")
            levels_count = metadata.get("levels_count")

            info_parts: list[str] = []
            if survey:
                info_parts.append(f"Survey: {survey}")
            if subject:
                info_parts.append(f"Subject: {subject}")
            if period_start or period_end:
                if period_start and period_end and period_start != period_end:
                    info_parts.append(f"Period {period_start}-{period_end}")
                else:
                    info_parts.append(f"Period {period_start or period_end}")
            if muni_count:
                try:
                    formatted = f"{int(muni_count):,}"
                except ValueError:
                    formatted = muni_count
                coverage = f"Municipalities: {formatted}"
                if covers_national in {"1", "True", "true", "yes"}:
                    coverage += " (national)"
                info_parts.append(coverage)
            if levels_sample:
                suffix = f" (total {levels_count})" if levels_count else ""
                info_parts.append(f"Levels: {levels_sample}{suffix}")
            if url:
                info_parts.append(f"URL: {url}")
            if info_parts:
                print(f"   {' | '.join(info_parts)}")

            if item.child_matches:
                print("   Why it matched:")
                for child in item.child_matches:
                    child_meta = dict(child.metadata)
                    details: list[str] = []
                    unit = child_meta.get("unit")
                    if unit:
                        details.append(f"unit={unit}")
                    classification_id = child_meta.get("classification_id")
                    if classification_id:
                        details.append(f"classification={classification_id}")
                    detail_text = f" ({', '.join(details)})" if details else ""
                    print(
                        f"      - [{child.entity_type}] {child.title}{detail_text} "
                        f"lex={child.lexical_score:.3f}"
                    )
            print()
        return

    if args.command == "list":
        rows = list_agregados(
            requires_national_munis=args.requires_national_munis,
            min_municipality_count=args.min_municipalities,
            limit=args.limit,
            order_by=args.order_by,
        )
        if not rows:
            print("No agregados matched the provided filters.")
            return
        for row in rows:
            coverage = f"municipalities={row.municipality_locality_count:,}"
            if row.covers_national_municipalities:
                coverage += " (national)"
            subject = f"assunto={row.assunto}" if row.assunto else "assunto=?"
            survey = f"pesquisa={row.pesquisa}" if row.pesquisa else "pesquisa=?"
            print(f"{row.id}: {row.nome}")
            print(f"   {subject} | {survey} | {coverage} | fetched={row.fetched_at}")
        return

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
