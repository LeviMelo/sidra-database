"""Command-line interface for sidra-database."""
from __future__ import annotations

import argparse
import asyncio
from typing import Sequence

from .config import get_settings
from .embedding import EmbeddingClient
from .ingest import ingest_agregado
from .bulk_ingest import ingest_by_coverage
from .search import semantic_search_with_metadata
from .catalog import list_agregados


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

    search_parser = subparsers.add_parser("search", help="Run semantic search over stored embeddings")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument(
        "--types",
        nargs="*",
        choices=["agregado", "variable", "classification", "category"],
        help="Optional entity types to filter",
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


async def _run_ids(agregado_ids: Sequence[int], concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(agregado_id: int) -> None:
        async with semaphore:
            await ingest_agregado(agregado_id)

    await asyncio.gather(*(worker(agregado_id) for agregado_id in agregado_ids))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    print(f"Using database at {settings.database_path}")

    if args.command == "ingest":
        asyncio.run(_run_ids(args.agregado_ids, args.concurrent))
        return

    if args.command == "ingest-coverage":
        any_levels = args.any_levels or ["N3", "N6"]
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

    if args.command == "search":
        client = EmbeddingClient(model=args.model) if args.model else EmbeddingClient()
        results = semantic_search_with_metadata(
            args.query,
            entity_types=args.types,
            limit=args.limit,
            embedding_client=client,
            model=args.model,
        )
        if not results:
            print("No results found.")
            return
        for index, item in enumerate(results, start=1):
            score = f"{item.score:.3f}" if item.score else "0.000"
            header = f"{index}. [{item.entity_type}] score={score} table={item.agregado_id}"
            print(header)
            print(f"   {item.title}")
            if item.description:
                print(f"   {item.description}")
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
