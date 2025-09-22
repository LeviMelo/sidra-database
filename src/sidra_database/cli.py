"""Command-line interface for sidra-database."""
from __future__ import annotations

import argparse
import asyncio
from typing import Sequence

from .config import get_settings
from .embedding import EmbeddingClient
from .ingest import ingest_agregado
from .search import semantic_search_with_metadata


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

    parser.error("Unknown command")


if __name__ == "__main__":
    main()
