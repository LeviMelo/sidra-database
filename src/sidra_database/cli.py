"""Simple command-line interface for sidra-database."""
from __future__ import annotations

import argparse
import asyncio
from typing import Sequence

from .config import get_settings
from .ingest import ingest_agregado


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SIDRA metadata management")
    parser.add_argument(
        "agregado_ids",
        metavar="AGREGADO",
        type=int,
        nargs="+",
        help="One or more agregados IDs to ingest",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=4,
        help="Number of concurrent ingestion tasks (default: 4)",
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
    asyncio.run(_run_ids(args.agregado_ids, args.concurrent))


if __name__ == "__main__":
    main()
