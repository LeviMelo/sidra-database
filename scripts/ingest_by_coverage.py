"""Convenience script to ingest agregados exposing UF (N3) or municipality (N6) coverage."""
from __future__ import annotations

import asyncio

from sidra_database import ingest_by_coverage

DEFAULT_LEVELS = ("N3", "N6")


async def main() -> None:
    report = await ingest_by_coverage(require_any_levels=DEFAULT_LEVELS, concurrency=8)
    discovered = len(report.discovered_ids)
    ingested = len(report.ingested_ids)
    skipped = len(report.skipped_existing)
    print(
        "Discovered {0} agregados matching UF/municipality coverage; "
        "ingested {1}, skipped {2} already present".format(discovered, ingested, skipped)
    )
    if report.failed:
        print(f"Failed to ingest {len(report.failed)} agregados:")
        for agregado_id, message in report.failed[:10]:
            print(f"   {agregado_id}: {message[:180]}")
        if len(report.failed) > 10:
            print("   ...")


if __name__ == "__main__":
    asyncio.run(main())
