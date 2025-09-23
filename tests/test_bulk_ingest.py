from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sidra_database import CatalogEntry, ingest_by_coverage, get_settings
from sidra_database.discovery import filter_catalog_entries
from sidra_database.db import ensure_schema, sqlite_session


@pytest.fixture(autouse=True)
def override_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "sidra-bulk-test.db"
    monkeypatch.setenv("SIDRA_DATABASE_PATH", str(db_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _entry(table_id: int, levels: dict[str, list[str]], *, subject: str | None = None, survey: str | None = None) -> CatalogEntry:
    return CatalogEntry(
        id=table_id,
        nome=f"Tabela {table_id}",
        pesquisa=survey,
        pesquisa_id=None,
        assunto=subject,
        assunto_id=None,
        periodicidade=None,
        nivel_territorial=levels,
    )


def test_filter_catalog_entries_by_levels_and_terms():
    entries = [
        _entry(1, {"Administrativo": ["N3"]}, subject="População"),
        _entry(2, {"Administrativo": ["N3", "N6"]}, subject="Agricultura", survey="Censo"),
        _entry(3, {"Administrativo": ["N2"]}, subject="População"),
    ]

    with_n6 = filter_catalog_entries(entries, require_any_levels=["n6"])
    assert [entry.id for entry in with_n6] == [2]

    with_both = filter_catalog_entries(entries, require_all_levels=["N3", "N6"])
    assert [entry.id for entry in with_both] == [2]

    subject_filtered = filter_catalog_entries(entries, subject_contains="popula")
    assert [entry.id for entry in subject_filtered] == [1, 3]

    survey_filtered = filter_catalog_entries(entries, survey_contains="censo")
    assert [entry.id for entry in survey_filtered] == [2]


def test_ingest_by_coverage_skips_existing_and_records_failures(monkeypatch: pytest.MonkeyPatch):
    entries = [
        _entry(10, {"Administrativo": ["N3", "N6"]}),
        _entry(20, {"Administrativo": ["N3"]}),
        _entry(30, {"Administrativo": ["N6"]}),
    ]

    async def fake_discover(**_kwargs):
        return entries

    recorded: list[int] = []

    async def fake_ingest(agregado_id: int, *, client=None, embedding_client=None, **_kwargs):
        if agregado_id == 30:
            raise RuntimeError("boom")
        recorded.append(agregado_id)
        with sqlite_session() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agregados (
                    id, nome, pesquisa, assunto, url, freq, periodo_inicio, periodo_fim, raw_json, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agregado_id,
                    f"Table {agregado_id}",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    b"{}",
                    "2024-01-01T00:00:00Z",
                ),
            )
            conn.commit()

    monkeypatch.setattr(
        "sidra_database.bulk_ingest.discover_agregados_by_coverage",
        fake_discover,
    )
    monkeypatch.setattr("sidra_database.bulk_ingest.ingest_agregado", fake_ingest)

    ensure_schema()
    with sqlite_session() as conn:
        conn.execute(
            "INSERT INTO agregados (id, nome, pesquisa, assunto, url, freq, periodo_inicio, periodo_fim, raw_json, fetched_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                10,
                "Existing",
                None,
                None,
                None,
                None,
                None,
                None,
                b"{}",
                "2024-01-01T00:00:00Z",
            ),
        )
        conn.commit()

    report = asyncio.run(
        ingest_by_coverage(
            require_any_levels=["N3", "N6"],
            skip_existing=True,
            concurrency=2,
        )
    )

    assert report.discovered_ids == [10, 20, 30]
    assert report.skipped_existing == [10]
    assert sorted(report.scheduled_ids) == [20, 30]
    assert report.ingested_ids == [20]
    assert report.failed == [(30, "boom")]
    assert sorted(recorded) == [20]
