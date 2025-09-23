# SIDRA Search Engine Migration Plan

## 1. API Understanding
- **Aggregados catalog**: `/api/v3/agregados` lists all table IDs grouped by survey; filters for period, subject, classification, periodicity, or geographic level.
- **Metadata**: `/api/v3/agregados/{agregado}/metadados` returns table name, survey, subject, periodicity, available geographic levels, variables (id, name, unit, summarization flags), and classifications with full category lists.
- **Temporal coverage**: `/api/v3/agregados/{agregado}/periodos` delivers all valid period literals per table.
- **Geographic coverage**: `/api/v3/agregados/{agregado}/localidades/{nivel}` enumerates locality ids and labels for each supported level (e.g. `N1`, `N2`, `N6`).
- **Data retrieval**: `/api/v3/agregados/{agregado}/periodos/{periodos}/variaveis/{variavel}` (or the `/{agregado}/variaveis/{variavel}` shortcut for last six periods) returns records in SIDRA format (`NC/NN`, `D#C/D#N`, `V`). Query params `localidades`, `classificacao`, `view` follow SIDRA syntax (`N6[all]`, `58[all]|2[4,5]`, `view=flat`).
- Responses are UTF-8 JSON and may arrive gzipped; all descriptive text needed for embeddings is present, so HTML scraping is unnecessary.

## 2. Project Goals
- Replace ad-hoc `.rds` metadata with a structured SQLite database capturing agregados, variables, classifications, categories, periods, and locality mappings.
- Build a Python ingestion pipeline that:
  1. Discovers target agregados (seeded from current demographic usage).
  2. Pulls metadata, periods, and locality catalogs via the v3 API and upserts into SQLite (including raw JSON for auditing).
  3. Generates canonical text strings and stores embeddings computed via LM Studio (`text-embedding-qwen3-embedding-0.6b@f16`).
  4. Exposes query utilities combining relational filters (variables, categories, geography, period coverage) with semantic search.
- Keep scope local-only (SQLite + optional FAISS); no external service deployment required for now.

## 3. Delivered
- Python project scaffold with `sidra_database` package.
- SQLite schema and ingestion pipeline.
- Async SIDRA API client with retry/backoff.
- Embedding client targeting LM Studio.
- CLI helper and development tooling (pytest, Ruff, MyPy).

## 4. Immediate Tasks
- ✅ Limit embeddings to agregados titles and skip variable/classification/category vectors during ingestion (completed September 2025).
- ✅ Extend hybrid search so variables, classifications, and categories surface via lexical enrichment (completed September 2025).
- ➡️ Add developer-facing docs/CLI examples that walk through the hybrid search workflow end-to-end.
- 🔄 Evaluate caching for large locality pulls and outline an incremental refresh strategy.

## 5. Validation & Next Steps
- Dashboard the search ranking (hit@k) for representative agregados/variable/classification/category queries.
- Cross-check counts against legacy `.rds` exports to ensure coverage.
- Extend ingestion to additional agregados as demographic requirements grow.
- Plan optional value retrieval layer if future workflows need direct data pulls.
