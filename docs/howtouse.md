# How to Use `sidra-database`

> Local-first metadata warehouse and semantic explorer for IBGE SIDRA agregados.

This guide walks through installing the toolkit, configuring dependencies, ingesting SIDRA metadata at scale, enriching it with embeddings, and exploring the resulting SQLite catalog both from the command line and programmatically. The goal is to replace ad-hoc scraping with a reproducible, coverage-aware pipeline: you can ingest thousands of tables, flag the ones with near-national municipality coverage, and query them semantically or via structured filters.

## 1. Prerequisites

### 1.1 Runtime
- **Python** 3.10 or newer.
- **LM Studio** (or a compatible OpenAI-compatible embedding server) running locally with the model `text-embedding-qwen3-embedding-0.6b@f16`.
- Optional: SQLite browser (DB Browser for SQLite, `sqlite3`, Datasette, etc.) for manual inspection.

### 1.2 Repository Setup
- Clone the repository into a Windows-accessible path, e.g. `C:\Users\Galaxy\LEVI\projects\sidra-database`.
- Use a Python virtual environment (recommended):
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  ```

### 1.3 Installation
- Install the package and development extras:
  ```powershell
  pip install -e .[dev]
  ```
- Optional tools for linting/type-checking: `ruff`, `mypy` (already included in `[dev]`).

## 2. Configuration

`sidra-database` reads configuration from environment variables or a local `.env` file (copy `.env.example` in the project root to `.env`). When using environment variables directly on Windows, remember to restart the shell after calling `setx`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `SIDRA_DATABASE_PATH` | `sidra.db` (repo root) | SQLite file path for metadata store. Use an absolute path if you want the DB elsewhere. |
| `SIDRA_EMBEDDING_API_URL` | `http://127.0.0.1:1234/v1/embeddings` | HTTP endpoint for LM Studio (or other embedding service). |
| `SIDRA_EMBEDDING_MODEL` | `text-embedding-qwen3-embedding-0.6b@f16` | Model ID sent to the embedding API. |
| `SIDRA_REQUEST_TIMEOUT` | `30` seconds | HTTP timeout for SIDRA API calls and embedding requests. |
| `SIDRA_REQUEST_RETRIES` | `3` | Retry attempts for transient HTTP failures. |
| `SIDRA_MUNICIPALITY_NATIONAL_THRESHOLD` | `4000` | Minimum municipality count (N6 level) to label a table as "national municipal coverage". |

### 2.1 Using `sidra.config.json`
1. Copy the template: `copy sidra.config.json.example sidra.config.json`
2. Edit the JSON values (paths, endpoints, thresholds).

Example:
```json
{
  "database_path": "C:/data/sidra.db",
  "embedding_api_url": "http://127.0.0.1:1234/v1/embeddings",
  "embedding_model": "text-embedding-qwen3-embedding-0.6b@f16",
  "municipality_national_threshold": 4000
}
```

### 2.2 Using `setx` (optional)
If you prefer global environment variables:
```powershell
setx SIDRA_DATABASE_PATH C:\data\sidra.db
setx SIDRA_EMBEDDING_API_URL http://127.0.0.1:1234/v1/embeddings
setx SIDRA_MUNICIPALITY_NATIONAL_THRESHOLD 4000
```
Restart the shell so Python sees the new values.

## 3. LM Studio Setup
1. Launch LM Studio.
2. Download the embedding model `text-embedding-qwen3-embedding-0.6b@f16`.
3. Start the local server (`File > Start Server`). Note the port (default `1234`).
4. Test with curl or PowerShell:
   ```powershell
   Invoke-WebRequest -Method Post -Uri http://127.0.0.1:1234/v1/embeddings -Body '{"model":"text-embedding-qwen3-embedding-0.6b@f16","input":"hello"}' -ContentType 'application/json'
   ```
   You should receive a JSON response containing an `embedding` array.

## 4. Understanding the Data Model

After ingestion, the SQLite database contains normalized tables:

- `agregados`: core table metadata (names, `assunto`, `pesquisa`, periods, URL), raw JSON payload, timestamp, plus derived coverage columns (`municipality_locality_count`, `covers_national_municipalities`).
- `agregados_levels`: territorial levels per table (e.g., `N1`, `N3`, `N6`) with counts of localities.
- `variables`, `classifications`, `categories`: detailed metadata for each variable/classification/category in the table.
- `periods`: available time periods.
- `localities`: locality IDs/names per level.
- `embeddings`: float vectors stored as blobs for agregados, variables, classifications, and categories.
- `ingestion_log`: audit trail of ingestion runs.

This structure lets you combine SQL filters (geography coverage, subject, survey, period windows) with semantic search results.

## 5. Ingesting SIDRA Tables

### 5.1 Basic CLI Ingestion
Ingest one or more known table IDs:
```powershell
python -m sidra_database.cli ingest 6579 2093
```
- Uses `asyncio` to fetch metadata, periods, localities, and calculate embeddings.
- Respects `SIDRA_DATABASE_PATH` and coverage threshold settings.
- `--concurrent N`: adjust concurrent requests (default 4).

### 5.2 Bulk Ingestion with Coverage Criteria
To ingest every table that exposes UF (`N3`) or municipality (`N6`) coverage, use a small helper script. Create `scripts/ingest_cov_tables.py`:
```python
import asyncio
import httpx
from sidra_database import ingest_agregado

BASE_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"
TARGET_CODES = {"N3", "N6"}

async def fetch_ids() -> list[int]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(BASE_URL)
        resp.raise_for_status()
    ids = set()
    for survey in resp.json():
        for table in survey.get("agregados", []):
            levels = set(table.get("nivelTerritorial") or [])
            if levels & TARGET_CODES:
                ids.add(int(table["id"]))
    return sorted(ids)

async def main() -> None:
    ids = await fetch_ids()
    print(f"Ingesting {len(ids)} agregados with UF or municipality coverage")
    for idx in range(0, len(ids), 50):  # chunk to stay polite with the API
        batch = ids[idx:idx + 50]
        await asyncio.gather(*(ingest_agregado(table_id) for table_id in batch))

asyncio.run(main())
```
Run it:
```powershell
python scripts/ingest_cov_tables.py
```
This script:
- Pulls the public agregados catalog once.
- Filters for tables containing level `N3` or `N6`.
- Ingests in batches of 50 to balance throughput and rate limits.

### 5.3 Incremental Refreshes
Re-run the ingestion command for any table to update metadata and embeddings. Hashing avoids redundant embedding calls if nothing changed.

## 6. Exploring the Local Catalog

### 6.1 Listing Coverage
```powershell
# Show agregados flagged as having national municipal coverage
python -m sidra_database.cli list --requires-national-munis

# Require a minimum municipality count and sort by name
python -m sidra_database.cli list --min-municipalities 3000 --order-by name --limit 30
```
Each row prints `assunto`, `pesquisa`, municipality count, coverage flag, and the ingestion timestamp.

### 6.2 Semantic Search
```powershell
python -m sidra_database.cli search "população municipal" --types variable classification
```
- Embeds the query using LM Studio, retrieves stored vectors, scores by cosine similarity, and prints the top matches.
- Metadata lines include municipality counts and whether the table meets the national threshold.
- `--limit N` controls result count; `--types` narrows entity types (agregado, variable, classification, category); `--model` overrides the embedding model.

### 6.3 Programmatic Access
```python
from sidra_database import list_agregados, semantic_search_with_metadata

# All tables with national municipal coverage
tables = list_agregados(requires_national_munis=True)
for table in tables[:5]:
    print(table.id, table.nome, table.municipality_locality_count)

# Semantic discovery across variables
results = semantic_search_with_metadata(
    "agricultura familiar municípios",
    entity_types=["variable"],
    limit=10,
)
for match in results:
    print(match.title, match.metadata.get("municipality_locality_count"))
```
Use these helpers in notebooks, scripts, or services to build custom dashboards.

### 6.4 Direct SQL
Open the SQLite file (`SIDRA_DATABASE_PATH`) in your favourite browser. Example SQL queries:
```sql
-- All agregados with municipality coverage >= 5000
SELECT id, nome, municipio_locality_count
FROM agregados
WHERE municipality_locality_count >= 5000
ORDER BY municipality_locality_count DESC;

-- Variables within a high-coverage table
SELECT v.id, v.nome, v.unidade
FROM variables v
JOIN agregados a ON a.id = v.agregado_id
WHERE a.covers_national_municipalities = 1 AND a.id = 6579;
```

## 7. Testing & Maintenance
- Run the suite after making changes: `pytest -q` (one Pydantic deprecation warning remains until we move to `ConfigDict`).
- Linters: `ruff check src tests`, `mypy src`.
- Ingestion log (`ingestion_log`) records every run; use it to monitor bulk operations.

## 8. Troubleshooting

| Issue | Remedy |
|-------|--------|
| `pytest` not found | Activate the venv or install dev extras. |
| LM Studio request fails | Ensure the server is running and `SIDRA_EMBEDDING_API_URL` matches the host/port. |
| Slow ingestion (municipalities) | The API returns large payloads; consider increasing `--concurrent` cautiously or caching `localities` payloads between runs. |
| Missing coverage columns | `apply_schema` automatically adds new columns; rerun ingestion if you created the DB before upgrading. |
| Wrong coverage flag | Adjust `SIDRA_MUNICIPALITY_NATIONAL_THRESHOLD` to suit your definition of "national" coverage. |

## 9. Next Steps & Extensions
- Add CLI support for ingestion ranges or manifest files.
- Layer structured filters (period range, `assunto`, variable keywords) on top of the search command.
- Integrate FAISS or sqlite-vss if embedding volume grows and you need faster searches.
- Produce notebooks demonstrating end-to-end ingestion, coverage analysis, and semantic exploration for new team members.

With the pipeline in place, you can ingest the entire SIDRA catalog, isolate tables covering all UFs or municipalities, and explore them using natural-language queries backed by reliable metadata. Happy hunting.
