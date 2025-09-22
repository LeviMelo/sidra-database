# SIDRA Database

Python toolkit for building a local SQLite metadata store and search layer for IBGE SIDRA agregados.

## Features
- Asynchronous client for the aggregated-data v3 API.
- SQLite schema and ingestion pipeline capturing tables, variables, classifications, periods, and localities.
- LM Studio embedding client for local vector generation.
- CLI helper to ingest one or more agregados.
- Pytest, Ruff, and MyPy configuration for development.

## Quickstart
```bash
# Install in editable mode with dev extras
pip install -e .[dev]

# Ingest one or more agregados into SQLite
python -m sidra_database.cli ingest 6579 2093

# Run a semantic search against stored embeddings
python -m sidra_database.cli search "população por sexo" --types variable classification

# Run tests and linters
pytest
ruff check src tests
mypy src
```

Set `SIDRA_DATABASE_PATH` to control the SQLite location and `SIDRA_EMBEDDING_API_URL` if LM Studio runs on a different port. The CLI search command will call the configured LM Studio embedding endpoint to embed the query and return enriched metadata for each match.
