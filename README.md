# SIDRA Database

Python toolkit for building a local SQLite metadata store and search layer for IBGE SIDRA agregados.

## Features
- Asynchronous client for the aggregated-data v3 API.
- SQLite schema and ingestion pipeline capturing tables, variables, classifications, periods, and localities.
- LM Studio embedding client for local vector generation.
- Coverage-aware catalog of agregados, including municipality counts and national coverage flags.
- CLI helper to ingest, inspect, and semantically search agregados.
- Pytest, Ruff, and MyPy configuration for development.

## Quickstart
```bash
# Install in editable mode with dev extras
pip install -e .[dev]

# Ingest one or more agregados into SQLite
python -m sidra_database.cli ingest 6579 2093

# Run a semantic search against stored embeddings
python -m sidra_database.cli search "população por sexo" --types variable classification

# List agregados covering most municipalities
python -m sidra_database.cli list --requires-national-munis

# Ingest every table with UF or municipality coverage
python scripts/ingest_by_coverage.py

# Run tests and linters
pytest
ruff check src tests
mypy src
```

Configuration values can be supplied via a simple JSON file (`sidra.config.json`) or environment variables. Copy `sidra.config.json.example` to `sidra.config.json` and adjust values as needed. The CLI search command will call the configured LM Studio embedding endpoint to embed the query and return enriched metadata for each match. Use the `list` subcommand to quickly surface agregados whose municipality coverage exceeds the configured national threshold (`SIDRA_MUNICIPALITY_NATIONAL_THRESHOLD`, default 4000).
