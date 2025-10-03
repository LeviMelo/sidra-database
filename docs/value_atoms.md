# Value Atom (VA) Search System

The Value Atom subsystem (`sidra_va`) provides a value-centric index over the
IBGE SIDRA metadata already ingested by the `sidra_database` package. A **Value
Atom (VA)** is the smallest slice of a table that identifies a variable, an
optional set of classification categories, the territorial coverage, the period
range and contextual metadata (survey, subject, table title).

```
VA := (
  agregado_id,
  variable_id,
  variable_name,
  unit,
  dims = [(classification_id, classification_name, category_id, category_name), ...],
  supported_levels (N1..N15),
  period_start, period_end,
  survey, subject, table_title
)
```

The project keeps the existing ingestion pipeline untouched. The new package
lives alongside the original modules and can be adopted incrementally.

## Getting Started

1. Ingest SIDRA metadata with the existing CLI (unchanged).
2. Run the VA migration and build the VA index:

```bash
python -m sidra_va.cli db migrate
# prints: "VA schema version: 1"
python -m sidra_va.cli db stats
python -m sidra_va.cli index build-va --all  # defaults to --concurrent 1 for safe writes
```

3. (Optional) Embed Value Atoms for semantic search:

```bash
python -m sidra_va.cli index embed-va --all --model your-model-name  # defaults to --concurrent 1
```

## Searching Value Atoms

Use the new CLI to issue hybrid lexical + semantic VA searches:

```bash
python -m sidra_va.cli search va "alfabetização indígena" \
  --require-level N6 \
  --period 2010-2020 \
  --json
```

The search pipeline uses SQLite FTS for lexical retrieval, optional embeddings
for semantic recall, Reciprocal Rank Fusion to combine candidate sets and a
structure-aware scorer that rewards exact matches on variables, units, classes,
categories, periods and territorial coverage.

Each result exposes a `why` explanation string so analysts can understand which
parts of the request matched.

## Managing Synonyms

Lexical normalization and fingerprinting leverage a simple synonym table. You
can import or export CSV files with:

```bash
python -m sidra_va.cli index synonyms import synonyms.csv
python -m sidra_va.cli index synonyms export current_synonyms.csv
```

The CSV must contain the columns `kind,key,alt` where `kind` is one of
`classification`, `category`, `variable` or `unit`.

## Neighbor Discovery (Concatenation)

To find compatible VAs across different tables (same variable/unit and matching
categories), use the `link` commands:

```bash
python -m sidra_va.cli link neighbors 6579::v123::c45:7
```

The neighbor score blends variable identity or fingerprint matches, unit
compatibility, dimension overlap, geographic coverage and period overlap. This
helps analysts identify value slices that can be concatenated even when variable
IDs differ between tables.

## Internals

* **Schema** – `sidra_va.schema_migrations.apply_va_schema` creates additive
  tables (`value_atoms`, `value_atom_dims`, `value_atoms_fts`, `synonyms`,
  `variable_fingerprints`) without altering the original schema.
* **Index Build** – `sidra_va.value_index` materializes variable-only and
  single-dimension VAs from the existing metadata, stores canonical textual
  representations and populates the FTS index.
* **Embeddings** – `sidra_va.embed.embed_vas_for_agregados` reuses the global
  `embeddings` table with `entity_type='va'`.
* **Search** – `sidra_va.search_va` implements lexical + semantic retrieval,
  Reciprocal Rank Fusion, structure-aware scoring and result explanations.
* **Compatibility** – `sidra_va.neighbors` compares fingerprints, units, dims
  and coverage to produce compatibility scores.

Two-dimension combinations are supported behind an optional flag to prevent
index explosion.

## Performance Notes

* The SQLite database lives at `sidra.db` in the project root by default. Set
  `SIDRA_DB_PATH=/path/to/custom.db` to target a different location.
* VA creation runs per table and is idempotent – rerunning updates texts and FTS
  content without duplication.
* Default CLI concurrency for write-heavy tasks is `1` to keep SQLite happy on
  Windows and network file systems. Increase cautiously with
  `--concurrent N` if your environment tolerates more writers.
* FTS keeps queries responsive even with many VAs. Rebuild the FTS table with
  `python -m sidra_va.cli index rebuild-fts` when needed.
* Embeddings are cached by hashing the VA text, avoiding unnecessary API calls.

Refer to the tests in `tests/test_va_*.py` for small, self-contained examples of
schema application, index building, searching and neighbor discovery.
