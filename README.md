
# sidra-search

Table-centric search and ingestion for **IBGE SIDRA** datasets (agregados).  
Ingest SIDRA table metadata into a local **SQLite** database, build fast name indexes (variables, classifications, categories), query with a **unified boolean language** (title/survey/subject + var/class/cat + coverage + periods), and print **concise “cards”** for tables via `show`.

> Works offline once tables are ingested. Optional **semantic title ranking** via an OpenAI-compatible embeddings endpoint.

---

## Contents

- [sidra-search](#sidra-search)
  - [Contents](#contents)
  - [Highlights](#highlights)
  - [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [CLI Overview](#cli-overview)
    - [db](#db)
    - [ingest](#ingest)
    - [ingest-coverage](#ingest-coverage)
    - [build-links](#build-links)
    - [embed-titles (optional)](#embed-titles-optional)
    - [search](#search)
    - [show](#show)
  - [Unified Query Language](#unified-query-language)
    - [Fields](#fields)
    - [Contains (`~`)](#contains-)
    - [Coverage (N-levels)](#coverage-n-levels)
    - [Periods](#periods)
    - [Examples](#examples)
  - [Output Examples](#output-examples)
  - [Architecture \& Design](#architecture--design)
    - [Database Schema (base)](#database-schema-base)
    - [Search Schema](#search-schema)
    - [Search Flow](#search-flow)
    - [Performance Notes](#performance-notes)
  - [Troubleshooting](#troubleshooting)
  - [FAQ](#faq)
  - [Development](#development)
    - [Project Layout](#project-layout)
    - [Coding Notes](#coding-notes)

## Highlights

- **Local ingest** of `/metadados`, `/periodos`, `/localidades/{N*}` into SQLite.
- **Fast link indexes** (`var`, `class`, `cat`, `var×class`) for filtering and matching.
- **Unified boolean query language**: `title~"...", var~"...", class~"...", cat~"...", N6>=5000, period in [2016..2024]`, etc.
- **Fuzzy matching** for `var` and `class` (RapidFuzz), tunable thresholds.
- **FTS on titles** (SQLite FTS5) for lexical ranking.
- **Semantic title ranking** (optional) blended via **RRF** with lexical scores.
- **Concise cards** via `show` (title, URL, coverage, full period list, variables, classifications, categories).

---

## Quick Start

```bash
# 0) Python ≥ 3.10 recommended
python -V

# 1) Create & activate a virtualenv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) Install dependencies
pip install httpx orjson tenacity rapidfuzz

# 3) Initialize the local DB (creates ./sidra.db by default)
python -m sidra_search db migrate

# 4) Ingest specific tables
python -m sidra_search ingest 10145 1758 1572

# 5) Build name links (var/class/cat) for all ingested tables
python -m sidra_search build-links --all

# 6) Search with the unified query language
python -m sidra_search search --q 'title~"pessoas" AND period in [2020..2025] AND subject~"educação" AND N6>=5000' --limit 20

# 7) Show concise cards
python -m sidra_search show 10145 1758 1572
````

> You can invoke either `python -m sidra_search ...` or `python -m sidra_search.cli ...` (both work).

---

## Installation

* **Dependencies:**

  ```
  httpx>=0.27.0
  orjson>=3.10.0
  tenacity>=8.3.0
  rapidfuzz>=3.6.1
  ```
* **Database file:** default is `./sidra.db`. Override with `SIDRA_DATABASE_PATH=/abs/path/sidra.db`.

---

## Environment Variables

All are optional; sensible defaults exist.

| Variable                                       | Purpose                                 | Default                                             |
| ---------------------------------------------- | --------------------------------------- | --------------------------------------------------- |
| `SIDRA_SEARCH_SIDRA_BASE_URL`                  | SIDRA API base                          | `https://servicodados.ibge.gov.br/api/v3/agregados` |
| `SIDRA_SEARCH_REQUEST_TIMEOUT`                 | HTTP timeout (sec)                      | `30.0`                                              |
| `SIDRA_SEARCH_REQUEST_RETRIES`                 | Retries for transient HTTP errors       | `3`                                                 |
| `SIDRA_SEARCH_USER_AGENT`                      | HTTP user agent                         | `sidra-search/0.1`                                  |
| `SIDRA_SEARCH_DATABASE_TIMEOUT`                | SQLite busy timeout (sec)               | `60.0`                                              |
| `SIDRA_SEARCH_MUNICIPALITY_NATIONAL_THRESHOLD` | N6 count considered “national coverage” | `5000`                                              |
| `SIDRA_SEARCH_EMBEDDING_API_URL`               | OpenAI-compatible embeddings endpoint   | `http://127.0.0.1:1234/v1/embeddings`               |
| `SIDRA_SEARCH_EMBEDDING_MODEL`                 | Embedding model name                    | `text-embedding-qwen3-embedding-0.6b@f16`           |
| `SIDRA_SEARCH_ENABLE_TITLES_FTS`               | Enable FTS search on titles             | `true`                                              |
| `SIDRA_SEARCH_ENABLE_TITLE_EMBEDDINGS`         | Enable semantic title ranking           | `true`                                              |
| `SIDRA_DATABASE_PATH`                          | SQLite file path                        | `./sidra.db`                                        |

> To leverage semantic title ranking, run an embeddings server that speaks the OpenAI embeddings API at `SIDRA_SEARCH_EMBEDDING_API_URL`, and run `embed-titles`.

---

## CLI Overview

Print a compact manual:

```bash
python -m sidra_search --manual
```

### db

```bash
# Create/upgrade base + search schemas
python -m sidra_search db migrate

# Row counts for key tables
python -m sidra_search db stats
```

### ingest

```bash
# Ingest one or more tables by ID (metadata + periods + localities)
python -m sidra_search ingest 10145 1758 1572
```

### ingest-coverage

Discover by **coverage** (boolean expression over N-level counts), optionally narrowed by survey/subject; then ingest.

```bash
python -m sidra_search ingest-coverage \
  --coverage "(N3 OR N6>=5000)" \
  --survey-contains "Censo" \
  --limit 25 \
  --concurrent 8 \
  --probe-concurrent 8
```

* Coverage expression examples: `N6>=5000`, `N3 OR (N6>=5000)`, `NOT N1`, `(N3 AND N6>=5000)`.
* Uses catalog level hints to minimize probing; subject narrowing can fall back to `/metadados` if needed (bounded concurrency, early-stop).

### build-links

Build (or rebuild) normalized name indexes and link tables.

```bash
# All ingested tables
python -m sidra_search build-links --all

# Or specific IDs
python -m sidra_search build-links 10145 1758
```

> Rebuilding resets the in-process fuzzy cache so new names are immediately visible.

### embed-titles (optional)

Compute/refresh embeddings of canonicalized table title text.

```bash
# Idempotent; only update missing by default with --only-missing
python -m sidra_search embed-titles --only-missing

# Force a limited batch
python -m sidra_search embed-titles --limit 200
```

### search

Unified boolean query over title/survey/subject, var/class/cat, coverage, and periods.

```bash
python -m sidra_search search --q 'title~"taxa" AND (N6>=5000)' --limit 20
# Extras:
#   --no-fuzzy             exact matching for var/class (disable RapidFuzz)
#   --var-th 0.74          fuzzy threshold for variables (0..1)
#   --class-th 0.78        fuzzy threshold for classes (0..1)
#   --semantic             blend semantic title ranking
#   --explain              show match rationale and scores
#   --show-classes         print up to 3 class names per hit
#   --json                 JSON output
```

> **Deprecated** flags `--title`, `--var`, `--class`, `--coverage`, `--survey-contains`, `--subject-contains` still work, but are internally translated into `--q`.

### show

Pretty, **concise** card for each table from the local DB.

```bash
python -m sidra_search show 10145 1758 1572
# Options:
#   --wrap WIDTH         wrap width (defaults to terminal width)
#   --max-vars N         variables to list inline (default 8)
#   --max-cats N         categories listed per classification (default 12)
#   --json               JSON instead of pretty output
```

---

## Unified Query Language

### Fields

* **Text fields** (normalized substring):

  * `TITLE`, `SURVEY`, `SUBJECT`, `VAR`, `CLASS`, `CAT`
* **Coverage counters**: `N1`, `N2`, `N3`, `N6`, `N14`, … (from `agregados_levels`)
* **Period**: `PERIOD` (year comparisons) / `period in [YYYY..YYYY]` (range overlap)

Logical operators: `AND`, `OR`, `NOT`, parentheses `(...)`.

### Contains (`~`)

* `title~"pessoas"` — accent/punctuation-insensitive substring match.
* `var~"taxa de alfabetização"`
* `class~"sexo"`

**Categories**:

* Loose: `cat~"Branca"` → any class that has category “Branca”
* Strict: `cat~"Cor ou raça::Branca"` → require that exact class/category pair

### Coverage (N-levels)

* Shorthand: bare `N3` means `N3 >= 1`
* Comparators: `>=`, `>`, `<=`, `<`, `==`, `!=`
* Combine with `AND/OR/NOT`, e.g. `(N3 OR N6>=5000) AND NOT N1`

### Periods

* **Range overlap**:
  `period in [2010..2019]` → table qualifies if **any** data point overlaps the range.
* **Comparators** (by year):
  `PERIOD >= 2016`, `PERIOD == 2022`, etc.

### Examples

```bash
# Education, 2020..2025, at least 5000 municipalities, title contains "pessoas"
python -m sidra_search search --q \
  'title~"pessoas" AND period in [2020..2025] AND subject~"educação" AND N6>=5000' \
  --limit 20

# Variables and classes (fuzzy enabled by default)
python -m sidra_search search --q 'var~"taxa de alfabetização" AND class~"sexo"' --limit 10

# Categories (loose and strict)
python -m sidra_search search --q 'cat~"Branca"' --limit 5
python -m sidra_search search --q 'cat~"Cor ou raça::Branca"' --limit 5

# Semantic fallback when lexical title contains fails
python -m sidra_search search --q 'title~"população etnia"' --semantic --limit 5 --explain
```

---

## Output Examples

Search (abridged):

```
10061: Pessoas de 18 anos ou mais de idade, por nível de instrução, segundo os grupos de idade, o sexo e a cor ou raça | 2022 | N3=27 N6=5570
10057: Pessoas de até 5 anos de idade que frequentavam escola ou creche, ... | 2022 | N3=27 N6=5570
...
```

Show:

```
[10145] População residente, total e diagnosticada com autismo, por sexo e grupo de idade
-----------------------------------------------------------------------------------------
Título: População residente, total e diagnosticada com autismo, por sexo e grupo de idade
URL: https://sidra.ibge.gov.br/tabela/10145
Pesquisa: Censo Demográfico
Assunto: Pessoas
Periodicidade: anual | 2022 (1 períodos)
Períodos: 2022
Cobertura: N3=27, N6=5570, municípios=5570, cobre_malha_municipal=nacional
Níveis: Administrativo=[N1, N2, N3, N6]
Variáveis (3):
  - [93] População residente (Pessoas) Σ[nivelTerritorial]
  - [13267] População residente diagnosticada com autismo (Pessoas) Σ[nivelTerritorial]
  - [13408] Percentual da população residente diagnosticada com autismo no total da população residente (%)
Classificações (2, categorias=25):
  • [2] Sexo — sumarização: on
     categorias: Homens [n1], Mulheres [n1], Total [n0]
  • [58] Grupo de idade — sumarização: on
     categorias: 0 a 4 anos [n1], 5 a 9 anos [n1], 10 a 14 anos [n1], … (+10)
```

Errors are explicit:

```
[5125] ERROR: table 5125 not found in local DB (ingest it first)
```

---

## Architecture & Design

### Database Schema (base)

* `agregados` — table header (id, **nome (title)**, pesquisa, assunto, URL, frequency, period start/end, raw JSON, timestamps, coverage hints)
* `agregados_levels` — per-level (N1, N2, N3, N6, …) counts + names
* `variables` — id, name, unit, summarization, text hash
* `classifications` — id, name, summarization flags/exceptions
* `categories` — per classification: category id, name, unit, level, text hash
* `periods` — full list of period IDs (with literals, kind, sortable ord)
* `localities` — exact membership for each level (N*, locality id, name)
* `ingestion_log` — success/error stages for observability

### Search Schema

* `name_keys` — normalized name registry (`var`, `class`, `cat`)
* `link_var` / `link_class` / `link_cat` — normalized name → (table, id)
* `link_var_class` — cross-product (or inferred) pairs for var×class
* `table_titles_fts` — FTS5 over (title, survey, subject)
* `embeddings` — generic store for title embeddings (entity_type='agregado')

### Search Flow

1. **Parse** `--q` to an AST (`search/where_expr.py`).
2. **Fuzzy expansions** for `var` and `class` (RapidFuzz), thresholds per flag.
3. **Cheap prefilters**: link tables, FTS, survey/subject substring scans.
4. **Bulk-load contexts** for candidate tables (batched SQL), including:

   * normalized vars/classes/cats, var×class map
   * coverage counts (N3/N6, …)
   * years (from `periods`)
   * title/survey/subject normalized
5. **Evaluate** the boolean query over the in-memory context (`search/where_eval.py`):

   * category logic supports loose & strict `Class::Category`
   * `PERIOD` comparisons and range overlap
6. **Rank**:

   * structural match score (var/class/cat/var×class + exact boosts)
   * **RRF** blend of lexical (FTS) and **semantic** title ranks

### Performance Notes

* SQLite in WAL mode; tuned PRAGMAs (`busy_timeout`, `cache_size`, `temp_store`, `mmap_size` where available).
* **Bulk context loading** avoids per-table query storms.
* Build links after ingest (`build-links`) to power prefilters & fuzzing.
* Re-ingesting safely replaces child rows within a transaction.

---

## Troubleshooting

* **“table … not found in local DB”**
  You must ingest first: `python -m sidra_search ingest <id>`.

* **429/500 from SIDRA**
  Requests are retried (exponential backoff). If persistent, retry later.

* **Semantic title ranking does nothing**

  * Ensure an embeddings server is reachable at `SIDRA_SEARCH_EMBEDDING_API_URL`.
  * Run: `python -m sidra_search embed-titles --only-missing`.
  * Use `--semantic` with your search.
  * Check that `SIDRA_SEARCH_ENABLE_TITLE_EMBEDDINGS=true`.

* **No results but you expected some**

  * Start by removing strict pieces, then add back:

    * Drop strict categories (`Class::Cat`) → try loose `cat~"Cat"`.
    * Relax fuzzy thresholds (`--var-th`, `--class-th`).
    * Try only `title~"..."` + `--semantic`.
    * Verify that you ingested the right tables and ran `build-links`.

---

## FAQ

**Q: What does `period in [A..B]` do?**
A: It matches tables that have **any** period overlapping `[A..B]`. It does **not** require full containment.

**Q: Are `--title/--var/--class/--coverage` supported?**
A: They’re accepted for convenience but **deprecated**; they’re converted into `--q` internally.

**Q: Do I need embeddings?**
A: No. All searches work lexically/fuzzily. Embeddings improve ranking or recover hits when `title~` literals are weak.

**Q: When to run `build-links`?**
A: After any ingestion that introduces new variables/classes/categories, or after clearing/rebuilding the DB.

---

## Development

### Project Layout

```
src/sidra_search/
  __init__.py
  __main__.py                 # makes `python -m sidra_search` work
  cli/                        # CLI entrypoints
    __init__.py
    __main__.py
  config.py                   # settings & env var loader
  db/
    __init__.py
    base_schema.py            # base SQLite tables
    migrations.py             # versioned search schema management
    search_schema.py          # search-specific tables (links, FTS, embeddings)
    session.py                # connection, PRAGMAs, WAL, helpers
  ingest/
    __init__.py
    bulk.py                   # coverage discovery + batch ingest orchestrator
    ingest_table.py           # per-table ingest (+ embeddings, FTS refresh)
    links.py                  # build (var/class/cat) link indexes
  net/
    __init__.py
    api_client.py             # HTTPX client, retries (tenacity)
    embedding_client.py       # OpenAI-compatible embeddings client
  search/
    __init__.py
    coverage.py               # coverage expression parser/eval (N-levels)
    fuzzy3gram.py             # fuzzy expansions for var/class
    normalize.py              # accent/punct normalization
    tables.py                 # main search pipeline & ranking
    title_rank.py             # RRF util
    where_eval.py             # boolean evaluator over table context
    where_expr.py             # unified query parser/AST
  util/
    __init__.py               # small helpers (time/hash)
```

### Coding Notes

* All DB writes are within transactions; child rows are replaced on ingest.
* FTS refresh (`table_titles_fts`) happens on every ingest.
* Embeddings are optional; failures don’t abort ingestion.
* `build-links` clears and rebuilds link tables, then resets the in-RAM fuzzy cache.
* SQLite connection uses WAL, tuned timeouts, large page cache, and in-memory temp store for speed.

---