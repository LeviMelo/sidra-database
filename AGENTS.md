Here’s a drop-in **AGENTS.md** you can paste at the repo root. It documents the project end-to-end for a coding AI (CODEX). It’s written to reflect the **current** code you posted (module names, schemas, CLI, behavior).

---

# AGENTS.md

**Project:** `sidra_search`
**Goal:** fast, table-centric search over IBGE SIDRA “agregados” (tables) with robust textual + structural filters, fuzzy matching, and optional semantic (embedding) re-ranking.
**Non-goals right now:** changing visible behavior or data semantics. We accept optimizations and tooling, not feature drift.

> **Invariant:** We must preserve **exact locality membership** per table+level. We can optimize storage/ingest, but the system must still know exactly which localities each table supports.

---

## 1) High-level architecture

```
             ┌──────────────────────────┐
             │        CLI entry         │
             │  python -m sidra_search  │
             └────────────┬─────────────┘
                          │
                          ▼
                ┌───────────────────┐
                │       CLI         │  sidra_search/cli/__init__.py
                └─────┬─────────────┘
                      │ calls into
     ┌────────────────┴────────────────┐
     │                                 │
     ▼                                 ▼
┌───────────────┐               ┌───────────────┐
│  Ingestion     │               │    Search     │
│ (bulk + one)   │               │ (tables, FTS, │
│ ingest/*.py    │               │  fuzzy, cov)  │
└─────┬──────────┘               └─────┬─────────┘
      │                                │
      ▼                                ▼
┌───────────────┐               ┌───────────────┐
│   Net/API     │               │    DB Layer   │
│ net/api_client│<──────────────│ db/session    │ (SQLite)
└───────────────┘               └───────────────┘
        ▲                                  ▲
        │                                  │
        │                 ┌────────────────┴────────────┐
        │                 │  Search schema + base schema │
        │                 │  db/search_schema.py +       │
        │                 │  db/base_schema.py           │
        │                 └──────────────────────────────┘
        │
        ▼
┌────────────────┐
│ Embeddings API │  net/embedding_client.py  (optional)
└────────────────┘
```

* **CLI** provides commands: `db`, `ingest`, `ingest-coverage`, `build-links`, `search`, `embed-titles`.
* **Ingestion** fetches SIDRA metadata, periods, and localities; persists to SQLite; builds link indexes; refreshes FTS; optionally writes title embeddings.
* **Search** combines:

  * *Structural matching* via link tables (variables/classes/categories).
  * *Coverage constraints* via a tiny boolean expression parser over counts (e.g., `N6>=5000`).
  * *Title ranking* with FTS (lexical) and optional *semantic* re-ranking via embeddings.
* **DB** stores all raw and derived data. WAL + pragmas tuned for ingestion speed.

---

## 2) Domain terms (SIDRA)

* **Agregado**: a “table” in SIDRA (primary entity). Stored in `agregados`.
* **Variáveis / Classificações / Categorias**: facets within a table; stored in `variables`, `classifications`, `categories`.
* **Períodos**: time keys per table; stored in `periods`.
* **N-levels**: territorial coverage levels. In our code:

  * `N3` = **UFs (states)** (as per project convention in this codebase).
  * `N6` = **Municípios (municipalities)**.
  * Others may exist, but the coverage expressions and probing typically focus on `N3`, `N6`.
* **Localidades**: the concrete localities supported per table & level—**we persist exact membership** in `localities`.

---

## 3) Data model (SQLite)

### 3.1 Base schema (db/base_schema.py)

* `agregados(id PK, nome, pesquisa, assunto, url, freq, periodo_inicio, periodo_fim, raw_json, fetched_at, municipality_locality_count, covers_national_municipalities)`
* `agregados_levels(agregado_id, level_id, level_name, level_type, locality_count, PK(agregado_id,level_id,level_type))`
* `variables( (agregado_id,id) PK, nome, unidade, sumarizacao JSON text, text_hash )`
* `classifications( (agregado_id,id) PK, nome, sumarizacao_status, sumarizacao_excecao )`
* `categories( (agregado_id,classification_id,categoria_id) PK, nome, unidade, nivel, text_hash )`
* `periods( (agregado_id,periodo_id) PK, literals JSON text, modificacao, periodo_ord, periodo_kind )`
* `localities( (agregado_id,level_id,locality_id) PK, nome )`
* `ingestion_log(id, agregado_id, stage, status, detail, run_at)`

**Indexes**:

* `idx_variables_agregado`, `idx_categories_agregado`,
  `idx_localities_agregado`,
  `u_agregados_levels_pair`,
  `idx_periods_agregado_ord`.

### 3.2 Search schema (db/search_schema.py)

* `name_keys(kind, key, raw)` — normalized keys for var/class/cat (kind ∈ {var,class,cat}).
* Link tables:

  * `link_var(var_key, table_id, variable_id)`
  * `link_class(class_key, table_id, class_id)`
  * `link_cat(class_key, cat_key, table_id, class_id, category_id)`
  * `link_var_class(var_key, class_key, table_id, variable_id, class_id)`
* `table_titles_fts` — FTS5 over (title, survey, subject).
* `embeddings(entity_type, entity_id, agregado_id, text_hash, model, dimension, vector, created_at)` — for title embeddings.

**Indexes**:

* On link keys (var/class/cat, pairs), on `embeddings(agregado_id, model)`.

**Invariants / guarantees**

* **Exact locality membership** per (table_id, level_id) lives in `localities`.
* `agregados_levels` stores **counts** per level; counts are used by coverage expressions.
* Link tables are **idempotently** rebuilt per table ingestion.

---

## 4) Settings & environment (config.py)

* **HTTP to SIDRA**

  * `SIDRA_SEARCH_SIDRA_BASE_URL` (default: `https://servicodados.ibge.gov.br/api/v3/agregados`)
  * `SIDRA_SEARCH_REQUEST_TIMEOUT` (float sec), `SIDRA_SEARCH_REQUEST_RETRIES` (int)
  * `SIDRA_SEARCH_USER_AGENT`
* **DB**

  * `SIDRA_DATABASE_PATH` (path to sqlite file; default `./sidra.db`)
  * `SIDRA_SEARCH_DATABASE_TIMEOUT`
  * `SIDRA_SEARCH_MUNICIPALITY_NATIONAL_THRESHOLD` (default 5000)
* **Embeddings (optional)**

  * `SIDRA_SEARCH_ENABLE_TITLE_EMBEDDINGS` (bool)
  * `SIDRA_SEARCH_EMBEDDING_API_URL` (LM Studio-style `/v1/embeddings`)
  * `SIDRA_SEARCH_EMBEDDING_MODEL` (model name)

---

## 5) Networking & clients

### 5.1 SIDRA API (net/api_client.py)

* Async `httpx.AsyncClient` with bounded connection pool and explicit timeouts.
* **Retry** (tenacity) for transport errors and 5xx/429.
* Endpoints used:

  * `/{id}/metadados` → table metadata (title, survey, subject, territorial levels…)
  * `/{id}/periodos` → periods list
  * `/{id}/localidades/{level}` → list of localities for a given level (e.g., N6)
  * Catalog root `""` with query params: `assunto=`, `periodicidade=`, `nivel=` → discovery

### 5.2 Embeddings API (net/embedding_client.py)

* Simple sync client posting to `/v1/embeddings` (LM Studio compatible).
* Used for **title embeddings** only (semantic re-ranking in search).

---

## 6) Ingestion flow

### 6.1 Single table (ingest/ingest_table.py)

1. **Fetch** `metadados` and `periodos`.
2. From metadata `nivelTerritorial`, for each `(level_type → [codes])`:

   * For each **code** (e.g., `N3`, `N6`), **fetch localities** (or reuse prefetched cache from bulk probe).
   * Write **counts** to `agregados_levels` and **membership** rows to `localities`.
   * Track municipality count for `covers_national_municipalities` flag (`>= threshold`).
3. Persist `variables`, `classifications`, `categories`, `periods`.
4. Refresh **FTS** row in `table_titles_fts` (title/survey/subject).
5. Build **link tables** (`build_links_for_table`):

   * Insert normalized names into `name_keys`.
   * Populate `link_var`, `link_class`, `link_cat`, and `link_var_class` (cross-product fallback).
6. Optionally compute **title embeddings**; upsert into `embeddings`.
7. Log to `ingestion_log`.

**Idempotency:** the function DELETEs child rows and re-INSERTs fresh rows per table.

### 6.2 Coverage-driven bulk discovery (ingest/bulk.py)

`ingest_by_coverage(coverage, subject_contains?, survey_contains?, limit?, concurrency, probe_concurrent?)`

1. **Parse coverage** expression (see §7).
2. **Fetch catalog** pruned by mentioned levels (server side).
3. Optional **survey substring** filter (cheap).
4. Optional **subject substring** filter with metadata fallback:

   * Try DB (“already ingested”).
   * Else check catalog’s subject.
   * Else fetch `/metadados` for candidates (bounded concurrency; early-stop if `limit` reached).
5. **Probe coverage**:

   * For each candidate, check counts for the levels that appear in the coverage expression.
   * Try **DB counts first** (if ingested); if missing, **call `/localidades/{level}`** and compute counts.
   * Evaluate coverage expression; **prefetch** the raw locality payloads for accepted tables (for reuse by `ingest_table`).
   * Concurrency is bounded; early-stop when `limit` hits.
6. **Schedule ingestion**:

   * New accepted tables not in `agregados`.
   * **Plus** historical failures from `ingestion_log` with `status='error'` (not already ingested).
7. **Ingest workers** run with short **per-table retries** and small backoffs. Hard 500s are allowed to “fail-fast” after 2 attempts. Other errors exhaust 3 attempts.

**Progress logs**:

* `[probe] candidates=..., levels=[...], parallel=...`
* During probe: `\r[probe] done=X/Y (accepted=K • P%) | inflight=M`
* During subject filter: `\r[subject] checked=X/Y kept=K (P%)`
* During ingest: `ingested {id}` or `failed {id}: ...`

---

## 7) Coverage expression language (search/coverage.py)

**Purpose:** Evaluate boolean constraints over **counts of localities** per level.

**Examples**

* `N6`                 (shorthand for `N6 >= 1`)
* `(N6>=5000) OR N3`
* `(N6<5000) AND (N3>=27 OR N3==27)`
* `NOT (N6>=5000)`

**Grammar (informal)**

```
expr   := and_expr ( OR and_expr )*
and_expr := unary ( AND unary )*
unary  := NOT unary | primary
primary:= '(' expr ')' | cmp
cmp    := ID (OP NUM)?        # OP ∈ <,<=,>,>=,==,!= ; '=' accepted as '=='
ID     := [A-Za-z_][A-Za-z0-9_]*
NUM    := [0-9]+
```

**Tokenizer** supports `AND/OR/NOT`, `&&` as AND, `||` as OR.
**Evaluation** maps `counts: {ID → int}` to boolean via the op table.

**Extracted levels**: `extract_levels(ast)` returns the set of IDs seen (e.g., `{N3,N6}`) to **prune catalog** and **probe only what’s necessary**.

---

## 8) Search flow (search/tables.py)

**Inputs:** `SearchArgs`

* `title` (FTS + optional embeddings)
* `vars`: list of variable names (free text)
* `classes`: list; each item `"Class"` or `"Class:Category"`
* `coverage`: same language as §7 (applied post structural filters)
* `limit`, `allow_fuzzy`, `var_th`, `class_th`, `semantic`, `debug_fuzzy`

**Steps**

1. **Expand** variable and class names with fuzzy (RapidFuzz) into normalized keys.

   * Variables → union of tables with any matching var.
   * Classes → per requested class, union inside the group, then **intersect across groups**.
   * `Class:Category` → class is fuzzy; **category is strict**.
   * If both var and classes, the pipeline intersects accordingly.
2. If no structural inputs, **start with all ingested tables**.
3. **Coverage filter**: read `agregados_levels` counts per table, evaluate expression.
4. **Title ranking**:

   * **Lexical** FTS (`table_titles_fts`) → a rank list.
   * **Semantic** (optional): cosine between query embedding and title embedding; rank list.
   * **RRF** merge of the rank lists → `rrf_score`.
5. **Structural score** blends best matching var with best per-group class keys (exact matches get small boosts).
6. **Final score** = `0.75 * structural + 0.25 * rrf`.
7. Return top `limit` tables, including `why`, `period_start/end`, `N3/N6` counts.

---

## 9) CLI commands (cli/**init**.py)

> Program: `sidra-search` (module: `python -m sidra_search.cli`)

* `db migrate`
  Create/ensure base+search schema.

* `db stats`
  Print table row counts.

* `ingest {id...}`
  Ingest specific table IDs (async loop).

* `ingest-coverage --coverage "<expr>" [--survey-contains STR] [--subject-contains STR] [--limit N] [--concurrent N] [--probe-concurrent N]`
  Discover via coverage, filter by survey/subject strings, ingest.

* `build-links [--all] [ids…]`
  Rebuild link indexes for tables.

* `search [options]`
  Options:

  * `-t/--title TEXT` (FTS; add `--semantic` to use embeddings)
  * `--var NAME` (repeatable)
  * `--class NAME` or `--class "Class:Category"` (repeatable)
  * `--coverage "<expr>"`
  * `--limit N`
  * `--no-fuzzy`, `--var-th FLOAT`, `--class-th FLOAT`
  * `--semantic` (requires embeddings)
  * `--explain`, `--json`, `--show-classes`, `--debug-fuzzy`

* `embed-titles [--model NAME] [--only-missing] [--limit N]`
  Backfill/upsert title embeddings, idempotent.

**Windows vs Bash env note**

* PowerShell: `$env:SIDRA_SEARCH_ENABLE_TITLE_EMBEDDINGS="1"`
* Bash: `export SIDRA_SEARCH_ENABLE_TITLE_EMBEDDINGS=1`

---

## 10) Performance notes

* **HTTP client** has bounded connections; per-request timeouts; retry for transient issues.
* **SQLite** is tuned (WAL, NORMAL sync, busy_timeout, larger page cache, in-memory temp).
* **Bulk probe**: bounded concurrency; **early-stop** when enough accepted; **prefetch** locality payloads to avoid re-fetch in ingestion.
* **Ingestion**: child tables are bulk-inserted with `executemany`.
* **Embeddings**: computed **off-path** (ingestion never blocks on embedding failure).

---

## 11) Error handling / retry strategy

* **Discovery probe**: soft-fails on locality fetch; just treats counts as 0.
* **Ingestion workers**: per-table retries `(1s, 3s, 7s)`; short-circuit after two 500-class failures that look permanent.
* **ingestion_log** keeps a record. Next runs of `ingest-coverage` will include **historical failures** (that are not ingested yet) in the **schedule**.

---

## 12) What CODEX may change vs must not change

**Must not change (behavioral invariants)**

* **Exact locality membership** must be persisted (`localities` table).
* Coverage expression **syntax and semantics**.
* Search scoring **blend** and structural logic.
* FTS surface (title/survey/subject as today).
* Link tables semantics and normalization steps.

**May change (optimizations / tooling)**

* Batch sizing, concurrency caps, progress logging clarity.
* Reusing probe payloads more aggressively; minimizing duplicate locality calls.
* SQLite pragmas (safe values), transaction boundaries, bulk insert strategies.
* Clearer CLI logs (e.g., `done/probed/accepted/inflight`).
* Non-semantic refactors (typing, docstrings, small helpers).

**Strong preferences**

* Keep code idempotent and safe to rerun.
* Favor **bounded concurrency** & backpressure over unbounded fan-out.
* Prefer **database reads** (counts) over network calls when possible.

---

## 13) Test/Run cookbook

**Quick DB sanity**

```bash
python -m sidra_search.cli db migrate
python -m sidra_search.cli db stats
```

**Discover + ingest a small slice**

```bash
python -m sidra_search.cli ingest-coverage \
  --coverage "N3 OR (N6>=5000)" \
  --survey-contains "Censo" \
  --limit 10
```

**Build link indexes for everything**

```bash
python -m sidra_search.cli build-links --all
```

**Search (lexical only)**

```bash
python -m sidra_search.cli search --q 'title~"frequência escolar indígena"' --limit 10 --explain
```

**Enable semantic**

```bash
# Start LM Studio (serving /v1/embeddings), then:
export SIDRA_SEARCH_ENABLE_TITLE_EMBEDDINGS=1
python -m sidra_search.cli embed-titles --only-missing
python -m sidra_search.cli search --q 'title~"tabela sobre sexo"' --semantic --limit 10
```

---

## 14) Known pitfalls / gotchas

* **Probe logging**: `done=… accepted=… inflight=…` counts are distinct; avoid misreading “probed” as “accepted”.
* **Windows env**: use PowerShell syntax, not `export`.
* **Permanent 500s** (e.g., certain tables) will stay in `failed`. Re-runs will not re-ingest them unless the API starts serving them again; they will appear in discovery but be skipped as “existing/failed”.
* **FTS**: ensure a row is inserted per table (ingestion does this); otherwise lexical title search won’t surface that table.
* **Embeddings off**: `--semantic` has no effect without embeddings populated.
* Use `python -m sidra_search.cli --manual` for a quick CLI refresher. When semantic ranking
  is requested but prerequisites are missing, the CLI now prints diagnostics instead of
  silently falling back to lexical-only results.
* **Unified query flag**: prefer `search --q '<expr>'` for all filters. Legacy `--title/--var/--class/--coverage`
  flags still work for now but only as a translation shim.
* **Text matching semantics**: `~` checks compare normalized, accent-stripped substrings. Internal
  prefilters on VAR/CLASS/CAT link keys stay exact for precision; the boolean AST still enforces
  the user-visible contains logic.
* **Facet semantics**: `TITLE` literals power lexical/semantic ranking; `SURVEY`/`SUBJECT`
  literals filter the catalog only. When both VAR and CLASS clauses are present, matches
  reflect actual `link_var_class` pairs, and `cat~"Class::Category"` requires the exact
  class/category combination.

---

## 15) Future hooks (not in scope for behavior change now)

* **Auto-search loop** (LLM agent): propose facets → run search → judge recall/precision → broaden/narrow → converge. Must call the *existing* `search` pipeline only; no new ranking logic.
* **Light ingest mode** (optional): persist only `agregados_levels` counts for search; fetch localities on demand when the user drills into a table. (Keep exact-membership invariant available via on-demand fetch + cache.)

---

## 16) File-by-file quick reference

* `__init__.py`: exposes schema helpers for external callers.
* `__main__.py`: `python -m sidra_search` → delegates to CLI main.
* `cli/__init__.py`: defines all subcommands; prints JSON for machine consumption where applicable.
* `config.py`: settings & env overrides.
* `db/base_schema.py`: core data tables & indexes.
* `db/search_schema.py`: search-specific tables: link keys, FTS, embeddings.
* `db/migrations.py`: versioned application of search schema.
* `db/session.py`: connection factory, pragmas, schema ensure.
* `ingest/ingest_table.py`: single-table ingestion; writes children; builds links; FTS; optional embeddings.
* `ingest/bulk.py`: coverage-driven discovery+probe; schedules ingestion; retries; progress logs.
* `ingest/links.py`: builds `name_keys`, `link_*`, cross-product fallback for var×class.
* `net/api_client.py`: async HTTP client to SIDRA with retries.
* `net/embedding_client.py`: sync embeddings client to LM Studio-style API.
* `search/coverage.py`: tokenizer, parser, evaluator for coverage expressions; level extraction.
* `search/normalize.py`: normalization (lowercase, accent strip, hyphen handling).
* `search/fuzzy3gram.py`: RapidFuzz expansions for var/class names; in-RAM corpus; reset_cache().
* `search/tables.py`: the main search pipeline (structural filters, coverage, title ranking, scoring).
* `search/title_rank.py`: RRF merge utility.
* `smoke.sh`: end-to-end sanity script (bash).

---

## 17) Acceptance checklist for CODEX changes

* [ ] No change in **public CLI arguments** behavior (except clearer logs).
* [ ] No change in **coverage semantics** or grammar.
* [ ] No loss of **exact locality membership** persistence.
* [ ] Idempotent re-runs; “failed” tables handled per current strategy.
* [ ] Pylance clean; type hints preserved or improved.
* [ ] Unit/integration smoke via `smoke.sh` passes.
* [ ] All schema changes (if any) gated by migrations version bump; **prefer none**.

---

**End of AGENTS.md**
